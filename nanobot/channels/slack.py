"""Slack channel implementation using Socket Mode."""

import asyncio
import re
from typing import Any
import os
import aiohttp
from pathlib import Path

from loguru import logger
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.socket_mode.websockets import SocketModeClient
from slack_sdk.web.async_client import AsyncWebClient
from slackify_markdown import slackify_markdown

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import SlackConfig


class SlackChannel(BaseChannel):
    """Slack channel using Socket Mode."""

    name = "slack"

    def __init__(self, config: SlackConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: SlackConfig = config
        self._web_client: AsyncWebClient | None = None
        self._socket_client: SocketModeClient | None = None
        self._bot_user_id: str | None = None

    async def start(self) -> None:
        """Start the Slack Socket Mode client."""
        if not self.config.bot_token or not self.config.app_token:
            logger.error("Slack bot/app token not configured")
            return
        if self.config.mode != "socket":
            logger.error("Unsupported Slack mode: {}", self.config.mode)
            return

        self._running = True

        self._web_client = AsyncWebClient(token=self.config.bot_token)
        self._socket_client = SocketModeClient(
            app_token=self.config.app_token,
            web_client=self._web_client,
        )

        self._socket_client.socket_mode_request_listeners.append(self._on_socket_request)

        # Resolve bot user ID for mention handling
        try:
            auth = await self._web_client.auth_test()
            self._bot_user_id = auth.get("user_id")
            logger.info("Slack bot connected as {}", self._bot_user_id)
        except Exception as e:
            logger.warning("Slack auth_test failed: {}", e)

        logger.info("Starting Slack Socket Mode client...")
        await self._socket_client.connect()

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the Slack client."""
        self._running = False
        if self._socket_client:
            try:
                await self._socket_client.close()
            except Exception as e:
                logger.warning("Slack socket close failed: {}", e)
            self._socket_client = None

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Slack."""
        if not self._web_client:
            logger.warning("Slack client not running")
            return
        try:
            slack_meta = msg.metadata.get("slack", {}) if msg.metadata else {}
            thread_ts = slack_meta.get("thread_ts")
            channel_type = slack_meta.get("channel_type")
            # Only reply in thread for channel/group messages; DMs don't use threads
            use_thread = thread_ts and channel_type != "im"
            thread_ts_param = thread_ts if use_thread else None

            if msg.content:
                await self._web_client.chat_postMessage(
                    channel=msg.chat_id,
                    text=self._to_mrkdwn(msg.content),
                    thread_ts=thread_ts_param,
                )

            for media_path in msg.media or []:
                try:
                    await self._web_client.files_upload_v2(
                        channels=[msg.chat_id],
                        file=media_path,
                        thread_ts=thread_ts_param,
                    )
                except Exception as e:
                    logger.error("Failed to upload file {}: {}", media_path, e)
        except Exception as e:
            logger.error("Error sending Slack message: {}", e)
    
    async def _download_file(self, url: str, filename: str) -> str | None:
        """Download a private Slack file to a local temp folder."""
        if not self.config.bot_token:
            return None
            
        # Create a temp directory for downloads
        download_dir = Path("/home/ubuntu/.nanobot/workspace/download")
        download_dir.mkdir(parents=True, exist_ok=True)
        
        local_path = download_dir / filename
        
        headers = {"Authorization": f"Bearer {self.config.bot_token}"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        with open(local_path, "wb") as f:
                            f.write(await resp.read())
                        return str(local_path)
                    else:
                        logger.error(f"Failed to download file: {resp.status}")
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
        return None
        
    async def _on_socket_request(
        self,
        client: SocketModeClient,
        req: SocketModeRequest,
    ) -> None:
        """Handle incoming Socket Mode requests."""
        if req.type != "events_api":
            return

        # Acknowledge immediately
        await client.send_socket_mode_response(
            SocketModeResponse(envelope_id=req.envelope_id)
        )

        payload = req.payload or {}
        event = payload.get("event") or {}
        event_type = event.get("type")

        # 1. Allow 'message' and 'app_mention'
        if event_type not in ("message", "app_mention"):
            return

        # 2. Allow 'file_share' subtype
        subtype = event.get("subtype")
        if subtype and subtype != "file_share":
            return

        sender_id = event.get("user")
        chat_id = event.get("channel")
        
        # Filter out bot's own messages
        if self._bot_user_id and sender_id == self._bot_user_id:
            return

        text = event.get("text") or ""
        
        # 3. Process Files (Download & Append to Text)
        files = event.get("files")
        if files:
            logger.info(f"files detected: {len(files)}")
            downloaded_paths = []
            
            for f in files:
                url = f.get("url_private")
                name = f.get("name")
                if url and name:
                    logger.info(f"Downloading {name}...")
                    local_path = await self._download_file(url, name)
                    if local_path:
                        downloaded_paths.append(local_path)
            
            # Append local paths to the text so the Agent knows where to find them
            if downloaded_paths:
                file_msg = "\n".join([f"User uploaded file: {path}" for path in downloaded_paths])
                text = f"{text}\n{file_msg}".strip()

        # ... (Rest of standard checks) ...
        if not sender_id or not chat_id:
            return
            
        channel_type = event.get("channel_type") or ""
        if not self._is_allowed(sender_id, chat_id, channel_type):
            return

        # Strip bot mention
        text = self._strip_bot_mention(text)

        # Threading logic
        thread_ts = event.get("thread_ts")
        if self.config.reply_in_thread and not thread_ts:
            thread_ts = event.get("ts")

        # Session Key
        session_key = f"slack:{chat_id}:{thread_ts}" if thread_ts and channel_type != "im" else None

        try:
            await self._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=text,
                metadata={
                    "slack": {
                        "event": event,
                        "thread_ts": thread_ts,
                        "channel_type": channel_type,
                    },
                },
                session_key=session_key,
            )
        except Exception:
            logger.exception("Error handling Slack message from {}", sender_id)

    def _is_allowed(self, sender_id: str, chat_id: str, channel_type: str) -> bool:
        if channel_type == "im":
            if not self.config.dm.enabled:
                return False
            if self.config.dm.policy == "allowlist":
                return sender_id in self.config.dm.allow_from
            return True

        # Group / channel messages
        if self.config.group_policy == "allowlist":
            return chat_id in self.config.group_allow_from
        return True

    def _should_respond_in_channel(self, event_type: str, text: str, chat_id: str) -> bool:
        if self.config.group_policy == "open":
            return True
        if self.config.group_policy == "mention":
            if event_type == "app_mention":
                return True
            return self._bot_user_id is not None and f"<@{self._bot_user_id}>" in text
        if self.config.group_policy == "allowlist":
            return chat_id in self.config.group_allow_from
        return False

    def _strip_bot_mention(self, text: str) -> str:
        if not text or not self._bot_user_id:
            return text
        return re.sub(rf"<@{re.escape(self._bot_user_id)}>\s*", "", text).strip()

    _TABLE_RE = re.compile(r"(?m)^\|.*\|$(?:\n\|[\s:|-]*\|$)(?:\n\|.*\|$)*")
    _CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
    _INLINE_CODE_RE = re.compile(r"`[^`]+`")
    _LEFTOVER_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
    _LEFTOVER_HEADER_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
    _BARE_URL_RE = re.compile(r"(?<![|<])(https?://\S+)")

    @classmethod
    def _to_mrkdwn(cls, text: str) -> str:
        """Convert Markdown to Slack mrkdwn, including tables."""
        if not text:
            return ""
        text = cls._TABLE_RE.sub(cls._convert_table, text)
        return cls._fixup_mrkdwn(slackify_markdown(text))

    @classmethod
    def _fixup_mrkdwn(cls, text: str) -> str:
        """Fix markdown artifacts that slackify_markdown misses."""
        code_blocks: list[str] = []

        def _save_code(m: re.Match) -> str:
            code_blocks.append(m.group(0))
            return f"\x00CB{len(code_blocks) - 1}\x00"

        text = cls._CODE_FENCE_RE.sub(_save_code, text)
        text = cls._INLINE_CODE_RE.sub(_save_code, text)
        text = cls._LEFTOVER_BOLD_RE.sub(r"*\1*", text)
        text = cls._LEFTOVER_HEADER_RE.sub(r"*\1*", text)
        text = cls._BARE_URL_RE.sub(lambda m: m.group(0).replace("&amp;", "&"), text)

        for i, block in enumerate(code_blocks):
            text = text.replace(f"\x00CB{i}\x00", block)
        return text

    @staticmethod
    def _convert_table(match: re.Match) -> str:
        """Convert a Markdown table to a Slack-readable list."""
        lines = [ln.strip() for ln in match.group(0).strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            return match.group(0)
        headers = [h.strip() for h in lines[0].strip("|").split("|")]
        start = 2 if re.fullmatch(r"[|\s:\-]+", lines[1]) else 1
        rows: list[str] = []
        for line in lines[start:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            cells = (cells + [""] * len(headers))[: len(headers)]
            parts = [f"**{headers[i]}**: {cells[i]}" for i in range(len(headers)) if cells[i]]
            if parts:
                rows.append(" · ".join(parts))
        return "\n".join(rows)

