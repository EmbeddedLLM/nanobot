# Personal Install Notes

```bash
# Base install
uv tool install -e "/home/ubuntu/git/nanobot"

# With Matrix support
uv tool install --reinstall -e "/home/ubuntu/git/nanobot[matrix]"
```

> Use `--reinstall` when adding new extras to an existing install.