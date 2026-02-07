# Deploy this bot on Railway

1. **Push this folder to GitHub** (or connect your repo to Railway).

2. **In [Railway](https://railway.com):**
   - New Project → Deploy from GitHub repo.
   - Select this repo/folder.
   - Railway will detect the **Dockerfile** and build the image (installs `unzip`, `p7zip-full`, Python deps).

3. **Set the bot token (optional but recommended):**
   - In your Railway project → your service → **Variables**.
   - Add: `BOT_TOKEN` = `your_telegram_bot_token`.
   - If you don’t set it, the bot uses the token hardcoded in `1.py`.

4. **Deploy:**
   - Start command is already in the Dockerfile: `python 1.py`.
   - No web port needed (this is a Telegram polling bot).

5. **Included for max speed and compatibility:**
   - **Unzip** + **7-Zip** (`p7zip-full`) installed so the bot can extract ZIP, 7Z, and RAR.
   - Download chunk size set to 64 MB for faster downloads.
   - Bot token can be overridden with `BOT_TOKEN` env var.
