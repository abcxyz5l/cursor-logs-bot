# Railway-ready image: Python + unzip, 7z, unrar for max compatibility
FROM python:3.11-slim

WORKDIR /app

# Enable non-free for unrar (needed for RAR extraction)
RUN echo "deb http://deb.debian.org/debian bookworm non-free non-free-firmware" >> /etc/apt/sources.list.d/non-free.list
# Install unzip + 7z + unrar so RAR always works (7z alone often fails on RAR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    p7zip-full \
    unrar \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bot token: set BOT_TOKEN in Railway dashboard (or keep in 1.py)
CMD ["python", "1.py"]
