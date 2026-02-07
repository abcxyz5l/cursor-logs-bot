# Railway-ready image: Python + unzip, 7z, unrar for max compatibility
FROM python:3.11-slim

WORKDIR /app

# Install unzip + 7z (Linux): 7z handles zip/7z/rar; unzip for fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    p7zip-full \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bot token: set BOT_TOKEN in Railway dashboard (or keep in 1.py)
CMD ["python", "1.py"]
