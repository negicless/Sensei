FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for WeasyPrint (font & rendering deps)
RUN apt-get update && apt-get install -y     curl wget gnupg     libffi-dev libcairo2 pango1.0-tools libpango-1.0-0 libgdk-pixbuf2.0-0     libxml2 libxslt1.1 shared-mime-info fonts-noto-cjk     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright chromium
RUN python -m playwright install chromium

COPY . .

CMD ["python", "-m", "bot.main"]
