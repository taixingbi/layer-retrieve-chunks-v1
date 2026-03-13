FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure logs directory exists for Promtail
RUN mkdir -p /app/logs

ENV LOG_JSON=true
ENV LOG_FILE=/app/logs/app.log

EXPOSE 8000

CMD ["python", "main.py"]
