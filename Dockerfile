FROM python:3.11-slim

RUN useradd -ms /bin/bash appuser
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir ccxt rich python-dotenv pyyaml

# Copy code cần thiết (KHÔNG copy .env vào image)
COPY botV1.py /app/botV1.py
COPY telegram_manager.py /app/telegram_manager.py

# Tạo thư mục log và set quyền cho toàn bộ /app
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

ENV TZ=Etc/UTC \
    PYTHONUNBUFFERED=1
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10s \
  CMD python -c "import os; print('Health OK' if os.path.exists('/app/logs') else exit(1))" || exit 1

CMD ["python", "-u", "/app/botV1.py"]