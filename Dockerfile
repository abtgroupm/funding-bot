FROM python:3.11-slim

RUN useradd -ms /bin/bash appuser
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir ccxt rich python-dotenv pyyaml

COPY bot.py /app/bot.py

RUN mkdir -p /app/logs && chown -R appuser:appuser /app

ENV TZ=Etc/UTC
USER appuser

# HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
#   CMD python -c "import ccxt; print('ok')" || exit 1

# CMD ["python", "/app/bot.py"]
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10s \
  CMD python -c "import os; print('Health OK' if os.path.exists('/app/logs') else exit(1))" || exit 1

CMD ["python", "/app/bot.py"]