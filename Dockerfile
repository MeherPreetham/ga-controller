# 1. Base image
FROM python:3.11-slim

# 2. Set working dir
WORKDIR /app

# 3. Copy & install dependencies (no pip cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the application code
COPY app.py .

# 5. Create and switch to a non-root user
RUN adduser --system --group appuser \
 && chown -R appuser:appuser /app
USER appuser

# 6. Expose FastAPI port (non-privileged)
EXPOSE 8000

# 7. Start Uvicorn  on port 8000 with a single worker
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
