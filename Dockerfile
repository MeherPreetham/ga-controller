# ga-controller/Dockerfile
FROM python:3.10-slim

# 2) Set working directory
WORKDIR /app

# 3) Copy & install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# 4) Copy service code
COPY app.py .

# 5) Expose ports:
#    - 80   → FastAPI HTTP API
#    - 8000 → Prometheus metrics (start_http_server)
EXPOSE 80 8000

# 6) Run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
