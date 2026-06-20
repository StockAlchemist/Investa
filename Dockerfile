FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src /app/src
COPY data /app/data

# Expose the port
EXPOSE 8000

# Run the application
CMD ["python", "src/server/main.py"]
