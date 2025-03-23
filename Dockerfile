# Stage 1: build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Copy only the requirements file to leverage cache
COPY requirements.txt /app/

# Install Python dependencies in a temporary directory
RUN pip install --upgrade pip && \
    pip install --target=/install -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Stage 2: final stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed dependencies from the builder stage
COPY --from=builder /install /usr/local/lib/python3.11/site-packages

# Copy application code
COPY . /app/

EXPOSE 8000 8001

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
