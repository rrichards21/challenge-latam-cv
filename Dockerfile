# STAGE 1: BUILDER - Installs Dependencies
FROM python:3.9-slim as builder

# Set env vars
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for compiling C-extensions (like those in OpenCV/PyTorch)
# We ensure we clean up apt cache immediately to keep this layer small
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/* \
    libgl1

# Create a clean virtual environment
RUN python -m venv /opt/venv
# Add venv to PATH so pip/python commands use the venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install in one layer for cache efficiency
COPY requirements.txt .
# Use --no-cache-dir to prevent caching installation files
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN find /opt/venv/ -type f -name "*.pyc" -delete && \
find /opt/venv/ -type d -name "__pycache__" -delete && \
rm -rf /opt/venv/lib/python3.9/site-packages/pip \
/opt/venv/lib/python3.9/site-packages/pip-*.dist-info && \
rm -rf /opt/venv/lib/python3.9/site-packages/*.dist-info/RECORD \
/opt/venv/lib/python3.9/site-packages/wheel \
/opt/venv/lib/python3.9/site-packages/setuptools \
/opt/venv/lib/python3.9/site-packages/numpy/doc \
/opt/venv/lib/python3.9/site-packages/scipy/doc \
/opt/venv/lib/python3.9/site-packages/setuptools-*.dist-info && \
rm -rf /root/.cache/pip

RUN rm -rf /opt/venv/lib/python3.9/site-packages/torch/lib/tmp
# STAGE 2: RUNNER (Final Production Image)
# Use the same base image for binary compatibility, but start fresh
FROM python:3.9-slim as runner

WORKDIR /app

# Install ONLY runtime system dependencies
# libgl1 is required by OpenCV (which Ultralytics/YOLO needs for image handling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy the Model Artifacts (ONNX, PT, and classes.json)
COPY challenge/artifacts/ /app/artifacts/

# Copy the Application Code
COPY challenge/api.py .

# 🔐 Security Best Practice: Create and switch to a non-root user
RUN useradd -m appuser
USER appuser

# Cloud Run expects the application to listen on the $PORT environment variable
ENV PORT=8080

# Command to run the application using the installed uvicorn package
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]