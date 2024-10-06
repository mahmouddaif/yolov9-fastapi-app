# Dockerfile

# Default to CPU base image
ARG BASE_IMAGE=python:3.9-slim

# Use the base image specified by the build argument
FROM ${BASE_IMAGE} as base

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone the YOLOv9 repository
RUN git clone https://github.com/WongKinYiu/yolov9.git

# Install YOLOv9 dependencies
RUN pip install --no-cache-dir -r yolov9/requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set the environment variable for PyTorch (optional)
ENV TORCH_HOME=/app/weights

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
