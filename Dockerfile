# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies and Rust
RUN apt-get update && apt-get install -y curl build-essential \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && apt-get clean

# Add Rust and Cargo to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Copy the rest of the application code
COPY . /app

# Ensure the uploads directory exists
RUN mkdir -p uploads


# Set environment variables
ENV FLASK_ENV=production

# Use Gunicorn to serve the app in production
CMD ["python", "app.py"]
