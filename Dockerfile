# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install the dependencies specified in the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV OPENAI_API_KEY = "sk-my-aihub-xsqem7zSnZ6DQI7sdRnzT3BlbkFJB9vRWBeocaut9kv7oFef"

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["flask", "run"]
