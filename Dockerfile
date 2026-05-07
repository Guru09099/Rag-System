# Use a lightweight Python base image (Linux)
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (for caching)
COPY requirements.txt .

# Install dependencies
# We add build-essential to ensure C++ libraries (for FAISS) compile correctly
RUN apt-get update && apt-get install -y build-essential \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Sanic runs on
EXPOSE 8000

# The command to start the server
CMD ["python", "-m", "src.serve"]