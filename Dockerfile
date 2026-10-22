
# Use the official Python image from the Docker Hub

FROM python:3.12-slim

# Set the working directory in the container

WORKDIR /app

# Install any needed packages specified in requirements.txt

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app

COPY . /app

# Expose the port the app runs on

EXPOSE 9696


# Run the Flask app


CMD ["python", "predict.py"]