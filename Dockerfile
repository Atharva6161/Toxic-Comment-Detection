# Specify the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Sentinel project files (code, weights, etc.) into the container
COPY . .

# Expose the necessary port (if running a web service/API)
EXPOSE 5000

# Define the command to run the application
CMD ["streamlit","run","app.py"]