FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir flask flask-socketio opencv-python-headless

# Expose port
EXPOSE 8090

# Command to run the web server
CMD ["python", "web_server.py"]
