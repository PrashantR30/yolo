#!/bin/bash

# Setup script for YOLO detection system
# This script creates the directory structure and all necessary files

set -e  # Exit immediately if a command exits with a non-zero status

echo "Setting up YOLO detection system..."

# Create directory structure
mkdir -p yolo-detector
mkdir -p web-server/templates
mkdir -p web-server/static

echo "Directory structure created."

# Create Docker Compose files
echo "Creating Docker Compose files..."

cat > docker-compose.yml << 'EOF'
version: '3.3'

services:
  web-server:
    image: web-server:latest
    ports:
      - "8090:8090"
    volumes:
      - ./web-server:/app
    networks:
      - yolo-network
    depends_on:
      - yolo-detector

  yolo-detector:
    image: yolo-detector:latest
    volumes:
      - ./yolo-detector:/app
      - /dev/video0:/dev/video0
    devices:
      - /dev/video0:/dev/video0
    environment:
      - PYTHONUNBUFFERED=1
    command: ["python", "detector.py", "--device", "cpu"]
    networks:
      - yolo-network

networks:
  yolo-network:
    driver: bridge
EOF

cat > docker-compose.gpu.yml << 'EOF'
version: '3.3'

services:
  web-server:
    image: web-server:latest
    ports:
      - "8090:8090"
    volumes:
      - ./web-server:/app
    networks:
      - yolo-network
    depends_on:
      - yolo-detector

  yolo-detector:
    image: yolo-detector:gpu
    volumes:
      - ./yolo-detector:/app
      - /dev/video0:/dev/video0
    devices:
      - /dev/video0:/dev/video0
    environment:
      - PYTHONUNBUFFERED=1
    command: ["python", "detector.py", "--device", "gpu"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - yolo-network

networks:
  yolo-network:
    driver: bridge
EOF

# Create Dockerfiles
echo "Creating Dockerfiles..."

cat > web-server/Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir flask flask-socketio opencv-python-headless

# Expose port
EXPOSE 8090

# Command to run the web server
CMD ["python", "web_server.py"]
EOF

cat > yolo-detector/Dockerfile.cpu << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python \
    numpy

# Expose port for streaming
EXPOSE 5555

# Command will be provided by docker-compose
EOF

cat > yolo-detector/Dockerfile.gpu << 'EOF'
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python \
    numpy

# Expose port for streaming
EXPOSE 5555

# Command will be provided by docker-compose
EOF

# Create Python scripts
echo "Creating Python scripts..."

cat > web-server/web_server.py << 'EOF'
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import socket
import threading
import logging
import time
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yolo-detection-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variable to store the latest frame
latest_frame = None
frame_ready = False

def receive_frames():
    """
    Receive frames from the YOLO detector service
    """
    global latest_frame, frame_ready
    
    logger.info("Starting frame receiver...")
    
    # Wait for the yolo-detector service to start
    time.sleep(5)
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', 5555))
    server_socket.listen(1)
    
    logger.info("Waiting for connection from YOLO detector...")
    
    while True:
        try:
            client_socket, addr = server_socket.accept()
            logger.info(f"Connected to YOLO detector at {addr}")
            
            data_buffer = b''
            payload_size = None
            
            while True:
                try:
                    # First, receive the message length (4 bytes)
                    if payload_size is None:
                        if len(data_buffer) < 4:
                            data = client_socket.recv(4096)
                            if not data:
                                break
                            data_buffer += data
                            continue
                        
                        payload_size = int.from_bytes(data_buffer[:4], byteorder='big')
                        data_buffer = data_buffer[4:]
                    
                    # Then, receive the actual frame data
                    if len(data_buffer) < payload_size:
                        data = client_socket.recv(4096)
                        if not data:
                            break
                        data_buffer += data
                        continue
                    
                    # We have a complete frame
                    frame_data = data_buffer[:payload_size]
                    data_buffer = data_buffer[payload_size:]
                    payload_size = None
                    
                    # Decode the frame
                    frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        latest_frame = frame
                        frame_ready = True
                        
                        # Emit the frame to all connected clients
                        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        socketio.emit('video_frame', {'image': frame_base64})
                    
                except Exception as e:
                    logger.error(f"Error receiving frame: {e}")
                    break
            
            client_socket.close()
            logger.info("Connection to YOLO detector closed")
            
        except Exception as e:
            logger.error(f"Socket error: {e}")
            time.sleep(2)  # Wait before trying to accept another connection

@app.route('/')
def index():
    """
    Render the main page
    """
    return render_template('index.html')

def gen_frames():
    """
    Generate frames for the video feed
    """
    global latest_frame, frame_ready
    
    while True:
        if frame_ready and latest_frame is not None:
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            
            # Convert to bytes
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in the format expected by Response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If no frame is available, send a placeholder
            with open('static/waiting.jpg', 'rb') as f:
                placeholder = f.read()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS

@app.route('/video_feed')
def video_feed():
    """
    Route for the video feed
    """
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the frame receiver in a separate thread
    receiver_thread = threading.Thread(target=receive_frames)
    receiver_thread.daemon = True
    receiver_thread.start()
    
    # Create the static directory if it doesn't exist
    import os
    os.makedirs('static', exist_ok=True)
    
    # Create a placeholder image if it doesn't exist
    if not os.path.exists('static/waiting.jpg'):
        # Create a simple placeholder image
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Waiting for YOLO detector...", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite('static/waiting.jpg', placeholder)
    
    # Start the Flask server
    logger.info("Starting web server on http://0.0.0.0:8090")
    socketio.run(app, host='0.0.0.0', port=8090, debug=False, allow_unsafe_werkzeug=True)
EOF

cat > yolo-detector/detector.py << 'EOF'
import argparse
import cv2
import socket
import time
import logging
import numpy as np
from ultralytics import YOLO
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'],
                      help='Device to run inference on (cpu or gpu)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                      help='YOLOv8 model to use')
    parser.add_argument('--camera', type=int, default=0,
                      help='Camera device ID')
    parser.add_argument('--conf', type=float, default=0.5,
                      help='Confidence threshold')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Log the device being used
    logger.info(f"Using device: {args.device}")
    
    # Set device
    device = 'cuda:0' if args.device == 'gpu' else 'cpu'
    
    # Initialize the YOLO model
    try:
        logger.info(f"Loading YOLO model: {args.model} on {device}")
        model = YOLO(args.model)
        model.to(device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Open the camera
    try:
        logger.info(f"Opening camera: {args.camera}")
        cap = cv2.VideoCapture(args.camera)
        
        # Check if the camera opened successfully
        if not cap.isOpened():
            logger.error("Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Camera opened successfully")
    except Exception as e:
        logger.error(f"Error opening camera: {e}")
        return
    
    # Start the streaming thread
    stream_thread = threading.Thread(target=stream_to_server, args=(cap, model, device, args.conf))
    stream_thread.daemon = True
    stream_thread.start()
    
    logger.info("Detector running. Press Ctrl+C to exit.")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exiting...")
    finally:
        # Clean up
        cap.release()
        logger.info("Camera released")

def stream_to_server(cap, model, device, conf_threshold):
    """Stream video frames to the web server"""
    logger.info("Starting streaming thread")
    
    # Connect to the web server
    retry_delay = 2  # seconds
    
    while True:
        try:
            # Create a TCP socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(('web-server', 5555))
            logger.info("Connected to web server")
            
            while True:
                # Read a frame from the camera
                ret, frame = cap.read()
                
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Run inference with YOLOv8
                results = model(frame, verbose=False, conf=conf_threshold)
                
                # Draw the results on the frame
                annotated_frame = results[0].plot()
                
                # Compress the frame
                _, encoded_frame = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_data = encoded_frame.tobytes()
                
                # Send the frame size first (4 bytes)
                frame_size = len(frame_data)
                client_socket.sendall(frame_size.to_bytes(4, byteorder='big'))
                
                # Then send the frame data
                client_socket.sendall(frame_data)
                
                # Limit the frame rate
                time.sleep(0.03)  # ~30 FPS
            
            # Close the socket
            client_socket.close()
            logger.info("Disconnected from web server")
            
        except ConnectionRefusedError:
            logger.warning(f"Connection to web server refused. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            continue
        except BrokenPipeError:
            logger.warning(f"Connection to web server broken. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            continue
        except Exception as e:
            logger.error(f"Error in streaming thread: {e}")
            time.sleep(retry_delay)
            continue

if __name__ == "__main__":
    main()
EOF

# Create HTML template
echo "Creating HTML template..."

cat > web-server/templates/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO Object Detection</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .video-container {
            margin: 20px auto;
            text-align: center;
        }
        #video-feed {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .status {
            margin-top: 20px;
            padding: 10px;
            background-color: #e0f7fa;
            border-radius: 5px;
            text-align: center;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>YOLO Object Detection</h1>
        
        <div class="video-container">
            <img id="video-feed" src="{{ url_for('video_feed') }}" alt="Video Stream">
        </div>
        
        <div class="status" id="status">
            Status: Waiting for connection to YOLO detector...
        </div>
        
        <div class="footer">
            <p>Real-time object detection using YOLOv8</p>
        </div>
    </div>

    <script>
        // Connect to Socket.IO server
        const socket = io();
        const videoFeed = document.getElementById('video-feed');
        const statusDisplay = document.getElementById('status');
        
        // Listen for video frames from the server
        socket.on('video_frame', function(data) {
            // Update the image with the received frame
            videoFeed.src = 'data:image/jpeg;base64,' + data.image;
            
            // Update status
            statusDisplay.innerText = 'Status: Connected to YOLO detector';
            statusDisplay.style.backgroundColor = '#e8f5e9';
        });
        
        // Handle disconnect events
        socket.on('disconnect', function() {
            statusDisplay.innerText = 'Status: Disconnected from server';
            statusDisplay.style.backgroundColor = '#ffebee';
        });
        
        // Handle connect events
        socket.on('connect', function() {
            statusDisplay.innerText = 'Status: Connected to server, waiting for YOLO detector...';
            statusDisplay.style.backgroundColor = '#fff8e1';
        });
    </script>
</body>
</html>
EOF

# Create build script
echo "Creating build script..."

cat > build.sh << 'EOF'
#!/bin/bash

# Build script for YOLO detection system
# This script builds the Docker images for both CPU and GPU configurations

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting build process..."

# Build web-server image
echo "Building web-server image..."
docker build -t web-server:latest ./web-server

# Build yolo-detector CPU image
echo "Building yolo-detector CPU image..."
docker build -t yolo-detector:latest -f ./yolo-detector/Dockerfile.cpu ./yolo-detector

# Build yolo-detector GPU image (optional)
echo "Building yolo-detector GPU image..."
docker build -t yolo-detector:gpu -f ./yolo-detector/Dockerfile.gpu ./yolo-detector

echo "Build completed successfully!"
echo ""
echo "The following images are now available:"
echo "  - web-server:latest    (Web server image)"
echo "  - yolo-detector:latest (CPU-based YOLO detector image)"
echo "  - yolo-detector:gpu    (GPU-based YOLO detector image)"
echo ""
echo "To run with CPU:"
echo "  docker-compose up"
echo ""
echo "To run with GPU:"
echo "  docker-compose -f docker-compose.gpu.yml up"
echo ""
EOF

# Make scripts executable
chmod +x build.sh

echo "Setup completed successfully!"
echo ""
echo "Directory structure:"
echo "  - yolo-detector/        (YOLO detector files)"
echo "  - web-server/           (Web server files)"
echo "  - web-server/templates/ (HTML templates)"
echo "  - web-server/static/    (Static files)"
echo ""
echo "Files created:"
echo "  - docker-compose.yml    (Docker Compose for CPU)"
echo "  - docker-compose.gpu.yml (Docker Compose for GPU)"
echo "  - build.sh              (Script to build Docker images)"
echo ""
echo "To build the Docker images:"
echo "  ./build.sh"
echo ""
echo "To run with CPU:"
echo "  docker-compose up"
echo ""
echo "To run with GPU:"
echo "  docker-compose -f docker-compose.gpu.yml up"
echo ""
