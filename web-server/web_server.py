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
