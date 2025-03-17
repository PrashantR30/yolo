import argparse
import cv2
import socket
import time
import logging
import numpy as np
import threading
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default class names for SSD MobileNet V1 (VOC dataset)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MobileNet-SSD Detection')
    parser.add_argument('--camera', type=int, default=0,
                      help='Camera device ID')
    parser.add_argument('--conf', type=float, default=0.4,
                      help='Confidence threshold')
    parser.add_argument('--model-dir', type=str, default='/app/models',
                      help='Directory where models are stored')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Check if models exist
    model_dir = args.model_dir
    prototxt_path = os.path.join(model_dir, 'MobileNetSSD_deploy.prototxt')
    model_path = os.path.join(model_dir, 'MobileNetSSD_deploy.caffemodel')
    
    logger.info(f"Looking for model files in: {model_dir}")
    
    # List files in model directory for debugging
    try:
        model_files = os.listdir(model_dir)
        logger.info(f"Files in model directory: {model_files}")
    except Exception as e:
        logger.error(f"Error listing model directory: {e}")
    
    if not os.path.exists(prototxt_path):
        logger.error(f"Prototxt file not found: {prototxt_path}")
        logger.error("Please run download_models.py first")
        return
        
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please run download_models.py first")
        return
    
    logger.info("Model files found successfully")
    
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
    stream_thread = threading.Thread(target=stream_to_server, 
                                    args=(cap, prototxt_path, model_path, args.conf))
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

def get_color_for_id(class_id):
    """Get a unique color based on class ID"""
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (128, 128, 0)   # Olive
    ]
    return colors[class_id % len(colors)]

def stream_to_server(cap, prototxt_path, model_path, conf_threshold):
    """Stream video frames to the web server"""
    logger.info("Starting streaming thread")
    
    # Load the MobileNet-SSD model
    try:
        logger.info("Loading MobileNet-SSD model...")
        logger.info(f"Prototxt path: {prototxt_path}")
        logger.info(f"Model path: {model_path}")
        
        # Check if files are readable
        if not os.access(prototxt_path, os.R_OK):
            logger.error(f"Cannot read prototxt file: {prototxt_path}")
            return
            
        if not os.access(model_path, os.R_OK):
            logger.error(f"Cannot read model file: {model_path}")
            return
            
        # Load model
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Connect to the web server
    retry_delay = 2  # seconds
    
    # For FPS calculation
    prev_time = time.time()
    frame_counter = 0
    fps = 0
    
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
                
                # Get frame dimensions
                (h, w) = frame.shape[:2]
                
                # Calculate FPS
                frame_counter += 1
                current_time = time.time()
                elapsed = current_time - prev_time
                
                if elapsed > 1.0:  # Update FPS every second
                    fps = frame_counter / elapsed
                    prev_time = current_time
                    frame_counter = 0
                    logger.info(f"Current FPS: {fps:.1f}")
                
                # Create a copy of the frame for annotations
                annotated_frame = frame.copy()
                
                # Process for detection
                # Prepare the frame for detection
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
                
                # Set the input to the network
                net.setInput(blob)
                
                # Forward pass to get detections
                detections = net.forward()
                
                # Process the detections
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    # Filter by confidence threshold
                    if confidence > conf_threshold:
                        # Get the class ID
                        class_id = int(detections[0, 0, i, 1])
                        
                        # Get the coordinates
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x1, y1, x2, y2) = box.astype("int")
                        
                        # Ensure coordinates are within frame boundaries
                        x1 = max(0, min(x1, w - 1))
                        y1 = max(0, min(y1, h - 1))
                        x2 = max(0, min(x2, w - 1))
                        y2 = max(0, min(y2, h - 1))
                        
                        # Get class name and color
                        class_name = CLASSES[class_id] if class_id < len(CLASSES) else "Unknown"
                        color = get_color_for_id(class_id)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Create label with class name and confidence
                        label = f"{class_name}: {confidence:.2f}"
                        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        
                        # Draw label background
                        y1_label = max(y1, label_size[1] + 10)
                        cv2.rectangle(annotated_frame, (x1, y1_label - label_size[1] - 10),
                                    (x1 + label_size[0], y1_label), color, cv2.FILLED)
                        
                        # Draw label text
                        cv2.putText(annotated_frame, label, (x1, y1_label - 7),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Add FPS to the frame
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Compress the frame
                _, encoded_frame = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_data = encoded_frame.tobytes()
                
                # Send the frame size first (4 bytes)
                frame_size = len(frame_data)
                client_socket.sendall(frame_size.to_bytes(4, byteorder='big'))
                
                # Then send the frame data
                client_socket.sendall(frame_data)
            
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
