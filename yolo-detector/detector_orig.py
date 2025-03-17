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
