import argparse
import cv2
import socket
import time
import logging
import numpy as np
import threading
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the target classes (COCO dataset) for vehicles and people
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorbike, bus, truck

# COCO class names
CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", 
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
    "toothbrush"
]

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLO Vehicle and Person Detection')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'],
                      help='Device to run inference on (cpu or gpu)')
    parser.add_argument('--camera', type=int, default=0,
                      help='Camera device ID')
    parser.add_argument('--conf', type=float, default=0.3,
                      help='Confidence threshold')
    parser.add_argument('--img-size', type=int, default=640,
                      help='Input image size for inference')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                      help='Path to YOLOv8 model file')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Set environment variable for CUDA if using CPU
    if args.device == 'cpu':
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Log the device being used
    logger.info(f"Using device: {args.device}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Detecting classes: {', '.join([CLASS_NAMES[cls_id] for cls_id in TARGET_CLASSES])}")
    
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
                                     args=(cap, args.device, args.conf, args.model, args.img_size))
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

# Background subtractor for motion detection
bg_subtractor = None

def detect_motion(frame, min_contour_area=500):
    """
    Detect motion in a frame using background subtraction
    Returns True if motion is detected, False otherwise
    """
    global bg_subtractor
    if bg_subtractor is None:
        # Initialize on first use
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=50, detectShadows=False)
    
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)
    
    # Apply some noise filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contour has significant area
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            return True, contours
    
    return False, []

# Function to draw a corner rectangle like cvzone
def draw_corner_rect(img, bbox, l=30, t=5, rt=1, colorR=(255, 0, 255), colorC=(0, 255, 0)):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    
    # Top Left
    cv2.line(img, (x, y), (x + l, y), colorR, t)
    cv2.line(img, (x, y), (x, y + l), colorR, t)
    # Top Right
    cv2.line(img, (x1, y), (x1 - l, y), colorR, t)
    cv2.line(img, (x1, y), (x1, y + l), colorR, t)
    # Bottom Left
    cv2.line(img, (x, y1), (x + l, y1), colorR, t)
    cv2.line(img, (x, y1), (x, y1 - l), colorR, t)
    # Bottom Right
    cv2.line(img, (x1, y1), (x1 - l, y1), colorR, t)
    cv2.line(img, (x1, y1), (x1, y1 - l), colorR, t)
    
    return img

# Function to put text with rectangle background like cvzone
def put_text_rect(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255), 
                 colorR=(0, 0, 0), font=cv2.FONT_HERSHEY_PLAIN, offset=10, border=None, colorB=(0, 255, 0)):
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    
    x1, y1, x2, y2 = ox - offset, oy - offset, ox + w + offset, oy + h + offset
    
    cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorB, border)
    cv2.putText(img, text, (ox, oy + h - offset // 2), font, scale, colorT, thickness)
    
    return img

def stream_to_server(cap, device, conf_threshold, model_path, img_size):
    """Stream video frames to the web server"""
    logger.info(f"Starting streaming thread using {device}")
    
    # Variables for frame rate control
    target_fps = 15  # Target processing FPS
    frame_interval = 1.0 / target_fps
    last_processed_time = time.time()
    
    # Load YOLO model
    try:
        import torch
        import os
        
        # Force YOLO to use CPU only, regardless of what's available
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.backends.cudnn.enabled = False
        
        # Import YOLO only after environment setup
        from ultralytics import YOLO
        
        logger.info(f"Loading YOLO model: {model_path} with device forcing to CPU only")
        # Directly pass device="cpu" to ensure CPU usage
        model = YOLO(model_path)
        # Force model to CPU
        model.to("cpu")
        logger.info("Model loaded successfully on CPU")
        
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        return
    
    # Connect to the web server
    retry_delay = 2  # seconds
    
    # Frame counter
    frame_counter = 0
    
    while True:
        try:
            # Create a TCP socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(('web-server', 5555))
            logger.info("Connected to web server")
            
            while True:
                current_time = time.time()
                
                # Control processing rate
                if current_time - last_processed_time < frame_interval:
                    time.sleep(0.001)  # Small sleep to not consume CPU
                    continue
                
                # Reset timer
                delta_time = current_time - last_processed_time
                last_processed_time = current_time
                
                # Read a frame from the camera
                ret, frame = cap.read()
                
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                frame_counter += 1
                
                # Create a copy for annotations
                annotated_frame = frame.copy()
                
                # Resize for motion detection
                small_frame = cv2.resize(frame, (320, 240))
                
                # Detect motion
                motion_detected, motion_contours = detect_motion(small_frame)
                
                # Process with YOLO
                process_with_yolo = (frame_counter % 2 == 0) or motion_detected  # Every other frame or on motion
                
                if process_with_yolo:
                    # Run inference
                    results = model(frame, stream=True, conf=conf_threshold, classes=TARGET_CLASSES)
                    
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            w, h = x2-x1, y2-y1
                            
                            # Draw corner rectangle
                            draw_corner_rect(annotated_frame, (x1, y1, w, h))
                            
                            # Get confidence and class
                            conf = math.ceil((box.conf[0]*100))/100
                            cls = int(box.cls[0])
                            name = CLASS_NAMES[cls]
                            
                            # Put text with background
                            put_text_rect(annotated_frame, f'{name} {conf:.2f}', 
                                        (max(0,x1), max(35,y1)), scale=0.5, thickness=1)
                
                # Draw motion contours if detected
                if motion_detected:
                    cv2.putText(annotated_frame, "Motion", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add FPS info
                fps = 1.0 / delta_time if delta_time > 0 else 0
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Compress the frame
                _, encoded_frame = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
