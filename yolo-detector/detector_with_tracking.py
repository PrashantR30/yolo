import argparse
import cv2
import socket
import time
import logging
import numpy as np
from ultralytics import YOLO
import threading
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define vehicle classes from COCO dataset
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
PERSON_CLASS = 0
CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"]

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLO Object Detection with Tracking')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'],
                      help='Device to run inference on (cpu or gpu)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                      help='YOLOv8 model to use')
    parser.add_argument('--camera', type=int, default=0,
                      help='Camera device ID')
    parser.add_argument('--conf', type=float, default=0.5,
                      help='Confidence threshold')
    parser.add_argument('--track', action='store_true',
                      help='Enable tracking')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Disable CUDA for CPU mode to avoid NVIDIA check
    if args.device == 'cpu':
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Log the device being used
    logger.info(f"Using device: {args.device}")
    logger.info(f"Tracking enabled: {args.track}")
    
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
    stream_thread = threading.Thread(target=stream_to_server, 
                                    args=(cap, model, device, args.conf, args.track))
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

class VehicleTracker:
    """Class to track vehicles across frames"""
    def __init__(self):
        self.next_id = 1
        self.tracked_vehicles = {}  # id -> {box, class_id, confidence, last_seen, trajectory}
        self.max_disappeared = 30  # frames before dropping a track
        self.matching_threshold = 0.3  # IOU threshold for considering it the same vehicle
    
    def update(self, detections):
        """
        Update tracker with new detections
        detections: list of (box, class_id, confidence) where box is (x1, y1, x2, y2)
        """
        # If no vehicles are being tracked yet, add all
        if len(self.tracked_vehicles) == 0:
            for box, class_id, confidence in detections:
                self.add_new_vehicle(box, class_id, confidence)
            return
        
        # Match detections to existing tracked vehicles
        matched_indices = set()
        detection_matched = [False] * len(detections)
        
        # For each tracked vehicle, find best matching detection
        for vehicle_id, vehicle_data in list(self.tracked_vehicles.items()):
            best_match_idx = -1
            best_iou = self.matching_threshold
            
            # Get the last known box
            last_box = vehicle_data['box']
            
            for i, (box, _, _) in enumerate(detections):
                if detection_matched[i]:
                    continue
                
                iou = self.calculate_iou(last_box, box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = i
            
            # If we found a match, update the tracked vehicle
            if best_match_idx >= 0:
                box, class_id, confidence = detections[best_match_idx]
                self.tracked_vehicles[vehicle_id]['box'] = box
                self.tracked_vehicles[vehicle_id]['class_id'] = class_id
                self.tracked_vehicles[vehicle_id]['confidence'] = confidence
                self.tracked_vehicles[vehicle_id]['last_seen'] = 0
                self.tracked_vehicles[vehicle_id]['trajectory'].append((
                    (box[0] + box[2]) // 2,  # center x
                    (box[1] + box[3]) // 2   # center y
                ))
                # Limit trajectory length
                if len(self.tracked_vehicles[vehicle_id]['trajectory']) > 30:
                    self.tracked_vehicles[vehicle_id]['trajectory'] = \
                        self.tracked_vehicles[vehicle_id]['trajectory'][-30:]
                
                detection_matched[best_match_idx] = True
                matched_indices.add(vehicle_id)
        
        # Increment disappeared counter for unmatched vehicles
        for vehicle_id in list(self.tracked_vehicles.keys()):
            if vehicle_id not in matched_indices:
                self.tracked_vehicles[vehicle_id]['last_seen'] += 1
                
                # Remove if disappeared for too long
                if self.tracked_vehicles[vehicle_id]['last_seen'] > self.max_disappeared:
                    del self.tracked_vehicles[vehicle_id]
        
        # Add new vehicles for unmatched detections
        for i, (box, class_id, confidence) in enumerate(detections):
            if not detection_matched[i]:
                self.add_new_vehicle(box, class_id, confidence)
    
    def add_new_vehicle(self, box, class_id, confidence):
        """Add a new vehicle to tracking"""
        self.tracked_vehicles[self.next_id] = {
            'box': box,
            'class_id': class_id,
            'confidence': confidence,
            'last_seen': 0,
            'trajectory': [((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)]
        }
        self.next_id += 1
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area
        return iou
    
    def draw_tracks(self, frame):
        """Draw tracked vehicles and their trajectories on the frame"""
        for vehicle_id, vehicle_data in self.tracked_vehicles.items():
            box = vehicle_data['box']
            class_id = vehicle_data['class_id']
            confidence = vehicle_data['confidence']
            trajectory = vehicle_data['trajectory']
            
            # Get color based on vehicle ID (to ensure each vehicle has a unique color)
            color = self.get_color_for_id(vehicle_id)
            
            # Draw bounding box
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw ID and class name
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
            label = f"ID:{vehicle_id} {class_name} {confidence:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw trajectory (connect points with lines)
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    pt1 = trajectory[i-1]
                    pt2 = trajectory[i]
                    cv2.line(frame, pt1, pt2, color, 2)
        
        return frame
    
    def get_color_for_id(self, id):
        """Get a unique color for a vehicle ID"""
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
        return colors[id % len(colors)]

def stream_to_server(cap, model, device, conf_threshold, enable_tracking=False):
    """Stream video frames to the web server"""
    logger.info("Starting streaming thread")
    logger.info(f"Tracking enabled: {enable_tracking}")
    
    # Initialize vehicle tracker if tracking is enabled
    vehicle_tracker = VehicleTracker() if enable_tracking else None
    
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
                
                # Calculate FPS
                frame_counter += 1
                if frame_counter >= 10:
                    current_time = time.time()
                    fps = frame_counter / (current_time - prev_time)
                    prev_time = current_time
                    frame_counter = 0
                
                # Run inference with YOLOv8
                if enable_tracking:
                    # Use tracking mode in YOLO
                    results = model.track(frame, verbose=False, conf=conf_threshold, 
                                         classes=VEHICLE_CLASSES + [PERSON_CLASS], persist=True)
                else:
                    results = model(frame, verbose=False, conf=conf_threshold)
                
                # Create a copy of the frame for annotations
                annotated_frame = frame.copy()
                
                if enable_tracking and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                    # If YOLO's built-in tracker is working
                    if hasattr(results[0], 'boxes') and hasattr(results[0].boxes, 'id'):
                        # Get boxes with tracking IDs from YOLO
                        boxes = results[0].boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cls = int(box.cls)
                            conf = float(box.conf)
                            track_id = int(box.id.item()) if box.id is not None else -1
                            
                            # Only draw if it's a vehicle or person
                            if cls in VEHICLE_CLASSES or cls == PERSON_CLASS:
                                class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
                                
                                # Create unique color based on track ID
                                if track_id != -1:
                                    color_r = int((track_id * 123) % 255)
                                    color_g = int((track_id * 87) % 255)
                                    color_b = int((track_id * 47) % 255)
                                    color = (color_b, color_g, color_r)
                                else:
                                    color = (0, 255, 0)  # Green for untracked
                                
                                # Draw bounding box
                                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                                
                                # Draw label with ID if available
                                label = f"{class_name} {conf:.2f}"
                                if track_id != -1:
                                    label = f"ID:{track_id} " + label
                                
                                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        # Use our custom tracker if YOLO's tracker isn't available
                        detections = []
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cls = int(box.cls)
                            conf = float(box.conf)
                            
                            # Only track vehicles and persons
                            if cls in VEHICLE_CLASSES or cls == PERSON_CLASS:
                                detections.append(((x1, y1, x2, y2), cls, conf))
                        
                        # Update tracker with new detections
                        vehicle_tracker.update(detections)
                        
                        # Draw tracking results
                        annotated_frame = vehicle_tracker.draw_tracks(annotated_frame)
                else:
                    # Just draw detection boxes without tracking
                    annotated_frame = results[0].plot()
                
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
