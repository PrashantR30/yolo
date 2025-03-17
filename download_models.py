import os
import sys
import urllib.request
import tarfile
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_mobilenet_v1(model_dir):
    """Download MobileNet-SSD V1 model"""
    prototxt_path = os.path.join(model_dir, 'MobileNetSSD_deploy.prototxt')
    model_path = os.path.join(model_dir, 'MobileNetSSD_deploy.caffemodel')
    
    # Download prototxt if it doesn't exist
    if not os.path.exists(prototxt_path):
        logger.info("Downloading MobileNetSSD_deploy.prototxt...")
        try:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
                prototxt_path
            )
        except Exception as e:
            logger.error(f"Failed to download prototxt: {e}")
            return False
    
    # Download model if it doesn't exist
    if not os.path.exists(model_path):
        logger.info("Downloading MobileNetSSD_deploy.caffemodel...")
        try:
            urllib.request.urlretrieve(
                "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
                model_path
            )
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False
    
    return True

def download_mobilenet_v2(model_dir):
    """Download SSD MobileNet V2 model"""
    config_path = os.path.join(model_dir, 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
    model_path = os.path.join(model_dir, 'ssd_mobilenet_v2_coco_2018_03_29.pb')
    
    # Download config if it doesn't exist
    if not os.path.exists(config_path):
        logger.info("Creating SSD MobileNet V2 config...")
        try:
            with open(config_path, 'w') as f:
                f.write("""input: "image_tensor"
output_result_name: "detection_boxes"
output_result_name: "detection_classes"
output_result_name: "detection_scores"
output_result_name: "num_detections"
input_size: 300
input_size: 300
input_size: 3
""")
        except Exception as e:
            logger.error(f"Failed to create config file: {e}")
            return False
    
    # Download model if it doesn't exist
    if not os.path.exists(model_path):
        logger.info("Downloading SSD MobileNet V2 model...")
        try:
            tar_path = os.path.join(model_dir, "ssd_mobilenet_v2_coco_2018_03_29.tar.gz")
            urllib.request.urlretrieve(
                "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz",
                tar_path
            )
            logger.info("Extracting model...")
            extract_dir = os.path.join(model_dir, "tmp_extract")
            os.makedirs(extract_dir, exist_ok=True)
            
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=extract_dir)
            
            # Move the frozen inference graph
            shutil.copy(
                os.path.join(extract_dir, "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb"),
                model_path
            )
            
            # Clean up
            shutil.rmtree(extract_dir)
            os.remove(tar_path)
        except Exception as e:
            logger.error(f"Failed to download or extract model: {e}")
            return False
    
    return True

def download_yolo_tiny_v3(model_dir):
    """Download YOLOv3-tiny model"""
    config_path = os.path.join(model_dir, 'yolov3-tiny.cfg')
    model_path = os.path.join(model_dir, 'yolov3-tiny.weights')
    names_path = os.path.join(model_dir, 'coco.names')
    
    # Download config if it doesn't exist
    if not os.path.exists(config_path):
        logger.info("Downloading YOLOv3-tiny config...")
        try:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
                config_path
            )
        except Exception as e:
            logger.error(f"Failed to download config: {e}")
            return False
    
    # Download model if it doesn't exist
    if not os.path.exists(model_path):
        logger.info("Downloading YOLOv3-tiny weights...")
        try:
            urllib.request.urlretrieve(
                "https://pjreddie.com/media/files/yolov3-tiny.weights",
                model_path
            )
        except Exception as e:
            logger.error(f"Failed to download weights: {e}")
            return False
        
    # Download names file if it doesn't exist
    if not os.path.exists(names_path):
        logger.info("Downloading COCO class names...")
        try:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
                names_path
            )
        except Exception as e:
            logger.error(f"Failed to download class names: {e}")
            return False
    
    return True

def download_ssd_inception(model_dir):
    """Download SSD Inception V2 model"""
    config_path = os.path.join(model_dir, 'ssd_inception_v2_coco_2017_11_17.pbtxt')
    model_path = os.path.join(model_dir, 'ssd_inception_v2_coco_2017_11_17.pb')
    
    # Download config if it doesn't exist
    if not os.path.exists(config_path):
        logger.info("Creating SSD Inception V2 config...")
        try:
            with open(config_path, 'w') as f:
                f.write("""input: "image_tensor"
output_result_name: "detection_boxes"
output_result_name: "detection_classes"
output_result_name: "detection_scores"
output_result_name: "num_detections"
input_size: 300
input_size: 300
input_size: 3
""")
        except Exception as e:
            logger.error(f"Failed to create config file: {e}")
            return False
    
    # Download model if it doesn't exist
    if not os.path.exists(model_path):
        logger.info("Downloading SSD Inception V2 model...")
        try:
            tar_path = os.path.join(model_dir, "ssd_inception_v2_coco_2017_11_17.tar.gz")
            urllib.request.urlretrieve(
                "http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz",
                tar_path
            )
            logger.info("Extracting model...")
            extract_dir = os.path.join(model_dir, "tmp_extract")
            os.makedirs(extract_dir, exist_ok=True)
            
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=extract_dir)
            
            # Move the frozen inference graph
            shutil.copy(
                os.path.join(extract_dir, "ssd_inception_v2_coco_2017_11_17", "frozen_inference_graph.pb"),
                model_path
            )
            
            # Clean up
            shutil.rmtree(extract_dir)
            os.remove(tar_path)
        except Exception as e:
            logger.error(f"Failed to download or extract model: {e}")
            return False
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python download_models.py <model_dir> [model_name]")
        print("Available models: all, mobilenet_v1, mobilenet_v2, yolo_tiny_v3, ssd_inception")
        return
    
    model_dir = sys.argv[1]
    os.makedirs(model_dir, exist_ok=True)
    
    model_name = 'all'
    if len(sys.argv) >= 3:
        model_name = sys.argv[2].lower()
    
    result = True
    
    if model_name in ['all', 'mobilenet_v1']:
        logger.info("Downloading MobileNet-SSD V1 model...")
        result = download_mobilenet_v1(model_dir) and result
    
    if model_name in ['all', 'mobilenet_v2']:
        logger.info("Downloading SSD MobileNet V2 model...")
        result = download_mobilenet_v2(model_dir) and result
    
    if model_name in ['all', 'yolo_tiny_v3']:
        logger.info("Downloading YOLOv3-tiny model...")
        result = download_yolo_tiny_v3(model_dir) and result
    
    if model_name in ['all', 'ssd_inception']:
        logger.info("Downloading SSD Inception V2 model...")
        result = download_ssd_inception(model_dir) and result
    
    if result:
        logger.info("All models downloaded successfully.")
    else:
        logger.error("There were errors downloading some models.")
        sys.exit(1)

if __name__ == "__main__":
    main()
