import torch
import cv2
import math
import numpy as np
import os
import logging
from flask import Flask, request, jsonify
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

# Setup Logging
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "fall_detection.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = Flask(__name__)

def get_pose_model():
    """Loads the YOLOv7 model for human pose estimation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    try:
        weights = torch.load('yolov7-w6-pose.pt', map_location=device,weights_only=False)
        model = weights['model'].to(device)
        model.float().eval()
        if torch.cuda.is_available():
            model = model.half().eval()
        
        logging.info("Pose model loaded successfully.")
        return model, device
    except Exception as e:
        logging.error(f"Error loading pose model: {e}")
        raise

# Load Model
model, device = get_pose_model()

def fall_detection(poses):
    """Determines if a fall has occurred based on detected human keypoints."""
    try:
        for pose in poses:
            left_shoulder_y, left_shoulder_x = pose[23], pose[22]
            right_shoulder_y = pose[26]
            left_body_y, left_body_x = pose[41], pose[40]
            right_body_y = pose[44]
            len_factor = math.sqrt((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2)
            left_foot_y, right_foot_y = pose[53], pose[56]
            
            if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2) or \
               (right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (len_factor / 2) and right_shoulder_y > right_body_y - (len_factor / 2)):
                return True
        return False
    except Exception as e:
        logging.error(f"Error in fall detection logic: {e}")
        return False

def get_pose(image):
    """Processes an image frame and returns detected keypoints."""
    try:
        image = letterbox(image, 960, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        
        if torch.cuda.is_available():
            image = image.half().to(device)

        with torch.no_grad():
            output, _ = model(image)

        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)

        with torch.no_grad():
            output = output_to_keypoint(output)

        return output
    except Exception as e:
        logging.error(f"Error processing pose from frame: {e}")
        return []

@app.route('/upload', methods=['POST'])
def detect_fall():
    """API endpoint to detect falls from an uploaded video."""
    if 'video' not in request.files:
        logging.warning("No video file uploaded in the request.")
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    video_path = "temp_video.mp4"
    
    try:
        video_file.save(video_path)
        logging.info(f"Received video file: {video_file.filename}, saved temporarily as {video_path}")

        vid_cap = cv2.VideoCapture(video_path)
        if not vid_cap.isOpened():
            logging.error("Error reading the uploaded video file.")
            return jsonify({'error': 'Error reading video file'}), 500
        
        fall_detected = False
        frame_count = 0
        success, frame = vid_cap.read()
        
        while success:
            frame_count += 1
            output = get_pose(frame)
            
            if fall_detection(output):
                fall_detected = True
                logging.info(f"Fall detected in frame {frame_count}")
                break
            
            success, frame = vid_cap.read()

        vid_cap.release()
        logging.info(f"Fall detection completed. Fall detected: {fall_detected}")

        return jsonify({'fall_detected': fall_detected})

    except Exception as e:
        logging.error(f"Error during fall detection processing: {e}")
        return jsonify({'error': 'Internal server error'}), 500
    
    finally:
        # Ensure the temporary file is deleted even in case of errors
        if os.path.exists(video_path):
            os.remove(video_path)
            logging.info(f"Temporary video file {video_path} deleted successfully.")

if __name__ == '__main__':
    logging.info("Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
