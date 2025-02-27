import torch
import cv2
import math
import numpy as np
import os
from flask import Flask, request, jsonify
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

app = Flask(__name__)

def get_pose_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device detected:", device)
    print("Loading YOLOv7 pose model...")
    weights = torch.load('yolov7-w6-pose.pt', map_location=device,weights_only=False)
    model = weights['model'].to(device)
    _ = model.float().eval()
    if torch.cuda.is_available():
        model = model.half().eval()
    print("Model loaded successfully!")
    return model, device

model, device = get_pose_model()

def fall_detection(poses):
    print("Checking for fall detection...")
    for pose in poses:
        left_shoulder_y = pose[23]
        left_shoulder_x = pose[22]
        right_shoulder_y = pose[26]
        left_body_y = pose[41]
        left_body_x = pose[40]
        right_body_y = pose[44]
        len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))
        left_foot_y = pose[53]
        right_foot_y = pose[56]
        
        if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (len_factor / 2) \
                and left_shoulder_y > left_body_y - (len_factor / 2) or \
                (right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (len_factor / 2) \
                 and right_shoulder_y > right_body_y - (len_factor / 2)):
            print("Fall detected!")
            return True
    print("No fall detected.")
    return False

def get_pose(image):
    print("Processing frame for pose estimation...")
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
    print("Pose estimation completed.")
    return output

@app.route('/upload', methods=['POST'])
def detect_fall():
    print("Received video for fall detection.")
    if 'video' not in request.files:
        print("No video file uploaded.")
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    video_path = "temp_video.mp4"
    video_file.save(video_path)
    print("Video saved locally: temp_video.mp4")
    
    vid_cap = cv2.VideoCapture(video_path)
    if not vid_cap.isOpened():
        print("Error reading video file.")
        return jsonify({'error': 'Error reading video file'}), 500
    
    fall_detected = False
    frame_count = 0
    success, frame = vid_cap.read()
    while success:
        frame_count += 1
        print(f"Processing frame {frame_count}...")
        output = get_pose(frame)
        if fall_detection(output):
            fall_detected = True
            print("Fall detected in video! Aborting further processing.")
            break
        success, frame = vid_cap.read()
    
    vid_cap.release()
    os.remove(video_path)
    print("Temporary video file deleted.")
    print("Fall detection process completed.")
    
    return jsonify({'fall_detected': fall_detected})

if __name__ == '__main__':
    print("Starting Fall Detection API...")
    app.run(host='0.0.0.0', port=5000, debug=True)
