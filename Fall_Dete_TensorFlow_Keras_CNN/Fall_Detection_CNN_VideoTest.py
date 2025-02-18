import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('fall_detection_model.h5')

# Function to preprocess a frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (128, 128))  # Resize to match model input
    frame = frame / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Path to the video file
video_path = 'E:/Users/Alister Clietus/Fall_detection_Deepseek/video.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the frames per second (fps) of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))

frame_interval = fps  # Process one frame per second

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Process only one frame per second
    if frame_count % frame_interval == 0:
        processed_frame = preprocess_frame(frame)
        prediction = model.predict(processed_frame)[0][0]

        if prediction >= 0.5:
            print("Fall detected! Stopping processing.")
            break  # Stop processing if fall is detected

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
