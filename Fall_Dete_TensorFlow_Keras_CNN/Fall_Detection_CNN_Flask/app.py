import os
import cv2
import numpy as np
import tensorflow as tf
import mysql.connector
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Set Upload Folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "password",  # Change this
    "database": "fall_detection"
}

# Load Model
MODEL_PATH = "fall_detection_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Image Preprocessing
IMG_SIZE = (128, 128)

def preprocess_frame(frame):
    """Preprocesses a video frame for model prediction."""
    frame = cv2.resize(frame, IMG_SIZE)  # Resize
    frame = frame / 255.0  # Normalize
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

def save_frame_to_db(frame, video_name, frame_number):
    """Saves extracted frames to MySQL database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Convert frame to bytes
        _, buffer = cv2.imencode(".jpg", frame)
        frame_data = buffer.tobytes()

        query = """
        INSERT INTO frames (video_name, frame_number, frame_data, timestamp)
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (video_name, frame_number, frame_data, datetime.now()))
        conn.commit()

        cursor.close()
        conn.close()
    except Exception as e:
        print("Database Error:", e)

def process_video(video_path, video_name):
    """Processes the video, detects falls, and saves frames."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Could not open video."}

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps  # Process one frame per second
    frame_count = 0
    fall_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Process one frame per second
        if frame_count % frame_interval == 0:
            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame)[0][0]

            # Save frame to DB
            save_frame_to_db(frame, video_name, frame_count)

            if prediction >= 0.5:
                fall_detected = True
                break  # Stop processing if fall is detected

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return {"fall_detected": fall_detected, "video_name": video_name}

@app.route("/upload", methods=["POST"])
def upload_video():
    """API Endpoint to Upload and Process Video"""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files["video"]

    if video.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    video.save(video_path)

    result = process_video(video_path, filename)
    return jsonify(result)

@app.route("/get_frames", methods=["GET"])
def get_frames():
    """Fetch all saved frames from the database and return as JSON."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)  # Returns data as dictionary

        query = "SELECT id, video_name, frame_number, timestamp FROM frames"
        cursor.execute(query)
        frames = cursor.fetchall()  # Fetch all rows

        cursor.close()
        conn.close()

        return jsonify(frames)  # Return data as JSON array

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
