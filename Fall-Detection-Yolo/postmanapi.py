import requests
import tkinter as tk
from tkinter import filedialog

def upload_video():
    # Open file dialog to select video
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    video_path = filedialog.askopenfilename(title="Select a video file", filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    
    if not video_path:
        print("No file selected.")
        return
    
    url = "http://127.0.0.1:5000/upload"  # Flask API endpoint
    
    try:
        with open(video_path, 'rb') as video_file:
            files = {'video': video_file}
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                print("Upload successful!", response.json())
            else:
                print("Upload failed!", response.text)
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    upload_video()