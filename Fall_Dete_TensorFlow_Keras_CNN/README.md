# Fall Detection System - Backend Setup Guide

This guide provides step-by-step instructions to set up the backend of the **Fall Detection System** on your local machine.

---

## Prerequisites

- **Python Version**: 3.11.4 or below
- **Pip Version**: 24.2

---

## Step 1: Set Up Virtual Environment

1. **Create a Virtual Environment**:
   - Navigate to the `WatchIt Backend` folder (or the folder where you cloned the repository).
   - Run the following command to create a virtual environment:
     ```bash
     python -m venv venv_name
     ```
   - Replace `venv_name` with your desired name for the virtual environment.

2. **Activate the Virtual Environment**:
   - On Windows, run:
     ```bash
     venv_name\Scripts\activate
     ```
   - If you encounter a script execution error, run the following command in PowerShell:
     ```bash
     Set-ExecutionPolicy Unrestricted -Scope Process
     ```
   - Then, try activating the virtual environment again.

---

## Step 2: Install Required Packages

1. **Install Dependencies**:
   - Ensure the virtual environment is activated.
   - Run the following command to install all required packages listed in `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

---

## Step 3: Understanding the Files

The repository contains the following key files and folders:

1. **`Fall_Detection_CNN.py`**:
   - This script trains the fall detection model and saves it as `fall_detection_model.h5`.
   - The `train` and `val` folders contain images used for training and validation.

2. **`Fall_Detection_CNN_ImageTest.py`**:
   - Use this script to test the model on a single image.

3. **`Fall_Detection_CNN_VideoTest.py`**:
   - Use this script to test the model on a video.

4. **`fall_detection_cnn_flask`**:
   - This folder contains the Flask application for running the fall detection system via APIs.
   - Run `app.py` to start the Flask server.

---

## Step 4: Run the Flask Application

1. **Navigate to the Flask Folder**:
   - Go to the `fall_detection_cnn_flask` folder:
     ```bash
     cd fall_detection_cnn_flask
     ```

2. **Run the Flask Server**:
   - Start the Flask application by running:
     ```bash
     python app.py
     ```

3. **API Endpoint**:
   - The API endpoint for uploading a video is:
     ```
     http://127.0.0.1:5000/upload
     ```
   - Use a tool like Postman or `curl` to send a video file as a multipart request to this endpoint.
   - The API will return whether a fall was detected or not.

---

## Troubleshooting

- Ensure that the virtual environment is activated before running any scripts.
- If you encounter issues with package installations, ensure that your Python and pip versions match the prerequisites.
- For Flask-related issues, check the logs in the terminal for detailed error messages.

---

## Support

For further assistance, please contact the development team or refer to the project documentation.`