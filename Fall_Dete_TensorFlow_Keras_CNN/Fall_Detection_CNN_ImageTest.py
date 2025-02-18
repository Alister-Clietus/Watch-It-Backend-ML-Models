import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('fall_detection_model.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read image
    img = cv2.resize(img, (128, 128))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Path to the test image
test_image_path = 'E:/Users/Alister Clietus/Fall_detection_Deepseek/image.png'

# Preprocess and predict
processed_image = preprocess_image(test_image_path)
prediction = model.predict(processed_image)[0][0]

# Interpret the result
if prediction >= 0.5:
    print("Fall detected!")
else:
    print("No fall detected.")
