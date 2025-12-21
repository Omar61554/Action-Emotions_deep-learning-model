import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================================
# 1. SETUP
# ==========================================
# Load your trained model
# Note: We use compile=False because we only need to Predict, not Train.
# This avoids errors if the optimizer state is weird.
model = load_model('/home/mohamed/autonom_ws/src/Action-Emotions_deep-learning-model/scripts/emotion_model_sparse_vgg_deep.keras', compile=False)

# Define the Emotion Labels (Must match the order of your training folders)
# Usually alphabetical: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
class_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

# # Load the Face Detector (Haar Cascade)
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Update the classifier line to use the built-in data path
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Start the Webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("System Ready. Press 'q' to quit.")

# ==========================================
# 2. THE MAIN LOOP
# ==========================================
while True:
    # A. Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # B. Convert to Grayscale (Haar Cascade & Model both need Gray)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # C. Detect Faces
    # scaleFactor=1.1: Reduces image by 10% each pass to find faces
    # minNeighbors=5: Higher quality, fewer false positives
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # --- Preprocessing for the Model ---
        
        # 1. Extract the Region of Interest (ROI) - The Face
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        # 2. Resize to 48x48 (Model Requirement)
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        # 3. Normalize pixel values (0-255 -> 0-1)
        roi_normalized = roi_gray.astype('float32') / 255.0
        
        # 4. Expand dimensions to match model input: (1, 48, 48, 1)
        # We need a 4D tensor: [Batch_Size, Height, Width, Channels]
        roi_input = np.expand_dims(roi_normalized, axis=0) # Add Batch dimension
        roi_input = np.expand_dims(roi_input, axis=-1)     # Add Channel dimension
        
        # --- Prediction ---
        prediction = model.predict(roi_input, verbose=0)
        max_index = int(np.argmax(prediction))
        predicted_emotion = class_labels[max_index]
        confidence = prediction[0][max_index] * 100

        # --- Visualization ---
        # Draw the bounding box (Blue)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Draw the Label and Confidence
        label_text = f"{predicted_emotion} ({confidence:.1f}%)"
        cv2.putText(frame, label_text, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # D. Display the resulting frame
    cv2.imshow('Real-Time Emotion Recognition', frame)

    # E. Quit logic (Press 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()