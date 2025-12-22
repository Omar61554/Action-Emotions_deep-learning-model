import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================

# --- Action Model Config ---
ACTION_MODEL_PATH = "/home/mohamed/autonom_ws/src/Action-Emotions_deep-learning-model/models/new_action_model_adam.keras"
ACTION_CLASSES = ["JumpingJack", "Lunges", "Punch", "PushUps", "TaiChi", "Typing", "WallPushUps"]
SEQUENCE_LENGTH = 15
ACTION_IMG_SIZE = (224, 224)
DESIRED_ACTION_FPS = 3
FRAME_TIME_INTERVAL = 1 / DESIRED_ACTION_FPS 

# --- Emotion Model Config ---
EMOTION_MODEL_PATH = '/home/mohamed/autonom_ws/src/Action-Emotions_deep-learning-model/notebooks/3LayersVGG_Adam/emotion_model_sparse.keras'
EMOTION_LABELS = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTION_IMG_SIZE = (48, 48)

# ==========================================
# 2. INITIALIZATION
# ==========================================

def main():
    print("--- Loading Models... ---")
    
    # Load Action Model
    action_model = load_model(ACTION_MODEL_PATH)
    print("Action Model Loaded.")

    # Load Emotion Model
    emotion_model = load_model(EMOTION_MODEL_PATH, compile=False)
    print("Emotion Model Loaded.")

    # Load Face Detector
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    # Initialize Video Capture
    cap = cv2.VideoCapture(0)
    # Optional: Request 30FPS from camera hardware
    cap.set(cv2.CAP_PROP_FPS, 30) 

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get width for positioning the FPS text on the right
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # --- Runtime Variables ---
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    current_action_label = "Initializing..."
    current_action_confidence = 0.0  # <--- NEW VARIABLE
    
    # Timers
    prev_action_time = time.time()
    prev_frame_time = 0
    new_frame_time = 0
    
    print("--- System Ready. Press 'q' to quit. ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ==========================================
        # CALCULATE FPS
        # ==========================================
        new_frame_time = time.time()
        # Avoid division by zero
        if new_frame_time != prev_frame_time:
            fps = 1 / (new_frame_time - prev_frame_time)
        else:
            fps = 0
        prev_frame_time = new_frame_time
        
        # Convert to integer string
        fps_text = f"FPS: {int(fps)}"

        # Current time for action limiting
        current_time = time.time()
        
        # ------------------------------------------
        # TASK A: ACTION RECOGNITION (Time-Limited)
        # ------------------------------------------
        # Only process for Action Model if 0.33s has passed
        if (current_time - prev_action_time) > FRAME_TIME_INTERVAL:
            prev_action_time = current_time
            
            # 1. Preprocess for Action (Resize -> RGB -> MobileNet Preprocess)
            resized_frame = cv2.resize(frame, ACTION_IMG_SIZE)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            normalized_frame = preprocess_input(rgb_frame.astype(np.float32))
            
            # 2. Add to Queue
            frames_queue.append(normalized_frame)
            
            # 3. Predict (only if we have 15 frames)
            if len(frames_queue) == SEQUENCE_LENGTH:
                # Shape: (1, 15, 224, 224, 3)
                input_sequence = np.expand_dims(list(frames_queue), axis=0)
                
                # Predict
                preds = action_model.predict(input_sequence, verbose=0)[0]
                
                # --- NEW LOGIC: Get Label AND Confidence ---
                best_idx = np.argmax(preds)
                current_action_label = ACTION_CLASSES[best_idx]
                current_action_confidence = preds[best_idx] * 100

        # ------------------------------------------
        # TASK B: EMOTION RECOGNITION (Every Frame)
        # ------------------------------------------
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            try:
                roi_gray = cv2.resize(roi_gray, EMOTION_IMG_SIZE)
            except Exception:
                continue # Skip if face is too small/weird
                
            roi_normalized = roi_gray.astype('float32') / 255.0
            roi_input = np.expand_dims(roi_normalized, axis=0) # (1, 48, 48)
            roi_input = np.expand_dims(roi_input, axis=-1)     # (1, 48, 48, 1)

            emotion_preds = emotion_model.predict(roi_input, verbose=0)
            max_index = int(np.argmax(emotion_preds))
            emotion_label = EMOTION_LABELS[max_index]
            confidence = emotion_preds[0][max_index] * 100

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{emotion_label} {confidence:.0f}%", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ------------------------------------------
        # VISUALIZATION
        # ------------------------------------------
        # 1. Action Label (Top Left) - UPDATED TEXT
        cv2.rectangle(frame, (0,0), (640, 40), (0,0,0), -1) # Black banner at top
        
        if current_action_label == "Initializing...":
            display_text = f"Action: {current_action_label}"
        else:
            display_text = f"Action: {current_action_label} ({current_action_confidence:.0f}%)"

        cv2.putText(frame, display_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 2. FPS Counter (Top Right)
        # Position: Width - 150 pixels from right edge, 30 pixels down
        cv2.putText(frame, fps_text, (width - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Action + Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()