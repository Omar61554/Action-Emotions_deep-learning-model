import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time

# --- Configuration ---
SEQUENCE_LENGTH = 15
IMG_HEIGHT = 224
IMG_WIDTH = 224
# Update this list to match the SORTED list of folder names from your training data
CLASSES_LIST = ["Basketball", "CliffDiving", "Diving", "Haircut", "MilitaryParade", "SalsaSpin", "TaiChi", "WritingOnBoard"] 
MODEL_PATH = "/home/mohamed/autonom_ws/src/Action-Emotions_deep-learning-model/models/final_action_recognition_model_SGD_new_data.keras"
# --- FPS CONTROL ---
DESIRED_FPS = 3 # Set this to match the effective FPS of your training data
FRAME_TIME_INTERVAL = 1 / DESIRED_FPS 

def predict_on_live_video():
    model = tf.keras.models.load_model(MODEL_PATH)
    video_reader = cv2.VideoCapture(0)
    
    # Optional: Set camera to high FPS to ensure smooth preview, 
    # even if we process slowly for the model
    video_reader.set(cv2.CAP_PROP_FPS, 30) 

    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''
    
    # Track time for FPS limiting
    prev_time = time.time()

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok: break

        # Current time
        current_time = time.time()
        
        # --- FPS LIMITER LOGIC ---
        # Only process the frame for the MODEL if enough time has passed
        if (current_time - prev_time) > FRAME_TIME_INTERVAL:
            
            # Reset timer
            prev_time = current_time

            # 1. Preprocess
            resized_frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            normalized_frame = preprocess_input(rgb_frame.astype(np.float32))

            # 2. Add to Queue
            frames_queue.append(normalized_frame)

            # 3. Predict (only if queue is full)
            if len(frames_queue) == SEQUENCE_LENGTH:
                input_sequence = np.expand_dims(list(frames_queue), axis=0)
                predicted_probs = model.predict(input_sequence, verbose=0)[0]
                predicted_class_name = CLASSES_LIST[np.argmax(predicted_probs)]

        # --- VISUALIZATION (Runs at full speed for smooth video) ---
        cv2.putText(frame, f"Action: {predicted_class_name}", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        
        cv2.imshow('Action Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_reader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_on_live_video()
