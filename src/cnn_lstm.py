import numpy as np
import os
import cv2
import random
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam,SGD,Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
# --- Configuration (Model Hyperparameters) ---
SEQUENCE_LENGTH = 15
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
NUM_CLASSES = 8 # This will be updated based on actual data
LSTM_UNITS = 128
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4

# --- 1. Data Gathering and Splitting Function ---

def get_all_clips_and_split(data_root, train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Scans the data_root, compiles a list of all clips, and splits them into 
    train, validation, and test sets based on the specified ratios.

    Args:
        data_root (str): Root directory containing class folders, which contain clip folders.
        ... ratios and random_seed ...

    Returns:
        tuple: (train_clips, validation_clips, test_clips)
               Each is a list of tuples: [(clip_path, label_index), ...]
    """
    if round(train_ratio + validation_ratio + test_ratio, 4) != 1.0:
        raise ValueError("The sum of train, validation, and test ratios must equal 1.0")

    print(f"Scanning directory: {data_root}")

    clips = []
    # 1. Map class names to integer labels (0 to NUM_CLASSES-1)
    class_names = sorted([d for d in os.listdir(data_root) 
                          if os.path.isdir(os.path.join(data_root, d))])
    class_map = {name: idx for idx, name in enumerate(class_names)}
    
    # 2. Compile list of all clips and their corresponding labels
    for class_name, class_idx in class_map.items():
        class_dir = os.path.join(data_root, class_name)
        for clip_name in os.listdir(class_dir):
            clip_path = os.path.join(class_dir, clip_name)
            if os.path.isdir(clip_path):
                # We store the full path and the integer label
                clips.append((clip_path, class_idx))

    if not clips:
        raise FileNotFoundError(f"No video clips found in '{data_root}' structure.")

    # 3. Perform Two-Step Splitting
    labels = [item[1] for item in clips]
    temp_ratio = validation_ratio + test_ratio
    
    # --- Split 1: Train vs. Temporary (Val + Test) ---
    train_clips, temp_clips, _, temp_labels = train_test_split(
        clips, labels, 
        test_size=temp_ratio, 
        random_state=random_seed,
        stratify=labels
    )
    
    # --- Split 2: Validation vs. Test ---
    test_size_relative = test_ratio / temp_ratio
    
    validation_clips, test_clips, _, _ = train_test_split(
        temp_clips, temp_labels,
        test_size=test_size_relative,
        random_state=random_seed,
        stratify=temp_labels
    )
    
    print(f"Total Clips Found: {len(clips)}")
    print(f"Split Results: Train={len(train_clips)}, Val={len(validation_clips)}, Test={len(test_clips)}")
    print(f"Class Names: {class_map}")
    
    return train_clips, validation_clips, test_clips, class_map

# --- 2. Updated VideoFrameGenerator Class ---

class VideoFrameGenerator(Sequence):
    def __init__(self, clip_list, sequence_length, img_size, batch_size, num_classes, shuffle=True):
        """
        Initializes the generator using a pre-split list of clips.

        Args:
            clip_list (list): The list of (clip_path, label_index) tuples 
                              specific to this split (train, val, or test).
            ... other hyperparameters ...
        """
        self.clips = clip_list
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        
        # Ensure NumPy is seeded correctly for workers
        np.random.seed(42) 
        self.on_epoch_end()
        
    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.clips) / self.batch_size))

    def on_epoch_end(self):
        """Updates indices after each epoch."""
        self.indices = np.arange(len(self.clips))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        """Generates one batch of data."""
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_clips = [self.clips[k] for k in batch_indices]
        X, Y = self._data_generation(batch_clips)
        return X, Y

    def _data_generation(self, batch_clips):
        """
        Loads, processes, and formats a batch of video sequences.
        """
        X = np.empty((self.batch_size, self.sequence_length, self.img_size[0], self.img_size[1], CHANNELS), dtype=np.float32)
        Y = np.empty((self.batch_size), dtype=int)

        for i, (clip_path, class_idx) in enumerate(batch_clips):
            frame_files = sorted(os.listdir(clip_path))
            total_frames = len(frame_files)
            
            # --- Frame Sampling Logic ---
            if total_frames > self.sequence_length:
                # Sub-sample: Select frames uniformly spaced
                indices = np.linspace(0, total_frames - 1, self.sequence_length).astype(int)
            else:
                # Padding: Repeat sequence elements and truncate (simple padding)
                indices = np.arange(self.sequence_length) % total_frames
                
            sampled_files = [frame_files[idx] for idx in indices]

            # Process each sampled frame
            for j, frame_file in enumerate(sampled_files):
                frame_path = os.path.join(clip_path, frame_file)
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    # Fallback for missing/corrupt frame
                    frame = np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.uint8)
                else:
                    frame = cv2.resize(frame, (self.img_size[1], self.img_size[0]))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                X[i, j,] = frame.astype(np.float32)

            Y[i] = class_idx

        # Preprocess (MobileNet specific normalization) and One-Hot Encode
        X = preprocess_input(X) 
        Y = to_categorical(Y, num_classes=self.num_classes)
        
        return X, Y

# --- 3. Model Definition (Same as before) ---

def create_cnn_feature_extractor():
    """
    Creates the MobileNetV2 feature extractor, truncated and frozen.
    """
    cnn_base = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    )
    x = cnn_base.output
    x = GlobalAveragePooling2D()(x)
    
    feature_extractor_model = Model(inputs=cnn_base.input, outputs=x, name='mobile_net_feature_model')
    feature_extractor_model.trainable = False
    
    return feature_extractor_model

def build_cnn_lstm_model():
    """
    Builds the complete CNN-LSTM model with TimeDistributed and stacked LSTMs.
    """
    feature_extractor = create_cnn_feature_extractor()
    
    # 1. Input layer: (Batch, T, H, W, C)
    video_input = Input(
        shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, CHANNELS), 
        name='video_input'
    )
    
    # 2. TimeDistributed CNN: Applies MobileNet to every frame
    cnn_features = TimeDistributed(feature_extractor, name='time_distributed_cnn')(video_input)
    
    # 3. Stacked LSTM Layers
    lstm_1 = LSTM(
        LSTM_UNITS, 
        return_sequences=True, 
        name='lstm_layer_1'
    )(cnn_features)
    lstm_1 = Dropout(0.5)(lstm_1)

    lstm_2 = LSTM(
        LSTM_UNITS, 
        return_sequences=False, 
        name='lstm_layer_2'
    )(lstm_1)
    
    # 4. Final Classification Layer
    output_tensor = Dense(NUM_CLASSES, activation='softmax', name='classification_output')(lstm_2)
    
    model = Model(inputs=video_input, outputs=output_tensor)
    
    return model

# --- 4. Main Training Pipeline ---

def run_training_pipeline(data_root_path):
    """
    Runs the full pipeline: split data, create generators, build model, and train.
    """
    # 1. Gather and Split Data
    train_clips, val_clips, test_clips, class_map = get_all_clips_and_split(
        data_root=data_root_path,
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15
    )
    
    # Update NUM_CLASSES based on actual data discovered
    global NUM_CLASSES
    NUM_CLASSES = len(class_map)

    # 2. Create Generator Instances
    train_generator = VideoFrameGenerator(
        clip_list=train_clips,
        sequence_length=SEQUENCE_LENGTH,
        img_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        shuffle=True
    )

    validation_generator = VideoFrameGenerator(
        clip_list=val_clips,
        sequence_length=SEQUENCE_LENGTH,
        img_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        shuffle=False
    )
    
    test_generator = VideoFrameGenerator(
        clip_list=test_clips,
        sequence_length=SEQUENCE_LENGTH,
        img_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        shuffle=False
    )
    
    # 3. Build and Compile Model
    print("\n--- Building Model ---")
    model = build_cnn_lstm_model()
    
    model.compile(
        optimizer= SGD(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 4. Train the Model
    print("\n--- Starting Training ---")
    checkpoint_callback = ModelCheckpoint(
    filepath='best_weights_SGD.keras',
    monitor='val_loss',
    mode='min',
    save_best_only=True
    )
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint_callback]
    )
   
    
    # 5. Evaluate on Test Set
    print("\n--- Evaluating on Test Set ---")
    loss, accuracy = model.evaluate(test_generator)
    print(f"Final Test Loss: {loss:.4f}")
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")
    
    # --- Detailed Class Performance Analysis ---
    print("\n--- Detailed Classification Report ---")
    
    # 1. Get predictions from the trained model
    Y_pred_raw = model.predict(test_generator)
    
    # 2. Convert raw probabilities to predicted class indices (e.g., 0, 1, 2...)
    Y_pred = np.argmax(Y_pred_raw, axis=1)
    # 3. Get true labels from the generator and convert to class indices
    Y_true_one_hot = np.concatenate([test_generator[i][1] for i in range(len(test_generator))])
    Y_true = np.argmax(Y_true_one_hot, axis=1) 
    
    # Slice Y_true to match Y_pred length if the last batch was incomplete
    Y_true = Y_true[:len(Y_pred)]
    
    # Ensure class_labels is correctly defined earlier in the function
    # Display the Classification Report (Per-Class Metrics)
    print("\nClassification Report (Per-Class Metrics):")
    class_labels = [name for name, index in sorted(class_map.items(), key=lambda item: item[1])]
    print(classification_report(Y_true, Y_pred, target_names=class_labels, zero_division=0))

    # Display the Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(Y_true, Y_pred)
    print(cm)
    model.save('final_action_recognition_model_SGD_new_data.keras')
    #plot confusion matrix using matplotlib if needed
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    
    

# --- 5. Execution ---
if __name__ == '__main__':
    DATA_ROOT = 'CV Data\extracted_frames'  # Update this path as needed 
    if not os.path.exists(DATA_ROOT):
       print(f"Data root '{DATA_ROOT}' does not exist. Please provide a valid path.")
       exit(1)

    # Run the full pipeline
    run_training_pipeline(DATA_ROOT)
