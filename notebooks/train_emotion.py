# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
# Update these paths to match your folder structure
train_dir = '/home/mohamed/autonom_ws/src/Action-Emotions_deep-learning-model/emotions/train'  # e.g., /home/mohamed/dataset/train
test_dir = '/home/mohamed/autonom_ws/src/Action-Emotions_deep-learning-model/emotions/test'    # e.g., /home/mohamed/dataset/test

img_size = (48, 48) # FER-2013 standard size
batch_size = 64     # Number of images to process at once

# ==========================================
# 2. DATA GENERATORS ( The "Pipeline" )
# ==========================================

# A. Training Generator (With Data Augmentation)
# We randomly rotate, flip, and zoom images so the model doesn't just memorize specific pixels.
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixels to 0-1 range
    rotation_range=10,       # Rotate image slightly (10 degrees)
    width_shift_range=0.1,   # Shift image horizontally
    height_shift_range=0.1,  # Shift image vertically
    zoom_range=0.1,          # Zoom in/out slightly
    horizontal_flip=True,    # Flip image left-right
    fill_mode='nearest'
)

# B. Validation/Test Generator (No Augmentation)
# We only rescale the test data. We want to test on "real" images, not distorted ones.
test_datagen = ImageDataGenerator(rescale=1./255)

print("Loading Training Data:")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode='grayscale',  # IMPORTANT: Your model expects 1 channel
    batch_size=batch_size,
    class_mode='sparse', # Use 'categorical' for 7 emotions
    shuffle=True
)

print("Loading Test/Validation Data:")
validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False # Don't shuffle validation data so we can check confusion matrix later if needed
)

# %% [markdown]
# # Model Design #

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


model = Sequential()

# --- Block 1 ---
# Knowledge Applied: Two Conv layers followed by Pooling (VGG Style)
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization()) # Helps training stabilize
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))        # Prevent overfitting


# --- Block 2 ---
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# --- Block 3 ---
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# --- Classification Head ---
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Output Layer
# Knowledge Applied: 7 classes = 7 neurons. Softmax for probability distribution.
model.add(Dense(7, activation='softmax')) 

# Summary to check the architecture
model.summary()

# %% [markdown]
# ## Training the model ##

# %%
import os

# Allow TensorFlow to use the driver's JIT if ptxas is missing
os.environ['XLA_FLAGS'] = '--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true'
from tensorflow.keras.optimizers import Adam
# ==========================================
# 4. COMPILE AND TRAIN (UPDATED FOR SPARSE)
# ==========================================
from tensorflow.keras.callbacks import EarlyStopping

# 1. Define the Early Stopping Callback
early_stopping = EarlyStopping(
    monitor='val_loss',         # Watch the validation loss
    patience=5,                 # Stop if it doesn't improve for 5 epochs
    min_delta=0.001,            # Minimum change to qualify as an improvement
    restore_best_weights=True,  # IMPORTANT: Revert to the best model found, not the last one
    verbose=1
)

# 2. Add it to your training loop

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy', # <--- CHANGED: Matches the integer labels
    metrics=['accuracy']
)



epochs = 150 


# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.n // train_generator.batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.n // validation_generator.batch_size
# )
print(f"Starting training for {epochs} epochs...")

history = model.fit(
    train_generator,
    epochs=epochs,                 # Set a high max, early stopping will cut it short
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size,
    callbacks=[early_stopping] 
)

# ==========================================
# 5. SAVE THE MODEL
# ==========================================
model.save('emotion_model_sparse.keras')
print("Model saved as 'emotion_model_sparse.keras'")

# %%
import matplotlib
# CRITICAL: This line must be BEFORE you import pyplot
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# 6. VISUALIZATION & LOGGING
# ==========================================

# A. Save the raw numbers to CSV
hist_df = pd.DataFrame(history.history)
hist_df.to_csv('training_history2.csv', index=False)
print("Training history saved to 'training_history.csv'")

# B. Plot Accuracy and Loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='lower right')
ax1.grid(True)

# Plot 2: Loss
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper right')
ax2.grid(True)

# Save the training graphs
plt.savefig('training_graphs.png', dpi=300)
print("Graphs saved as 'training_graphs.png'")
plt.close() # Close to free up memory

# ==========================================
# 7. CONFUSION MATRIX
# ==========================================
print("\nGenerating Confusion Matrix...")

# 1. Get Predictions
# Important: Reset generator to start from the beginning
validation_generator.reset() 

# Predict on all validation data
preds = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(preds, axis=1) # Convert probabilities to class labels (0, 1, 2...)
y_true = validation_generator.classes # True labels from the folder structure

# 2. Compute Matrix
cm = confusion_matrix(y_true, y_pred)
class_labels = list(validation_generator.class_indices.keys()) # e.g. ['angry', 'happy', ...]

# 3. Plot Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, 
            yticklabels=class_labels)

plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 4. Save Confusion Matrix
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("Confusion Matrix saved as 'confusion_matrix.png'")
plt.close()

# Optional: Print a text report for precision/recall details
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))


