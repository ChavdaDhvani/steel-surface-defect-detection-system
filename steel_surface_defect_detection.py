#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Automatic Detection of Hot-Rolled Steel Strips Surface Defects
# using Convolutional Neural Network
#
# Trains on the NEU Metal Surface Defects Dataset
# ============================================================

# ------------------------------------------------------------
# 1. Importing Libraries
# ------------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from sklearn.datasets import load_files
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img
import pickle

# ------------------------------------------------------------
# 2. Define Data Paths
# ------------------------------------------------------------

train_dir = r"C:\steel-surface-defect-detection-system\NEU-DET\train\images"
val_dir   = r"C:\steel-surface-defect-detection-system\NEU-DET\validation\images"

# Check folders
print("✅ Found train classes:", os.listdir(train_dir))
print("✅ Found validation classes:", os.listdir(val_dir))

# Example image counts
for cls in os.listdir(train_dir):
    print(f"Class {cls} - Train Images:", len(os.listdir(os.path.join(train_dir, cls))))

# ------------------------------------------------------------
# 3. Data Pre-processing
# ------------------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(200, 200),
    batch_size=10,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(200, 200),
    batch_size=10,
    class_mode='categorical'
)

# Save the detected class names for later use
labels = list(train_generator.class_indices.keys())
print("✅ Labels saved:", labels)

# ------------------------------------------------------------
# Early Stopping Callback
# ------------------------------------------------------------

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.98:
            print("\nReached 98% accuracy, stopping training!")
            self.model.stop_training = True

# ------------------------------------------------------------
# 4. CNN Model Definition
# ------------------------------------------------------------

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(200, 200, 3)),
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

# Plot model architecture
try:
    plot_model(
        model,
        to_file='cnn_architecture.png',
        show_shapes=True
    )
    print("✅ Model architecture plot saved.")
except Exception as e:
    print("⚠️ Plot model skipped (pydot missing?):", e)

# ------------------------------------------------------------
# 5. Train Model
# ------------------------------------------------------------

callbacks = myCallback()

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[callbacks],
    verbose=1,
    shuffle=True
)

# ------------------------------------------------------------
# 6. Plot Accuracy & Loss
# ------------------------------------------------------------

sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))

plt.subplot(211)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

plt.subplot(212)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. Evaluate on Validation Data (Optional Visualization)
# ------------------------------------------------------------

# Load all images from validation for quick test predictions

def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files, targets, target_labels

x_val, y_val, target_labels = load_dataset(val_dir)
no_of_classes = len(np.unique(y_val))
y_val = to_categorical(y_val, no_of_classes)

def convert_image_to_array(files):
    images_as_array = []
    for file in files:
        images_as_array.append(img_to_array(load_img(file, target_size=(200, 200))))
    return images_as_array

x_val = np.array(convert_image_to_array(x_val))
print('Validation set shape:', x_val.shape)

x_val = x_val.astype('float32') / 255

# Predict
y_pred = model.predict(x_val, verbose=1)

# Plot sample predictions
fig = plt.figure(figsize=(10, 10))
for i, idx in enumerate(np.random.choice(x_val.shape[0], size=16, replace=False)):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_val[idx]))
    pred_idx = np.argmax(y_pred[idx])
    true_idx = np.argmax(y_val[idx])
    ax.set_title(
        f"{target_labels[pred_idx]} ({target_labels[true_idx]})",
        color="green" if pred_idx == true_idx else "red"
    )

plt.show()

# ------------------------------------------------------------
# 8. Save Model & Labels
# ------------------------------------------------------------

model.save("steel_defect_model.h5")
print("✅ Model saved successfully.")

with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)
print("✅ Labels saved successfully.")

print("Current working directory:", os.getcwd())
