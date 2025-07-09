import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import os

# Load model
model = tf.keras.models.load_model("steel_defect_model.h5")

# Load labels
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Path to single test image
img_path = r"C:/One/NEU-DET/train/images/patches/patches_10.jpg"

# Check if image exists
if not os.path.exists(img_path):
    print(f"❌ Image not found: {img_path}")
    exit()

print(f"\n📷 Testing: {img_path}")

# Load and preprocess image
image = Image.open(img_path).convert("RGB")
image = image.resize((200, 200))
img_array = np.array(image) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array, verbose=0)
pred_class = labels[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"✅ Prediction: {pred_class} ({confidence:.2f}%)")
