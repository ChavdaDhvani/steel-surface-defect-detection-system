import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
import tensorflow as tf
import pickle
import cv2
from PIL import Image

# Load model
model = tf.keras.models.load_model("steel_defect_model.h5")

# Load labels
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

st.title("🔎 Steel Surface Defect Detection")

st.sidebar.header("Select Mode")

mode = st.sidebar.radio("Choose an option:", ["Upload Image", "Live Webcam"])

# ---------- IMAGE UPLOAD ----------
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image of steel strip...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img_resized = image.resize((200, 200))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array, verbose=0)
        pred_class = labels[np.argmax(pred)]
        confidence = np.max(pred) * 100

        st.success(f"Predicted Defect Type: **{pred_class}** ({confidence:.1f}%)")

# ---------- LIVE WEBCAM ----------
elif mode == "Live Webcam":

    class SteelDefectDetector(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Flip horizontally for mirror effect (optional)
            img = cv2.flip(img, 1)

            # Resize for model input
            resized = cv2.resize(img, (200, 200))
            img_array = resized / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array, verbose=0)
            pred_class = labels[np.argmax(pred)]
            confidence = np.max(pred) * 100

            # Overlay prediction
            text = f"{pred_class} ({confidence:.1f}%)"
            cv2.putText(
                img,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            return img

    webrtc_streamer(
        key="steel-defect-detector",
        video_transformer_factory=SteelDefectDetector,
        media_stream_constraints={"video": True, "audio": False}
    )
