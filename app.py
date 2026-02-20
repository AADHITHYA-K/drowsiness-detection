import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os

model = load_model("drowsiness_model.h5")

SEQ_LEN = 10
IMG_SIZE = 64

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < SEQ_LEN:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, SEQ_LEN).astype(int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == SEQ_LEN:
        return np.expand_dims(frames, axis=0)

    return None

st.title("Drowsiness Detection")
st.write("Upload a video to detect drowsiness")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.write("Processing video...")

    sequence = extract_frames(tfile.name)

    if sequence is not None:
        prediction = model.predict(sequence)[0][0]

        st.write(f"Prediction Probability: {prediction:.4f}")

        if prediction > 0.7:
            st.error("DROWSY DETECTED")
        else:
            st.success("ALERT DRIVER")

    else:
        st.warning("Video too short or frame extraction failed.")