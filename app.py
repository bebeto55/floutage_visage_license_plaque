import torch
torch.set_num_threads(2)

import streamlit as st
import cv2
import numpy as np
import tempfile
import av
import os

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# ====================================================
# MODELS
# ====================================================

@st.cache_resource
def load_models():
    face_model = YOLO("yolov8n-face.pt")
    plate_model = YOLO("best.pt")
    return face_model, plate_model

face_model, plate_model = load_models()

# ====================================================
# UI
# ====================================================

st.title("🔒 YOLO Floutage Visages & Plaques")

mode = st.sidebar.radio(
    "Mode",
    ["Image", "Vidéo", "Webcam", "Téléphone"]
)

target = st.sidebar.selectbox(
    "Que flouter ?",
    ["Visages", "Plaques", "Les deux"]
)

# ====================================================
# VISAGE
# ====================================================

def blur_faces(img, model):

    results = model.predict(
        img,
        conf=0.35,
        iou=0.45,
        imgsz=416,
        verbose=False
    )

    boxes = results[0].boxes.xyxy.cpu().numpy()

    for x1, y1, x2, y2 in boxes.astype(int):

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

        roi = img[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)

    cv2.putText(
        img,
        f"Visages: {len(boxes)}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    return img

# ====================================================
# PLAQUES
# ====================================================

def blur_plates(img, model):

    results = model.predict(
        img,
        conf=0.35,
        iou=0.45,
        imgsz=416,
        verbose=False
    )

    boxes = results[0].boxes.xyxy.cpu().numpy()

    for x1, y1, x2, y2 in boxes.astype(int):

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

        roi = img[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)

    cv2.putText(
        img,
        f"Plaques: {len(boxes)}",
        (50, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    return img

# ====================================================
# PROCESS
# ====================================================

def process(img):

    if target == "Visages":
        return blur_faces(img, face_model)

    elif target == "Plaques":
        return blur_plates(img, plate_model)

    else:
        img = blur_faces(img, face_model)
        img = blur_plates(img, plate_model)
        return img

# ====================================================
# IMAGE
# ====================================================

if mode == "Image":

    file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img = process(img)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# ====================================================
# VIDEO
# ====================================================

elif mode == "Vidéo":

    video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

    if video_file:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()

        skip = st.slider("Skip frames", 1, 5, 2)

        i = 0

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            i += 1
            if i % skip != 0:
                continue

            frame = process(frame)

            frame_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

        cap.release()

# ====================================================
# WEBCAM / TELEPHONE
# ====================================================

elif mode in ["Webcam", "Téléphone"]:

    skip = st.slider("Skip frames", 1, 5, 2)

    class VideoProcessor(VideoProcessorBase):

        def __init__(self):
            self.frame_count = 0

        def recv(self, frame):

            img = frame.to_ndarray(format="bgr24")

            self.frame_count += 1

            if self.frame_count % skip == 0:
                img = process(img)

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )

    constraints = {"video": True, "audio": False}

    if mode == "Téléphone":
        constraints = {
            "video": True,
            "audio": False
        }

    webrtc_streamer(
        key="stream",
        video_processor_factory=VideoProcessor,
        media_stream_constraints=constraints,
        async_processing=False
    )
