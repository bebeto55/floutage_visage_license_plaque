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

st.title("🔒 YOLO Floutage Stable (visages + plaques)")

mode = st.sidebar.radio(
    "Mode",
    ["Image", "Vidéo", "Webcam", "Téléphone"]
)

target = st.sidebar.selectbox(
    "Que flouter ?",
    ["Visages", "Plaques", "Les deux"]
)

st.sidebar.markdown("## ⚙️ Paramètres YOLO")

conf = st.sidebar.slider("Confiance", 0.1, 0.9, 0.35, 0.05)
iou = st.sidebar.slider("IoU", 0.1, 0.9, 0.45, 0.05)
imgsz = st.sidebar.select_slider("Résolution", [320, 416, 512], value=320)

# ====================================================
# FACE BLUR (optimisé)
# ====================================================

def blur_faces(img, model):

    results = model.predict(
        img,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False
    )

    boxes = results[0].boxes.xyxy.cpu().numpy()

    for x1, y1, x2, y2 in boxes.astype(int):

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        blur = cv2.GaussianBlur(roi, (51, 51), 30)
        img[y1:y2, x1:x2] = blur

    cv2.putText(
        img,
        f"Visages: {len(boxes)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    return img

# ====================================================
# PLATE BLUR
# ====================================================

def blur_plates(img, model):

    results = model.predict(
        img,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False
    )

    boxes = results[0].boxes.xyxy.cpu().numpy()

    for x1, y1, x2, y2 in boxes.astype(int):

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

        roi = img[y1:y2, x1:x2]
        if roi.size:
            img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)

    cv2.putText(
        img,
        f"Plaques: {len(boxes)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    return img

# ====================================================
# PROCESS
# ====================================================

def process(img):

    if mode in ["Webcam", "Téléphone"]:
        imgsz_live = 320
    else:
        imgsz_live = imgsz

    global face_model, plate_model

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
# WEBCAM / TELEPHONE (STABLE VERSION)
# ====================================================

elif mode in ["Webcam", "Téléphone"]:

    skip = st.slider("Skip frames", 1, 5, 2)

    class VideoProcessor(VideoProcessorBase):

        def __init__(self):
            self.frame_count = 0
            self.last_frame = None

        def recv(self, frame):

            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            # traiter seulement 1 frame sur N
            if self.frame_count % skip == 0:
                self.last_frame = process(img)
            else:
                if self.last_frame is None:
                    self.last_frame = img

            return av.VideoFrame.from_ndarray(
                self.last_frame,
                format="bgr24"
            )

    webrtc_streamer(
        key="stream",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False
    )
