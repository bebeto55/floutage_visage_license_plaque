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
    face_model = YOLO("yolov8n-face.pt")   # <-- modèle visages
    plate_model = YOLO("best.pt")  # <-- modèle plaques
    return face_model, plate_model

face_model, plate_model = load_models()

# ====================================================
# UI
# ====================================================

st.title("🔒 Floutage intelligent YOLO (visages et plaques d'immatriculation)")

mode = st.sidebar.radio(
    "Mode",
    ["Image", "Vidéo", "Webcam", "Téléphone"]
)

target = st.sidebar.selectbox(
    "Que veux-tu flouter ?",
    ["Visages", "Plaques", "Les deux"]
)

# ====================================================
# CORE BLUR FUNCTION
# ====================================================

def blur_yolo(img, model, mode="rect"):

    results = model(img, conf=0.25, iou=0.45, imgsz=512, verbose=False)

    blurred = cv2.GaussianBlur(img, (99, 99), 30)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for box in results[0].boxes.xyxy.cpu().numpy():

        x1, y1, x2, y2 = map(int, box)

        if mode == "circle":
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            radius = max(x2 - x1, y2 - y1) // 2
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            nb_visages = len(results[0].boxes)

            cv2.putText(
                img,
                f"visages détectés : {nb_visages}",
                (45, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 20, 255),
                2
            )
        else:
            mask[y1:y2, x1:x2] = 255

    img[mask == 255] = blurred[mask == 255]
    nb_plaques = len(results[0].boxes)

    cv2.putText(
        img,
        f"Plaques : {nb_plaques}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 20, 255),
        2
    )
    return img

# ====================================================
# APPLY LOGIC
# ====================================================

def process(img):

    if target == "Visages":
        img = blur_yolo(img, face_model, mode="circle")

    elif target == "Plaques":
        img = blur_yolo(img, plate_model, mode="rect")

    else:
        img = blur_yolo(img, face_model, mode="circle")
        img = blur_yolo(img, plate_model, mode="rect")

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

        skip = st.slider("Skip frames", 1, 5, 1)

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

    skip = st.slider("Skip frames", 1, 5, 3)

    class VideoProcessor(VideoProcessorBase):

        def __init__(self):
            self.frame_count = 0
            self.last_frame = None

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            self.frame_count += 1

            if self.frame_count % skip == 0:
                self.last_frame = process(img)
            else:
                self.last_frame = img

            return av.VideoFrame.from_ndarray(
                self.last_frame,
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
        async_processing=True
    )