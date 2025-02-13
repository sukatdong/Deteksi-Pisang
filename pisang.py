import streamlit as st
import cv2
import torch
import numpy as np
import av
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load YOLO model
MODEL_PATH = "best.pt"  # Pastikan model ada di folder yang sama
model = YOLO(MODEL_PATH)

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="YOLO Object Detection", layout="wide")

# Sidebar untuk pengaturan
st.sidebar.title("🔧 Pengaturan Model")
confidence_threshold = st.sidebar.slider("Threshold Confidence", 0.1, 1.0, 0.5, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.5, 0.05)

# Sidebar untuk navigasi
st.sidebar.title("📸 Pilih Mode Deteksi")
mode = st.sidebar.radio("Mode Deteksi", ["Gambar", "Video", "Kamera"])

st.title("🚀 YOLO Object Detection")

# Fungsi deteksi objek
def detect_objects(image, conf_thresh, iou_thresh):
    results = model(image, conf=conf_thresh, iou=iou_thresh)
    annotated_frame = results[0].plot()  # Gambar hasil deteksi
    return annotated_frame

# Mode: Deteksi pada Kamera (Realtime)
if mode == "Kamera":
    st.write("🎥 Deteksi dari Kamera (WebRTC)")

    class YOLOVideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")  # Konversi frame ke NumPy
            processed_img = detect_objects(img, confidence_threshold, iou_threshold)
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

    # Stream kamera langsung dengan model YOLO
    webrtc_streamer(
        key="camera",
        video_transformer_factory=YOLOVideoTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
