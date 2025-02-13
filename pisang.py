import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLO model
model = YOLO("50nonaug.pt")

# Konfigurasi halaman
st.set_page_config(page_title="YOLO Object Detection", layout="wide")

# Sidebar untuk pengaturan model
st.sidebar.title("Pengaturan Model")
confidence_threshold = st.sidebar.slider("Threshold Confidence", 0.1, 1.0, 0.5, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.5, 0.05)

# Sidebar untuk navigasi
st.sidebar.title("Pilih Mode Input")
mode = st.sidebar.radio("Mode Deteksi", ["Gambar", "Video", "Kamera"])

st.title("YOLO Object Detection")

# Fungsi deteksi objek
def detect_objects(image, conf_thresh, iou_thresh):
    results = model(image, conf=conf_thresh, iou=iou_thresh)
    annotated_frame = results[0].plot()  # Menampilkan hasil deteksi dengan anotasi
    return annotated_frame

if mode == "Gambar":
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        col1, col2 = st.columns(2)  # Membuat layout 2 kolom

        image = Image.open(uploaded_file)
        image = np.array(image)
        processed_image = detect_objects(image, confidence_threshold, iou_threshold)

        with col1:
            st.image(image, caption="Gambar Asli", use_column_width=True)
        with col2:
            st.image(processed_image, caption="Hasil Deteksi", use_column_width=True)

elif mode == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = f"temp_video.{uploaded_video.name.split('.')[-1]}"
        with open(tfile, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile)
        stframe1, stframe2 = st.columns(2)  # Layout untuk menampilkan video berdampingan

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = detect_objects(frame_rgb, confidence_threshold, iou_threshold)

            with stframe1:
                st.image(frame_rgb, channels="RGB", caption="Video Asli")
            with stframe2:
                st.image(processed_frame, channels="RGB", caption="Hasil Deteksi")

        cap.release()

elif mode == "Kamera":
    st.write("Menggunakan Kamera untuk Deteksi Real-time")

    cap = cv2.VideoCapture(0)
    stframe1, stframe2 = st.columns(2)  # Layout untuk kamera sebelum & sesudah deteksi

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = detect_objects(frame_rgb, confidence_threshold, iou_threshold)

        with stframe1:
            st.image(frame_rgb, channels="RGB", caption="Kamera Asli")
        with stframe2:
            st.image(processed_frame, channels="RGB", caption="Hasil Deteksi")

    cap.release()
