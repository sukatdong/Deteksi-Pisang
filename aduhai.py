import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLOv8 model
model = YOLO('pakdo.pt')

# Streamlit UI
st.title('Deteksi Penyakit Pisang dengan YOLOv8n')
st.write("Aplikasi ini mendeteksi penyakit pada pisang dari gambar dan kamera langsung.")

# Opsi untuk deteksi gambar atau kamera
option = st.selectbox("Pilih Mode Deteksi", ["Deteksi Gambar", "Deteksi Kamera"])

# Fungsi untuk deteksi gambar
def detect_image(image):
    results = model.predict(image)  # Prediksi menggunakan YOLOv8
    result_image = results[0].plot()  # Visualisasi hasil
    return result_image

# Deteksi dari Gambar
if option == "Deteksi Gambar":
    uploaded_file = st.file_uploader("Unggah Gambar", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        # Deteksi penyakit pisang
        st.write("Mendeteksi...")
        result_image = detect_image(np.array(image))
        st.image(result_image, caption="Hasil Deteksi", use_column_width=True)

# Deteksi dari Kamera
if option == "Deteksi Kamera":
    st.write("Arahkan kamera pada objek untuk mendeteksi penyakit pisang.")

    # Membuka kamera
    cap = cv2.VideoCapture(0)

    # Placeholder untuk menampilkan hasil terbaru di atas
    camera_placeholder = st.empty()

    run_camera = st.button("Mulai Deteksi Kamera")
    stop_camera = st.button("Stop Deteksi Kamera")

    if run_camera:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal mengakses kamera.")
                break

            # Deteksi penyakit pisang di frame kamera
            result_image = detect_image(frame)

            # Tampilkan frame dengan deteksi di placeholder (selalu terbarui di atas)
            camera_placeholder.image(result_image, channels="BGR", use_column_width=True)

            # Break jika tombol stop ditekan
            if stop_camera:
                break

    cap.release()
