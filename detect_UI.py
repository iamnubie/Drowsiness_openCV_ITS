import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import pygame
import time
from threading import Thread
from PIL import Image
import os

# CÀI ĐẶT STREAMLIT 
st.set_page_config(page_title="Drowsiness Detector", layout="centered")

# CSS GIAO DIỆN 
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://i.pinimg.com/736x/03/bc/48/03bc4882cb83c09173b44886f3880177.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        h1 {
            color: black;
        }
        h2, h3, p {
            color: white;
        }
        .stButton > button {
            background-color: #0077b6;
            color: white;
            font-size: 18px;
            border-radius: 12px;
            padding: 10px 24px;
        }
        .stAlert {
            background-color: rgba(255, 0, 0, 0.7);
            color: white;
        }
        .stSelectbox select, .stButton button {
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề nổi bật "Tỉnh Lộ"
st.markdown("""
    <h2 style='
        text-align: center;
        color: #ffd60a;
        text-shadow: 2px 2px 4px #000000;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 0.5em;
    '>Tỉnh Lộ</h2>
""", unsafe_allow_html=True)

# GIAO DIỆN NGƯỜI DÙNG 
st.markdown('<h1>Hệ Thống Phát Hiện Buồn Ngủ</h1>', unsafe_allow_html=True)
st.markdown("Chọn âm thanh cảnh báo và nhấn 'Bắt đầu giám sát' để khởi động camera.")
alarm_option = st.selectbox("🔊 Chọn âm thanh cảnh báo", ("alarm.mp3", "alarm2.mp3", "alarm3.wav", "alarm4.mp3", "alarm5.wav"))
start_cam = st.button("🚗 Bắt đầu giám sát buồn ngủ")
EAR_text = st.empty()
alert_box = st.empty()
frame_placeholder = st.empty()

# HÀM PHÁT ÂM THANH 
def sound_alarm(path):
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"[ERROR] Không thể phát âm thanh: {e}")

# TÍNH EAR
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# THAM SỐ 
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 64
COUNTER = 0
ALARM_ON = False
ALARM_SOUND = os.path.join(os.path.dirname(__file__), alarm_option)

# LANDMARK MẮT 
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# MEDIAPIPE 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# CAMERA XỬ LÝ
if start_cam:
    cap = cv2.VideoCapture(0)
    time.sleep(1.0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape

                    eyes = {'left': LEFT_EYE, 'right': RIGHT_EYE}
                    eye_coords = {'left': [], 'right': []}

                    for eye_label, eye_indices in eyes.items():
                        for idx in eye_indices:
                            x = int(face_landmarks.landmark[idx].x * w)
                            y = int(face_landmarks.landmark[idx].y * h)
                            eye_coords[eye_label].append((x, y))

                    # Tính EAR
                    leftEAR = eye_aspect_ratio(eye_coords['left'])
                    rightEAR = eye_aspect_ratio(eye_coords['right'])
                    ear = (leftEAR + rightEAR) / 2.0
                    EAR_text.markdown(f"**👁️ EAR (Eye Aspect Ratio)**: `{ear:.3f}`")

                    # VẼ CÁC CHẤM XANH (KHÔNG VẼ ĐƯỜNG)
                    for (x, y) in eye_coords['left'] + eye_coords['right']:
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                    # Kiểm tra buồn ngủ
                    if ear < EYE_AR_THRESH:
                        COUNTER += 1
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            if not ALARM_ON:
                                ALARM_ON = True
                                t = Thread(target=sound_alarm, args=(ALARM_SOUND,))
                                t.daemon = True
                                t.start()
                            alert_box.error("⚠️ Buồn ngủ! Vui lòng nghỉ ngơi!")
                    else:
                        COUNTER = 0
                        if ALARM_ON:
                            ALARM_ON = False
                            pygame.mixer.music.stop()
                        alert_box.empty()

            # Hiển thị
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            frame_placeholder.image(img, channels="RGB")

           
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ALARM_ON:
            pygame.mixer.music.stop()
            ALARM_ON = False
        EAR_text.empty()
        alert_box.empty()
        frame_placeholder.empty()
