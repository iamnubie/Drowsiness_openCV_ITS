from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import pygame
from threading import Thread

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48
ALARM_SOUND = "alarm2.wav"

class CamLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.counter = 0
        self.alarm_on = False
        self.running = False

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def sound_alarm(self, path):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play(-1)
        except Exception as e:
            print(f"[ERROR] Không thể phát âm thanh: {e}")


    def start_detect(self):
        self.capture = cv2.VideoCapture(0)
        self.running = True
        Clock.schedule_interval(self.update, 1.0 / 30)

    def stop_detect(self):
        self.running = False
        if self.capture:
            self.capture.release()
        pygame.mixer.music.stop()
        Clock.unschedule(self.update)

    def update(self, dt):
        if not self.running:
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
                right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                print(f"EAR hiện tại: {ear:.3f}")

                for (x, y) in left_eye + right_eye:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                self.ids.status_label.text = f"EAR: {ear:.2f}"

                if ear < EYE_AR_THRESH:
                    self.counter += 1
                    if self.counter >= EYE_AR_CONSEC_FRAMES:
                        if not self.alarm_on:
                            self.alarm_on = True
                            t = Thread(target=self.sound_alarm, args=(ALARM_SOUND,))
                            t.daemon = True
                            t.start()
                        self.ids.status_label.text = "BUỒN NGỦ!"
                else:
                    self.counter = 0
                    if self.alarm_on:
                        self.alarm_on = False
                        pygame.mixer.music.stop()

        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.camera_view.texture = img_texture

