from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from camlayout import CamLayout  # ← sử dụng xử lý camera ở file riêng
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
import tkinter as tk
from tkinter import filedialog
from kivy.app import App

class MenuScreen(Screen):
    def open_image_chooser(self):
        # Khởi tạo cửa sổ ẩn của tkinter
        root = tk.Tk()
        root.withdraw()

        # Mở File Explorer để chọn ảnh
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh từ máy",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )

        if file_path:
            app = App.get_running_app()
            app.root.get_screen('camera').ids.camlayout.detect_face_in_image(file_path)
            app.root.current = 'camera'

class CameraScreen(Screen):
    pass

class DrowsinessApp(App):
    def build(self):
        self.title = "Phát hiện buồn ngủ"
        return Builder.load_file("main.kv")  # Load giao diện chính từ file .kv

if __name__ == '__main__':
    DrowsinessApp().run()
