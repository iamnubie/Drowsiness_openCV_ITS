from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from camlayout import CamLayout  # ← sử dụng xử lý camera ở file riêng

class MenuScreen(Screen):
    pass

class CameraScreen(Screen):
    pass

class DrowsinessApp(App):
    def build(self):
        self.icon = "icon.png"
        self.title = "Tỉnh lộ"
        return Builder.load_file("main.kv")  # Load giao diện chính từ file .kv

if __name__ == '__main__':
    DrowsinessApp().run()
