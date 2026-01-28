import cv2
import time


class Camera:
    
    def __init__(self, device_index=0, width=640, height=480, fps=30):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
    def open(self):
        self.cap = cv2.VideoCapture(self.device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.cap.isOpened()
    
    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None, 0
        ret, frame = self.cap.read()
        if not ret:
            return None, 0
        timestamp = time.time()
        return frame, timestamp
    
    def get_frame_rgb(self):
        frame, timestamp = self.get_frame()
        if frame is None:
            return None, 0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb, timestamp
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
