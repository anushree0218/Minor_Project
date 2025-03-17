import cv2
import time
import threading
import numpy as np
from typing import Dict, Tuple

class Camera:
    def __init__(self, config_path: str):
        """Initialize camera with configuration"""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device_id = self.config.get('device_id', 0)
        self.width = self.config.get('width', 640)
        self.height = self.config.get('height', 480)
        self.fps = self.config.get('fps', 30)
        
        self.cap = None
        self.frame = None
        self.timestamp = 0
        self.running = False
        self.thread = None
    
    def start(self):
        """Start camera in a separate thread"""
        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.device_id}")
        
        self.running = True
        self.thread = threading.Thread(target=self._update_frame)
        self.thread.daemon = True
        self.thread.start()
    
    def _update_frame(self):
        """Camera update loop"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame.copy()
                self.timestamp = time.time()
            else:
                print("Warning: Failed to capture frame")
            
            # Maintain frame rate
            time.sleep(1.0/self.fps)
    
    def get_frame(self) -> Tuple[np.ndarray, float]:
        """Get the latest frame"""
        if self.frame is None:
            return None, 0
        return self.frame.copy(), self.timestamp
    
    def stop(self):
        """Stop the camera"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()