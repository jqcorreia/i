from threading import Thread
from queue import Queue
import cv2

class VideoStream:
    def __init__(self, path, name):
        self.cap = cv2.VideoCapture(path)
        self.num_frames = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.name = name

    def read(self):
        running, image = self.cap.read()
        self.num_frames += 1
        self.seconds_elapsed = self.num_frames / self.fps
        return running, image

