import ujson
import utils
import numpy as np
import datetime
import cv2

class OccupancyProcessor():
    def __init__(self, min_confidence):
        self.min_confidence = min_confidence
        self.counts = []
        self.topic = 'occupancy'

    def update(self, data):
        (_, boxes, scores, classes, num) = data
        count = 0
        for i in range(int(num[0])):
            if scores[0][i] < self.min_confidence:
                continue
            count += 1

        self.counts.append(count)
    def serialize(self):
        msg = {
            'ts' : int(datetime.datetime.now().strftime("%s%f")),
            'occupancy' : max(self.counts)
        }
        return (self.topic, msg)

    def reset(self):
        self.counts = []

class PointsProcessor():
    def __init__(self, min_confidence):
        self.min_confidence = min_confidence
        self.points = []
        self.topic = 'points'

    def update(self, data):
        (image, boxes, scores, classes, num) = data
        (h, w) = image.shape[:2]

        for i in range(int(num[0])):
            if scores[0][i] > self.min_confidence:
                box = boxes[0][i] * np.array([h, w, h, w])
                (startY, startX, endY, endX) = box.astype("int")
                center = utils.rect_center((startX, startY, endX - startX, endY - startY))
                self.points.append(list(center))

    def serialize(self):
        msg = {
            'ts' : int(datetime.datetime.now().strftime("%s%f")),
            'data' : self.points
        }
        return (self.topic, msg)

    def reset(self):
        self.points = []
