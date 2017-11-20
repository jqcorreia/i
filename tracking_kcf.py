import numpy as np
import argparse
import imutils
import cv2
import datetime
import math
import sys
from imutils.object_detection import non_max_suppression
from imutils.video import fps
import profile

import time
import argparse
from utils import *

from scipy.interpolate import Rbf
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from collections import defaultdict
from detector import DetectorMobilenetSSD

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--stream", required=True,
    help="path to input image")
ap.add_argument("-v", "--view", type=bool, default=False,
    help="Display results in window")
args = vars(ap.parse_args())

W = 640
H = 480

TAU = 200

def main():
    j = 1
    people = {}
    passages = 0
    cap = cv2.VideoCapture(args["stream"])
    running = True

    detector = DetectorMobilenetSSD("zoo/mobilenet.pb")

    cframe = 0
    j = 0
    # line = [(350, 0), (250, 500)]
    line = [(500, 400), (900, 400)]
    while running:
        to_remove = set()

        current = datetime.datetime.now()

        # load the image and resize it to (1) reduce detection time
        # and (2) improve detection accuracy
        running, image = cap.read()
        cframe += 1

        # if cframe % 30 != 0:
        #     time.sleep(0.010)
        #     continue

        orig = image.copy()

        (h, w) = image.shape[:2]
        screen = (int(h * 0.10),int(w * 0.10),h - int(h * 0.10),w - int(w * 0.10))
        # blob = cv2.dnn.blobFromImage(cv2.resize(image, (255,255)), 0.007843,
        #     (255, 255), 127.5)

        # net.setInput(blob)
        # detections = net.forward()
        (boxes, scores, classes, num) = detector.detect(image)

        for pid, (tracker, prect) in people.items():
            _, prect = tracker.update(orig)
            people[pid] = (tracker, prect)

        # loop over the detections
        for i in range(int(num[0])):
            confidence = scores[0][i]
            box = boxes[0][i] * np.array([h, w, h, w])
            class_id = classes[0][i]

            if class_id == 1.0 and confidence > 0.3:
                (startY, startX, endY, endX) = box.astype("int")
                rect = (startX, startY, endX - startX, endY - startY)
                center = rect_center(rect)
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0,255,255), 2)

                found = False
                for pid, (tracker, prect) in people.items():
                    if rect_contains(prect, center):
                        found = True

                if not found:
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(image, rect)
                    prect = rect
                    j += 1
                    people[j] = (tracker, rect)

        cv2.line(orig, line[0], line[1], (255,0,0))

        for pid, (tracker, prect) in people.items():
            (x, y, w, h) = tuple(map(int, prect))

            cv2.rectangle(orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(orig, str(pid), (x+int(w/2), y + int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
            # for x in range(0,len(points)-1):
            #     cv2.circle(orig, points[x], 2, (0, 0, 255))
            #     cv2.line(orig, points[x], points[x+1], (0, 255, 0))

        for pid in to_remove:
            del people[pid]

        if args["view"]:
            cv2.imshow("Original", orig)
            cv2.waitKey(1)

        # print(values)
        # print(datetime.datetime.now() - current)


main()
