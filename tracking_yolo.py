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

import detectors

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--stream", required=True,
    help="path to stream")
ap.add_argument("-v", "--view", type=bool, default=False,
    help="Display results in window")
args = vars(ap.parse_args())

def main():
    j = 1
    people = {}
    passages = 0
    cap = cv2.VideoCapture(args["stream"])
    running = True

    print("[INFO] loading model...")
    detector = detectors.DetectorYOLO()

    cframe = 0
    j = 0
    line = [(350, 0), (250, 500)]

    while running:
        to_remove = set()

        current = datetime.datetime.now()

        # load the image and resize it to (1) reduce detection time
        # and (2) improve detection accuracy
        running, image = cap.read()
        cframe += 1

        # if cframe % 10 != 0:
        #     continue

        orig = image.copy()

        (h, w) = image.shape[:2]
        rects = detector.detect(image)

        for rect in rects:
            cv2.rectangle(orig, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0,255,255), 2)

        # blob = cv2.dnn.blobFromImage(cv2.resize(image, (255,255)), 0.007843,
        #     (255, 255), 127.5)

        # net.setInput(blob)
        # detections = net.forward()

        # # loop over the detections
        # for i in np.arange(0, detections.shape[2]):
        #     # extract the confidence (i.e., probability) associated with the
        #     # prediction
        #     confidence = detections[0, 0, i, 2]
        #     box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #     class_id = int(detections[0, 0, i, 1])

        #     if class_id == 15 and confidence > 0.7:
        #         (startX, startY, endX, endY) = box.astype("int")
        #         rect = (startX, startY, endX - startX, endY - startY)
        #         cv2.rectangle(orig, (startX, startY), (endX, endY), (0,255,255), 2)

        #         histogram = create_hist(crop_image(image, rect, 0))

        #         bestCorrelation = 0
        #         bestCorrelationID = -1
        #         for pid, p in people.items():
        #             dist = rect_dist(rect, p["rect"])
        #             histCorrelation = cv2.compareHist(histogram, p["histogram"], cv2.HISTCMP_CORREL)

        #             if histCorrelation > bestCorrelation and histCorrelation > 0.7:
        #                 bestCorrelationID = pid
        #                 bestCorrelation = histCorrelation

        #         if bestCorrelationID != -1:
        #             pid = bestCorrelationID
        #             # print("found", pid, histCorrelation)
        #             people[pid]["rect"] = rect
        #             people[pid]["histogram"] = histogram
        #             points = people[pid]["points"]
        #             points.append(rect_center_base(rect))
        #             people[pid]["points"] = points
        #             people[pid]["last_seen"] = datetime.datetime.now()
        #         else:
        #             j += 1
        #             # print("new person", j)
        #             people[j] = { "rect" : rect, "histogram": histogram, "points" : [rect_center_base(rect)], "last_seen" : datetime.datetime.now()}

        # cv2.line(orig, line[0], line[1], (255,0,0))

        # for pid, p in people.items():
        #     dt = (datetime.datetime.now() - p["last_seen"])
        #     if len(p['points']) > 1:
        #         for i in range(1, len(p["points"]) - 1):
        #             if segment_intersect((line[0], line[1]), (p["points"][i-1], p["points"][i])) != None:
        #                 people[pid]["points"] = []
        #                 passages += 1
        #                 print("Number of passages", passages)

        #     if dt.total_seconds() * 1000 > 1000:
        #         print("Remove", pid)
        #         to_remove.add(pid)

        #     rect = p["rect"]
        #     points = p["points"]
        #     (x, y, w, h) = rect
        #     cv2.rectangle(orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #     cv2.putText(orig, str(pid), (x+int(w/2), y + int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
        #     for x in range(0,len(points)-1):
        #         cv2.line(orig, points[x], points[x+1], (0, 255, 0))

        # for pid in to_remove:
        #     del people[pid]

        if args["view"]:
            cv2.imshow("Original", orig)
            cv2.waitKey(1)

        print(datetime.datetime.now() - current)


profile.run("main()")
