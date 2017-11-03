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

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--stream", required=True,
    help="path to input image")
ap.add_argument("-v", "--view", type=bool, default=False,
    help="Display results in window")
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
args = vars(ap.parse_args())

W = 640
H = 480

TAU = 100

visits = defaultdict(int)
XI, YI = np.meshgrid(np.arange(0,W), np.arange(0,H))

def main():
    j = 1
    people = {}
    passages = 0
    cap = cv2.VideoCapture(args["stream"])
    cap.set(cv2.CAP_PROP_FPS, 1)
    running = True

    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

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
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (255,255)), 0.007843,
            (255, 255), 127.5)

        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            class_id = int(detections[0, 0, i, 1])

            if class_id == 15 and confidence > 0.1:
                (startX, startY, endX, endY) = box.astype("int")
                rect = (startX, startY, endX - startX, endY - startY)
                center = rect_center_base(rect)

                visits[(center[0],center[1])] += 1
                # center = (center[0], int(center[1] * 1.2))
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0,255,255), 2)

                proposedPerson = -1
                closestDist = 10000

                for pid, p in people.items():
                    dist = point_dist(p["points"][-1], center)
                    if dist < TAU and dist < closestDist:
                        proposedPerson = pid
                        closestDist = dist

                if proposedPerson != -1:
                    pid = proposedPerson
                    # print("found", pid, histCorrelation)
                    people[pid]["rect"] = rect
                    points = people[pid]["points"]
                    points.append(center)
                    people[pid]["points"] = points
                    people[pid]["last_seen"] = datetime.datetime.now()
                else:
                    j += 1
                    # print("new person", j)
                    people[j] = { "rect" : rect, "points" : [center], "last_seen" : datetime.datetime.now(), "passed" : False}

        cv2.line(orig, line[0], line[1], (255,0,0))

        for pid, p in people.items():
            dt = (datetime.datetime.now() - p["last_seen"])
            if len(p['points']) > 1:
                if segment_intersect((line[0], line[1]), (p["points"][-1], p["points"][-2])) != None and not p["passed"]:
                    passages += 1
                    # people[pid]["passed"] = True
                    print("Number of passages", passages)

            if dt.total_seconds() * 1000 > 1000:
                print("Remove", pid)
                to_remove.add(pid)

            rect = p["rect"]
            points = p["points"]
            (x, y, w, h) = rect
            cv2.rectangle(orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(orig, str(pid), (x+int(w/2), y + int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
            for x in range(0,len(points)-1):
                cv2.circle(orig, points[x], 2, (0, 0, 255))
                cv2.line(orig, points[x], points[x+1], (0, 255, 0))

        for pid in to_remove:
            del people[pid]

        if args["view"]:
            grid_points = np.array(list(visits.keys()))
            lx = np.array(grid_points[:,0]).astype(float)
            ly = np.array(grid_points[:,1]).astype(float)
            lz = np.array([ visits[(gp[0], gp[1])] for gp in grid_points ]).astype(float)

            a,b,c = griddata(lx, ly, lz, (0,640), (0,480), binsize=10)
            cmap = cm.ScalarMappable(cmap="jet")
            norm = plt.Normalize(0, 3)

            img = cm.jet(norm(a))

            img = cv2.resize(img, (640,480))

            cv2.imshow("Heatmap", img)
            cv2.imshow("Original", orig)
            cv2.waitKey(1)

        # print(values)
        # print(datetime.datetime.now() - current)


main()
