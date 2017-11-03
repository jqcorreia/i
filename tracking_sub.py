import numpy as np
import argparse
import imutils
import cv2
import datetime
import math
import sys
from imutils.object_detection import non_max_suppression
import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--stream", required=True,
    help="path to input image")
ap.add_argument("-v", "--view", type=bool, default=False,
    help="Display results in window")
args = vars(ap.parse_args())

j = 1
people = {}

cap = cv2.VideoCapture(args["stream"])
subtractor = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=1000)
last = datetime.datetime.now()

def rect_intersect(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return (0,0,0,0)
    return (x, y, w, h)

def rect_contains(rec, point):
    return point[0] > rec[0] and point[0] < rec[0] + rec[2] and point[1] > rec[1] and point[1] < rec[1] + rec[3]

def rect_center(rec):
    return (int(rec[0] + rec[2] / 2),int(rec[1] + rec[3] / 2 ))

def rect_dist(rec1, rec2):
    cx1, cy1 = ((rec1[0] + rec1[2] )/ 2,(rec1[1] + rec1[3] )/ 2 )
    cx2, cy2 = ((rec2[0] + rec2[2] )/ 2,(rec2[1] + rec2[3] )/ 2 )

    return math.sqrt(math.pow(cx1 - cx2, 2) + math.pow(cy1 - cy2, 2))

def crop_image(image, rec, padding):
    return image[rect[1]-padding:rect[1]+rect[3]+padding, rect[0]-padding:rect[0]+rect[2]+padding]


def create_hist(image):
    hist = cv2.calcHist([image],[0,1,2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist,hist)

    return hist

running = True
stride = 8
people = {}
j = 0
print("Loading model...")
# model = VGG16(weights="imagenet")

while running:
    current = datetime.datetime.now()

    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    running, image = cap.read()

    delta = current - last

    image = imutils.resize(image, width=min(320, image.shape[1]))
    imageArea = image.shape[0] * image.shape[1]

    orig = image.copy()

    masked = subtractor.apply(orig)
    _,contour,_=cv2.findContours(masked,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    rects = np.array([ cv2.boundingRect(cnt) for cnt in contour])
    rects = non_max_suppression(rects, overlapThresh = 0.9)

    for rect in rects:
        area = rect[2] * rect[3]
        if area > imageArea * 0.05 and area <= imageArea - 10:
            point1 = (rect[0], rect[1])
            point2 = (rect[0] + rect[2], rect[1] + rect[3])
            cv2.rectangle(orig, point1, point2,(255,0,0), 2)

            found = False
            for pid, people_rect in people.items():
                res = rect_intersect(people_rect, rect)
                resArea = res[2] * res[3]
                # print(area, resArea, resArea / area)
                if resArea / area > 0.1:
                    found = True
                    print("Found person", pid)
                    people[pid] = people_rect

            if not found:
                j = j +1
                people[j] = rect
                print("New person", j, rect)

    if args["view"]:
        cv2.imshow("Masked", masked)
        cv2.imshow("Original", orig)
        cv2.waitKey(1)


    # print(datetime.datetime.now() - current)
