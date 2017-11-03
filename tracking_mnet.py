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
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
args = vars(ap.parse_args())

j = 1
people = {}

cap = cv2.VideoCapture(args["stream"])

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

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

cframe = 0
tracking = False
tracker = None
foo = 0

while running:
    current = datetime.datetime.now()

    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    running, image = cap.read()
    # cframe += 1

    # if cframe % 10 != 0:
    #     continue

    orig = image.copy()

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (255,255)), 0.007843,
        (255, 255), 127.5)

    net.setInput(blob)
    detections = net.forward()

    if tracking:
        tracked, rect = tracker.update(image)
        rect = [ int(i) for i in rect ]
        cv2.rectangle(orig, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255,0,0), 2)
        print(rect)
    else:
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            class_id = int(detections[0, 0, i, 1])

            if CLASSES[class_id] == 'person' and confidence > 0.5:
                (startX, startY, endX, endY) = box.astype("int")
                rect = (startX, startY, endX - startX, endY - startY)
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0,255,255), 2)
                # foo += 1
                # if foo > 20:
                tracker = cv2.TrackerKCF_create()
                tracker.init(image, rect)
                tracking = True

    # for rect in rects:
    #     point1 = (rect[0], rect[1])
    #     point2 = (rect[0] + rect[2], rect[1] + rect[3])
    #     cv2.rectangle(orig, point1, point2,(255,0,0), 2)

    if args["view"]:
        cv2.imshow("Original", orig)
        cv2.waitKey(1)


    print(datetime.datetime.now() - current)
