# import the necessary packages
import numpy as np
import argparse
import cv2
from imutils import paths
import time
import datetime
import sys

from stream import FileVideoStream
from svideo import VideoServer

from multiprocessing import Queue, Value, Process

import signal

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--stream", required=True,
    help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
ap.add_argument("-k", "--skip", type=int, default=0,
    help="frame skip")

args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
cap = cv2.VideoCapture(args["stream"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
running = True
num_frames = 0
counts = []
queue = Queue()
vs = VideoServer(queue)

p = Process(target=vs.start)
p.start()

def signal_handler(signal, frame):
    p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

last_measure_dt = datetime.datetime.now()
last_measure = 0

while True:
    # time.sleep(0.2)
    running, image = cap.read()
    num_frames += 1

    dt_since_last = datetime.datetime.now() - last_measure_dt

    if dt_since_last.total_seconds() > 10 and len(counts) > 0:
        last_measure = max(counts)
        last_measure_dt = datetime.datetime.now()
        print(last_measure_dt, last_measure)
        counts = []

    if args["skip"] > 0 and num_frames % args["skip"] != 0:
        continue

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
        (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    image = cv2.blur(image, (20,20))
    count = 0
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] != 'person':
                continue
            count += 1

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    cv2.putText(image, str(last_measure), (w - 100, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0,0), 2)

    counts.append(count)
    queue.put(image)

p.terminate()
