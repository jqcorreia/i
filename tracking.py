# import the necessary packages
import numpy as np
import argparse
import cv2
from imutils import paths
import time
import datetime
import math

from stream import FileVideoStream

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
cap = FileVideoStream(args["stream"]).start()
time.sleep(1.0)

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
running = True
num_frames = 0
counts = []

people = {}
num_people = 0

sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

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

while True:
    # time.sleep(0.2)
    image = cap.read()
    orig = image.copy()
    num_frames += 1

    if args["skip"] > 0 and num_frames % args["skip"] != 0:
        continue

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
        (300, 300), 127.5)

    for id, (prect, tracker, points) in people.items():
        tracked, prect = tracker.update(image)
        points.append(rect_center(prect))
        people[id] = (prect, tracker, points)


    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # extract the index of the class label from the `detections`,
        # then compute the (x, y)-coordinates of the bounding box for
        # the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        box_w, box_h = endX - startX, endY - startY
        print(box_w)
        if confidence > args["confidence"] and box_w > 20:
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] != 'person':
                continue


            endX = min(image.shape[0], endX)
            endY = min(image.shape[1], endY)

            rect = (startX, startY, endX - startX, endY - startY)

            found = False
            for _, (prect, _, _) in people.items():
                if rect_dist(prect, rect) < 100:
                    found = True


            if not found:
                print(image.shape, rect)
                tracker = cv2.TrackerKCF_create()
                tracker.init(image, tuple(rect))
                people[num_people] = (rect, tracker, [rect_center(rect)])
                num_people += 1


            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0,255,255), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    for id, (prect, _, points) in people.items():
        rect = list(map(int, prect))
        (x, y, w, h) = rect
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(orig, str(id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        for x in range(0,len(points)-1):
            cv2.line(orig, points[x], points[x+1], (0, 255, 0))


    # show the output image
    cv2.imshow("Output", orig)
    cv2.waitKey(1)
