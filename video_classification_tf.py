import tensorflow as tf
import cv2
import numpy as np
import sys
from detector import DetectorMobilenetSSD
import utils

cap = cv2.VideoCapture(sys.argv[1])
detector = DetectorMobilenetSSD("zoo/nas.pb")

max_count = 0
while True:
    running, image = cap.read()

    if running == False:
        break

    (boxes, scores, classes, num) = detector.detect(image)
    (h, w) = image.shape[:2]
    count = 0
    for i in range(int(num[0])):
        if scores[0][i] < 0.5 or classes[0][i] != 1.0:
            continue
        count +=1
        box = boxes[0][i] * np.array([h, w, h, w])
        (startY, startX, endY, endX) = box.astype("int")
        cv2.rectangle(image, (startX, startY), (endX, endY),
                (255,0,0), 2)
        y = startY - 15
        cv2.putText(image, str(scores[0][i]), (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    if count > max_count:
        print(count)
        max_count = count
    cv2.imshow("Output", image)
    cv2.waitKey(1)
