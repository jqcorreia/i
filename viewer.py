from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import datetime
import math
import sys
import time

from stream import FileVideoStream

j = 1
people = {}

running = True

cf = 0

cap = FileVideoStream(sys.argv[1]).start()
time.sleep(1.0)
while True:
    current = datetime.datetime.now()
    to_remove = set()
    image = cap.read()

    cf +=1

    # if cf % 30 == 0:
    #     continue

    current = datetime.datetime.now()

    image = imutils.resize(image, width=min(640, image.shape[1]))
    orig = image.copy()
    # show the output images

    # cv2.imshow("Original", orig)
    # cv2.waitKey(1)
    # print((datetime.datetime.now() - current).microseconds / 1000)
