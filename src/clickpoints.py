#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream

point = np.array([-1, -1])
def manejador(event, x, y, flags, param):
    global point
    if event == cv.EVENT_LBUTTONDOWN:
        point = np.array([x, y])
        print(f"({point[0]}, {point[1]})")

cv.namedWindow("webcam")
cv.setMouseCallback("webcam", manejador)

for key, frame in autoStream():
    if point[1] != -1:
        cv.circle(frame, (point[0], point[1]), 2, (0, 0, 255), 3)
        cv.putText(frame, f"({point[0]}, {point[1]})", (point[0], point[1]), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        
    cv.imshow('webcam',frame)
cv.destroyAllWindows()

