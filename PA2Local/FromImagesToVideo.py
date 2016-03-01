import cv2
import numpy as np
import sys
import time
import os

startIndex = 300
endIndex = 999
width = 720
height = 576
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30.0

cap = cv2.VideoCapture(" ")

gt_writer = cv2.VideoWriter('groundTruth.avi', 0, fps, (width, height))
writer = cv2.VideoWriter('input.avi', 0, fps, (width, height))

for i in range(startIndex, endIndex+1):

    path = "..\Datasets\PETS2006\groundtruth\gt000" + str(i) + ".png"
    pathOriginal = "..\Datasets\PETS2006\input\in000" + str(i) + ".jpg"

    # Read image
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    imOut = cv2.imread(pathOriginal)
    # Write Video
    channel = np.zeros((height,width,1), np.uint8)
    combined = cv2.merge((
                cv2.bitwise_or(channel, im),
                cv2.bitwise_or(channel, im),
                cv2.bitwise_or(channel, im)
            ))
    gt_writer.write(combined)
    writer.write(imOut)

# Show keypoints
writer.release()
gt_writer.release()
cv2.waitKey(0)