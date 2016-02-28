# Standard imports
import cv2
import numpy as np
import time


fourcc = cv2.VideoWriter_fourcc(*'XVID')

cap = cv2.VideoCapture("inGT.avi")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = float(cap.get(cv2.CAP_PROP_FPS))
codec = int(cap.get(cv2.CAP_PROP_FOURCC))

writer = cv2.VideoWriter("outMask.avi", codec, fps, (width, height))

k = 0

while not k == 27:

    ret, im = cap.read()
    cv2.imshow('image', im)
    print im.shape[2]
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #print len(contours)
    for contour in contours:
        if len(contour) > 45:
            minx = contour[0][0][0]
            maxx = contour[0][0][0]
            #print contour[0][0]
            miny = contour[0][0][1]
            maxy = contour[0][0][1]
            for pts in contour:
                #print pts[0]
                if pts[0][0] < minx:
                    minx = pts[0][0]
                if pts[0][0] > maxx:
                    maxx = pts[0][0]
                if pts[0][1] < miny:
                    miny = pts[0][1]
                if pts[0][1] > maxy:
                    maxy = pts[0][1]
        #print minx, miny, maxx, maxy
            cv2.rectangle(im, (minx, miny), (maxx, maxy), (0, 255, 0), 1)

    writer.write(im)

    k = cv2.waitKey(100)


# Show keypoints
writer.release()
cv2.waitKey(0)