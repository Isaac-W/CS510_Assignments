import sys
import numpy as np
import cv2
import skvideo.io
from sklearn.externals import joblib
from sklearn import svm
import os

# Define Non-Maximum Suppression (www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/)

def nms(boxes, overlapThreshold):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThreshold)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def applyDetection(filteredFrame, inputFrame, hog, clf):

    ret, thresh = cv2.threshold(filteredFrame, 127, 255, 0)

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    object_number = 0

    for contour in contours:
        if len(contour) > 45:
            object_number += 1
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
            template = inputFrame[miny:maxy, minx:maxx]
            #cv2.imshow("cropped", cropped_image)
            #order = frame_number*10 + object_number
            #cv2.imwrite(str(order) + ".png", cropped_image)

            # Extracting HOG feature vector from template center
            location = ((template.shape[1] / 2, template.shape[0] / 2),)
            h = hog.compute(template, (1, 1), (2, 2), location)

            # Classify template using pre-trained SVM classifier
            h = h.T
            output = clf.predict(h)
            output = output[0]
            if output == 0:
                output = "Shadow"
            else:
                output = "Pedestrian"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(inputFrame, str(output), (minx, miny), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(filteredFrame, (minx, miny), (maxx, maxy), (0, 255, 0), 1)
            cv2.rectangle(inputFrame, (minx, miny), (maxx, maxy), (255, 0, 0), 1)

def postProcessing(frame):


    # Create the kernel used for filtering
    kernel = np.ones((5,5),np.uint8)

    median = cv2.medianBlur(frame,5)

    # Removing small and random noise (like isolated white pts)
    opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
    # Expanding the white pixels to get more connected pixels
    #dilation = cv2.dilate(opening,kernel,iterations = 1)
    # Filling the holes in the white pixels, to improve detection
    #closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    #median = cv2.medianBlur(frame,5)

    return opening

def main():
    # Disabling OpenCL here cause it causes problem using BackgroundSubstractor
    # Found out online that OpenCL bindings for openCV 3.1 are not working
    cv2.ocl.setUseOpenCL(False)
    source_path = sys.argv[1]
    cap = skvideo.io.VideoCapture(source_path)
    print(str(cap.width))
    #cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print 'Error--Unable to open video:', source_path
        return

    # Get video parameters (try to retain same attributes for output video)

    width = cap.width
    height = cap.height
    #fps = float(cap.get(cv2.CAP_PROP_FPS))
    #codec = int(cap.get(cv2.CAP_PROP_FOURCC))

    skvideo_writer = skvideo.io.VideoWriter("HOGandSVMTrainStation.avi", frameSize=(width,height))
    skvideo_writer.open()
    #dst_writer = cv2.VideoWriter("output.avi", codec, fps, (width, height))
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    #fgbg = cv2.createBackgroundSubtractorMOG2()

    # initialize the HOG descriptor To extract feature from image center
    winSize = (8, 8)
    blockSize = (8, 8)
    blockStride = (8, 8)
    cellSize = (4, 4)
    nbins = 8
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)


    # Load pre-trained SVM classifier
    clf = joblib.load('SVM_Linear_TrainedOnGardenTemplates.pkl')

    frame_number = 0
    while True:
        # Get frame
        ret, frame = cap.read()
        if ret is False or frame is None:
            break

        frame_number += 1
        #cv2.imshow("Input", frame)
        # MOG
        fgmask = fgbg.apply(frame)

        filtered = postProcessing(fgmask)
        hogFrame = frame

        applyDetection(filtered, frame, hog, clf)

        #dst_writer.write(fgmask)

        cv2.imshow("HOG", hogFrame)
        cv2.imshow("Filtered", filtered)
        cv2.imshow("Input", frame)
        cv2.imshow("Foreground", fgmask)

        skvideo_writer.write(frame)
        #dst_writer.write(frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    skvideo_writer.release()
    #dst_writer.release()
    cap.release()


if __name__ == '__main__':
    main()