import sys
import numpy as np
import cv2
import os

def applyDetection(im, inFrame):
    #print im.shape[2]
    #imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im, 127, 255, 0)
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
            cv2.rectangle(inFrame, (minx, miny), (maxx, maxy), (255, 0, 0), 1)

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
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print 'Error--Unable to open video:', source_path
        return

    # Get video parameters (try to retain same attributes for output video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))

    dst_writer = cv2.VideoWriter("output.avi", codec, fps, (width, height))
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    #fgbg = cv2.createBackgroundSubtractorMOG2()

    while True:
        # Get frame
        ret, frame = cap.read()
        if ret is False or frame is None:
            break

        #cv2.imshow("Input", frame)
        # MOG
        fgmask = fgbg.apply(frame)

        filtered = postProcessing(fgmask)

        applyDetection(filtered, frame)

        #dst_writer.write(fgmask)

        cv2.imshow("Filtered", filtered)
        cv2.imshow("Input", frame)
        cv2.imshow("Foreground", fgmask)

        dst_writer.write(frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    dst_writer.release()
    cap.release()


if __name__ == '__main__':
    main()