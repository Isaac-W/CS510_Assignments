import sys
import numpy as np
import cv2
import math
import random
from matplotlib import pyplot as plt

RATIO_THRESHOLD = 0.8  # match1 / match2 must be less than this to consider a match good, and not arbitrary
MIN_MATCHES = 4  # Good matches required to consider matched!


# Init colors
random.seed(1)
colors = [(random.randrange(256), random.randrange(256), random.randrange(256)) for i in range(100)]

class Rectangle:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def center_x(self):
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self):
        return (self.y1 + self.y2) / 2

    @property
    def center(self):
        return self.center_x, self.center_y


class DetectedObject:
    def __init__(self, bounds, image, mask, keypoints, descriptors, objectNum):
        self.bounds = bounds
        self.image = image
        self.mask = mask
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.tracked = True
        self.kalman_filter = kalFilter(self.bounds, objectNum)

class kalFilter:
    def __init__(self, bounds, objectNum):
        self.meas=[]
        self.pred=[]
        self.objectNum = objectNum
        # self.frame = np.zeros((400,400,3), np.uint8) # drawing canvas
        self.mp = np.array((2,1), np.float32) # measurement
        # self.tp = np.zeros((2,1), np.float32) # tracked / prediction
        self.tp = np.array([[np.float32(bounds.center_x)],[np.float32(bounds.center_y)]])
        self.currentPrediction = (bounds.center_x,bounds.center_y)

        # cv2.namedWindow("kalman")
        # cv2.setMouseCallback("kalman",onmouse);
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * .003
        # self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03

        # APPLY KALMAN
        track_x, track_y = bounds.center_x,bounds.center_y
        self.meas.append( (track_x, track_y) )
        for i in range(100):
            self.mp = np.array([[np.float32(track_x)],[np.float32(track_y)]])
            self.kalman.correct(self.mp)
            self.predict()

        # self.fig = plt.figure(self.objectNum)
        # self.fig.add_axes([0,520,360,0])
        # self.ax = self.fig.add_subplot(2,1,1)
        # self.fig.plot([i[0] for i in self.meas],[i[1] for i in self.meas],'xr',label='measured')
        # plt.axis([0,520,360,0])
        # self.ax.hold(True)
        # self.fig.plot([i[0] for i in self.pred],[i[1] for i in self.pred],'ob',label='kalman output')
        # self.ax.legend(loc=2)
        # self.ax.title("Constant Velocity Kalman Filter")
        # self.ax.show()

    def update(self, track_x, track_y):

        # APPLY KALMAN
        self.meas.append( (track_x, track_y) )
        self.mp = np.array([[np.float32(track_x)],[np.float32(track_y)]])
        self.kalman.correct(self.mp)
        self.predict()


        # plt.figure(self.objectNum)
        # self.ax.plot([i[0] for i in self.meas],[i[1] for i in self.meas],'xr',label='measured')
        # plt.axis([0,520,360,0])
        # plt.hold(False)
        # self.ax.plot([i[0] for i in self.pred],[i[1] for i in self.pred],'ob',label='kalman output')
        # plt.legend(loc=2)
        # plt.title("Constant Velocity Kalman Filter")
        # plt.show()

    def predict(self):
        tp = self.kalman.predict()
        self.pred.append((int(tp[0]),int(tp[1])))
        self.currentPrediction = (int(tp[0]),int(tp[1]))


trackedObjects = []
objectNum = 0

def applyDetection(im, inFrame):
    #imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sift = cv2.xfeatures2d.SIFT_create()
    objects = []

    for contour in contours:
        if len(contour) > 45:
            minx = contour[0][0][0]
            maxx = contour[0][0][0]
            miny = contour[0][0][1]
            maxy = contour[0][0][1]

            for pts in contour:
                if pts[0][0] < minx:
                    minx = pts[0][0]
                if pts[0][0] > maxx:
                    maxx = pts[0][0]
                if pts[0][1] < miny:
                    miny = pts[0][1]
                if pts[0][1] > maxy:
                    maxy = pts[0][1]

            # Create DetectedObject
            bounds = Rectangle(minx, miny, maxx, maxy)
            mask = im[miny:maxy, minx:maxx]  # y1:y2, x1:x2
            image = inFrame[miny:maxy, minx:maxx]
            keypoints, descriptors = sift.detectAndCompute(image, None)

            #cv2.imshow("Current Object", image)
            global objectNum
            objects.append(DetectedObject(bounds, image, mask, keypoints, descriptors, objectNum))
            objectNum = objectNum + 1

    return objects


def postProcessing(frame):
    # Create the kernel used for filtering
    kernel = np.ones((5, 5), np.uint8)

    median = cv2.medianBlur(frame, 5)

    # Removing small and random noise (like isolated white pts)
    opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
    # Expanding the white pixels to get more connected pixels
    #dilation = cv2.dilate(opening,kernel,iterations = 1)
    # Filling the holes in the white pixels, to improve detection
    #closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    return opening


def trackObjects(frame, fgmask, objects):
    matchedTracks = {}  # Map of tracks matched to current objects

    # Find currently tracked objects
    print '*** Current tracks ****************'
    for track_index, track in enumerate(trackedObjects):
        if not track.tracked:
            # track_x = track.bounds.center_x
            # track_y = track.bounds.center_y
            # track.kalman_filter.update(track_x, track_y)
            track.kalman_filter.predict()
            continue

        # Look at new boxes, and look for matches
        # If we find a box with a match, mark it as an object
        # At the end, if an object has no box, add it to history
        # If there are boxes with no matched objects, search history
        # If no matches, add the box as a new track

        track_x = track.bounds.center_x
        track_y = track.bounds.center_y

        track.kalman_filter.update(track_x, track_y)

        # Sort objects by closest to last tracked position
        closestObjects = sorted(
            objects,
            key=lambda val:
            math.sqrt(math.pow(val.bounds.center_x - track_x, 2) +
                      math.pow(val.bounds.center_y - track_y, 2))
        )

        if track.tracked:
            closestObjects = closestObjects[:1]  # TODO Using only single closest for non-historical

        track.tracked = False  # Set to historical state

        if len(closestObjects) == 0:
            continue

        # TODO Testing -- ignore if object has moved too far!
        closestCenter = closestObjects[0].bounds.center
        dist = math.sqrt(math.pow(closestCenter[0] - track_x, 2) +
                         math.pow(closestCenter[1] - track_y, 2))
        if dist > 25:
            continue

        for detectedObject in closestObjects:
            # Match tracked object to detected object
            matcher = cv2.BFMatcher()

            # TODO There are sometimes empty descriptors (no features detected)!
            if detectedObject.descriptors is None:
                #cv2.imshow("Error objects", detectedObject.image)
                continue

            #matches = matcher.match(track.descriptors, detectedObject.descriptors)
            matches = matcher.knnMatch(track.descriptors, detectedObject.descriptors, k=2)

            # Apply Lowe's ratio test to determine a good match TODO
            good = []
            for match in matches:
                if len(match) != 2:
                    continue

                m, n = match
                if m.distance <= RATIO_THRESHOLD * n.distance:
                    good.append([m])

            # Sort them in the order of their distance
            #matches.sort(key=lambda val: val.distance)

            # Draw best matches
            match_img = cv2.drawMatchesKnn(track.image, track.keypoints,
                                           detectedObject.image, detectedObject.keypoints,
                                           good, None, flags=2)

            # TODO
            #cv2.imshow('Track ' + str(track_index), match_img)

            print 'Track ' + str(track_index) + ' and Object '+ str(objects.index(detectedObject))

            feature_count = min(len(track.keypoints), len(detectedObject.keypoints))
            match_percent = len(good) / float(feature_count) if feature_count > 0 else 0
            print str(len(good)) + ' good matches out of ' + str(feature_count) + ' features... ' + str(match_percent * 100) + '% match'

            # If match is close enough, consider the object found and update it
            #if len(matches) > 4 and matches[0].distance < 150:  # TODO More than 4 matches required
            if len(good) > MIN_MATCHES:
                # Mark as found
                index = objects.index(detectedObject)
                matchedTracks[index] = True

                # Update tracked object (TODO Get new bounds within box -- for multiple people per box) -- use homography to get transform
                # TODO Right now, the code is treating the entire bounds as the object... which is incorrect when objects overlap
                track.bounds = detectedObject.bounds
                track.image = detectedObject.image
                track.keypoints = detectedObject.keypoints
                track.descriptors = detectedObject.descriptors

                # KALMAN
                track_x = track.bounds.center_x
                track_y = track.bounds.center_y
                detectedObject.kalman_filter.update(track_x, track_y)

                track.tracked = True

                # TODO
                cv2.imshow('Track ' + str(track_index), match_img)

                break

        if not track.tracked:
            #print 'track ' + str(track_index) + ' has been lost'
            pass

    # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
    # TODO The code to look for historical tracked objects is copy-pasted from above
    # Find historical tracked objects
    print '*** Historical tracks *************'
    for track_index, track in enumerate(trackedObjects):
        if track.tracked:
            continue

        # Search all objects not matched
        for index, detectedObject in enumerate(objects):
            # Don't consider already matched objects
            if index in matchedTracks:
                continue

            # Match tracked object to detected object
            matcher = cv2.BFMatcher()

            if detectedObject.descriptors is None:
                continue

            # TODO When an object is dropped from tracking, the last image is often in a bad state (part of object offscreen, etc.)
            # TODO We need to keep track of the object's appearance historically and match to that instead (do it like ViBE)

            #matches = matcher.match(track.descriptors, detectedObject.descriptors)
            matches = matcher.knnMatch(track.descriptors, detectedObject.descriptors, k=2)

            # TODO
            # Apply Lowe's ratio test to determine a good match
            good = []
            for match in matches:
                if len(match) != 2:
                    continue

                m, n = match
                if m.distance <= RATIO_THRESHOLD * n.distance:
                    good.append([m])

            # Sort them in the order of their distance
            #matches.sort(key=lambda val: val.distance)

            # Draw best matches
            match_img = cv2.drawMatchesKnn(track.image, track.keypoints,
                                           detectedObject.image, detectedObject.keypoints,
                                           good, None, flags=2)

            # TODO
            #cv2.imshow('Track ' + str(track_index), match_img)

            feature_count = min(len(track.keypoints), len(detectedObject.keypoints))
            match_percent = len(good) / float(feature_count) if feature_count > 0 else 0
            print str(len(good)) + ' good matches out of ' + str(feature_count) + ' features... ' + str(match_percent * 100) + '% match'

            # If match is close enough, consider the object found and update it
            if len(good) > MIN_MATCHES:
                # Mark as found
                index = objects.index(detectedObject)
                matchedTracks[index] = True

                # Update tracked object (TODO Get new bounds within box -- for multiple people per box)
                track.bounds = detectedObject.bounds
                track.image = detectedObject.image
                track.keypoints = detectedObject.keypoints
                track.descriptors = detectedObject.descriptors

                # KALMAN
                track_x = track.bounds.center_x
                track_y = track.bounds.center_y
                detectedObject.kalman_filter.update(track_x, track_y)

                track.tracked = True

                # TODO
                cv2.imshow('Track ' + str(track_index), match_img)

                break


    # Add new objects to track
    for i in range(len(objects)):
        if i not in matchedTracks:
            trackedObjects.append(objects[i])


def drawObjects(frame, objects):
    global colors

    output = np.copy(frame)

    drawnBounds = []

    for i in range(len(objects)):
        item = objects[i]

        if not item.tracked:
            item.kalman_filter.predict()
            cv2.circle(output, item.kalman_filter.currentPrediction, 8, (0,0,255), 5)
            continue

        bounds = item.bounds

        if bounds.center in drawnBounds:
            continue

        cv2.rectangle(output, (bounds.x1, bounds.y1), (bounds.x2, bounds.y2), colors[i % len(colors)], 1)
        cv2.circle(output, bounds.center, 4, colors[i % len(colors)], 2)

        # KALMAN
        # print item.kalman_filter.currentPrediction
        cv2.circle(output, item.kalman_filter.currentPrediction, 8, (0,0,255), 5)

        cv2.putText(output, 'Object ' + str(i), (bounds.x1, bounds.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 1)

        drawnBounds.append(bounds.center)

    return output


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

    while True:
        # Get frame
        ret, frame = cap.read()
        cv2.rectangle(frame, (130, 60), (170, 200), (255,0,0), 40)
        if ret is False or frame is None:
            break

        cv2.imshow("Input", frame)
        # MOG FG Segmentation
        fgmask = fgbg.apply(frame)
        filtered = postProcessing(fgmask)

        objects = applyDetection(filtered, frame)
        #print 'Number of objects: ' + str(len(objects))

        trackObjects(frame, filtered, objects)
        #print 'Number of tracked: ' + str(len(trackedObjects))

        tracked = drawObjects(frame, trackedObjects)

        cv2.imshow("Input", frame)
        cv2.imshow("Foreground", filtered)
        cv2.imshow("Tracks", tracked)

        dst_writer.write(tracked)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    dst_writer.release()
    cap.release()


if __name__ == '__main__':
    main()