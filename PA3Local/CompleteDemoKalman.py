import cv2
import numpy as np
# import skvideo.io
from sklearn.externals import joblib
import sys
from scipy import stats

# The length of history labels for each track
MAX_LABEL_HISTORY = 20

# The number of frames that a track can stay without being deleted
NON_TRACKED_LIMIT = 120

# Contour elimination based on number of pixels per contour
CONTOUR_SIZE_THRESHOLD = 45

# Labels for classification
CAR_LABEL = 0
PEDESTRIAN_LABEL = 1
RANDOM_LABEL = 2

# Global ID to have unique ID for each track
TRACK_ID_COUNTER = 0

# Colors of bounding boxes of each type of track
TRACKED_COLOR = (0, 255, 0)
PREDICTED_COLOR = (255, 0, 0)
TEXT_COLOR = (0, 0, 0)

# Minimum number of point to match between object and track
MIN_PTS_THRESHOLD = 4

# Lowe good matches threshold
RATIO_THRESHOLD = 0.7


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
    def __init__(self, bounds, image, keypoints, descriptors, label):
        self.bounds = bounds
        self.image = image
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.label = label
        self.matched = 0

    def update(self):
        self.matched = 1

class kalFilter:
    def __init__(self, bounds, id):
        self.meas=[]
        self.pred=[]
        self.id = id
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


class Track:
    def __init__(self, detectedObject, id):
        self.currentBounds = detectedObject.bounds
        self.currentImage = detectedObject.image
        self.currentKeypoints = detectedObject.keypoints
        self.currentDescriptor = detectedObject.descriptors
        self.labelsList = [detectedObject.label]
        self.modeLabel = detectedObject.label
        self.id = id
        self.tracked = 1
        self.numberOfFramesNotTracked = 0

        #Kalman
        self.kalman_filter = kalFilter(self.currentBounds, self.id)

    def update(self, detectedObject):
        self.currentBounds = detectedObject.bounds
        self.currentImage = detectedObject.image
        self.currentKeypoints = detectedObject.keypoints
        self.currentDescriptor = detectedObject.descriptors
        self.tracked = 1
        self.numberOfFramesNotTracked = 0

        # Updating the labels history
        if len(self.labelsList) >= MAX_LABEL_HISTORY:
            self.labelsList.pop()
        self.labelsList.append(detectedObject.label)
        self.modeLabel = stats.mode(self.labelsList)
        self.modeLabel = self.modeLabel[0][0]


def applyDetection(im, inFrame, hog, clf):

    # Preparing Image for processing
    ret, thresh = cv2.threshold(im, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sift = cv2.xfeatures2d.SIFT_create()

    objects = []

    for contour in contours:
        if len(contour) > CONTOUR_SIZE_THRESHOLD:
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
            image = inFrame[miny:maxy, minx:maxx]
            keypoints, descriptors = sift.detectAndCompute(image, None)

            # Extracting HOG feature vector from template center
            location = ((image.shape[1] / 2, image.shape[0] / 2),)
            h = hog.compute(image, (1, 1), (2, 2), location)
            # Adding width and height as features
            addition = np.array([image.shape[0], image.shape[1]]).reshape(-1, 1)
            h = np.vstack((h, addition))
            # Classify template using pre-trained SVM classifier
            h = h.T
            label = clf.predict(h)
            label = label[0]

            #cv2.imshow("Current Object", image)
            # Add object to the list of detected objects
            objects.append(DetectedObject(bounds, image, keypoints, descriptors, label))

    return objects


def matchObjectTrack(object, track, matcher, MIN_PTS_THRESHOLD, frame_number):

    if (len(object.keypoints) > 0) and (len(track.currentKeypoints) > 0):

        matches = matcher.knnMatch(track.currentDescriptor, object.descriptors, k=2)

        # Apply Lowe's ratio test to determine a good match
        good = []
        goodList = []
        for match in matches:
            if len(match) != 2:
                continue

            m, n = match
            if m.distance <= RATIO_THRESHOLD * n.distance:
                good.append(m)
                goodList.append([m])

        goodMatches = []
        if len(good) > 0:
            # Using RANSAC to further improve the matching and eliminate wrong matches
            src_pts = np.float32([track.currentKeypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([object.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            mask = []
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if mask != None:

                matchesMask = mask.ravel().tolist()
                goodMatches = []
                index = 0
                # print len(good)
                # print mask.shape
                for match in good:
                    # print match
                    m = match
                    # print index
                    # print len(matchesMask)
                    if matchesMask[index]:
                        goodMatches.append([m])
                    index += 1

            if len(goodMatches) > MIN_PTS_THRESHOLD:
                match_img = cv2.drawMatchesKnn(track.currentImage, track.currentKeypoints, object.image, object.keypoints,
                                           goodMatches, None, flags=2)
                #cv2.imshow(str(track.id), match_img)
                #cv2.imwrite(str(frame_number) + "-" + str(track.id) + ".png",match_img)
                return True
    return False


def updateAllTracks(trackList, detectedObjectsList, frame_number):

    global TRACK_ID_COUNTER

    # Mark all tracks as not tracked and start matching
    for track in trackList:
        track.tracked = 0

    matcher = cv2.BFMatcher()

    # Check for matches between objects and tracks
    for object in detectedObjectsList:
        # Object can match with more than one track
        matchedTracks = []
        for track in trackList:
            if matchObjectTrack(object, track, matcher, MIN_PTS_THRESHOLD, frame_number):
                matchedTracks.append(track)

        #print "object matched with " + str(len(matchedTracks)) + " Tracks"
        # Selecting the oldest track and deleting the others as they are duplicates
        if len(matchedTracks) > 0:
            object.update()
            trackToKeep = matchedTracks[0]
            for track in matchedTracks:
                if track.id < trackToKeep.id:
                    trackList.remove(trackToKeep)
                    trackToKeep = track
            # Update the oldest matched track
            trackToKeep.update(object)
            # KALMAN
            track_x = trackToKeep.currentBounds.center_x
            track_y = trackToKeep.currentBounds.center_y
            trackToKeep.kalman_filter.update(track_x, track_y)

    # Increment the counter for tracks with no matches
    for track in trackList:
        if not track.tracked:
            track.kalman_filter.predict()
            track.numberOfFramesNotTracked += 1

    # Add new tracks for non matched objects
    for object in detectedObjectsList:
        if (not object.matched) and (not object.label == RANDOM_LABEL):
            trackList.append(Track(object, TRACK_ID_COUNTER))
            TRACK_ID_COUNTER += 1

    # Remove tracks that have been idle for a certain number of frames and have a consistent label of RANDOM
    trackList[:] = [track for track in trackList if (not track.numberOfFramesNotTracked > NON_TRACKED_LIMIT) and
                    (not track.modeLabel == RANDOM_LABEL)]
    print len(trackList)


def drawTracks(trackList, frame):

    outputFrame = np.copy(frame)
    """ NOTE : Use different colors for tracked tracks and predicted tracks """
    for track in trackList:

        if track.tracked:
            #print track.modeLabel
            if int(track.modeLabel) == CAR_LABEL:
                label = "Car"
            else:
                if int(track.modeLabel) == PEDESTRIAN_LABEL:
                    label = "Pedestrian"

            cv2.rectangle(outputFrame, (track.currentBounds.x1, track.currentBounds.y1),
                                    (track.currentBounds.x2, track.currentBounds.y2), TRACKED_COLOR, 1)
            cv2.putText(outputFrame, '# ' + str(track.id) + " - " + label, (track.currentBounds.x1,
                                                                            track.currentBounds.y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

        """ TODO add drawing for predicted tracks """
        # KALMAN
        # print item.kalman_filter.currentPrediction
        cv2.circle(outputFrame, track.kalman_filter.currentPrediction, 8, (0,0,255), 5)

    return outputFrame


def postProcessing(frame):


    # Create the kernel used for filtering
    kernel = np.ones((5,5),np.uint8)

    median = cv2.medianBlur(frame,5)

    # Removing small and random noise (like isolated white pts)
    opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)

    """# Expanding the white pixels to get more connected pixels
    dilation = cv2.dilate(opening,kernel,iterations = 1)

    # Filling the holes in the white pixels, to improve detection
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)"""

    return opening


def main():
    # Disabling OpenCL here cause it causes problem using BackgroundSubstractor
    # Found out online that OpenCL bindings for openCV 3.1 are not working
    cv2.ocl.setUseOpenCL(False)
    source_path = sys.argv[1]

    # Using skvideo to load and write videos because of codecs issue under Linux
    cap = cv2.VideoCapture(source_path)
    # cap = skvideo.io.VideoCapture(source_path)

    if not cap.isOpened():
        print 'Error--Unable to open video:', source_path
        return

    # Get video parameters (try to retain same attributes for output video)
    # width = cap.width
    # height = cap.height
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))

    # skvideo_writer = skvideo.io.VideoWriter("HOG_SVM_3ClassesPark.avi", frameSize=(width,height))
    # skvideo_writer.open()
    dst_writer = cv2.VideoWriter("output.avi", codec, fps, (width, height))

    # Creating the Foreground/Background segmentation based on MOG
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    # initialize the HOG descriptor To extract feature from image center - These parameters will give 32 features vector
    winSize = (8, 8)
    blockSize = (8, 8)
    blockStride = (8, 8)
    cellSize = (2, 2)
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
    clf = joblib.load('SVM_OVOLinear_CarsPeopleRandom.pkl')

    trackList = []
    # Main loop
    frame_number = 0
    while True:
        # Get frame
        ret, frame = cap.read()

        # Testing Occlusion
        cv2.rectangle(frame, (130, 60), (170, 200), (255,0,0), 40)

        if ret is False or frame is None:
            break

        frame_number += 1
        #cv2.imshow("Input", frame)

        # Apply MOG to frame
        fgmask = fgbg.apply(frame)

        # Filter the fgmask to remove noise
        filtered = postProcessing(fgmask)

        # Detect moving object in frame
        detectedObjectsList = applyDetection(filtered, frame, hog, clf)

        # Update all tracks
        updateAllTracks(trackList, detectedObjectsList, frame_number)

        outputFrame = drawTracks(trackList, frame)

        cv2.imshow("Filtered", filtered)
        cv2.imshow("Input", outputFrame)
        cv2.imshow("Foreground", fgmask)

        # Write Video To file
        # skvideo_writer.write(outputFrame)
        dst_writer.write(outputFrame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # skvideo_writer.release()
    dst_writer.release()
    cap.release()


if __name__ == '__main__':
    main()