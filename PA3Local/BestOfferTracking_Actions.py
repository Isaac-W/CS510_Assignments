import math
import cv2
import numpy as np
import skvideo.io
from sklearn.externals import joblib
import sys
from scipy import stats
import os
import MOSSE
import copy


DEBUG_OUTPUT = True

# The length of history labels for each track
MAX_LABEL_HISTORY = 20

# The number of frames that a track can stay without being deleted
NON_TRACKED_LIMIT = 60

# Contour elimination based on number of pixels per contour
CONTOUR_SIZE_THRESHOLD = 45

# Labels for classification
CAR_LABEL = 0
PEDESTRIAN_LABEL = 1
RANDOM_LABEL = 2

# Colors of bounding boxes of each type of track
OBJECT_COLOR = (255, 0, 0)
TRACKED_COLOR = (0, 255, 0)
PREDICTED_COLOR = (0, 0, 255)
TEXT_COLOR = (0, 0, 0)

# Minimum number of point to match between object and track
MIN_PTS_THRESHOLD = 4

# Lowe good matches threshold
RATIO_THRESHOLD = 0.7

# Track offer acceptance threshold (offers above this level are accepted)
# Set to a negative number to accept any offers that intersect
TRACK_OFFER_THRESHOLD = -1

# The size of the predicted search box for track resolution
PREDICTED_WINDOW_MULTIPLIER = 1.1

# How many times larger can the destination be to still be considered the same size
# Destination windows more than this amount will only partially contribute to the track window size
WINDOW_GROWTH_THRESHOLD = 1.4

# The percentage of the destination window size to include in the track size
WINDOW_GROWTH_RATE = 0.1

# Global feature extractors
sift = None
hog = None
clf = None

# Action Recognition Parameters
ACTION_RECOGNITION_WAIT_PERIOD = 30
TRACKLET_LENGTH = 30
CUBE_X = 20
CUBE_Y = 20

BOXING = 0
HAND_CLAPPING = 1
HAND_WAVING = 2
RUNNING = 3
WALKING = 4

DIMENSIONS_TO_KEEP = 10


def initFeatureExtractors():
    global sift, hog, clf, imgsize

    # Create SIFT feature detector
    sift = cv2.xfeatures2d.SIFT_create()

    # initialize the HOG descriptor To extract feature from image center - These parameters will give 32 features vector
    imgsize = 32
    winLen = 16
    blockLen = 8
    strideLen = 8
    cellLen = 2

    winSize = (winLen,winLen)  # Size of input window
    blockSize = (blockLen,blockLen)  # Size of blocks (each block will have blockSize/cellSize cells)
    blockStride = (strideLen,strideLen)  # Block shift parameter (e.g. window of 64, block of 32, stride of 16 will have 3 blocks per row)
    cellSize = (cellLen,cellLen)  # Size of cell (a histogram is computed for each cell in a block)
    nbins = 9  # The number of bins in the histogram
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)


    # Load pre-trained SVM classifier
    clf = joblib.load('SVM_HOG_WANG.pkl')

initFeatureExtractors()


def distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


class Rectangle(object):
    CORNER_NW = 0
    CORNER_NE = 1
    CORNER_SE = 2
    CORNER_SW = 3

    SIDE_N = 0
    SIDE_E = 1
    SIDE_S = 2
    SIDE_W = 3

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @staticmethod
    def create_centered_rect(center_x, center_y, width, height):
        half_width = width / 2
        half_height = height / 2
        return Rectangle(
            center_x - half_width, center_y - half_height,
            center_x + half_width, center_y + half_height
        )

    @property
    def points(self):
        return self.x1, self.y1, self.x2, self.y2

    @property
    def corners(self):
        return [(self.x1, self.y1), (self.x2, self.y1), (self.x2, self.y2), (self.x1, self.y2)]

    @property
    def size(self):
        return self.width, self.height

    @size.setter
    def size(self, value):
        self.width = value[0]
        self.height = value[1]

    @property
    def width(self):
        return self.x2 - self.x1

    @width.setter
    def width(self, value):
        delta = value - self.width
        self.x1 = int(self.x1 - delta / 2.0)
        self.x2 = int(self.x2 + delta / 2.0)

    @property
    def height(self):
        return self.y2 - self.y1

    @height.setter
    def height(self, value):
        delta = value - self.height
        self.y1 = int(self.y1 - delta / 2.0)
        self.y2 = int(self.y2 + delta / 2.0)

    @property
    def center_x(self):
        return (self.x1 + self.x2) / 2

    @center_x.setter
    def center_x(self, value):
        delta = value - self.center_x
        self.x1 += delta
        self.x2 += delta

    @property
    def center_y(self):
        return (self.y1 + self.y2) / 2

    @center_y.setter
    def center_y(self, value):
        delta = value - self.center_y
        self.y1 += delta
        self.y2 += delta

    @property
    def center(self):
        return self.center_x, self.center_y

    @center.setter
    def center(self, value):
        self.center_x = value[0]
        self.center_y = value[1]

    def intersects(self, rect):
        if (self.x1 > rect.x2) or (rect.x1 > self.x2):
            return False

        if (self.y1 > rect.y2) or (rect.y1 > self.y2):
            return False

        return True

    def contains_point(self, point):
        x, y = point
        if (x >= self.x1) and (x <= self.x2) and (y >= self.y1) and (y <= self.y2):
            return True
        return False

    def contains_rect(self, rect):
        if rect.x1 < self.x1 or rect.x2 > self.x2 or rect.y1 < self.y1 or rect.y2 > self.y2:
            return False
        return True

    def closest_corner_to_point(self, point):
        min_index = 0
        min_dist = sys.maxint
        for index, corner in enumerate(self.corners):
            dist = distance(corner, point)
            if dist < min_dist:
                min_dist = dist
                min_index = index

        return min_index, min_dist

    def find_closest_corner(self, rect):
        min_index = 0
        min_dist = sys.maxint
        for index, corner in enumerate(self.corners):
            dist = distance(corner, rect.corners[index])
            if dist < min_dist:
                min_dist = dist
                min_index = index

        return min_index, index

    def clip_rect(self, rect):
        x1 = max(self.x1, rect.x1)
        x2 = min(self.x2, rect.x2)
        y1 = max(self.y1, rect.y1)
        y2 = min(self.y2, rect.y2)

        return Rectangle(x1, y1, x2, y2)


class DetectedObject(object):
    def __init__(self, bounds, image, keypoints, descriptors, label):
        self.bounds = bounds
        self.image = image
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.label = label
        self.matched = 0

        self.bestOffer = -1
        self.bestTrack = None

    @staticmethod
    def createFromFrame(frame, bounds):
        image_bounds = Rectangle(0, 0, frame.shape[1], frame.shape[0])
        if not image_bounds.contains_rect(bounds):
            print 'Detected object out of bounds!'

            # Resize bounds to fit window
            bounds = image_bounds.clip_rect(bounds)

        # Create DetectedObject
        image = frame[bounds.y1:bounds.y2, bounds.x1:bounds.x2]

        # Get SIFT features
        keypoints, descriptors = sift.detectAndCompute(image, None)

        # Extracting HOG feature vector from template center
        orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #"""
        # Get normalized size (scale so that one side is equal to imgsize; keep ratio fixed)

        h = orig.shape[0]
        w = orig.shape[1]

        if h < w:
            w = int(w * float(imgsize) / h)
            h = imgsize
        else:
            h = int(h * float(imgsize) / w)
            w = imgsize

        im = cv2.resize(orig, (w, h))
        """
        # Use full size
        im = cv2.resize(orig, (imglen, imglen))
        #"""

        cv2.imshow('Object', im)

        location = ((im.shape[1] / 2, im.shape[0] / 2),)
        h = hog.compute(im, (1, 1), (2, 2), location)

        # Adding width and height as features
        addition = np.array([image.shape[0] / float(image.shape[1])]).reshape(-1, 1)
        #h = np.vstack((h, addition))

        # Classify template using pre-trained SVM classifier
        h = h.T
        label = clf.predict(h)
        label = label[0]

        return DetectedObject(bounds, image, keypoints, descriptors, label)

    def makeOffer(self, track, offer):
        """
        Make an offer to the object, highest bidder wins.
        :param track:
        :param offer:
        :return: True if offer won over the previous, False if declined
        """
        if offer > self.bestOffer:
            self.bestOffer = offer
            self.bestTrack = track
            return True

        return False

    def acceptOffer(self):
        """
        Checks for an acceptable offer, and confirms it with the track.
        :return: True if an offer was accepted, False if declined
        """
        if self.bestTrack is not None and self.bestOffer > TRACK_OFFER_THRESHOLD:
            # Tell the track it won
            self.bestTrack.confirmOffer(self)
            return True

        return False


class kalFilter(object):
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
        return self.currentPrediction


class Track(object):
    TRACK_ID_COUNTER = 0  # Assign a unique id to each created track

    def __init__(self, detectedObject):
        self.id = Track.TRACK_ID_COUNTER
        Track.TRACK_ID_COUNTER += 1

        self.spaceBound = detectedObject.bounds
        self.waitPeriodforActionRecognition = 0
        self.actionLabel = ""

        self.currentBounds = detectedObject.bounds
        self.currentImage = detectedObject.image
        self.currentKeypoints = detectedObject.keypoints
        self.currentDescriptor = detectedObject.descriptors

        self.tracked = True
        self.acceptedOffers = []

        self.lifeTime = 1
        self.numberOfFramesNotTracked = 0

        self.labelsList = [detectedObject.label]
        self.modeLabel = detectedObject.label

        self.most_recent_frames = []

        # Create Kalman filter
        self.kalman_filter = kalFilter(self.currentBounds, self.id)

        self.mosse_active = False
        self.mosse_filter = None

    @property
    def offer_count(self):
        return len(self.acceptedOffers)

    @property
    def predicted_center(self):
        return self.kalman_filter.currentPrediction

    def confirmOffer(self, detectedObject):
        """
        Adds the offer to the list of accepted offers. Sets self.tracked to True.
        :param detectedObject:
        :return:
        """
        self.tracked = True
        self.acceptedOffers.append(detectedObject)



    def update(self, detectedObject, current_frame):
        """
        Matches the track to the given object
        :param detectedObject:
        :return:
        """
        self.currentBounds = detectedObject.bounds
        self.currentImage = detectedObject.image
        self.currentKeypoints = detectedObject.keypoints
        self.currentDescriptor = detectedObject.descriptors

        if self.waitPeriodforActionRecognition > 0:
            if self.spaceBound.x1 > detectedObject.bounds.x1:
                self.spaceBound.x1 = detectedObject.bounds.x1
            if self.spaceBound.x2 < detectedObject.bounds.x2:
                self.spaceBound.x2 = detectedObject.bounds.x2
            if self.spaceBound.y1 > detectedObject.bounds.y1:
                self.spaceBound.y1 = detectedObject.bounds.y1
            if self.spaceBound.y2 < detectedObject.bounds.y2:
                self.spaceBound.y2 = detectedObject.bounds.y2
        else:
            self.spaceBound.x1 = detectedObject.bounds.x1
            self.spaceBound.x2 = detectedObject.bounds.x2
            self.spaceBound.y1 = detectedObject.bounds.y1
            self.spaceBound.y2 = detectedObject.bounds.y2



        self.tracked = True
        self.acceptedOffers = []

        self.lifeTime += 1
        self.numberOfFramesNotTracked = 0

        # Updating the labels history
        if len(self.labelsList) >= MAX_LABEL_HISTORY:
            self.labelsList.pop()

        # Update the frame history
        self.most_recent_frames.append(current_frame)

        self.labelsList.append(detectedObject.label)
        self.modeLabel = stats.mode(self.labelsList)
        self.modeLabel = self.modeLabel[0][0]

        # Update Kalman filter
        self.kalman_filter.update(self.currentBounds.center_x, self.currentBounds.center_y)

        # Clear MOSSE filter
        self.mosse_active = False
        self.mosse_filter = None

    def notifyOrphaned(self, frame):
        """
        # Use MOSSE filter to track when object not detected
        if not self.mosse_filter:
            # Create MOSSE filter from previous known image
            self.mosse_filter = MOSSE.MOSSE(self.currentImage,
                                            (0, 0, self.currentBounds.width, self.currentBounds.height))

        self.mosse_filter.update(frame, self.predicted_center)

        if self.mosse_filter.good:
            self.lifeTime += 1
            self.mosse_active = True
            # Don't reset the number of frames not tracked... MOSSE is only temporary if we lose tracking
            # If MOSSE is missing it intermittently, we will destroy the track eventually

            # Create new meta-object and update
            new_center = self.mosse_filter.pos
            new_bounds = Rectangle.create_centered_rect(new_center[0], new_center[1],
                                                        self.currentBounds.width, self.currentBounds.height)
            detectedObject = DetectedObject.createFromFrame(frame, new_bounds)

            self.currentBounds = detectedObject.bounds
            self.currentImage = detectedObject.image
            # Don't update the descriptor... don't know if object is in good state (may be partially occluded)

            # Update Kalman filter
            self.kalman_filter.update(self.currentBounds.center_x, self.currentBounds.center_y)
        else:
            self.mosse_active = False
            self.kalman_filter.predict()
            self.numberOfFramesNotTracked += 1
        """
        self.kalman_filter.predict()
        self.numberOfFramesNotTracked += 1
        #"""


def applyDetection(im, inFrame):
    # Preparing Image for processing
    ret, thresh = cv2.threshold(im, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detectedObjects = []

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

            bounds = Rectangle(minx, miny, maxx, maxy)

            #cv2.imshow("Current Object", image)
            # Add object to the list of detected objects
            detectedObjects.append(DetectedObject.createFromFrame(inFrame, bounds))

    return detectedObjects


def combineObjectBounds(detectedObjects):
    minx = sys.maxint
    maxx = 0
    miny = sys.maxint
    maxy = 0

    for detectedObject in detectedObjects:
        bounds = detectedObject.bounds

        if bounds.x1 < minx:
            minx = bounds.x1
        if bounds.x2 > maxx:
            maxx = bounds.x2
        if bounds.y1 < miny:
            miny = bounds.y1
        if bounds.y2 > maxy:
            maxy = bounds.y2

    return Rectangle(minx, miny, maxx, maxy)


def calculateMatchOffer(track, detectedObject):
    """
    Determines the match between the track and the detected object, and returns a match score.
    :param track:
    :param detectedObject:
    :return: match score >= 0, 0 means no match, < 0 means a match is impossible
    """

    # TODO Test different matching methods here! (Just return the same range of scores)

    # Factor in the ratio of sizes
    obj_area = detectedObject.bounds.width * detectedObject.bounds.height
    trk_area = track.currentBounds.width * track.currentBounds.height

    if obj_area < trk_area:
        size_ratio = obj_area / float(trk_area)
    else:
        size_ratio = trk_area / float(obj_area)

    # Penalize by difference in size
    return calculateSIFTMatch(track, detectedObject) * (size_ratio**2)


def calculateSIFTMatch(track, detectedObject):
    matcher = cv2.BFMatcher()

    if (len(detectedObject.keypoints) > 0) and (len(track.currentKeypoints) > 0):
        matches = matcher.knnMatch(track.currentDescriptor, detectedObject.descriptors, k=2)

        # Keep a tally of ratio scores
        ratio_sum = 0

        # Apply Lowe's ratio test to determine a good match
        good = []
        for match in matches:
            if len(match) != 2:
                continue

            m, n = match
            ratio = m.distance / n.distance

            if ratio <= RATIO_THRESHOLD:
                ratio_sum += RATIO_THRESHOLD / ratio if ratio > 0 else 0.001  # Ratio of n vs. m with respect to RATIO_THRESHOLD
                good.append(m)

        goodMatches = []
        if len(good) > 0:
            # Using RANSAC to further improve the matching and eliminate wrong matches
            # TODO Sometimes the homography transform results in a bounding box OUTSIDE the destination
            src_pts = np.float32([track.currentKeypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([detectedObject.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if mask is not None:
                matchesMask = mask.ravel().tolist()

                for index, match in enumerate(good):
                    if matchesMask[index]:
                        goodMatches.append([match])

            if len(goodMatches) > MIN_PTS_THRESHOLD:
                match_img = cv2.drawMatchesKnn(track.currentImage, track.currentKeypoints,
                                               detectedObject.image, detectedObject.keypoints,
                                               goodMatches, None, flags=2)
                #cv2.imshow(str(track.id), match_img)
                #cv2.imwrite(str(frame_number) + "-" + str(track.id) + ".png",match_img)

            # TODO What's a better score here?

            # If we can find a transform, we should have a very good match
            return ratio_sum * (len(goodMatches) + 1)

        # This should always be equal to 0
        return ratio_sum

    # Failure condition -- track or object had no features to match
    return -1


def vectorizeCubebyTime(cube):

    outputMatrix = np.zeros((CUBE_X*CUBE_Y, TRACKLET_LENGTH))
    for i in range(TRACKLET_LENGTH):
        column = cube[:, :, i].reshape((CUBE_X*CUBE_Y))
        outputMatrix[:, i] = column

    #print outputMatrix
    mean = np.mean(outputMatrix, axis=1)
    std = np.std(outputMatrix, axis=1)
    #print std
    #print mean
    #print mean.shape, std.shape, outputMatrix.shape
    for i in range(TRACKLET_LENGTH):
        for x in range(CUBE_X*CUBE_Y):
            outputMatrix[x, i] = (outputMatrix[x, i] - mean[x,]) / (std[x,] + 1)

    #print outputMatrix
    return outputMatrix


def vectorizeCubebyWidth(cube):

    outputMatrix = np.zeros((TRACKLET_LENGTH*CUBE_Y, CUBE_X))
    for i in range(CUBE_X):
        column = cube[i, :, :].reshape((TRACKLET_LENGTH*CUBE_Y))
        outputMatrix[:, i] = column

    #print outputMatrix
    mean = np.mean(outputMatrix, axis=1)
    std = np.std(outputMatrix, axis=1)
    #print std
    #print mean
    #print mean.shape, std.shape, outputMatrix.shape
    for i in range(CUBE_X):
        for x in range(TRACKLET_LENGTH*CUBE_Y):
            outputMatrix[x, i] = (outputMatrix[x, i] - mean[x, ]) / (std[x, ] + 1)

    #print outputMatrix
    return outputMatrix


def vectorizeCubebyHeight(cube):

    outputMatrix = np.zeros((TRACKLET_LENGTH*CUBE_X, CUBE_Y))
    for i in range(CUBE_Y):
        column = cube[:, i, :].reshape((TRACKLET_LENGTH*CUBE_X))
        outputMatrix[:, i] = column

    #print outputMatrix
    mean = np.mean(outputMatrix, axis=1)
    std = np.std(outputMatrix, axis=1)
    #print std
    #print mean
    #print mean.shape, std.shape, outputMatrix.shape
    for i in range(CUBE_Y):
        for x in range(TRACKLET_LENGTH*CUBE_X):
            outputMatrix[x, i] = (outputMatrix[x, i] - mean[x, ]) / (std[x, ] + 1)

    #print outputMatrix
    return outputMatrix


def getEigenVectors(cube):
    matrixTime = vectorizeCubebyTime(cube)
    matrixWidth = vectorizeCubebyWidth(cube)
    matrixHeight = vectorizeCubebyHeight(cube)


    PCAresultTime = np.linalg.svd(np.dot(matrixTime.T, matrixTime))
    PCAresultWidth = np.linalg.svd(np.dot(matrixWidth.T, matrixWidth))
    PCAresultHeight = np.linalg.svd(np.dot(matrixHeight.T, matrixHeight))


    #print PCAresultTime[1]
    return PCAresultTime[0], PCAresultWidth[0], PCAresultHeight[0]


def loadActionsDatabase(actionEigenvectorsPathsList):

    database = []
    for path in actionEigenvectorsPathsList:
        vectors = np.load(path)
        database.append(vectors)
    return database


def subspaceSimilarity(testEigenvectors, gestEigenvectors, dim):

    testEigenvectors = testEigenvectors[0:dim]
    gestEigenvectors = gestEigenvectors[0:dim]

    finalPCA = np.linalg.svd(np.dot(testEigenvectors, gestEigenvectors.T))

    return finalPCA[1][0]


def getPrincipalAnglesScores(database, testEigenvectors):

    scores = []

    for actionSet in database:
        #print actionSet.shape
        maxPrincipalAngle = 0
        for i in range(actionSet.shape[0]):
            gestureEigenvectors = actionSet[i, :, :]
            score = subspaceSimilarity(testEigenvectors, gestureEigenvectors, DIMENSIONS_TO_KEEP)
            if score > maxPrincipalAngle:
                maxPrincipalAngle = score
        scores.append(maxPrincipalAngle)

    return scores


def anglesDistance(scoresTime, scoresWidth, scoresHeight):
    scores = []
    for i in range(len(scoresTime)):
        dist = math.sqrt(scoresTime[i] * scoresTime[i] + scoresWidth[i]*scoresWidth[i] + scoresHeight[i]*scoresHeight[i])
        scores.append(dist)

    return scores


def actionRecognition(cube, databaseTime, databaseWidth, databaseHeight):

    eigenvectorsTime, eigenvectorsWidth, eigenvectorsHeight = getEigenVectors(cube)
    scoresTime = getPrincipalAnglesScores(databaseTime, eigenvectorsTime)
    scoresWidth = getPrincipalAnglesScores(databaseWidth, eigenvectorsWidth)
    scoresHeight = getPrincipalAnglesScores(databaseHeight, eigenvectorsHeight)

    scores = anglesDistance(scoresTime, scoresWidth, scoresHeight)

    maxScore = scores[0]
    maxIndex = 0
    for i in range(len(scores)):
        if scores[i] > maxScore:
            maxScore = scores[i]
            maxIndex = i

    if maxIndex == BOXING:
        return "Boxing"
    elif maxIndex == HAND_CLAPPING:
        return "Hand Clapping"
    elif maxIndex == HAND_WAVING:
        return "Hand Waving"
    elif maxIndex == RUNNING:
        return  "Running"
    elif maxIndex == WALKING:
        return "Walking"
    else:
        return "Undefined Motion"


def updateAllTracks(trackList, detectedObjectsList, frame_number, frame, databaseTime, databaseWidth, databaseHeight):
    # TODO Note:
    # Before, we were assuming objects could match with more than one track...
    # Now, we are assuming that a track can match with more than one object!
    # Tracks will find the object they want to match with, and "offer" the objects a match score.
    # Objects will keep the highest match score they receive, and will associate with that track.
    # We are exploiting track/object locality seeing as feature comparison isn't robust enough.

    if DEBUG_OUTPUT:
        print '--- Frame:', frame_number
        print '|  Tracks:', len(trackList)
        print '| Objects:', len(detectedObjectsList)

    # Ask the tracks to make offers to the objects
    for track in trackList:
        # Get check bounds
        # TODO For new tracks (1 frame lifetime), may just snap to closest object it can find
        pred_center = track.predicted_center
        pred_bounds = Rectangle.create_centered_rect(pred_center[0], pred_center[1],
                                                     track.currentBounds.width, track.currentBounds.height)

        if not track.tracked:
            # Make predicted window larger if not previously tracked (to compensate for prediction error)
            pred_bounds.width *= PREDICTED_WINDOW_MULTIPLIER
            pred_bounds.height *= PREDICTED_WINDOW_MULTIPLIER

        # Clear tracked flag
        track.tracked = False

        # Make an offer to all potential objects
        for objectId, detectedObject in enumerate(detectedObjectsList):
            # Skip objects that do not intersect with the predicted track bounds
            if not pred_bounds.intersects(detectedObject.bounds):
                continue

            offerAmt = calculateMatchOffer(track, detectedObject)

            # TODO Force offer to match, even if failed
            #if offerAmt < 0:
                #offerAmt = 0  # Insignificant amount, but will match if no other tracks have a better offer

            offer_accepted = detectedObject.makeOffer(track, offerAmt)

            if DEBUG_OUTPUT:
                print 'Offer made from Track', track.id, 'to Object', objectId, \
                    'for amount', offerAmt, '[Best]' if offer_accepted else ''

    # Accept all offers
    for objectId, detectedObject in enumerate(detectedObjectsList):
        if detectedObject.acceptOffer():
            if DEBUG_OUTPUT:
                print 'Best offer -- Track', detectedObject.bestTrack.id, 'matched to Object', objectId
        else:
            # Object had no offer, make into new track
            if detectedObject.label != RANDOM_LABEL:
                trackList.append(Track(detectedObject))

    # Update tracks
    for track in trackList:
        if track.offer_count > 0:
            # Combine offers and update track
            new_bounds = combineObjectBounds(track.acceptedOffers)

            # Only take the full size of the object if it is similar enough
            track_area = track.currentBounds.width * track.currentBounds.height
            object_area = new_bounds.width * new_bounds.height

            if max(track_area, object_area) / float(min(track_area, object_area)) > WINDOW_GROWTH_THRESHOLD:
                # Adjust size with weights (slow expanding)
                adj_bounds = copy.copy(track.currentBounds)
                adj_bounds.width = int((1 - WINDOW_GROWTH_RATE) * adj_bounds.width + WINDOW_GROWTH_RATE * new_bounds.width)
                adj_bounds.height = int((1 - WINDOW_GROWTH_RATE) * adj_bounds.height + WINDOW_GROWTH_RATE * new_bounds.height)

                # Move adjusted window to the best location in the destination bounds
                corner, dist = adj_bounds.find_closest_corner(new_bounds)

                # Is center closer?
                if dist > distance(adj_bounds.center, new_bounds.center):
                    # Snap to center
                    adj_bounds.center = new_bounds.center
                else:
                    # Snap to nearest corner
                    adj_corners = adj_bounds.corners
                    new_corners = new_bounds.corners

                    delta_x = new_corners[corner][0] - adj_corners[corner][0]
                    delta_y = new_corners[corner][1] - adj_corners[corner][1]

                    adj_bounds.center = (adj_bounds.center_x + delta_x, adj_bounds.center_y + delta_y)

                new_bounds = adj_bounds

                # TODO We have only partially filled the destination window; the rest could be a new object???
                # TODO Then we could potentially match TWO OR MORE tracks in the same "object"

            # Create new meta-object
            new_object = DetectedObject.createFromFrame(frame, new_bounds)
            track.update(new_object, frame)
        elif not track.tracked:
            # Orphaned track (no one accepted an offer)
            track.notifyOrphaned(frame)
        # else, track was just created

        # Action recognition part
        track.waitPeriodforActionRecognition += 1
        if track.waitPeriodforActionRecognition >= ACTION_RECOGNITION_WAIT_PERIOD:
            if len(track.most_recent_frames) >= TRACKLET_LENGTH:
                #print track.most_recent_frames.shape
                spaceBound = track.spaceBound
                #print spaceBound.x1, spaceBound.x2, spaceBound.y1, spaceBound.y2
                frames = track.most_recent_frames[-TRACKLET_LENGTH:]
                imageCube = np.zeros((CUBE_X, CUBE_Y, TRACKLET_LENGTH))
                for i in range(TRACKLET_LENGTH):
                    #print frames[i].shape
                    gray_image = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                    box = gray_image[spaceBound.y1:spaceBound.y2, spaceBound.x1:spaceBound.x2]
                    cv2.imshow('Cube', box)
                    box = cv2.resize(box, (CUBE_X, CUBE_Y))

                    imageCube[:, :, i] = box
                    #print imageCube.shape
                actionLabel = actionRecognition(imageCube, databaseTime, databaseWidth, databaseHeight)
                track.actionLabel = actionLabel
                track.waitPeriodforActionRecognition = 0



    # Delete old, orphaned tracks (short-term memory approach)
    trackList[:] = [track for track in trackList if
                    (track.numberOfFramesNotTracked < NON_TRACKED_LIMIT) and
                    (not track.modeLabel == RANDOM_LABEL)]


def drawObjects(detectedObjectList, frame):
    outputFrame = np.copy(frame)

    for detectedObject in detectedObjectList:
        cv2.rectangle(outputFrame, (detectedObject.bounds.x1, detectedObject.bounds.y1),
                      (detectedObject.bounds.x2, detectedObject.bounds.y2), OBJECT_COLOR, 1)

    return outputFrame

imagecount = 0
def drawTracks(trackList, frame):
    global imagecount  # TODO Save image

    outputFrame = np.copy(frame)

    for track in trackList:
        # TODO DEBUG
        #cv2.imshow('Track' + str(track.id), track.currentImage)

        if track.tracked:
            # TODO Save image
            #cv2.imwrite('temp/' + str(imagecount) + '_sample.bmp', track.currentImage)
            imagecount += 1
            # TODO

            #print track.modeLabel
            if int(track.modeLabel) == CAR_LABEL:
                label = "Car"
            elif int(track.modeLabel) == PEDESTRIAN_LABEL:
                label = "Pedestrian"
            else:
                label = "Unknown"

            cv2.rectangle(outputFrame, (track.currentBounds.x1, track.currentBounds.y1),
                          (track.currentBounds.x2, track.currentBounds.y2), TRACKED_COLOR, 1)
            cv2.putText(outputFrame, '# ' + str(track.id) + " - " + label + " - " + track.actionLabel, (track.currentBounds.x1,
                                                                            track.currentBounds.y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

            cv2.circle(outputFrame, track.currentBounds.center, 8, (255,0,0), 5)
        else:
            pred_bounds = copy.copy(track.currentBounds)
            pred_bounds.center = track.predicted_center
            cv2.rectangle(outputFrame, (pred_bounds.x1, pred_bounds.y1),
                          (pred_bounds.x2, pred_bounds.y2), PREDICTED_COLOR, 1)

            if track.mosse_active:
                cv2.putText(outputFrame, '# ' + str(track.id), (track.currentBounds.x1,
                                                                track.currentBounds.y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

        # KALMAN
        cv2.circle(outputFrame, track.predicted_center, 4, PREDICTED_COLOR, 2)

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


def identifyTrackMovements(track_list, gestures):

    clip_length = 30

    for track in track_list:
        print "-.-.-.-.-.-.-.-.-.-."
        print track.id, len(track.most_recent_frames[-1*clip_length:])

        frames_from_track = track.most_recent_frames[-1*clip_length:]

        correct_gesture = get_gesture(frames_from_track, gestures)
        print correct_gesture[1]


def get_gesture(frames_from_track, gestures):

    closest_diff = float("inf")
    correct_gesture = None

    for gesture in gestures:
        num_frames = len(frames_from_track)
        current_diff = compute_subspace_difference(frames_from_track, gesture[0][-1*num_frames:])
        print current_diff
        if current_diff < closest_diff:
            closest_diff = current_diff
            correct_gesture = gesture

    return correct_gesture


def compute_subspace_difference(track, gesture):
    print(len(track) == len(gesture))


def load_gestures(paths):

    gestures = [([], path) for i, path in enumerate(paths)]

    for index, current_path in enumerate(paths):
        print current_path
        cap = cv2.VideoCapture(current_path)

        if not cap.isOpened():
            print 'Error--Unable to open video:', current_path
            return

        # Get video parameters (try to retain same attributes for output video)
        # width = cap.width
        # height = cap.height
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))

        while True:
            # Get frame
            ret, frame = cap.read()

            gestures[index][0].append(frame)

            if ret is False or frame is None:
                break

    return gestures

def main():
    # Disabling OpenCL here cause it causes problem using BackgroundSubstractor
    # Found out online that OpenCL bindings for openCV 3.1 are not working
    cv2.ocl.setUseOpenCL(False)
    source_path = sys.argv[1] if len(sys.argv) > 1 else 0

    # Using skvideo to load and write videos because of codecs issue under Linux
    #cap = cv2.VideoCapture(source_path)
    cap = skvideo.io.VideoCapture(source_path)

    if not cap.isOpened():
        print 'Error--Unable to open video:', source_path
        return

    truthCap = cv2.VideoCapture(os.path.dirname(source_path) + '/groundTruth.avi')
    truthAvailable = truthCap.isOpened()

    # Get video parameters (try to retain same attributes for output video)
    width = cap.width
    height = cap.height
    #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #fps = float(cap.get(cv2.CAP_PROP_FPS))
    #codec = int(cap.get(cv2.CAP_PROP_FOURCC))

    skvideo_writer = skvideo.io.VideoWriter("Actions.avi", frameSize=(width, height))
    skvideo_writer.open()
    #dst_writer = cv2.VideoWriter("output.avi", codec, fps if fps > 0 else 30, (width, height))

    # Creating the Foreground/Background segmentation based on MOG
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    #Load training gestures
    databaseTime = loadActionsDatabase(['Data/boxingEigenvectorsTime.npy', 'Data/handClappingEigenvectorsTime.npy',
                                    'Data/handWavingEigenvectorsTime.npy', 'Data/runningEigenvectorsTime.npy',
                                        'Data/walkingEigenvectorsTime.npy'])

    databaseWidth = loadActionsDatabase(['Data/boxingEigenvectorsWidth.npy', 'Data/handClappingEigenvectorsWidth.npy',
                                        'Data/handWavingEigenvectorsWidth.npy', 'Data/runningEigenvectorsWidth.npy',
                                         'Data/walkingEigenvectorsWidth.npy'])

    databaseHeight = loadActionsDatabase(['Data/boxingEigenvectorsHeight.npy', 'Data/handClappingEigenvectorsHeight.npy',
                                        'Data/handWavingEigenvectorsHeight.npy', 'Data/runningEigenvectorsHeight.npy',
                                          'Data/walkingEigenvectorsHeight.npy'])

    trackList = []
    # Main loop
    frame_number = 0
    while True:
        # Get frame
        ret, frame = cap.read()

        # Testing Occlusion
        #cv2.rectangle(frame, (130, 60), (170, 200), (255,0,0), 40)

        if ret is False or frame is None:
            break

        frame_number += 1
        #cv2.imshow("Input", frame)

        if not truthAvailable:
            # Apply MOG to frame
            fgmask = fgbg.apply(frame)
        else:
            # Read from ground truth
            truthRet, fgmask = truthCap.read()
            if ret is False or fgmask is None:
                print 'HELP!'
                break

            # Filter out classifications
            fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
            temp, fgmask = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)

        # Filter the fgmask to remove noise
        filtered = postProcessing(fgmask)

        # Detect moving object in frame
        detectedObjectsList = applyDetection(filtered, frame)

        # Update all tracks
        updateAllTracks(trackList, detectedObjectsList, frame_number, frame, databaseTime, databaseWidth, databaseHeight)

        outputFrame = drawObjects(detectedObjectsList, frame)
        outputFrame = drawTracks(trackList, outputFrame)

        cv2.imshow("Filtered", filtered)
        cv2.imshow("Input", outputFrame)
        cv2.imshow("Foreground", fgmask)




        # Implement movement comparisons
        # TODO
        #identifyTrackMovements(trackList, gestures)



        # Write Video To file
        skvideo_writer.write(outputFrame)
        #dst_writer.write(outputFrame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    skvideo_writer.release()

    #dst_writer.release()
    cap.release()


if __name__ == '__main__':
    main()