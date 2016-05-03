import numpy as np
import cv2
import os
from sklearn import svm
from sklearn.externals import joblib
from random import shuffle
import time


LABEL_CARS = 0
LABEL_PEDS = 1
LABEL_RAND = 2
LABEL_SIZE = 3


imglen = 64

nbins = 9  # The number of bins in the histogram
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64

hog = None


def initHog(imgLen, winLen, blockLen, strideLen, cellLen):
    global imglen, hog
    imglen = imgLen
    hog = cv2.HOGDescriptor(
        (winLen, winLen), (blockLen, blockLen), (strideLen, strideLen), (cellLen, cellLen),
        nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels
    )


def trainSamples(X, Y, imgs, label):
    for img in imgs:
        #print img

        # Load Image to perform HOG descriptor
        orig = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        #"""
        # Get normalized size (scale so that one side is equal to imgsize; keep ratio fixed)

        h = orig.shape[0]
        w = orig.shape[1]

        if h < w:
            w = int(w * float(imglen) / h)
            h = imglen
        else:
            h = int(h * float(imglen) / w)
            w = imglen

        im = cv2.resize(orig, (w, h))
        """
        # Use full size
        im = cv2.resize(orig, (imglen, imglen))
        #"""

        #cv2.imshow('Sample', im)
        #cv2.waitKey(20)

        # Extract HOG descriptor from image center
        location = ((im.shape[1] / 2, im.shape[0] / 2),)
        h = hog.compute(im, (1, 1), (2, 2), location)

        # Use ratio of height to width
        #addition = np.array([orig.shape[0] / float(orig.shape[1])]).reshape(-1, 1)
        #h = np.vstack((h, addition))

        # Add to lists of features and labels
        X.append(h)
        Y.append(label)

        #####################
        # Flip image
        im = cv2.flip(im, 1)

        # Extract HOG descriptor from image center
        location = ((im.shape[1] / 2, im.shape[0] / 2),)
        h = hog.compute(im, (1, 1), (2, 2), location)

        # Add to lists of features and labels
        X.append(h)
        Y.append(label)

    #print len(h)
    #print h


def testRecognition(clf, imgs, trueLabel, noout=False):
    if not noout:
        print '--- Label: ' + str(trueLabel)

    success = 0
    failure = 0

    confusion = [0 for x in range(LABEL_SIZE)]

    for img in imgs:
        #print img

        # Load Image to perform HOG descriptor
        orig = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        # Get largest square that can fit
        #"""
        h = orig.shape[0]
        w = orig.shape[1]

        if h < w:
            w = int(w * float(imglen) / h)
            h = imglen
        else:
            h = int(h * float(imglen) / w)
            w = imglen

        im = cv2.resize(orig, (w, h))
        """
        im = cv2.resize(orig, (imglen, imglen))
        #"""

        # Extract HOG descriptor from image center
        location = ((im.shape[1] / 2, im.shape[0] / 2),)
        h = hog.compute(im, (1, 1), (2, 2), location)

        # Classify it!
        h = h.T
        label = clf.predict(h)
        label = int(label[0])

        if label == trueLabel:
            success += 1
        else:
            failure += 1

        confusion[label] += 1

    rate = success / float(success + failure)

    if not noout:
        print 'Success: ' + str(success)
        print 'Failure: ' + str(failure)
        print 'Rate: ' + str(rate)

    return success, failure, rate, confusion


def getImgList(path):
    imgs = os.listdir(path)
    imgs = [os.path.join(path, x) for x in imgs]
    return imgs


def shuffleAndSplit(mylist):
    shuffle(mylist)
    return mylist[:len(mylist)/2], mylist[len(mylist)/2:]


def trainAndTest(cars_train, peds_train, rand_train, cars_test, peds_test, rand_test, noout=False, clfSave=False, clfName=''):
    X = []
    Y = []

    trainSamples(X, Y, cars_train, LABEL_CARS)
    trainSamples(X, Y, peds_train, LABEL_PEDS)
    trainSamples(X, Y, rand_train, LABEL_RAND)

    Y = np.asarray(Y, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    X = X[:, :, 0]

    #print Y.shape
    #print X.shape

    clf = svm.SVC(decision_function_shape='ovo', kernel='linear')
    clf.fit(X, Y)

    if clfSave:
        joblib.dump(clf, clfName)

    #################
    # Test recognition

    total_success = 0
    total_failure = 0

    confusion = []

    s, f, r, c = testRecognition(clf, cars_test, LABEL_CARS, noout)
    total_success += s
    total_failure += f
    confusion.append(c)

    s, f, r, c = testRecognition(clf, peds_test, LABEL_PEDS, noout)
    total_success += s
    total_failure += f
    confusion.append(c)

    s, f, r, c = testRecognition(clf, rand_test, LABEL_RAND, noout)
    total_success += s
    total_failure += f
    confusion.append(c)

    total_rate = total_success / float(total_success + total_failure)

    if not noout:
        print '---------------------------'
        print 'Total Success: ' + str(total_success)
        print 'Total Failure: ' + str(total_failure)
        print 'Total Rate: ' + str(total_rate)

    return total_success, total_failure, total_rate, confusion


def run_it(imgLen, winLen, blockLen, strideLen, cellLen, noout=False, full=False):
    initHog(imgLen, winLen, blockLen, strideLen, cellLen)

    # Two fold cross validation
    cars_list = getImgList(r'C:\Working\Datasets\Training\TrainingSets\Cars')
    peds_list = getImgList(r'C:\Working\Datasets\Training\TrainingSets\Pedestrians')
    rand_list = getImgList(r'C:\Working\Datasets\Training\TrainingSets\Random')

    cars_1, cars_2 = shuffleAndSplit(cars_list)
    peds_1, peds_2 = shuffleAndSplit(peds_list)
    rand_1, rand_2 = shuffleAndSplit(rand_list)

    if not noout:
        print '========================================='
        print 'PART I'
    s1, f1, r1, c1 = trainAndTest(cars_1, peds_1, rand_1, cars_2, peds_2, rand_2, noout)
    #print c1

    if not noout:
        print '========================================='
        print 'PART II'
    s2, f2, r2, c2 = trainAndTest(cars_2, peds_2, rand_2, cars_1, peds_1, rand_1, noout)
    #print c2

    # Add c1 to c2
    confusion = [[0, 0, 0] for x in range(LABEL_SIZE)]
    for truth in range(LABEL_SIZE):
        for predicted in range(LABEL_SIZE):
            confusion[truth][predicted] = c1[truth][predicted] + c2[truth][predicted]

    avg_rate = (r1 + r2) / 2

    if not noout:
        print '========================================='
        print 'Complete Success: ' + str(s1 + s2)
        print 'Complete Failure: ' + str(f1 + f2)
        print 'Average Rate: ' + str(avg_rate)

    # Train full classifier
    if full:
        if not noout:
            print '========================================='
            print 'FULL CLASSIFIER'
        trainAndTest(cars_list, peds_list, rand_list, cars_list, peds_list, rand_list, noout, True, 'SVM_HOG_WANG.pkl')

    return avg_rate, confusion


# TEST ALL COMBINATIONS
#"""
imgSizes = [64, 32, 16, 8]  # Size of scaled image
winSizes = [32, 16, 8, 4]  # Size of input window
blockSizes = [32, 16, 8, 4]  # Size of blocks (each block will have blockSize/cellSize cells)
blockStrides = [32, 16, 8, 4, 2]  # Block shift (e.g. window 64, block 32, stride 16 will have 3 blocks per row)
cellSizes = [32, 16, 8, 4, 2]  # Size of cell (a histogram is computed for each cell in a block)

print 'Image, Window, Block, Stride, Cell, CarsP, CarsR, CarsF, PedsP, PedsR, PedsF, RandP, RandR, RandF, AvgP, AvgR, AvgF'

# Test all sizes
for imgSize in imgSizes:
    for winSize in winSizes:
        if winSize >= imgSize:
            continue

        for blockSize in blockSizes:
            if blockSize > winSize:
                continue

            for blockStride in blockStrides:
                if blockStride > blockSize:
                    continue

                for cellSize in cellSizes:
                    if cellSize > blockSize:
                        continue

                    try:
                        accuracy, confusion = run_it(imgSize, winSize, blockSize, blockStride, cellSize, True, False)

                        # confusion[TRUTH][PREDICTED]
                        #P = confusion[TRUTH][TRUTH] / float(confusion[LABEL_CARS][TRUTH] + confusion[LABEL_PEDS][TRUTH] + confusion[LABEL_RAND][TRUTH])
                        #R = confusion[TRUTH][TRUTH] / float(confusion[TRUTH][LABEL_CARS] + confusion[TRUTH][LABEL_PEDS] + confusion[TRUTH][LABEL_RAND])
                        
                        carsP = confusion[LABEL_CARS][LABEL_CARS] / float(confusion[LABEL_CARS][LABEL_CARS] + confusion[LABEL_PEDS][LABEL_CARS] + confusion[LABEL_RAND][LABEL_CARS])
                        carsR = confusion[LABEL_CARS][LABEL_CARS] / float(confusion[LABEL_CARS][LABEL_CARS] + confusion[LABEL_CARS][LABEL_PEDS] + confusion[LABEL_CARS][LABEL_RAND])
                        carsF = 2 * (carsP * carsR) / (carsP + carsR)

                        pedsP = confusion[LABEL_PEDS][LABEL_PEDS] / float(confusion[LABEL_CARS][LABEL_PEDS] + confusion[LABEL_PEDS][LABEL_PEDS] + confusion[LABEL_RAND][LABEL_PEDS])
                        pedsR = confusion[LABEL_PEDS][LABEL_PEDS] / float(confusion[LABEL_PEDS][LABEL_CARS] + confusion[LABEL_PEDS][LABEL_PEDS] + confusion[LABEL_PEDS][LABEL_RAND])
                        pedsF = 2 * (pedsP * pedsR) / (pedsP + pedsR)
                        
                        randP = confusion[LABEL_RAND][LABEL_RAND] / float(confusion[LABEL_CARS][LABEL_RAND] + confusion[LABEL_PEDS][LABEL_RAND] + confusion[LABEL_RAND][LABEL_RAND])
                        randR = confusion[LABEL_RAND][LABEL_RAND] / float(confusion[LABEL_RAND][LABEL_CARS] + confusion[LABEL_RAND][LABEL_PEDS] + confusion[LABEL_RAND][LABEL_RAND])
                        randF = 2 * (randP * randR) / (randP + randR)

                        avgP = (carsP + pedsP + randP) / 3
                        avgR = (carsR + pedsR + randR) / 3
                        avgF = (carsF + pedsF + randF) / 3

                        print str([imgSize, winSize, blockSize, blockStride, cellSize,
                                   carsP, carsR, carsF,
                                   pedsP, pedsR, pedsF,
                                   randP, randR, randF,
                                   avgP, avgR, avgF
                                   ]).strip('[]')
                    except:
                        pass
"""

# Run best HOG classifier
result, confusion = run_it(32, 16, 8, 8, 2, full=True)

# Print confusion matrix
print '--- Confusion Matrix'
print 'Truth, Car, Ped, Rnd'
print 'Car, ' + str(confusion[LABEL_CARS]).strip('[]')
print 'Ped, ' + str(confusion[LABEL_PEDS]).strip('[]')
print 'Rnd, ' + str(confusion[LABEL_RAND]).strip('[]')
#"""