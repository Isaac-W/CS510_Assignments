import math
import cv2
import numpy as np
import skvideo.io
from sklearn.externals import joblib
import sys
from scipy import stats
import os
# import MOSSE
import copy

gesturePath = "Data/person01_boxing_d1_uncomp_sample0.avi"
testPath = "Data/person02_running_d3_uncomp_sample2.avi"

TRACK_LENGTH = 30
BOX_X = 20
BOX_Y = 20


def loadCube(Path):
    Cube = np.zeros((BOX_Y, BOX_X, TRACK_LENGTH))
    cap = skvideo.io.VideoCapture(Path)
    if not cap.isOpened():
        print 'Error--Unable to open video:', Path
        return
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if ret is False or frame is None:
            break
        print frame_index
        Cube[:, :, frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_index += 1

    return Cube


def vectorizeCube(cube):

    outputMatrix = np.zeros((BOX_X*BOX_Y, TRACK_LENGTH))
    for i in range(TRACK_LENGTH):
        column = cube[:, :, i].reshape((BOX_X*BOX_Y))
        outputMatrix[:, i] = column

    #print outputMatrix
    mean = np.mean(outputMatrix, axis=1)
    std = np.std(outputMatrix, axis=1)
    print std
    #print mean
    #print mean.shape, std.shape, outputMatrix.shape
    for i in range(TRACK_LENGTH):
        for x in range(BOX_X*BOX_Y):
            outputMatrix[x, i] = (outputMatrix[x, i] - mean[x,]) / (std[x,] + 1)

    #print outputMatrix
    return outputMatrix


def subspaceSimilarity(testCube, gestureCube):
    testMatrix = vectorizeCube(testCube)
    gestureMatrix = vectorizeCube(gestureCube)

    testPCAresult = np.linalg.svd(np.dot(testMatrix.T, testMatrix))
    gesturePCAresult = np.linalg.svd(np.dot(gestureMatrix.T, gestureMatrix))

    testPCAeigenvectors = testPCAresult[0]
    print testPCAresult[1]
    gesturePCAeigenvectors = gesturePCAresult[0]

    testPCAeigenvectors = testPCAeigenvectors[0:14]
    #print gesturePCAresult[1]
    gesturePCAeigenvectors = gesturePCAeigenvectors[0:14]

    #final = np.dot(testPCAeigenvectors.T, gesturePCAeigenvectors)

    finalPCA = np.linalg.svd(np.dot(testPCAeigenvectors, gesturePCAeigenvectors.T))
    print finalPCA[1]

    return 0


def main():

    testCube = loadCube(testPath)
    gestureCube = loadCube(gesturePath)

    score = subspaceSimilarity(testCube, gestureCube)
    print score

if __name__ == '__main__':
    main()
