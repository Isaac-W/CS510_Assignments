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


TRACK_LENGTH = 30
BOX_X = 20
BOX_Y = 20

BOXING = 0
HAND_CLAPPING = 1
HAND_WAVING = 2
JOGGING = 3
RUNNING = 4
WALKING = 5

DIMENSIONS_TO_KEEP = 5


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
        #print frame_index
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
    #print std
    #print mean
    #print mean.shape, std.shape, outputMatrix.shape
    for i in range(TRACK_LENGTH):
        for x in range(BOX_X*BOX_Y):
            outputMatrix[x, i] = (outputMatrix[x, i] - mean[x,]) / (std[x,] + 1)

    #print outputMatrix
    return outputMatrix


def getEigenVectors(Cube):
    Matrix = vectorizeCube(Cube)


    PCAresult = np.linalg.svd(np.dot(Matrix.T, Matrix))

    return PCAresult[0]


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
        print actionSet.shape
        maxPrincipalAngle = 0
        for i in range(actionSet.shape[0]):
            gestureEigenvectors = actionSet[i, :, :]
            score = subspaceSimilarity(testEigenvectors, gestureEigenvectors, DIMENSIONS_TO_KEEP)
            if score > maxPrincipalAngle:
                maxPrincipalAngle = score
        scores.append(maxPrincipalAngle)

    return scores


def main():
    database = loadActionsDatabase(['Data/boxingEigenvectors.npy', 'Data/handClappingEigenvectors.npy',
                                    'Data/handWavingEigenvectors.npy', 'Data/joggingEigenvectors.npy',
                                    'Data/runningEigenvectors.npy', 'Data/walkingEigenvectors.npy'])

    testCube = loadCube("Data/Walking/person01_walking_d1_uncomp_sample0.avi")
    testEigenvectors = getEigenVectors(testCube)

    scores = getPrincipalAnglesScores(database, testEigenvectors)

    print scores

if __name__ == '__main__':
    main()