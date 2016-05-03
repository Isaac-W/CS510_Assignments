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


def vectorizeCubebyTime(cube):

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


def vectorizeCubebyWidth(cube):

    outputMatrix = np.zeros((TRACK_LENGTH*BOX_Y, BOX_X))
    for i in range(BOX_X):
        column = cube[i, :, :].reshape((TRACK_LENGTH*BOX_Y))
        outputMatrix[:, i] = column

    #print outputMatrix
    mean = np.mean(outputMatrix, axis=1)
    std = np.std(outputMatrix, axis=1)
    #print std
    #print mean
    #print mean.shape, std.shape, outputMatrix.shape
    for i in range(BOX_X):
        for x in range(TRACK_LENGTH*BOX_Y):
            outputMatrix[x, i] = (outputMatrix[x, i] - mean[x, ]) / (std[x, ] + 1)

    #print outputMatrix
    return outputMatrix


def vectorizeCubebyHeight(cube):

    outputMatrix = np.zeros((TRACK_LENGTH*BOX_X, BOX_Y))
    for i in range(BOX_Y):
        column = cube[:, i, :].reshape((TRACK_LENGTH*BOX_X))
        outputMatrix[:, i] = column

    #print outputMatrix
    mean = np.mean(outputMatrix, axis=1)
    std = np.std(outputMatrix, axis=1)
    #print std
    #print mean
    #print mean.shape, std.shape, outputMatrix.shape
    for i in range(BOX_Y):
        for x in range(TRACK_LENGTH*BOX_X):
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
        print actionSet.shape
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


def main():
    databaseTime = loadActionsDatabase(['Data/boxingEigenvectorsTime.npy', 'Data/handClappingEigenvectorsTime.npy',
                                    'Data/handWavingEigenvectorsTime.npy', 'Data/joggingEigenvectorsTime.npy',
                                    'Data/runningEigenvectorsTime.npy', 'Data/walkingEigenvectorsTime.npy'])

    databaseWidth = loadActionsDatabase(['Data/boxingEigenvectorsWidth.npy', 'Data/handClappingEigenvectorsWidth.npy',
                                        'Data/handWavingEigenvectorsWidth.npy', 'Data/joggingEigenvectorsWidth.npy',
                                        'Data/runningEigenvectorsWidth.npy', 'Data/walkingEigenvectorsWidth.npy'])

    databaseHeight = loadActionsDatabase(['Data/boxingEigenvectorsHeight.npy', 'Data/handClappingEigenvectorsHeight.npy',
                                        'Data/handWavingEigenvectorsHeight.npy', 'Data/joggingEigenvectorsHeight.npy',
                                        'Data/runningEigenvectorsHeight.npy', 'Data/walkingEigenvectorsHeight.npy'])

    testCube = loadCube("Data/Walking/person01_walking_d1_uncomp_sample0.avi")
    testEigenvectorsTime, testEigenvectorsWidth, testEigenvectorsHeight = getEigenVectors(testCube)

    scoresTime = getPrincipalAnglesScores(databaseTime, testEigenvectorsTime)
    scoresWidth = getPrincipalAnglesScores(databaseWidth, testEigenvectorsWidth)
    scoresHeight = getPrincipalAnglesScores(databaseHeight, testEigenvectorsHeight)

    scores = anglesDistance(scoresTime, scoresWidth, scoresHeight)

    print scoresTime
    print scoresWidth
    print scoresHeight
    print scores


if __name__ == '__main__':
    main()