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


def main():
    eigenvectorsTime = []
    eigenvectorsWidth = []
    eigenvectorsHeight = []
    index = 0
    with open("Data/walkingPath.txt") as f:
        for line in f:
            index += 1
            print index
            # print line
            path = line[:-1]
            # Load Image to perform HOG descriptor
            cube = loadCube(path)

            vectorsTime, vectorsWidth, vectorsHeight = getEigenVectors(cube)
            eigenvectorsTime.append(vectorsTime)
            eigenvectorsWidth.append(vectorsWidth)
            eigenvectorsHeight.append(vectorsHeight)
    eigenvectorsTime = np.array(eigenvectorsTime)
    eigenvectorsWidth = np.array(eigenvectorsWidth)
    eigenvectorsHeight = np.array(eigenvectorsHeight)
    np.save("Data/walkingEigenvectorsTime", eigenvectorsTime)
    np.save("Data/walkingEigenvectorsWidth", eigenvectorsWidth)
    np.save("Data/walkingEigenvectorsHeight", eigenvectorsHeight)

if __name__ == '__main__':
    main()