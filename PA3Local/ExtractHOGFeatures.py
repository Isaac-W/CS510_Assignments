import numpy as np
import cv2
from sklearn import svm
from sklearn.externals import joblib


paths = np.genfromtxt('labeledDataPaths.txt', dtype='str')
print paths.shape



winSize = (8,8)
blockSize = (8,8)
blockStride = (8,8)
cellSize = (4,4)
nbins = 8
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)


X = []
Y = []
for path in paths:
    # Extract Label from Image Path
    label = str(path[2])
    path = str(path[0]) + " " + str(path[1]) + " " + str(path[2])
    #print path
    label = label.split('.')
    label = (label[0])[1]
    # Add label to labels vector
    Y.append(label)

    # Load Image to perform HOG descriptor
    im = cv2.imread(path)
    # Extract HOG descriptor from image center
    location = ((im.shape[1]/2, im.shape[0]/2),)
    h = hog.compute(im, (1, 1), (2, 2), location)

    X.append(h)

    #print len(h)
    #print h

Y = np.asarray(Y)
Y = np.float32(Y)
X = np.asarray(X)
X = np.float32(X)
X = X[:, :, 0]

print Y.shape
print X.shape

clf = svm.SVC(kernel='linear')
clf.fit(X, Y)
joblib.dump(clf, 'SVM_Linear_TrainedOnGardenTemplates.pkl')

count = 0
index = 0
for path in paths:
    path = str(path[0]) + " " + str(path[1]) + " " + str(path[2])

    # Load Image to perform HOG descriptor
    im = cv2.imread(path)
    # Extract HOG descriptor from image center
    location = ((im.shape[1]/2, im.shape[0]/2),)
    h = hog.compute(im, (1, 1), (2, 2), location)
    h = h.T
    output = clf.predict(h)
    output = output[0]
    if output == Y[index]:
        count +=1
    #print output


    index += 1

print "Total count : " + str(count) + " / " + str(len(paths))
print "Percentage : " + str((count*100.0)/len(paths))