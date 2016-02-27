"""
Description:
    This module houses the logic for the ViBe algorithm. The module is layed out
    such that there is a class Model available for a program interface. The
    instance methods call global methods where the bulk of the work occurs.


Module Summary:

Constants -- help control and support the ViBe algorithm.
class Params -- Stores the main control params for the ViBe algorithm.
class Model -- Houses the analysis results and pixel history (samples).

generator TwoVariableIterator -- allows for easy iteration of a 2D array.
function init_model -- creates the initial foreground and sample arrays based
    on the Params.
function processChannel -- Processes
function processFrame -- For multiple channels of a frame,
    updates the ViBe samples and detects the foreground.
function processChannel -- For a simple channel,
    updates the ViBe samples and detects the foreground.
function IsPixelPartOfBackground -- Detects if a pixel is part of the bg.

Author:
    Joshua Gillham

Date:
    2016-02-24
"""

# Module imports.
import cv2
import numpy as np
import random
import math
import time

# BEGIN MODULE CONSTANTS

# Declare vector constants for cardinal compass directions.
NW=(-1,-1)
N=(0,-1)
NE=(1,-1)
W=(-1,0)
E=(1,0)
SW=(-1,1)
S=(0,1)
SE=(1,1)

# Put compass directions into an easily accessible array.
EightCardinalDirections = [NW, N, NE, W, E, SW, S, SE]

# The constants determine how the fg and bg will be marked.
PIXEL_MARKER_FOREGROUND=255
PIXEL_MARKER_BACKGROUND=0

# The data type for pixel arrays in memory.
NP_ELEMENT_TYPE = np.uint8

# END MODULE CONSTANTS

class Params:
    """
    Description:
        Holds the params that ViBe will use such as the maximum number of
        samples.

    Args:
        maxHistory (int): The maximum samples to keep or, in-other-words,
            the pixel history.
        noMin (int): The threshold that determines if a pixel should belong to
            the background. For example, 2 would mean that there are 2 samples
            that need to be similar.
        R (int): The threshold that determines if a pixel is similar. For
            example, 20 means that pixels can have a difference of 20 for one of
            their channels such as the red channel.
    """
    # def __init__(self, maxHistory = 20, noMin = 2, R = 20, initFrames = 20): #For color images
    def __init__(self, maxHistory = 20, noMin = 2, R = 20, initFrames = 20): #For black and white images
        """
        Description:
            Stores params.
        """
        self.maxHistory = maxHistory
        self.noMin = noMin
        self.R = R
        self.initFrames = initFrames

class Model:
    """
    Description:
        Holds the information ViBe will use such as the current foreground
        pixels and the sample space.
    """
    def __init__(self, videoCapture, params = Params()):
        """
        Description:
            Initializes the model.

        Args:
            firstSample (ndarray (h, w, 3)): the first frame that will be used
                to initialize the algorithm.
            params (Params): the params control how the algorithm works.
        """
        self.foreGround, self.samples = init_model(videoCapture, params)
        self.params=params

    def update(self, frame ):
        """
        Description:
            Updates ViBe's information using the current frame.

        Args:
            frame (ndarray (h, w, 3)): The current frame will be processed with
                the ViBe algorithm to determine which pixels are foreground
                using the samples.
        """

        #Preprocess the frame
        frame = preprocess_frame(frame)

        processFrame(
            frame,
            self.samples,
            self.foreGround,
            self.params
        )

def preprocess_frame(frame):
        frame = cv2.pyrDown(frame)
        # frame = cv2.pyrDown(cv2.pyrDown(frame))
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB);
        return frame

def TwoVariableIterator(h, w, sr=0,sc=0):
    """
    Description:
        Provides an easy way to iterate through a 2D space.

    Args:
        h (uint): The maximum value for the vertical iterator.
        w (uint): The maximum value for the horizontal iterator.
        sr (uint): The starting value for the vertical iterator.
        sc (uint): The starting value for the horizontal iterator.
    Returns:
        Tuple: returns the current row and column.
    """
    for r in range(sr, h):
        for c in range(sc, w):
            yield (r,c)

def init_model(videoCapture, params ):
    """
    Description:
        Initializes the memory for the algorithm such as the sample space.

    Args:
        params (Params): the params control how the algorithm works.
    """

    # Initialize the samples to the background.
    # Allow the samples to vary a little bit.
    samples = []
    avg = averageFrames(videoCapture, params.initFrames )
    cv2.imshow('average frame', avg)
    for n in range(0, params.maxHistory ):
        samples.append(np.copy(avg))

    ret, frame = videoCapture.read()

    #Preprocess the frame
    frame = preprocess_frame(frame)


    h, w, channels = frame.shape
    # h, w = frame.shape


    #w = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    #h = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    channels = 3#XXX

    ndSize = h, w, 1
    foreGround=np.zeros(ndSize, dtype=NP_ELEMENT_TYPE)

    return foreGround, samples

def averageFrames( videoCapture, numberOfFrames ):
    ret, frame = videoCapture.read()

    #Preprocess the frame
    frame = preprocess_frame(frame)

    h, w, channels = frame.shape
    # h, w = frame.shape
    avg = np.zeros(frame.shape, dtype=NP_ELEMENT_TYPE)
    # Sum
    for n in range(0, numberOfFrames):
        #cv2.imshow('average frame', frame)
        for (r, c) in TwoVariableIterator(h, w):
            #print avg[r, c]
            avg[r,c] = avg[r,c]+(frame[r,c]/numberOfFrames)
            #print avg[r, c]
        ret, frame = videoCapture.read()
        #Preprocess the frame
        frame = preprocess_frame(frame)
    return avg

def processFrame( frame, samples, foreGroundChannel, params ):
    """
    Description:
        Process the frame by checking for the bg pixels and updating them

    Args:
        frame (ndarray (h, w, ch)): A 2D array of the pixels of the
            current frame for one pixel.
        samples (list): A list samples. Each element is a 2D array of pixels.
            So this param is kind of like a 3D cube.
        foreGroundChannel: The foreground mask
        params (Params): the params control how the algorithm works.
    """

    h, w, ch = frame.shape
    # h, w = frame.shape
    # This step below actually speeds python up.
    # Turns out that accessing the properties of an object slow it down.
    maxHistory=params.maxHistory
    R=params.R
    noMin=params.noMin
    # Check each pixel to see if it belongs with the background
    # Leave a border of 1 unit thick.

    for (r,c) in TwoVariableIterator(h-1,w-1,1,1):
        # Force the current pixel into an 32-bit.
        # Python has trouble with the subtraction other wise.
        isPixelPartOfBackground = IsPixelPartOfBackground(samples, r, c, frame, noMin, maxHistory, R)
        #print isPixelPartOfBackground
        # Plant seeds for pixels that are part of the background.
        if ( isPixelPartOfBackground ):
            # Set this pixel as part of the bg.
            foreGroundChannel[r,c]=PIXEL_MARKER_BACKGROUND
            # Detect slow changes in the background
            phi = 16
            rand = np.random.random_integers(0,phi-1) #Updating bg with 1/16 probability
            if (rand==0):
                rndSample=np.random.random_integers(0,maxHistory-1)
                (samples[rndSample])[r,c]=(frame[r,c])
                # Allow ghosts to disappear by planting seeds in nearby pixels.
                rndSample=np.random.random_integers(0,maxHistory-1)
                rndD=np.random.random_integers(0,7)
                nc,nr = EightCardinalDirections[rndD]
                (samples[rndSample])[nr+r,nc+c]=(frame[r,c])
        # Set this pixel as part of the fg.
        else:
            foreGroundChannel[r,c]=PIXEL_MARKER_FOREGROUND

def IsPixelPartOfBackground(channelSamples, r,c, frame, noMin, maxHistory, R ):
    """
    Description:
        Determines if the pixel should be part of the background by looking for
        similarities in the sample space.

    Args:
        channelSamples (list): A list samples. Each element is a 2D array of pixels.
            So this param is kind of like a 3D cube.
        r (int): The current row.
        c (int): The current column.
        frame : the frame of the current pixel.
        noMin (int): The threshold that determines if a pixel should belong to
            the background. For example, 2 would mean that there are 2 samples
            that need to be similar.
        maxHistory (int): The maximum samples to keep or, in-other-words,
            the pixel history.
        R (int): The threshold that determines if a pixel is similar. For
            example, 20 means that pixels can have a difference of 20 for one of
            their channels such as the red channel.

    Returns:
        True if the pixel is background, False if foreground
    """
    count=0
    index=0
    # Go through the samples.
    while (index<maxHistory):
        # Find a measure of the similarity.
        #print (channelSamples[index])[r, c]
        #print frame.shape
        if(False):
            pass
        if(frame.shape[2] == 3): #If frame has 3 channels (RGB or Lab)
            #print (channelSamples[index])[r, c]
            Rs = int((channelSamples[index])[r, c, 0])
            Gs = int((channelSamples[index])[r, c, 1])
            Bs = int((channelSamples[index])[r, c, 2])
            R = int(frame[r, c, 0])
            G = int(frame[r, c, 1])
            B = int(frame[r, c, 2])
            #print Rs, R, Gs, G, Bs, B
            RR = abs(Rs - R)*abs(Rs - R)
            GG = abs(Gs - G)*abs(Gs - G)
            BB = abs(Bs - B)*abs(Bs - B)
            dist = np.sqrt(RR + GG + BB)
            #print dist

        else: #Frame has only one channel
            dist = abs(channelSamples[index][r, c] - frame[r, c])
        # Count similar pixels.
        if (dist <= R ):
            count = count + 1
            # The pixel is part of the background
            # if we have reached the threshold.
            if (count >= noMin):
                return True
        index = index + 1
    return False
