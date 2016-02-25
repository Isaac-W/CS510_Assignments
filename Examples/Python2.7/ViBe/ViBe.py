"""
Description:
    This model houses the logic for the ViBe algorithm.

Author:
    Joshua Gillham

Date:
    2016-02-24
"""
import cv2

import numpy as np

NW=(-1,-1)
N=(0,-1)
NE=(1,-1)
W=(-1,0)
E=(1,0)
SW=(-1,1)
S=(0,1)
SE=(1,1)
EightCardinalDirections = [NW, N, NE, W, E, SW, S, SE]

PIXEL_MARKER_FOREGROUND=255
PIXEL_MARKER_BACKGROUND=0
NP_ELEMENT_TYPE=np.uint8

class Params:
    """
    Description:
        Holds the params that ViBe will use such as the maximum number of
        samples.
    """
    def __init__(self, maxHistory = 20, noMin = 2, R = 20):
        """
        Description:
            Stores params.
        """
        self.maxHistory = maxHistory
        self.noMin = noMin
        self.R = R

class Model:
    """
    Description:
        Holds the information ViBe will use such as the current foreground
        pixels and the sample space.
    """
    def __init__(self, firstSample, params = Params()):
        """
        Description:
            Initializes the model.
        """
        self.foreGroundChannels, self.samplesArray, self.foreGround = init_model(firstSample, params)
        self.params=params
    def update(self, frame ):
        """
        Description:
            Updates ViBe's information using the current frame.
        """
        self.foreGround = processFrame( frame, self.foreGroundChannels, self.samplesArray, self.foreGround, self.params )

def TwoVariableIterator(h, w, sr=0,sc=0):
    """
    Description:
        Provides an easy way to iterate through a 2D space.
    """
    for r in range(sr, h):
        for c in range(sc, w):
            yield (r,c)

def init_model(firstSample, params ):
    """
    Description:
        Initializes the memory for the algorithm such as the sample space.
    """
    h, w, channelCount = firstSample.shape
    singleChannelSize = h, w
    channels = cv2.split(firstSample)
    foreGroundChannels=[]
    channelSamples=[]
    foreGround=np.zeros(singleChannelSize, dtype=NP_ELEMENT_TYPE)
    for ch in range(0, channelCount):
        chMat = np.zeros(singleChannelSize, dtype=NP_ELEMENT_TYPE)
        foreGroundChannels.append(chMat)
        samples = []
        for n in range(0, params.maxHistory ):
            variance=np.random.random_integers(0,20)-10
            samples.append(np.copy(channels[ch])+variance)

        channelSamples.append(samples)
    return foreGroundChannels, channelSamples, foreGround

def processChannel( frameChannel, h, w, samples, foreGroundChannel, params ):
    """
    Description:
        Processes a channel to detect the foreground and updates the sample
        space.
    """
    # This step below actually speeds python up.
    # Turns out that accessing the properties of an object slow it down.
    maxHistory=params.maxHistory
    R=params.R
    noMin=params.noMin
    # Leave a border of 1 unit thick.
    for (r,c) in TwoVariableIterator(h-1,w-1,1,1):
        frameChannelValue=int((frameChannel[r,c]))
        isPixelPartOfBackground = IsPixelPartOfBackground(samples, r, c, frameChannelValue, noMin, maxHistory, R )
        if ( isPixelPartOfBackground ):
            foreGroundChannel[r,c]=PIXEL_MARKER_BACKGROUND
            rand=0#XXX 
            if (rand==0):
                rndSample=np.random.random_integers(0,maxHistory-1)
                (samples[rndSample])[r,c]=(frameChannel[r,c])
            rand=0#XXX random phi?
            if ( rand == 0 ):
                rndSample=np.random.random_integers(0,maxHistory-1)
                rndD=np.random.random_integers(0,7)
                nc,nr = EightCardinalDirections[rndD]
                (samples[rndSample])[nr+r,nc+c]=(frameChannel[r,c])
        else:
            foreGroundChannel[r,c]=PIXEL_MARKER_FOREGROUND

def IsPixelPartOfBackground(samples, r,c, frameValue, noMin, maxHistory, R ):
    """
    Description:
        Determines if the pixel should be part of the background by looking for
        similarities in the sample space.
    """
    count=0
    index=0
    while ( count < noMin and index<maxHistory):
        dist=int((samples[index])[r,c])-frameValue
        if ( dist <= R and dist >= -R):
            count = count + 1
            if (count >= noMin):
                return True
        index = index + 1
    return False
    
def processFrame( frame, foreGroundChannels, channelSamples, foreGround, params ):
    """
    Description:
        Breaks the frames into channels and processes each one.
    """
    h, w, channelCount = frame.shape
    channels = cv2.split(frame)
    for ch in range(0, channelCount):
        processChannel(channels[ch], h, w, channelSamples[ch], foreGroundChannels[ch], params )
        if ( ch > 0 and ch < 2 ):
            foreGround=cv2.bitwise_or(foreGroundChannels[ch-1],foreGroundChannels[ch])
        if ( ch >= 2 ):
            foreGround=cv2.bitwise_or(foreGround,foreGroundChannels[ch])
    if(channelCount==1):
        return foreGroundChannels[0];
    return foreGround
    