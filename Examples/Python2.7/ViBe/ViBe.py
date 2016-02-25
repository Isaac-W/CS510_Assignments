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
NP_ELEMENT_TYPE=np.uint8

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

        Args:
            firstSample (ndarray (h, w, 3)): the first frame that will be used
                to initialize the algorithm.
            params (Params): the params control how the algorithm works.
        """
        self.foreGroundChannels, self.samplesArray, self.foreGround = init_model(firstSample, params)
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
        self.foreGround = processFrame( 
            frame,
            self.foreGroundChannels,
            self.samplesArray,
            self.foreGround,
            self.params
        )

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

def init_model(firstSample, params ):
    """
    Description:
        Initializes the memory for the algorithm such as the sample space.

    Args:
        params (Params): the params control how the algorithm works.
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

        # Initialize the samples to the background.
        # Allow the samples to vary a little bit.
        samples = []
        for n in range(0, params.maxHistory ):
            variance=np.random.random_integers(0,20)-10
            samples.append(np.copy(channels[ch])+variance)

        channelSamples.append(samples)
    return foreGroundChannels, channelSamples, foreGround

def processChannel( frameChannel, samples, params ):
    """
    Description:
        Processes a channel to detect the foreground and updates the sample
        space.

    Args:
        frameChannel (ndarray (h, w, 1)): A 2D array of the pixels of the
            current frame for one pixel.
        samples (list): A list samples. Each element is a 2D array of pixels.
            So this param is kind of like a 3D cube.
        params (Params): the params control how the algorithm works.
    """
    h, w = frameChannel.shape
    newForeGroundChannel = np.zeros(frameChannel.shape, dtype=NP_ELEMENT_TYPE)
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
        frameChannelValue=int((frameChannel[r,c]))
        isPixelPartOfBackground = IsPixelPartOfBackground(samples, r, c, frameChannelValue, noMin, maxHistory, R )
        # Plant seeds for pixels that are part of the background.
        if ( isPixelPartOfBackground ):
            # Set this pixel as part of the bg.
            newForeGroundChannel[r,c]=PIXEL_MARKER_BACKGROUND
            # Detect slow changes in the background
            rand=0#XXX 
            if (rand==0):
                rndSample=np.random.random_integers(0,maxHistory-1)
                (samples[rndSample])[r,c]=(frameChannel[r,c])
            # Allow ghosts to disappear by planting seeds in nearby pixels.
            rand=0#XXX random phi?
            if ( rand == 0 ):
                rndSample=np.random.random_integers(0,maxHistory-1)
                rndD=np.random.random_integers(0,7)
                nc,nr = EightCardinalDirections[rndD]
                (samples[rndSample])[nr+r,nc+c]=(frameChannel[r,c])
        # Set this pixel as part of the fg.
        else:
            newForeGroundChannel[r,c]=PIXEL_MARKER_FOREGROUND
    return newForeGroundChannel

def IsPixelPartOfBackground(samples, r,c, frameValue, noMin, maxHistory, R ):
    """
    Description:
        Determines if the pixel should be part of the background by looking for
        similarities in the sample space.

    Args:
        samples (list): A list samples. Each element is a 2D array of pixels.
            So this param is kind of like a 3D cube.
        r (int): The current row.
        c (int): The current column.
        frameValue (int): the value of the current pixel.
        noMin (int): The threshold that determines if a pixel should belong to
            the background. For example, 2 would mean that there are 2 samples
            that need to be similar.
        maxHistory (int): The maximum samples to keep or, in-other-words,
            the pixel history.
        R (int): The threshold that determines if a pixel is similar. For
            example, 20 means that pixels can have a difference of 20 for one of
            their channels such as the red channel.

    Returns:
        ndarray (shape=(h,w,1)): Returns the new foreground for the current
            channel.
    """
    count=0
    index=0
    # Go through the samples.
    while ( index<maxHistory):
        # Find a measure of the similarity.
        dist=int((samples[index])[r,c])-frameValue
        # Count similar pixels.
        if ( dist <= R and dist >= -R):
            count = count + 1
            # The pixel is part of the background
            # if we have reached the threshold.
            if (count >= noMin):
                return True
        index = index + 1
    return False

def processFrame( frame, foreGroundChannels, channelSamples, foreGround, params ):
    """
    Description:
        Breaks the frames into channels and processes each one.

    Args:
        frame (ndarray (shape = (h, w, chCount) )): A video or picture frame.
        [out] foreGroundChannels (list): the results from the ViBe algorithm
            are saved.
        channelSamples (list): the 3D cube of samples for each channel.
        [out] foreGround (ndarray (shape = (h, w, 1) )): the new foreground is
            saved.
        params (Params): the params control how the algorithm works.

    Returns:
        ndarray (shape=(h,w,1)): Returns the new foreground for the current
            channel.
    """
    h, w, channelCount = frame.shape
    channels = cv2.split(frame)
    # Process each channel and combine the results into the foreground.
    for ch in range(0, channelCount):
        # Process the current channel.
        newfgCh=processChannel(channels[ch], channelSamples[ch],params )
        # Replace the old foreground.
        if ( ch == 0 ):
            foreGround=newfgCh
        # Combine the rest of foreground results.
        else:
            foreGround=cv2.bitwise_or(foreGround,newfgCh)
        # Save results for this channel in case we need them later.
        foreGroundChannels[ch]=newfgCh
    if(channelCount==1):
        return foreGroundChannels[0];
    return foreGround
