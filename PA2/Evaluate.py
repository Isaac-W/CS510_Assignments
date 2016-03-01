import numpy as np
import cv2
import sys
import os
import csv
import time
import copy
import timeit

# BEGIN CONSTANTS

MIN_ARGS = 2

PIXEL_MAX = 255

# BEGIN Ground truth enum.
PX_STATIC = 0
PX_SHADOW = 50
PX_OUTSIDE = 85
PX_UNKNOWN = 170
PX_MOTION = 255
# END Ground truth enum.

PX_LABELS = [PX_STATIC, PX_SHADOW, PX_OUTSIDE, PX_UNKNOWN, PX_MOTION]

# END CONSTANTS

def createVideoWriter(output_file, codec, fps, width, height):
    """
    Description:
        Open a video writer and try to recover from errors.

    Args:
        output_file (string): the file path name to output to.
        codec: The openCV codec.
        fps (uint): Frames per second.
        width (int): the horizontal dimension.
        height (int): the vertical dimension.

    Returns:
        VideoWriter: Returns a video writer to the output file.
    """

    # Try to create the video with the specified codec and extension.
    try:
        writer = cv2.VideoWriter(output_file, codec, fps, (width, height))
        return writer
    except:
        print 'Error creating output video -- you may not have the right codecs installed!'
        print 'Defaulting to raw RGB format and .AVI extension...'
        # Try again with default BI_RGB codec and .AVI extension.
        try:
            writer = cv2.VideoWriter(output_file + '.avi', 0, fps, (width, height))
            return writer
        except:
            return None

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

def GetClosestPixelValue(pixel):
    """
    Finds the closest label and returns it. Fixup for ground truth pixel values.

    :param pixel: The grayscale value for a pixel
    :return: The label (PX_STATIC, etc.) for the pixel
    """

    closest = PIXEL_MAX
    out_label = PX_STATIC

    for label in PX_LABELS:
        diff = abs(pixel - label)
        if diff < closest:
            closest = diff
            out_label = label

    return out_label

def IsForeground(truth_pixel):
    """
    Checks if a pixel is considered foreground in the ground truth video.

    :param truth_pixel: The grayscale value for a pixel
    :return: True if is foreground, False if is background
    """

    if truth_pixel == PX_STATIC:
        return False
    elif truth_pixel == PX_SHADOW:
        return False
    elif truth_pixel == PX_OUTSIDE:
        return False
    elif truth_pixel == PX_UNKNOWN:
        return False
    elif truth_pixel == PX_MOTION:
        return True

    return False

# precision
def CalcP(tp, fp):
    if tp == 0:
        return 0
    return tp / float(tp + fp)

# recall
def CalcR(tp, fn):
    if tp == 0:
        return 0
    return tp / float(tp + fn)

def CalcF(p, r):
    if p == 0:
        return 0
    return (2 * p * r) / (p + r)

#pbc
def CalcPWC(tp, fp, tn, fn):
    if (fn + fp) == 0:
        return 0
    return (fn + fp) / float(fn + fp + tp + tn)

class FrameStats:
    def __init__(self, truthComparisonStats, truthMetaComparisonStats):
        self.metaStats = truthMetaComparisonStats
        self.stats = truthComparisonStats
    @staticmethod
    def GetCSVHeader():
        return [
            'Type', 'Index','TP', 'FP', 'TN', 'FN','P', 'R', 'F', 'PWC'
        ]
    def GetCSVArray(self, rowTitle, frame_number ):
        return [
            rowTitle, frame_number
        ]+\
        self.stats.GetCSVArray(rowTitle, frame_number)[2:]+\
        self.metaStats.GetCSVArray(rowTitle, frame_number)[2:]

class TruthMetaComparisonStats:
    def __init__(self, truthComparisonStats):
        self.tcs = truthComparisonStats
        self.p = CalcP(self.tcs.tp, self.tcs.fp)
        self.r = CalcR(self.tcs.tp, self.tcs.fn)
        self.f = CalcF(self.p, self.r)
        self.pwc = self.CalcPWC()
    def CalcPWC(self):
        return CalcPWC( self.tcs.tp, self.tcs.fp, self.tcs.tn, self.tcs.fn)

    def printOut(self, rowTitle, frame_number):
        print '%s : %5d -- P: %.3f, R: %.3f, F: %.3f, PWC: %.3f' \
              % (rowTitle, frame_number, self.p, self.r, self.f, self.pwc)
    def GetCSVArray(self, rowTitle, frame_number ):
        return [
            rowTitle, frame_number,
            self.p, self.r, self.f, self.pwc
        ]

class TruthComparisonStats:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.marker = [0,0,0]
    def accumulate( self, otherTruthComparisonStats ):
        self.tp = self.tp + otherTruthComparisonStats.tp
        self.tn = self.tn + otherTruthComparisonStats.tn
        self.fp = self.fp + otherTruthComparisonStats.fp
        self.fn = self.fn + otherTruthComparisonStats.fn
    def average( self, count ):
        self.tp = self.tp / float( count )
        self.tn = self.tn / float( count )
        self.fp = self.fp / float( count )
        self.fn = self.fn / float( count )

    def printOut(self, rowTitle, frame_number):
        print '%s : %5d -- TP: %4d, FP: %4d, TN: %4d, FN: %4d' \
              % (rowTitle, frame_number, self.tp, self.fp, self.tn, self.fn)
    def GetCSVArray(self, rowTitle, frame_number ):
        return [
            rowTitle, frame_number,
            self.tp, self.fp, self.tn, self.fn
        ]

def ComparePixels( truth_px, input_px):
    truthComparisonStats = TruthComparisonStats()
    # Fixup truth pixel
    truth_px = GetClosestPixelValue(truth_px)

    # The ground truth values are encoded.
    # Determine if this ground truth pixel represents the foreground.
    if IsForeground(truth_px):
        # Our results agree with the ground truth.
        if input_px:
            # FG (True: FG)
            truthComparisonStats.tp += 1
        # Our results report negative even though they should not.
        else:
            # BG (True: FG)
            truthComparisonStats.fn += 1
            truthComparisonStats.marker = [192, 192, 0]
    # The ground truth value is part of the background.
    else:
        # Our results report a positive even though they should not.
        if input_px:
            # FG (True: BG)
            truthComparisonStats.fp += 1
            truthComparisonStats.marker = [0, 0, 192]
        # Our results agree with the ground truth.
        else:
            # BG (True: BG)
            truthComparisonStats.tn += 1
    return truthComparisonStats

def CalculateFrameStats( h, w, truth_frame, input_frame, showDiff ):

    results = TruthComparisonStats()
    diffFrame = None
    if showDiff:
        diffFrame = np.zeros(
            (input_frame.shape[0], input_frame.shape[1], 3), dtype=np.uint8)

    for (y, x) in TwoVariableIterator(h, w):
        # Assume grayscale; then all channels have the same value (use R channel only)
        truth_px = truth_frame.item(y, x, 0)    # array.item is faster than array[]
        input_px = input_frame.item(y, x, 0)

        pixelResult = ComparePixels( truth_px, input_px )
        results.accumulate( pixelResult )
        if showDiff:
            diffFrame[y, x] = pixelResult.marker

    return results,diffFrame

def analyzeVideo( truth_cap, input_cap, csv_writer, no_out, no_csv ):
    # Truth and input must have same width/height, and have same number of frames!
    width = int(truth_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(truth_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Init running totals
    frame_number = 0
    videoStats = TruthComparisonStats()

    try:
        while True:
            truth_ret, truth_frame = truth_cap.read()
            input_ret, input_frame = input_cap.read()

            if truth_frame is None or input_frame is None:
                break

            truthComparisonStats = \
                CalculateFrameStats( height, width, truth_frame, input_frame )
            metaStats = TruthMetaComparisonStats( truthComparisonStats )

            # Output running stats
            if not no_out:
                truthComparisonStats.printOut( "Frame", frame_number)
                metaStats.printOut( "     ", frame_number)

            # Write running stats
            if not no_csv:
                frameStats = FrameStats(truthComparisonStats, metaStats)
                csvArray = frameStats.GetCSVArray( "Frame", frame_number )
                csv_writer.writerow( csvArray )

            # Update totals
            videoStats.accumulate(truthComparisonStats)

            frame_number += 1

    except KeyboardInterrupt:
        pass
    return frame_number, videoStats

def printSummary( frameCount, (avg, metaAvg, videoStats, metaVideoStats) ):
    avg.printOut( "Averages",  frameCount)
    metaAvg.printOut( "        ", frameCount)
    videoStats.printOut( "Totals  ",  frameCount)
    metaVideoStats.printOut( "        ", frameCount)

def writeSummaryToCSV( csv_writer, frameCount, (avg, metaAvg, stats, mStats) ):
    videoFullStats = FrameStats(stats, mStats)
    videoFullAverage = FrameStats(avg, metaAvg)
    # Average
    csvArray = videoFullStats.GetCSVArray( "Totals",  frameCount)
    csv_writer.writerow( csvArray )

    # Video
    csvArray = videoFullAverage.GetCSVArray( "Averages", frameCount)
    csv_writer.writerow( csvArray )

def createCSVWriter( csv_path ):
    csv_file = open(csv_path + '.csv', 'wb')
    return csv.writer(csv_file)

def calculateResults( frameCount, videoStats ):
    videoAverages = copy.deepcopy(videoStats)
    videoAverages.average(frameCount)

    metaStatsAverages = TruthMetaComparisonStats( videoAverages )
    metaVideoStats = TruthMetaComparisonStats( videoStats )

    return (videoAverages,metaStatsAverages, videoStats,metaVideoStats)

def main():
    if len(sys.argv) < MIN_ARGS + 1:
        print 'usage: %s <ground truth> <input video> [-s|-n]' % sys.argv[0]
        print '-----------------------------------------------'
        print 'flags: -s -- silent; run without console output'
        print '       -n -- no csv output; console stats only'
        return

    truth_path = sys.argv[1]
    input_path = sys.argv[2]

    no_out = False
    no_csv = False

    if len(sys.argv) > MIN_ARGS + 1:
        flag = sys.argv[3].lower()
        if flag == '-s':
            no_out = True
        elif flag == '-n':
            no_csv = True

    # Open the videos
    truth_cap = cv2.VideoCapture(truth_path)
    input_cap = cv2.VideoCapture(input_path)

    csv_file = None
    csv_writer = None

    # Create CSV writer
    if not no_csv:
        csv_path, extension = os.path.splitext(input_path)
        csv_writer = createCSVWriter( csv_path )

        # Write headers
        csv_writer.writerow( FrameStats.GetCSVHeader() )

    frame_number, videoStats = \
        analyzeVideo( truth_cap, input_cap, csv_writer, no_out,no_csv )

    # Calculate frame averages
    results = calculateResults( frame_number, videoStats )

    # Output totals and averages
    if not no_out:
        printSummary( frame_number, results )

    # Write video stats
    if not no_csv:
        writeSummaryToCSV( csv_writer, frame_number, results )

    # Cleanup
    if csv_file:
        csv_file.close()

if __name__ == '__main__':
    from timeit import Timer
    t = Timer(lambda: main())
    print t.timeit(number=1)
