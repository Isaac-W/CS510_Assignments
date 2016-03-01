"""
Description:
    This module runs the ViBe algorithm.

Module Summary:
function main -- starts the test runner.

Author:
    Joshua Gillham

Date:
    2016-02-24
"""

# Test runner imports.
import ViBe as vibe
import sys 
import cv2
import time
import os
import numpy as np
import Evaluate as e

# Minimum args required.
MIN_ARGS = 2

def processFrame( model, writer, frame, height, width ):
    # Run ViBe on the current frame to update the model.
    frameStartTime = time.time()
    model.update(frame)
    frameEndTime = time.time()
    
    print "seconds for ViBe processing: %f" % (frameEndTime - frameStartTime)

    # Overlay the current frame with the results.
    # channels = cv2.split(frame)
    # blank_image = numpy.zeros((height, width), numpy.uint8)
    # combined = model.foreGround

    channel = np.zeros((height, width, 1), np.uint8)
    fullSized = cv2.pyrUp(model.foreGround)
    resultOneChannel = cv2.bitwise_or(channel, fullSized)
    combined = cv2.merge((
        resultOneChannel,
        resultOneChannel,
        resultOneChannel
    ))
    
    return combined, combined

def processAndAnalyzeVideo( truth_cap, input_cap, csv_writer, params,
        model, writer, height, width ):

    # Calculate and display megapixels.
    megapixels = height * width / 1000000.0
    print "megapixels: %g" % megapixels
    
    # Truth and input must have same width/height, and have same number of frames!
    width = int(truth_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(truth_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Init running totals
    frame_number = 0
    videoStats = e.TruthComparisonStats()
    
    # Fast forward past <params.startFrame> frames.
    print "Skipping %d frames." % params.startFrame
    for i in range( 0, params.startFrame ):
        truth_ret = truth_cap.grab()
        input_ret = input_cap.grab()

        if not truth_ret or not input_ret:
            raise Exception( "Start time past the end of videos." )

    try:
        # Keep track of the last input key, the frames, and time.
        k = 0
        frames = 0
        startTime = time.time()
        
        while not k == 27:
            truth_ret, truth_frame = truth_cap.read()
            input_ret, input_frame = input_cap.read()

            if truth_frame is None or input_frame is None:
                break

            cv2.imshow('image22', input_frame)
            
            frame_number += 1
            print "Frame:", frame_number
            
            input_frame = vibe.preprocess_frame(input_frame)
            
            toShow, toFile = processFrame( 
                model, writer, input_frame, height, width )

            if params.doEval:
                truthComparisonStats = \
                    e.CalculateFrameStats( height, width, truth_frame, toFile )
                metaStats = e.TruthMetaComparisonStats( truthComparisonStats )

                # Output running stats
                if not params.no_out:
                    truthComparisonStats.printOut( "Frame", frame_number)
                    metaStats.printOut( "     ", frame_number)

                # Write running stats
                if not params.no_csv:
                    frameStats = e.FrameStats(truthComparisonStats, metaStats)
                    csvArray = frameStats.GetCSVArray( "Frame", frame_number )
                    csv_writer.writerow( csvArray )

                # Update totals
                videoStats.accumulate(truthComparisonStats)

            # Display statistics.
            endTime = time.time()
            totalTime = endTime - startTime
            timeForEachFrame = totalTime / frame_number
            print "average seconds for each frame: %f" % timeForEachFrame
            print "average megapixels a second: %f" % (megapixels / 
                timeForEachFrame)

            # Show the results and write it to the file buffer.
            cv2.imshow('image', toShow)
            writer.write(toFile)
            
            # Grab the key pressed.
            k = cv2.waitKey(100)

    except KeyboardInterrupt:
        pass
    return frame_number, videoStats


class Params:
    def __init__(self, truth_path, input_path):
        self.no_out = False
        self.no_csv = False
        self.doEval = False
        self.startFrame = 0
        self.truth_path = truth_path
        self.input_path = input_path

def readArgs( params ):
    global no_out, no_csv, doEval, startFrame
    i = MIN_ARGS + 1
    while i < len(sys.argv):
        currentArg = sys.argv[i]
        if currentArg == '-e':
            params.doEval = True
        elif currentArg == '-s':
            params.no_out = True
        elif currentArg == '-n':
            params.no_csv = True
        elif currentArg == '-t':
            i = i + 1
            params.startFrame = int(sys.argv[i])
        i = i + 1

def main():
    if len(sys.argv) < MIN_ARGS + 1:
        print ( 'usage: %s <ground truth> <input video> [-e] [-s|-n] ' + \
            '[-t <startFrame> ]') % sys.argv[0]
        print '-----------------------------------------------'
        print 'flags:'
        print '       -e -- evaluate the results.'
        print '       -n -- no csv output; console stats only.'
        print '       -s -- silent; run without console output.'
        print '       -t <startFrame> -- start running after <startFrame> frames.'
        return

    programParams = Params(truth_path = sys.argv[1], input_path = sys.argv[2] )

    # Split the extension of the source video.
    sourceName, file_extension = os.path.splitext(programParams.input_path)
    
    readArgs( programParams )

    # Open the videos
    truth_cap = cv2.VideoCapture(programParams.truth_path)
    input_cap = cv2.VideoCapture(programParams.input_path)

    csv_file = None
    csv_writer = None

    # Create CSV writer
    if not programParams.no_csv:
        csv_path, extension = os.path.splitext(programParams.input_path)
        csv_writer = e.createCSVWriter( csv_path )

        # Write headers
        csv_writer.writerow( e.FrameStats.GetCSVHeader() )

    # Get video parameters (try to retain same attributes for output video)
    width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(input_cap.get(cv2.CAP_PROP_FPS))
    codec = int(input_cap.get(cv2.CAP_PROP_FOURCC))

    # Create an a video writer so that the resuls can be saved.
    output_path = sourceName + '_outvibe' + file_extension
    writer = cv2.VideoWriter(output_path, codec, fps, (width, height))
    if not writer:
        print 'Error--Could not write to magnitude video:', output_path
        return

    # Initialize initial ViBE background model
    startTime = time.time()
    model = vibe.Model(input_cap)
    endTime = time.time()

    # Display time results.
    totalTime = endTime - startTime
    print "init time: %g" % totalTime

    frame_number, videoStats = \
        processAndAnalyzeVideo( truth_cap, input_cap, csv_writer, programParams, 
            model, writer, height, width )

    # Calculate frame averages
    if programParams.doEval:
        results = e.calculateResults( frame_number, videoStats )

    # Output totals and averages
    if not programParams.no_out and programParams.doEval:
        e.printSummary( frame_number, results )

    # Write video stats
    if not programParams.no_csv and programParams.doEval:
        e.writeSummaryToCSV( csv_writer, frame_number, results )

    # Cleanup
    if csv_file:
        csv_file.close()

# Run main if this was the main module.
if __name__ == '__main__':
    from timeit import Timer
    t = Timer(lambda: main())
    print "Running time: ", t.timeit(number=1)
