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
MIN_ARGS = 1

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
        model, writer, diffWriter, height, width ):

    # Calculate and display megapixels.
    megapixels = height * width / 1000000.0
    print "megapixels: %g" % megapixels
    
    # Truth and input must have same width/height, and have same number of frames!
    width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Init running totals
    frame_number = 0
    videoStats = e.TruthComparisonStats()
    
    # Fast forward past <params.startFrame> frames.
    print "Skipping %d frames." % params.startFrame
    for i in range( 0, params.startFrame ):
        truth_ret = 1 # Set to non null in case not in eval mode.
        if params.doEval:
            truth_ret = truth_cap.grab()
        input_ret = input_cap.grab()

        if not truth_ret or not input_ret:
            raise Exception( "Start time past the end of videos." )
        frame_number += 1

    try:
        # Keep track of the last input key, the frames, and time.
        k = 0
        processedFrames = 0
        evalFrames = 0
        startTime = time.time()
        
        while not k == 27:
            truth_frame = 1 # Set to non null in case not in eval mode.
            if params.doEval:
                truth_ret, truth_frame = truth_cap.read()
            input_ret, input_frame = input_cap.read()

            if truth_frame is None or input_frame is None:
                break

            if params.showInput:
                cv2.imshow('Input', input_frame)
            if params.showTruth and programParams.doEval:
                cv2.imshow('truth_frame', truth_frame )
            
            frame_number += 1
            processedFrames += 1
            print "Frame:", frame_number
            
            input_frame = vibe.preprocess_frame(input_frame)
            
            toShow, toFile = processFrame( 
                model, writer, input_frame, height, width )

            if params.doEval:
                evalFrames += 1
                truthComparisonStats, diffFrame = \
                    e.CalculateFrameStats( height, width, truth_frame, toFile,
                        params.showDiff )
                metaStats = e.TruthMetaComparisonStats( truthComparisonStats )
                
                if params.showDiff:
                    diffWriter.write( diffFrame )
                    cv2.imshow('diffFrame', diffFrame )

                # Output running stats
                if not params.no_out:
                    truthComparisonStats.printOut( "Frame", evalFrames)
                    metaStats.printOut( "     ", evalFrames)

                # Write running stats
                if not params.no_csv:
                    frameStats = e.FrameStats(truthComparisonStats, metaStats)
                    csvArray = frameStats.GetCSVArray( "Frame", evalFrames )
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
            cv2.imshow('Processing Results', toShow)
            writer.write(toFile)
            
            # Grab the key pressed.
            k = cv2.waitKey(100)

    except KeyboardInterrupt:
        pass
    return frame_number, processedFrames, evalFrames, videoStats


class Params:
    def __init__(self, input_path):
        self.no_out = False
        self.no_csv = False
        self.doEval = False
        self.showDiff = False
        self.showTruth = False
        self.showInput = False
        self.startFrame = 0
        self.input_path = input_path
        self.truth_path = None

class BadArgumentsException(Exception):
    pass
    
def assertFileExists( filePath ):
    if not os.path.isfile(filePath):
        raise BadArgumentsException( 
            "'%s' does not exist or does not specify a file." % filePath )

def assertHasArgumentIndex( index, argName, expectedArgName ):
    if index >= len(sys.argv):
        raise BadArgumentsException( 
            "Expected %s for argument '%s'" % (expectedArgName,argName) )

            
            
def readArgs( params ):
    global no_out, no_csv, doEval, startFrame
    i = MIN_ARGS + 1
    while i < len(sys.argv):
        currentArg = sys.argv[i]
        if currentArg == '-e':
            i = i + 1
            assertHasArgumentIndex( i, '-e', "<ground truth path>")
            params.truth_path = sys.argv[i]
            assertFileExists( params.truth_path )
            params.doEval = True
        elif currentArg == '-s':
            params.no_out = True
        elif currentArg == '-n':
            params.no_csv = True
        elif currentArg == '-Si':
            params.showInput = True
        elif currentArg == '-Sd':
            params.showDiff = True
        elif currentArg == '-St':
            params.showTruth = True
        elif currentArg == '-t':
            i = i + 1
            assertHasArgumentIndex( i, '-t', "<startFrame>")
            try:
                params.startFrame = int(sys.argv[i])
            except ValueError:
                raise BadArgumentsException(
                    "Expected int for argument '%s'" % '-t'  )
        i = i + 1
def showUsage():
    print ( 'usage: %s <input video> [flags..] ' + \
        '[-t <startFrame> ]') % sys.argv[0]
    print '-----------------------------------------------'
    print 'flags:'
    print '       -e <ground truth video> -- evaluate the results. Turn on evaluation mode.'
    print '       -s -- silent; run without console output.'
    print '       -t <startFrame> -- start running after <startFrame> frames.'
    print '       -Si -- show input frames.'
    print '       -St -- show truth frame.'
    print 'EVALUATION MODE:'
    print '       -n -- no csv output.'
    print '       -Sd -- show diff frame.'

def main():
    if len(sys.argv) < MIN_ARGS + 1:
        showUsage()
        return

    programParams = Params(input_path = sys.argv[1] )

    # Read and validate arguments
    try:
        assertFileExists( programParams.input_path )
        readArgs( programParams )
    except BadArgumentsException, err:
        print str(err)
        showUsage()
        return 1

    # Open the videos
    truth_cap = None
    if programParams.doEval:
        truth_cap = cv2.VideoCapture(programParams.truth_path)
    input_cap = cv2.VideoCapture(programParams.input_path)

    csv_file = None
    csv_writer = None

    # Create CSV writer
    if not programParams.no_csv and programParams.doEval:
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
    sourceName, file_extension = os.path.splitext(programParams.input_path)
    output_path = sourceName + '_outvibe' + file_extension
    writer = cv2.VideoWriter(output_path, codec, fps, (width, height))
    if not writer:
        print 'Error--Could not write to out video:', output_path
        return

    diffWriter = None
    if programParams.doEval and programParams.showDiff:
        # Create an a video writer so that the resuls can be saved.
        output_path = sourceName + '_diff' + file_extension
        diffWriter = cv2.VideoWriter(output_path, codec, fps, (width, height))
        if not writer:
            print 'Error--Could not write to diff video:', output_path
            return

    # Initialize initial ViBE background model
    startTime = time.time()
    model = vibe.Model(input_cap)
    endTime = time.time()

    # Display time results.
    totalTime = endTime - startTime
    print "init time: %g" % totalTime

    frame_number, processedFrames, evalFrames, videoStats = \
        processAndAnalyzeVideo( truth_cap, input_cap, csv_writer, programParams, 
            model, writer, diffWriter, height, width )

    # Calculate frame averages
    if programParams.doEval:
        results = e.calculateResults( evalFrames, videoStats )

    # Output totals and averages
    if not programParams.no_out and programParams.doEval:
        e.printSummary( evalFrames, results )

    # Write video stats
    if not programParams.no_csv and programParams.doEval:
        e.writeSummaryToCSV( csv_writer, evalFrames, results )

    # Cleanup
    if csv_file:
        csv_file.close()
    if diffWriter:
        diffWriter.release()
    if writer:
        writer.release()

# Run main if this was the main module.
if __name__ == '__main__':
    from timeit import Timer
    t = Timer(lambda: main())
    time = t.timeit(number=1)
    print "Running time: ", time
