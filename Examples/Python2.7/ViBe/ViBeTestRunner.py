"""
Description:
    This module runs the ViBe algorithm.

Module Summary:
function createVideoWriter -- Hardens video creation against errors.
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

# Minimum args required.
MIN_ARGS = 1

def main():
    # Display error if the program needs more args
    if len(sys.argv) < MIN_ARGS + 1:
        print("usage: %s <video file>" % sys.argv[0])
        return

    # Grab the argument.
    source_path=sys.argv[1]

    # Open the video.
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print 'Error--Unable to open video:', source_path
        return

    # Split the extension of the source video.
    sourceName, file_extension = os.path.splitext(source_path)

    # Get video parameters (try to retain same attributes for output video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Create an a video writer so that the resuls can be saved.
    writer = cv2.VideoWriter("outVibe.avi", codec, fps, (width, height))
    if not writer:
        print 'Error--Could not write to magnitude video:', outputPaths[0]
        return

    # Get the first frame and display it.
    #ret, frame = cap.read()
    ret, frame = cap.read()
    cv2.imshow('image', frame)
    k = cv2.waitKey(100)

    # Initialize and time the method.
    startTime=time.time()
    model = vibe.Model(cap)
    endTime=time.time()

    # Display time results.
    totalTime=endTime-startTime
    print "init time: %g" % totalTime

    # Calculate and display megapixels.
    h, w, channelCount = frame.shape
    megapixels = h*w/1000000.0
    print "megapixels: %g" % megapixels

    # Protect against an error, so the video can be saved regardless.
    try:
        # Keep track of the last input key, the frames, and time.
        k=0
        frames = 0
        startTime=time.time()

        # Loop until the user presses escape.
        while not k == 27:
            # Grab next frame.
            ret, frame = cap.read()
            if not ret:
                break

            # Run ViBe on the current frame to update the model.
            frameStartTime=time.time()
            model.update(frame)
            frameEndTime=time.time()

            # Display statistics.
            print "Frame:", frames
            frames = frames + 1
            endTime=time.time()
            totalTime=endTime-startTime
            print "seconds this frame: %f" % (frameEndTime-frameStartTime)
            timeForEachFrame=totalTime/frames
            print "average seconds for each frame: %f" % timeForEachFrame
            print "average megapixels a second: %f" % (megapixels/timeForEachFrame)

            # Overlay the current frame with the results.
            #channels = cv2.split(frame)
            #blank_image = numpy.zeros((height, width), numpy.uint8)
            #combined = model.foreGround
            channel = np.zeros((height,width,1), np.uint8)
            combined = cv2.merge((
                cv2.bitwise_or(channel, cv2.pyrUp(model.foreGround)),
                cv2.bitwise_or(channel, cv2.pyrUp(model.foreGround)),
                cv2.bitwise_or(channel, cv2.pyrUp(model.foreGround))
            ))

            # Show the results and write it to the file buffer.
            cv2.imshow('image', combined)
            writer.write(combined)

            # Grab the key pressed.
            k = cv2.waitKey(100)
    except KeyboardInterrupt:
        pass
    finally:
        print "Writing video to file."
        writer.release()
        cap.release()

# Run main if this was the main module.
if __name__ == '__main__':
    main()