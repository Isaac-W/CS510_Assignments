import ViBe as vibe
import sys 
import cv2
import time
import os

MIN_ARGS = 1


def createVideoWriter(output_path, codec, fps, width, height):
    try:
        writer = cv2.VideoWriter(output_path, codec, fps, (width, height))
        return writer
    except:
        print 'Error creating output video -- you may not have the right codecs installed!'
        print 'Defaulting to raw RGB format and .AVI extension...'

    try:
        # Try again with default BI_RGB codec and .AVI extension        
        writer = cv2.VideoWriter(output_path + '.avi', 0, fps, (width, height))
        return writer
    except:
        return None
        

def main():
    
    if len(sys.argv) < MIN_ARGS + 1:
        print("usage: %s <video file>" % sys.argv[0])
        return
    source_path=sys.argv[1]
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print 'Error--Unable to open video:', source_path
        return
    sourceName, file_extension = os.path.splitext(source_path)
    
    # Get video parameters (try to retain same attributes for output video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        
    writer = createVideoWriter("out" + file_extension, codec, fps, width, height)
    if not writer:
        print 'Error--Could not write to magnitude video:', outputPaths[0]
        return
    
    ret, frame = cap.read()
    startTime=time.time()
    model = vibe.Model(frame)
    endTime=time.time()
    totalTime=endTime-startTime
    print "init time: %g" % totalTime
    k=0
    h, w, channelCount = frame.shape
    megapixels = h*w/1000000.0
    print "megapixels: %g" % megapixels
    frames = 0
    startTime=time.time()
    try:
        while not k == 27:
            ret, frame = cap.read()
            channels = cv2.split(frame)
            combined = cv2.merge((
                cv2.bitwise_or(channels[0], model.foreGround ),
                cv2.bitwise_or(channels[1], model.foreGround ),
                cv2.bitwise_or(channels[2], model.foreGround )
            ))
            cv2.imshow('image', combined)
            writer.write(combined)
            k = cv2.waitKey(100)
            frameStartTime=time.time()
            model.update(frame)
            frameEndTime=time.time()
            print "Frame:", frames
            frames = frames + 1
            endTime=time.time()
            totalTime=endTime-startTime
            print "seconds this frame: %f" % (frameEndTime-frameStartTime)
            timeForEachFrame=totalTime/frames
            print "average seconds for each frame: %f" % timeForEachFrame
            print "average megapixels a second: %f" % (megapixels/timeForEachFrame)
    except KeyboardInterrupt:
        pass
    finally:
        print "Writing video to file."
        writer.release()
        cap.release()

if __name__ == '__main__':
    main()