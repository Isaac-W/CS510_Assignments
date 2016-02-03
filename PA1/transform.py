import sys
import numpy as np
import cv2

# XXX This code needs to be cleaned up.
# I hacked it together very quickly.
# I would think there would be lots of bugs.
def ApplyTranformationToEachFrameInTheVideo( sourceFile, transformation, destFile):
    

    cap = cv2.VideoCapture(sourceFile)
    videoCodec = cv2.VideoWriter_fourcc(*'XVID')
    
    M = transformation


    # take first frame of the video
    ret,frame = cap.read()
    rows, cols = np.size(frame,0),np.size(frame,1)
    destWriter = cv2.VideoWriter(destFile,videoCodec, 20.0, (cols,rows))
    while(1):
        ret ,frame = cap.read()
        if ret == True:
            rows, cols = np.size(frame,0),np.size(frame,1)
            
            # Draw it on image
            img2 = cv2.warpAffine(frame, M, (cols, rows))
            destWriter.write(img2)
            cv2.imshow('img2',img2)
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            #else:
            #    cv2.imwrite(chr(k)+".jpg",img2)
        else:
            break
    destWriter.release()
    cv2.destroyAllWindows()
    cap.release()

def main():
    args = sys.argv[1:] # Remove first argument

    if len(sys.argv) < 4:
        print 'usage: transform.py source points output'
        return

    # TODO Check for flags when we supply additional features

    # Parse arguments
    source_path = args[0]
    points_path = args[1]
    output_path = args[2]

    # Read points file
    try:
        with open(points_path) as points:
            # Get src -> dst point mappings
            src = []
            dst = []

            for line in points:
                values = line.split()
                if len(values) < 4:
                    # print 'Error: Less than 4 numerical values found in line:', line
                    continue

                # TODO Catch unexpected non-integer values
                src.append([int(values[1]), int(values[0])])    # Read values as  (x, y)
                dst.append([int(values[3]), int(values[2])])    # Store values as (y, x) -- for OpenCV
    except:
        print 'Error--File not found:', points_path
        return

    # Do OpenCV stuff

    # Convert to numpy arrays
    src = np.array(src, np.float32)
    dst = np.array(dst, np.float32)

    # Get appropriate transformation matrix
    if len(src) == 2:
        # TODO It takes a point and a rotation angle instead of two points.
        # Maybe there is another function.
        transform = cv2.getRotationMatrix2D()
    elif len(src) == 3:
        transform = cv2.getAffineTransform(src, dst)
    elif len(src) == 4:
        transform = cv2.getPerspectiveTransform(src, dst)
    else:
        print 'Error--Unexpected number of points (expected 2-4, got {})'.format(len(src))
        return

    # TODO Open source/destination videos and transform each frame
    print transform
    ApplyTranformationToEachFrameInTheVideo( source_path, transform, output_path )




if __name__ == '__main__':
    main()
