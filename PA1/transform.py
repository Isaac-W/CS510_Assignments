import sys
import numpy as np
import cv2


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
        # TODO
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


if __name__ == '__main__':
    main()
