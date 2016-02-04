import sys
import numpy as np
import cv2
import time


def getSimilarityTransform(src, dst):
    if len(src) < 2 or len(dst) < 2:
        return None

    if len(src[0]) < 2 or len(src[1]) < 2 or len(dst[0]) < 2 or len(dst[1]) < 2:
        return None

    # Separate points
    y1 = src[0][0]
    x1 = src[0][1]
    y2 = src[1][0]
    x2 = src[1][1]

    v1 = dst[0][0]
    u1 = dst[0][1]
    v2 = dst[1][0]
    u2 = dst[1][1]

    # Create matrix
    mat = np.array([
        [x1,  y1, 1, 0],
        [y1, -x1, 0, 1],
        [x2,  y2, 1, 0],
        [y2, -x2, 0, 1]
    ], np.float32)

    # Get the inverse matrix
    mat = np.linalg.inv(mat)

    # Create vector
    vec = np.array([
        [u1],
        [v1],
        [u2],
        [v2]
    ], np.float32)

    # Multiply matrices
    val = np.matmul(mat, vec)

    # Get elements
    a = val[0]
    b = val[1]
    c = val[2]
    d = val[3]

    # Make output transformation matrix (2 X 3)
    out = np.array([
        [ a, b, c],
        [-b, a, d]
    ], np.float32)

    return out


def main():
    if len(sys.argv) < 4:
        print 'usage: transform.py source points output'
        return

    # TODO Check for flags when we supply additional features

    # Parse arguments
    source_path = sys.argv[1]
    points_path = sys.argv[2]
    output_path = sys.argv[3]

    # Read points file
    try:
        with open(points_path) as points:
            # Get src -> dst point mappings
            src = []
            dst = []

            for line in points:
                values = line.split()
                if len(values) < 4:
                    # Skip lines with less than the 4 necessary values
                    continue

                src.append([int(values[1]), int(values[0])])    # Store values as (y, x) -- for OpenCV
                dst.append([int(values[3]), int(values[2])])
    except IOError:
        print 'Error--File not found:', points_path
        return
    except TypeError:
        print 'Error--Points must be integer values'
        return

    # Do OpenCV stuff

    # Convert to numpy arrays
    src = np.array(src, np.float32)
    dst = np.array(dst, np.float32)

    # Get appropriate transformation matrix (and normalize to perspective transformation matrix)
    if len(src) == 2:
        transform = getSimilarityTransform(src, dst)
        transform = np.append(transform, [[0, 0, 1]], 0)
    elif len(src) == 3:
        transform = cv2.getAffineTransform(src, dst)
        transform = np.append(transform, [[0, 0, 1]], 0)
    elif len(src) == 4:
        transform = cv2.getPerspectiveTransform(src, dst)
    else:
        print 'Error--Unexpected number of points (expected 2-4, got {})'.format(len(src))
        return

    # TODO Open source/destination videos and transform each frame


if __name__ == '__main__':
    main()
