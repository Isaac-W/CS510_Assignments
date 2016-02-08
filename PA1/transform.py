import sys
import numpy as np
import cv2


def getTransform(src, dst):
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
        #print 'Error--Unexpected number of points (expected 2-4, got {})'.format(len(src))
        print 'error'
        return None

    return transform


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
    a = val[0][0]
    b = val[1][0]
    c = val[2][0]
    d = val[3][0]

    # Make output transformation matrix (2 X 3)
    transform = np.array([
        [ a, b, c],
        [-b, a, d]
    ], np.float32)

    return transform


def applyVideoTransformation(source_path, transform, output_path):
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print 'Error--Unable to open video:', source_path
        return

    # Get video parameters (try to retain same attributes for output video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Fix codec
    #if codec == 0:
    #    codec = cv2.VideoWriter_fourcc(*'MJPG')

    dst_writer = cv2.VideoWriter(output_path, codec, fps, (width, height))
    if not dst_writer.isOpened():
        print 'Error--Could not write to video:', output_path
        return

    while True:
        # Get frame
        ret, frame = cap.read()
        if ret is False or frame is None:
            break

        # Transform frame
        frame = cv2.warpPerspective(frame, transform, (width, height))
        dst_writer.write(frame)

    dst_writer.release()
    cap.release()

def applyFourierTransform(source_path, output_path):
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print 'Error--Unable to open video:', source_path
        return

    # Get video parameters (try to retain same attributes for output video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Fix codec
    #if codec == 0:
    #    codec = cv2.VideoWriter_fourcc(*'MJPG')

    dst_writer = cv2.VideoWriter(output_path, codec, fps, (width, height))
    if not dst_writer.isOpened():
        print 'Error--Could not write to video:', output_path
        return

    while True:
        # Get frame
        ret, frame = cap.read()
        if ret is False or frame is None:
            break


        from matplotlib import pyplot as plt

        # img = cv2.imread('xfiles.jpg',0)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        # magnitude_spectrum = 20*np.log(np.abs(fshift))

        # plt.subplot(121),plt.imshow(img, cmap = 'gray')
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
        # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        # plt.show()

        rows, cols = img.shape
        crow,ccol = rows/2 , cols/2
        fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        plt.subplot(131),plt.imshow(img, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
        plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
        plt.subplot(133),plt.imshow(img_back)
        plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

        plt.show()


        # print type(frame)
        # print type(np.float32(frame))
        # dft = cv2.dft(np.float32(frame),flags = cv2.DFT_COMPLEX_OUTPUT)
        #dft_shift = np.fft.fftshift(dft)
        # print type(dft)
        # print type(dft_shift)
        # dst_writer.write(frame)


    dst_writer.release()
    cap.release()

def main():
    if len(sys.argv) < 4:
        print 'usage: transform.py source points output'
        return

    # TODO Check for flags when we supply additional features

    # Parse arguments
    source_path = sys.argv[1]
    points_path = sys.argv[2]
    output_path = sys.argv[3]
    flag = None
    if len(sys.argv) is 5:
        flag = sys.argv[4]

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

                src.append([int(values[0]), int(values[1])])
                dst.append([int(values[2]), int(values[3])])
    except IOError:
        print 'Error--File not found:', points_path
        return
    except TypeError:
        print 'Error--Points must be integer values'
        return

    # Convert to numpy arrays
    src = np.array(src, np.float32)
    dst = np.array(dst, np.float32)

    # Get transformation matrix
    transform = getTransform(src, dst)
    applyVideoTransformation(source_path, transform, output_path)
    if flag is 'f':
        applyFourierTransform(source_path, 'fourier.avi')




if __name__ == '__main__':
    main()
