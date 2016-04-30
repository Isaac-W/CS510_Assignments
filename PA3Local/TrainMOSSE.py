import numpy as np
import cv2
import MOSSE
import sys
import os


WAIT_FRAME = False


def main():
    if len(sys.argv) < 4:
        print 'USAGE: TrainMOSSE.py <path> <width> <height>'
        return

    path = sys.argv[1]
    w = int(sys.argv[2])
    h = int(sys.argv[3])

    mosse = MOSSE.MOSSEMatcher((w, h))

    for img in os.listdir(path):
        sample = cv2.imread(os.path.join(path, img))
        if sample is None:
            continue

        mosse.train(sample)

        cv2.imshow("Training MOSSE Filter", mosse.state_vis)

        if WAIT_FRAME:
            while True:
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    return
                elif k != 255:
                    break
        else:
            _ = cv2.waitKey(30)

    # Save MOSSE filter
    mosse.save_filter(path + '/mosse_matcher_nolines.pkl')


if __name__ == '__main__':
    main()
