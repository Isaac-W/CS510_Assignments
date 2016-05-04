import numpy as np
import cv2
import sys
import os


CROP_SIZE = (120, 120)
CROP_CENTER = (80, 60)
WINDOW_SIZE = (20, 20)
WINDOW_FRAMES = 30


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def create_training_videos(label, video_path, roi_list, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []

    while True:
        ret, frame = cap.read()
        if ret is False or frame is None:
            break

        # Preprocess frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.getRectSubPix(frame, CROP_SIZE, CROP_CENTER)
        frame = cv2.resize(frame, WINDOW_SIZE)

        # Add frame to list
        frames.append(frame)

    cap.release()

    for index, (start_frame, end_frame) in enumerate(roi_list):
        path = output_path + '/' + os.path.splitext(os.path.basename(video_path))[0] + '_sample' + str(index) + '.avi'
        print label + ', ' + path
        dst_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), 30, WINDOW_SIZE, False)

        roi_len = end_frame - start_frame + 1

        if roi_len > WINDOW_FRAMES:
            half = start_frame + (roi_len / 2)
            start_frame = half - (WINDOW_FRAMES / 2)
            end_frame = half + (WINDOW_FRAMES / 2)

        #print start_frame, end_frame, roi_len, len(frames)

        for i in range(WINDOW_FRAMES):
            frame_index = start_frame + (i % roi_len) - 1
            dst_writer.write(frames[frame_index])

        dst_writer.release()


def main():
    if len(sys.argv) < 3:
        print 'USAGE: GetTrainingActions <input_sequences_file> <output_path>'
        return

    with open(sys.argv[1], 'r') as f:
        for line in f:
            vals = line.strip().split(', ')
            if len(vals) < 2:
                continue

            label = vals[0]
            path = vals[1]
            roi = []

            for i in range(2, len(vals)):
                roi_vals = vals[i].split('-')
                roi.append((int(roi_vals[0]), int(roi_vals[1])))

            create_training_videos(label, path, roi, sys.argv[2])



if __name__ == '__main__':
    main()
