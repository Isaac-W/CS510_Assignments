import numpy as np
import cv2
import sys
import os
import csv
import time


# BEGIN CONSTANTS

MIN_ARGS = 3

PIXEL_MAX = 255

PX_STATIC = 0
PX_SHADOW = 50
PX_OUTSIDE = 85
PX_UNKNOWN = 170
PX_MOTION = 255

PX_LABELS = [PX_STATIC, PX_SHADOW, PX_OUTSIDE, PX_UNKNOWN, PX_MOTION]

# END CONSTANTS


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


def TwoVariableIterator(h, w, sr=0,sc=0):
    """
    Description:
        Provides an easy way to iterate through a 2D space.

    Args:
        h (uint): The maximum value for the vertical iterator.
        w (uint): The maximum value for the horizontal iterator.
        sr (uint): The starting value for the vertical iterator.
        sc (uint): The starting value for the horizontal iterator.
    Returns:
        Tuple: returns the current row and column.
    """
    for r in range(sr, h):
        for c in range(sc, w):
            yield (r,c)


def GetClosestPixelValue(pixel):
    """
    Finds the closest label and returns it. Fixup for ground truth pixel values.

    :param pixel: The grayscale value for a pixel
    :return: The label (PX_STATIC, etc.) for the pixel
    """

    closest = PIXEL_MAX
    out_label = PX_STATIC

    for label in PX_LABELS:
        diff = abs(pixel - label)
        if diff < closest:
            closest = diff
            out_label = label

    return out_label


def IsForeground(truth_pixel):
    """
    Checks if a pixel is considered foreground in the ground truth video.

    :param truth_pixel: The grayscale value for a pixel
    :return: True if is foreground, False if is background
    """

    if truth_pixel == PX_STATIC:
        return False
    elif truth_pixel == PX_SHADOW:
        return False
    elif truth_pixel == PX_OUTSIDE:
        return False
    elif truth_pixel == PX_UNKNOWN:
        return False
    elif truth_pixel == PX_MOTION:
        return True

    return False


def CalcP(tp, fp):
    if tp == 0:
        return 0
    return tp / float(tp + fp)


def CalcR(tp, fn):
    if tp == 0:
        return 0
    return tp / float(tp + fn)


def CalcF(p, r):
    if p == 0:
        return 0
    return (2 * p * r) / (p + r)


def CalcPWC(tp, fp, tn, fn):
    if (fn + fp) == 0:
        return 0
    return (fn + fp) / float(fn + fp + tp + tn)


def main():
    if len(sys.argv) < MIN_ARGS + 1:
        print 'usage: %s <ground truth> <input video> <start_frame> [-s|-n]' % sys.argv[0]
        print '-----------------------------------------------'
        print 'flags: -s -- silent; run without console output'
        print '       -n -- no csv output; console stats only'
        return

    truth_path = sys.argv[1]
    input_path = sys.argv[2]

    start_frame = int(sys.argv[3])

    # Check flags
    no_out = False
    no_csv = False

    if len(sys.argv) > MIN_ARGS + 1:
        flag = sys.argv[4].lower()
        if flag == '-s':
            no_out = True
        elif flag == '-n':
            no_csv = True

    # Open the videos
    truth_cap = cv2.VideoCapture(truth_path)
    input_cap = cv2.VideoCapture(input_path)

    # Truth and input must have same width/height, and have same number of frames!
    width = int(truth_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(truth_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    csv_file = None
    csv_writer = None

    # Create CSV writer
    if not no_csv:
        csv_path, extension = os.path.splitext(input_path)

        try:
            csv_file = open(csv_path + '.csv', 'wb')
            csv_writer = csv.writer(csv_file)

            # Write headers
            csv_writer.writerow([
                'Type', 'Index',
                'TP', 'FP', 'TN', 'FN',
                'P', 'R', 'F', 'PWC'
            ])
        except:
            print 'Error opening csv file (%s) for writing! ' % (csv_path + '.csv')
            return

    # Create difference output
    codec = int(input_cap.get(cv2.CAP_PROP_FOURCC))
    fps = int(input_cap.get(cv2.CAP_PROP_FPS))

    diff_path, extension = os.path.splitext(input_path)
    diff_writer = cv2.VideoWriter(diff_path + '_diff' + extension, codec, fps, (width, height))

    # Skip until start frame in truth video
    frame_number = 0

    for frame in range(start_frame):
        truth_ret, truth_frame = truth_cap.read()
        frame_number += 1

    # Init running totals
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    total_pwc = 0
    total_p = 0
    total_r = 0
    total_f = 0

    while True:
        truth_ret, truth_frame = truth_cap.read()
        input_ret, input_frame = input_cap.read()

        if truth_frame is None or input_frame is None:
            break

        # Show frames
        cv2.imshow('Ground Truth', truth_frame)
        cv2.imshow('Input Video', input_frame)
        cv2.waitKey(30)

        frame_tp = 0
        frame_tn = 0
        frame_fp = 0
        frame_fn = 0

        for (y, x) in TwoVariableIterator(height, width):
            # Assume grayscale; then all channels have the same value (use R channel only)
            truth_px = truth_frame.item(y, x, 0)    # array.item is faster than array[]
            input_px = input_frame.item(y, x, 0)

            # Fixup truth pixel
            truth_px = GetClosestPixelValue(truth_px)

            if IsForeground(truth_px):
                if input_px > 127:
                    # FG (True: FG)
                    frame_tp += 1
                else:
                    # BG (True: FG)
                    frame_fn += 1

                    # Flag false negatives as light blue
                    input_frame.itemset((y, x, 0), 192)
                    input_frame.itemset((y, x, 1), 192)
                    input_frame.itemset((y, x, 2), 0)
            else:
                if input_px > 127:
                    # FG (True: BG)
                    frame_fp += 1

                    # Flag false positives as red
                    input_frame.itemset((y, x, 0), 0)
                    input_frame.itemset((y, x, 1), 0)
                    input_frame.itemset((y, x, 2), 192)
                else:
                    # BG (True: BG)
                    frame_tn += 1

        # Output difference
        cv2.imshow('Difference', input_frame)
        diff_writer.write(input_frame)

        # Calculate per-frame metrics
        frame_p = CalcP(frame_tp, frame_fp)
        frame_r = CalcR(frame_tp, frame_fn)
        frame_f = CalcF(frame_p, frame_r)
        frame_pwc = CalcPWC(frame_tp, frame_fp, frame_tn, frame_fn)

        # Output running stats
        if not no_out:
            print 'Frame : %5d -- TP: %4d, FP: %4d, TN: %4d, FN: %4d' \
                  % (frame_number, frame_tp, frame_fp, frame_tn, frame_fn)
            print '      : %5d -- P: %.3f, R: %.3f, F: %.3f, PWC: %.3f' \
                  % (frame_number, frame_p, frame_r, frame_f, frame_pwc)

        # Write running stats
        if not no_csv:
            csv_writer.writerow([
                'Frame', frame_number,
                frame_tp, frame_fp, frame_tn, frame_fn,
                frame_p, frame_r, frame_f, frame_pwc
            ])

        # Update totals
        total_tp += frame_tp
        total_tn += frame_tn
        total_fp += frame_fp
        total_fn += frame_fn

        total_p += frame_p
        total_r += frame_r
        total_f += frame_f
        total_pwc += frame_pwc

        frame_number += 1

    # Calculate frame averages
    average_tp = total_tp / float(frame_number)
    average_tn = total_tn / float(frame_number)
    average_fp = total_fp / float(frame_number)
    average_fn = total_fn / float(frame_number)

    average_p = total_p / float(frame_number)
    average_r = total_r / float(frame_number)
    average_f = total_f / float(frame_number)
    average_pwc = total_pwc / float(frame_number)

    # Output frame averages
    if not no_out:
        print '-------------------------------------------------------------------'
        print '  Avg : %5d -- TP: %4.2f, FP: %4.2f, TN: %4.2f, FN: %4.2f' \
              % (frame_number, average_tp, average_fp, average_tn, average_fn)
        print '      : %5d -- P: %.3f, R: %.3f, F: %.3f, PWC: %.3f' \
              % (frame_number, average_p, average_r, average_f, average_pwc)

    # Calculate video stats
    video_p = CalcP(total_tp, total_fp)
    video_r = CalcR(total_tp, total_fn)
    video_f = CalcF(video_p, video_r)
    video_pwc = CalcPWC(total_tp, total_fp, total_tn, total_fn)

    # Output video stats
    if not no_out:
        print '-------------------------------------------------------------------'
        print 'Total : %5d -- TP: %4d, FP: %4d, TN: %4d, FN: %4d' \
              % (frame_number, total_tp, total_fp, total_tn, total_fn)
        print 'Video : %5d -- P: %.3f, R: %.3f, F: %.3f, PWC: %.3f' \
              % (frame_number, video_p, video_r, video_f, video_pwc)

    # Write video stats
    if not no_csv:
        # Average
        csv_writer.writerow([
            'Average', frame_number,
            average_tp, average_fp, average_tn, average_fn,
            average_p, average_r, average_f, average_pwc
        ])

        # Video
        csv_writer.writerow([
            'Video', frame_number,
            total_tp, total_fp, total_tn, total_fn,
            video_p, video_r, video_f, video_pwc
        ])

    # Cleanup
    if csv_file:
        csv_file.close()

    if diff_writer:
        diff_writer.release()


if __name__ == '__main__':
    main()
