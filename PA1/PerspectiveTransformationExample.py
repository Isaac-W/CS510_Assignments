import numpy as np
import cv2

# Initialize video capture, 0 is for using internal webcam
cap = cv2.VideoCapture(0)

# Getting video properties : width - height
width = cap.get(3)
height = cap.get(4)

src = np.array([[0,0], [0,width], [height, 0], [height, width]], np.float32)
dst = np.array([[0,0], [120,480], [height, 0], [360, 480]], np.float32)
# Computing the Perspective tranformation matrix
M = cv2.getPerspectiveTransform(src, dst)

print(width, height)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Applying the perspective transformation to the frame
    warped = cv2.warpPerspective(frame, M, (int(width), int(height)))

    # Display the resulting frame
    cv2.imshow('frame',warped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()