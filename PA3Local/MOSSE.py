"""
MOSSE tracking sample
Taken from OpenCV 3 Python samples
Modified by Isaac Wang

This sample implements correlation-based tracking approach, described in [1].

[1] David S. Bolme et al. "Visual Object Tracking using Adaptive Correlation Filters"
    http://www.cs.colostate.edu/~bolme/publications/Bolme2010Tracking.pdf
"""

import numpy as np
import cv2
import pickle


def load_filter(path):
    my_filter = None
    with open(path, 'rb') as f:
        my_filter = pickle.load(f)
    return my_filter


def rnd_warp(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return cv2.warpAffine(a, T, (w, h), borderMode = cv2.BORDER_REFLECT)


def divSpec(A, B):
    """
    Computes the optimal filter by dividing sum of spectrums
    :param A: matrix of sum G x F*
    :param B: matrix of sum F x F*
    :return: matrix representing optimal MOSSE filter
    """

    Ar, Ai = A[...,0], A[...,1]
    Br, Bi = B[...,0], B[...,1]
    C = (Ar+1j*Ai)/(Br+1j*Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C

eps = 1e-5


class MOSSE(object):
    def __init__(self, frame, rect):
        # Get frame size
        x1, y1, x2, y2 = rect

        # Resize to fft window
        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2

        # Store position and size relative to frame
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size = w, h

        # Crop template image from frame
        img = cv2.getRectSubPix(frame, (w, h), (x, y))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create Hanning window (weighting of values from center)
        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)

        # Create output image (centered Gaussian point--for correlation)
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        # Save the desired output image in frequency space
        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)

        # Create transformed variants of input
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        for i in xrange(128):
            # Preprocess, get DFT
            a = self.preprocess(rnd_warp(img))
            A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)

            self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)  # Sum of G x F*
            self.H2 += cv2.mulSpectrums(     A, A, 0, conjB=True)  # Sum of F x F*

        # Update filter
        self.update_kernel()
        self.update(frame)


    def update(self, frame, pos=None, rate=0.125):
        # Crop template image from last position
        (x, y), (w, h) = self.pos if pos is None else pos, self.size
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self.preprocess(img)

        # Correlate, find position of object
        self.last_resp, (dx, dy), self.psr = self.correlate(img)

        # Break if lost tracking (don't update filter)
        self.good = self.psr > 8.0
        if not self.good:
            return

        # Cut out new image based on tracked location
        self.pos = x+dx, y+dy
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.pos)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Preprocess, get DFT
        img = self.preprocess(img)
        A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

        H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)  # G x F*
        H2 = cv2.mulSpectrums(     A, A, 0, conjB=True)  # F x F*

        # Get weighted average based on the rate (using the new image)
        self.H1 = self.H1 * (1.0-rate) + H1 * rate
        self.H2 = self.H2 * (1.0-rate) + H2 * rate

        # Update filter
        self.update_kernel()

    @property
    def state_vis(self):
        """
        Get state of the filter in RGB space
        :return: combined image showing last tracked image, filter kernel, and output response
        """

        # Compute inverse DFT
        f = cv2.idft(self.H, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        # Reshape to template dimensions
        h, w = f.shape
        f = np.roll(f, -h//2, 0)
        f = np.roll(f, -w//2, 1)

        # Convert kernel
        kernel = np.uint8((f-f.min()) / f.ptp()*255)

        # Convert last correlation output response
        resp = self.last_resp
        resp = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)

        # Position the images side by side
        vis = np.hstack([self.last_img, kernel, resp])
        return vis

    def draw_state(self, vis):
        # Get template bounds
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)

        # Draw bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))

        if self.good:
            # Plot center
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        else:
            # Draw X on screen
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.line(vis, (x2, y1), (x1, y2), (0, 0, 255))

        #draw_str(vis, (x1, y2+16), 'PSR: %.2f' % self.psr)

    def preprocess(self, img):
        # Mean center and normalize image
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)

        # Apply Hanning window (weighting of values from center)
        return img * self.win

    def correlate(self, img):
        # Multiply the image with the filter (in frequency space)
        C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)

        # Convert back to RGB space
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        # Look for peak
        h, w = resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)

        # Get the isolated peak response (should be a Gaussian point)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)

        # Calculate the match probability
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval-smean) / (sstd+eps)

        return resp, (mx-w//2, my-h//2), psr

    def update_kernel(self):
        # Update the MOSSE filter kernel
        self.H = divSpec(self.H1, self.H2)
        self.H[...,1] *= -1


class MOSSEMatcher(object):
    def __init__(self, size):
        # Get frame size
        x1, y1 = 0, 0
        x2, y2 = size

        # Resize to fft window
        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2

        # Store position and size relative to frame
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size = w, h

        # Create Hanning window (weighting of values from center)
        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)

        # Create output image (centered Gaussian point--for correlation)
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        # Save the desired output image in frequency space
        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)

        # Create sum matrices
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)

    def save_filter(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def train(self, sample):
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        #sample = cv2.Sobel(sample, -1, 1, 1)
        self.last_img = sample

        # Apply sample
        self.apply_sample(sample)

        # Apply mirroring
        self.apply_sample(cv2.flip(sample, 1))

        # Apply random transformations
        #"""
        for i in xrange(64):
            # Preprocess, get DFT
            img = rnd_warp(sample)
            self.apply_sample(img)
            self.apply_sample(cv2.flip(img, 1))
        #"""

        # Update filter
        self.update_kernel()

        # Update visual state
        img = self.preprocess(sample)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)

        # TODO DEBUG
        print 'Pos:', (dx,dy), 'PSR:', self.psr

    def apply_sample(self, sample):
        # Preprocess, get DFT
        img = self.preprocess(sample)
        A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

        # Add to sum matrices
        self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)  # Sum of G x F*
        self.H2 += cv2.mulSpectrums(     A, A, 0, conjB=True)  # Sum of F x F*

    def match(self, frame, pos=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.Sobel(gray, -1, 1, 1)

        # Crop template image from position
        (x, y), (w, h) = self.pos if pos is None else pos, self.size
        self.last_img = img = cv2.getRectSubPix(gray, (w, h), (x, y))

        # Correlate, find position of object
        img = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)

        # Return match
        self.good = self.psr > 8.0
        return (dx, dy), self.psr

    @property
    def state_vis(self):
        """
        Get state of the filter in RGB space
        :return: combined image showing last tracked image, filter kernel, and output response
        """

        # Compute inverse DFT
        f = cv2.idft(self.H, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        # Reshape to template dimensions
        h, w = f.shape
        f = np.roll(f, -h//2, 0)
        f = np.roll(f, -w//2, 1)

        # Convert kernel
        kernel = np.uint8((f-f.min()) / f.ptp()*255)

        # Convert last correlation output response
        resp = self.last_resp
        resp = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)

        # Position the images side by side
        vis = np.hstack([self.last_img, kernel, resp])
        return vis

    def draw_state(self, vis):
        # Get template bounds
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)

        # Draw bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))

        if self.good:
            # Plot center
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        else:
            # Draw X on screen
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.line(vis, (x2, y1), (x1, y2), (0, 0, 255))

        #draw_str(vis, (x1, y2+16), 'PSR: %.2f' % self.psr)

    def preprocess(self, img):
        # Mean center and normalize image
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)

        # Apply Hanning window (weighting of values from center)
        return img * self.win

    def correlate(self, img):
        # Multiply the image with the filter (in frequency space)
        C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)

        # Convert back to RGB space
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        # Look for peak
        h, w = resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)

        # Get the isolated peak response (should be a Gaussian point)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)

        # Calculate the match probability
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval-smean) / (sstd+eps)

        return resp, (mx-w//2, my-h//2), psr

    def update_kernel(self):
        # Update the MOSSE filter kernel
        self.H = divSpec(self.H1, self.H2)
        self.H[...,1] *= -1


class ASEFMatcher(object):
    def __init__(self, size):
        # Get frame size
        x1, y1 = 0, 0
        x2, y2 = size

        # Resize to fft window
        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2

        # Store position and size relative to frame
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size = w, h

        # Create Hanning window (weighting of values from center)
        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)

        # Create output image (centered Gaussian point--for correlation)
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        # Save the desired output image in frequency space
        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)

        # Create sum matrices
        self.F = np.zeros_like(self.G)
        self.HSum = np.zeros_like(self.G)

        self.count = 0

    def save_filter(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def train(self, sample):
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        #sample = cv2.Sobel(sample, -1, 1, 1)
        self.last_img = sample

        # Apply sample
        self.apply_sample(sample)

        # Apply mirroring
        self.apply_sample(cv2.flip(sample, 1))

        # Apply random transformations
        """
        for i in xrange(64):
            # Preprocess, get DFT
            self.apply_sample(rnd_warp(sample))
        #"""

        # Update visual state
        img = self.preprocess(sample)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)

    def apply_sample(self, sample):
        # Preprocess, get DFT
        img = self.preprocess(sample)
        self.F = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

        self.count += 1

        # Update filter
        self.update_kernel()

    def match(self, frame, pos=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.Sobel(gray, -1, 1, 1)

        # Crop template image from position
        (x, y), (w, h) = self.pos if pos is None else pos, self.size
        self.last_img = img = cv2.getRectSubPix(gray, (w, h), (x, y))

        # Correlate, find position of object
        img = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)

        # Return match
        self.good = self.psr > 8.0
        return (dx, dy), self.psr

    @property
    def state_vis(self):
        """
        Get state of the filter in RGB space
        :return: combined image showing last tracked image, filter kernel, and output response
        """

        # Compute inverse DFT
        f = cv2.idft(self.H, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        # Reshape to template dimensions
        h, w = f.shape
        f = np.roll(f, -h//2, 0)
        f = np.roll(f, -w//2, 1)

        # Convert kernel
        kernel = np.uint8((f-f.min()) / f.ptp()*255)

        # Convert last correlation output response
        resp = self.last_resp
        resp = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)

        # Position the images side by side
        vis = np.hstack([self.last_img, kernel, resp])
        return vis

    def draw_state(self, vis):
        # Get template bounds
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)

        # Draw bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))

        if self.good:
            # Plot center
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        else:
            # Draw X on screen
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.line(vis, (x2, y1), (x1, y2), (0, 0, 255))

        #draw_str(vis, (x1, y2+16), 'PSR: %.2f' % self.psr)

    def preprocess(self, img):
        # Mean center and normalize image
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)

        # Apply Hanning window (weighting of values from center)
        return img * self.win

    def correlate(self, img):
        # Multiply the image with the filter (in frequency space)
        C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)

        # Convert back to RGB space
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        # Look for peak
        h, w = resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)

        # Get the isolated peak response (should be a Gaussian point)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)

        # Calculate the match probability
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval-smean) / (sstd+eps)

        return resp, (mx-w//2, my-h//2), psr

    def update_kernel(self):
        # Update the MOSSE filter kernel
        self.HSum += divSpec(self.G, self.F)
        self.H = self.HSum / self.count
        self.H[...,1] *= -1
