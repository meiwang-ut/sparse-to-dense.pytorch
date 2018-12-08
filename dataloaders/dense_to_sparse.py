import numpy as np
import cv2


def rgb2grayscale(rgb):
    return rgb[:, :, 0] * 0.2989 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114


def transmap(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return int(rightMin + (valueScaled * rightSpan))


def makeGaussian(size, fwhm = 10):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.linspace(-size, size, 2*size)
    y = x[:,np.newaxis]

    return np.exp(-4*np.log(2) * (x**2 + y**2) / fwhm**2)


class DenseToSparse:
    def __init__(self):
        pass

    def dense_to_sparse(self, rgb, depth):
        pass

    def __repr__(self):
        pass


class UniformSampling(DenseToSparse):
    name = "uar"

    def __init__(self, num_samples, max_depth=np.inf):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth

    def __repr__(self):
        return "%s{ns=%d,md=%f}" % (self.name, self.num_samples, self.max_depth)

    def dense_to_sparse(self, rgb, depth):
        """
        Samples pixels with `num_samples`/#pixels probability in `depth`.
        Only pixels with a maximum depth of `max_depth` are considered.
        If no `max_depth` is given, samples in all pixels
        """
        mask_keep = depth > 0
        if self.max_depth is not np.inf:
            mask_keep = np.bitwise_and(mask_keep, depth <= self.max_depth)
        n_keep = np.count_nonzero(mask_keep)
        if n_keep == 0:
            return mask_keep
        else:
            prob = float(self.num_samples) / n_keep
            return np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)


class SimulatedStereo(DenseToSparse):
    name = "sim_stereo"

    def __init__(self, num_samples, max_depth=np.inf, dilate_kernel=3, dilate_iterations=1):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.dilate_kernel = dilate_kernel
        self.dilate_iterations = dilate_iterations

    def __repr__(self):
        return "%s{ns=%d,md=%f,dil=%d.%d}" % \
               (self.name, self.num_samples, self.max_depth, self.dilate_kernel, self.dilate_iterations)

    # We do not use cv2.Canny, since that applies non max suppression
    # So we simply do
    # RGB to intensitities
    # Smooth with gaussian
    # Take simple sobel gradients
    # Threshold the edge gradient
    # Dilatate
    def dense_to_sparse(self, rgb, depth):
        gray = rgb2grayscale(rgb)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

        depth_mask = np.bitwise_and(depth != 0.0, depth <= self.max_depth)

        edge_fraction = float(self.num_samples) / np.size(depth)

        mag = cv2.magnitude(gx, gy)
        min_mag = np.percentile(mag[depth_mask], 100 * (1.0 - edge_fraction))
        mag_mask = mag >= min_mag

        if self.dilate_iterations >= 0:
            kernel = np.ones((self.dilate_kernel, self.dilate_kernel), dtype=np.uint8)
            cv2.dilate(mag_mask.astype(np.uint8), kernel, iterations=self.dilate_iterations)

        mask = np.bitwise_and(mag_mask, depth_mask)
        return mask


class SimulatedReflector(DenseToSparse):
    name = "sim_reflector"

    def __init__(self, num_samples, max_depth=np.inf, dilate_kernel=3, dilate_iterations=1):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.dilate_kernel = dilate_kernel
        self.dilate_iterations = dilate_iterations

    def __repr__(self):
        return "%s{ns=%d,md=%f,dil=%d.%d}" % \
               (self.name, self.num_samples, self.max_depth, self.dilate_kernel, self.dilate_iterations)

    def dense_to_sparse(self, rgb, depth):
        #         gray = np.float32(rgb2grayscale(rgb))
        gray = np.float32(rgb)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        depth_test = blurred.reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(depth_test, self.num_samples, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        mask = np.full(depth.shape, False)
        for i in range(self.num_samples):
            a0 = label.ravel() == i
            a0 = a0.reshape(depth.shape)
            idx = np.where(a0)
            x0 = np.mean(idx[0]).astype(np.int)
            y0 = np.mean(idx[1]).astype(np.int)
            mask[x0, y0] = True
        return mask


class SimulatedWireless(DenseToSparse):
    name = "sim_wireless"

    def __init__(self, num_samples, max_depth=np.inf, dilate_kernel=3, dilate_iterations=1):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.dilate_kernel = dilate_kernel
        self.dilate_iterations = dilate_iterations
        self.output_size = (228, 304)

    def __repr__(self):
        return "%s{ns=%d,md=%f,dil=%d.%d}" % \
               (self.name, self.num_samples, self.max_depth, self.dilate_kernel, self.dilate_iterations)

    def dense_to_spectrum(self, rgb, depth):
        gray = np.float32(rgb)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        depth_test = blurred.reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(depth_test, self.num_samples, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        spectrum = np.zeros(depth.shape)
        for i in range(self.num_samples):
            a0 = label.ravel() == i
            a0 = a0.reshape((depth.shape))
            idx = np.where(a0)
            x0 = np.mean(idx[0]).astype(np.int)
            s0 = np.sum(a0).astype(np.int)
            y0 = np.mean(idx[1]).astype(np.int)
            dd = depth[x0, y0]
            z0 = transmap(dd, np.min(depth), np.max(depth), 0, self.output_size[0]-1)
            rang = min(np.floor(2 * np.log(s0)).astype(np.int), 30)
            print(rang)
            spectrum[z0 - rang:z0 + rang, x0 - rang:x0 + rang] += makeGaussian(rang, rang)
        return spectrum

