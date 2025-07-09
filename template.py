
# when importing things that are not isntalled by default, you need to update requirements.txt
import cv2
import numpy as np
import numpy.linalg as la
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm
import scipy.optimize
import math


def computeOutline(picture, circleRadiusApproximate):
    """
    DO NOT MODIFY FUNCTION CALL
    This function finds best fit circle for the given picture
    Circle radius is given as input and is an indication of the size of the circle to be fitted.
    You should find and return correct diamter which is +/-3% of the given radius.
    The function should find the correct center of the circle and return it.
    """


    # Load grayscale image
    img = cv2.imread(str(picture), cv2.IMREAD_GRAYSCALE).astype(np.float32)

    ksize = 9 # Increase this to reduce noise before binarization.
    optimization_iters = 20 # Basin-hopping iteration count.
    optimization_step_scale = 0.04 # Basin-hopping step scale relative to image size.
    optimization_seed = 0xDEADBEEF # Basin-hopping RNG seed.

    # Compute variance around each pixel.
    local_mean = cv2.blur(img, ksize=(ksize, ksize))
    local_mean_of_squares = cv2.blur(img**2, ksize=(ksize, ksize))
    local_variance = local_mean_of_squares - local_mean**2

    # Find the high-variance region with Otsu's binarization
    _threshold_val, variance_thresh = cv2.threshold(local_variance.astype(np.uint16), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    var_ys, var_xs = np.nonzero(variance_thresh)
    points = np.transpose([ var_xs, var_ys ])

    centroid = np.mean(points, axis=0)

    center_distances = la.norm(points - centroid, axis=1)

    # In a disk points are on average R / 1.5 from its center.
    # Since our points are uniformly sampled in every direction, we can use this to roughly estimate the radius.
    radius_guess = 1.5 * np.mean(center_distances)

    def try_circle(args):
        center = args[:2]
        radius = args[2]

        distances_squared = np.sum((points - center)**2, axis=1)

        # Estimate the density of inliers inside this candidate disk.
        is_in = distances_squared < radius**2
        num_in = np.count_nonzero(is_in)
        area = math.pi * radius**2

        in_density = num_in / area

        # Estimate the density of outliers in an annulus around the candidate disk.
        around_radius = radius * 1.1
        num_around = np.count_nonzero(np.logical_and(np.logical_not(is_in), distances_squared < around_radius**2))

        around_area = math.pi * around_radius**2 - area

        around_density = num_around / around_area

        # The loss is the outlier density minus the inlier density.
        return around_density - in_density

    # Optimize the initial rough guess.
    result = scipy.optimize.basinhopping(
      func=try_circle,
      x0=np.array([ centroid[0], centroid[1], radius_guess ]),
      niter=optimization_iters,
      stepsize=max(img.shape[0], img.shape[1]) * optimization_step_scale,
      minimizer_kwargs={"method": "COBYLA"},
      seed=optimization_seed,
    )

    cx, cy, radius = result.x

    radius -= ksize / 2

    return radius, ( cx, cy )