
# when importing things that are not isntalled by default, you need to update requirements.txt
import cv2
import numpy as np
import skimage.transform
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm


def computeOutline(picture, circleRadiusApproximate):
    """
    DO NOT MODIFY FUNCTION CALL
    This function finds best fit circle for the given picture
    Circle radius is given as input and is an indication of the size of the circle to be fitted.
    You should find and return correct diamter which is +/-3% of the given radius.
    The function should find the correct center of the circle and return it.
    """


    # Load grayscale image
    img = cv2.imread(picture, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    ksize = 9 # Increase this to reduce noise before binarization.
    point_sampling_factor = 50 # Take every nth point to speed up circle calculation.

    # Compute variance around each pixel.
    local_mean = cv2.blur(img, ksize=(ksize, ksize))
    local_mean_of_squares = cv2.blur(img**2, ksize=(ksize, ksize))
    local_variance = local_mean_of_squares - local_mean**2

    # Find the high-variance region with Otsu's binarization
    _threshold_val, variance_thresh = cv2.threshold(local_variance.astype(np.uint16), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find white pixels surrounded mostly by black (like the edge of the disk).
    var_blur = cv2.blur(variance_thresh, ksize=(18, 18))
    edgy = (np.bitwise_and(variance_thresh != 0, var_blur < 128)).astype(np.uint8) * 255

    test_radii = np.arange(img.shape[0] // 4, img.shape[0] // 2)

    def try_radius(radius):
        edgy_circles = skimage.transform.hough_circle(
            edgy,
            radius=[ radius ],
            normalize=False,
        )

        return skimage.transform.hough_circle_peaks(
            hspaces=edgy_circles,
            radii=[ radius ],
            normalize=False,
            num_peaks=1,
        )

    most_votes = 0
    best_center_x = 0
    best_center_y = 0
    best_radius = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for votes, center_x, center_y, radius in tqdm(executor.map(try_radius, test_radii), desc="Fitting circles", total=len(test_radii)):
            if votes[0] > most_votes:
                most_votes = votes[0]
                best_center_x = center_x[0]
                best_center_y = center_y[0]
                best_radius = radius[0]

    return best_radius, (best_center_x, best_center_y)
