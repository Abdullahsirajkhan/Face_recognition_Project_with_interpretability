import cv2
import numpy as np
from skimage.feature import hog
import os
def extract_hog_features(image_path, visualize=False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    """ This function extracts HOG features from the grayscale.
        It takes the path to the image and an optional visualize flag.
        If visualize is True, it returns the HOG features and the HOG image.
        """

    if image is None: # This checks if the image is read correctly or not.
        raise ValueError(f"Image at {image_path} could not be read.") # this rasies an error if the image was not readed correctly.

    image_resized = cv2.resize(image, (256, 256))
    features, hog_image = hog(image_resized,
                              orientations=9,
                              pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2),
                              block_norm='L2-Hys',
                              visualize=True,
                              feature_vector=True)

    return (features, hog_image) if visualize else (features,None)
