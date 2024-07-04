from skimage.feature import local_binary_pattern 
from skimage.feature import hog
import numpy as np


def extract_hog_features(image):
    hog_features = hog(image,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       visualize=False,
                       feature_vector=True,
                       block_norm="L2-Hys")
    return hog_features

def extract_lbp_features(image, radius=2, n_points=8, eps = 1e-7) -> np.ndarray:
    lbp_features = local_binary_pattern(image, n_points*radius, radius, method='uniform')
    hist, _ = np.histogram(lbp_features.ravel(),
                           bins=np.arange(0, n_points + 3),
                           range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist
   

  