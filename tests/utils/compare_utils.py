import numpy as np
from matplotlib import image

def l2_error(img1: np.ndarray, img2: np.ndarray, epsilon = 1e-3):
    '''
    Compute and return error and variance between image img1 and image img2
    
    ## Paramters

    img1 : ndarray
        First image to compare represented as rgba numpy array
    img2 : ndarray
        Second image to compare represented as rgba numpy array
    epsilon : float
        Minimal difference underwhich difference between img1 and img2 is not considered as an error

    ## Return

    err : float
        The error between img1 and img2
    var : float
        The variance of the error between img1 and img2
    '''
    assert(img1.shape == img2.shape)

    # Compute diff
    diff = np.abs(img1 - img2)

    # Compute error and variance
    pixels = img1.shape[0] * img2.shape[1]
    error = diff.sum() / pixels
    variance = (diff * diff).sum() / pixels

    return error, variance, diff