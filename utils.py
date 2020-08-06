import cv2 as cv
import numpy as np
from numba import jit


def resize_image(image, scale_percent, interpolation=cv.INTER_AREA) -> np.ndarray:
	width = int(image.shape[1] * scale_percent / 100)
	height = int(image.shape[0] * scale_percent / 100)
	dim = (width, height)
	image = cv.resize(image, dim, interpolation=interpolation)
	return image