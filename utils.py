import cv2 as cv


def resize_image(image, scale_percent, interpolation=cv.INTER_AREA):
	width = int(image.shape[1] * scale_percent / 100)
	height = int(image.shape[0] * scale_percent / 100)
	dim = (width, height)
	image = cv.resize(image, dim, interpolation=interpolation)
	return image