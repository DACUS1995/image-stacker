import cv2 as cv
import numpy as np


def resize_image(image, scale_percent, interpolation=cv.INTER_AREA) -> np.ndarray:
	width = int(image.shape[1] * scale_percent / 100)
	height = int(image.shape[0] * scale_percent / 100)
	dim = (width, height)
	image = cv.resize(image, dim, interpolation=interpolation)
	return image


def draw_matches(base_image, second_image, method="akaze", display=True):
	kp1, des1 = None, None
	kp2, des2 = None, None

	if method == "akaze":
		detector = cv.AKAZE_create()
	elif method == "orb":
		detector = cv.ORB_create()
	else:
		raise Exception("Unhandled method.")

	kp1, des1 = detector.detectAndCompute(base_image, None)
	kp2, des2 = detector.detectAndCompute(second_image, None)

	bf = cv.BFMatcher()
	matches = bf.knnMatch(des1, des2, k=2)

	good_matches = []
	for m, n in matches:
		if m.distance < 0.75*n.distance:
			good_matches.append([m])
	
	image_combined = cv.drawMatchesKnn(
		base_image, 
		kp1, 
		second_image, 
		kp2, 
		good_matches, 
		None, 
		flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
	)

	if display:
		cv.imshow("Display window", image_combined)
		cv.waitKey(0)


	return image_combined