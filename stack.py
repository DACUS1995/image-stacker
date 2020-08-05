import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def draw_matches(image_one, image_two):
	akaze = cv.AKAZE_create()
	kp1, des1 = akaze.detectAndCompute(base_image, None)
	kp2, des2 = akaze.detectAndCompute(second_image, None)

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
	return image_combined


def simple_stack_images_ECC(base_image_path, image_list):
	M = np.eye(3, 3, dtype=np.float32)

	base_image = cv.imread(base_image_path, 1).astype(np.float32) / 255

	print('Original Dimensions : ', base_image.shape)
	scale_percent = 200 # percent of original size
	width = int(base_image.shape[1] * scale_percent / 100)
	height = int(base_image.shape[0] * scale_percent / 100)
	dim = (width, height)
	resized_base_image = cv.resize(base_image, dim, interpolation = cv.INTER_AREA)
	print('Resized Dimensions : ', resized_base_image.shape)

	resized_stacked_image = resized_base_image
	base_image = cv.cvtColor(base_image, cv.COLOR_BGR2GRAY)

	for image_path in tqdm(image_list):
		image = cv.imread(image_path, 1).astype(np.float32) / 255
		image = cv.resize(image, dim, interpolation = cv.INTER_AREA)

		s, M = cv.findTransformECC(
			cv.cvtColor(image, cv.COLOR_BGR2GRAY), 
			base_image, 
			M, 
			cv.MOTION_HOMOGRAPHY,
			(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 5000, 1e-10), inputMask=None, gaussFiltSize=1
		)

		w, h, _ = image.shape
		image = cv.warpPerspective(image, M, (h, w))

		resized_image = cv.resize(image, dim, interpolation = cv.INTER_AREA)
		resized_stacked_image += resized_image

	resized_stacked_image /= len(image_list)
	resized_stacked_image = (resized_stacked_image * 255)
	return resized_stacked_image


def main():
	action = "simple_stack_images_ecc"
	image_folder = "./images/noisy_images"

	file_list = os.listdir(image_folder)
	file_list = [os.path.join(image_folder, x) for x in file_list if x.endswith(('.jpg', '.png','.bmp'))]

	if action == "draw_matches":
		first_image = cv.imread(file_list[0], cv.IMREAD_GRAYSCALE)  
		second_image = cv.imread(file_list[1], cv.IMREAD_GRAYSCALE)
		result_image = draw_matches(first_image, second_image)
		cv.imwrite("matches.jpg", result_image)
	
	elif action == "simple_stack_images_ecc":
		stacked_image = simple_stack_images_ECC(file_list[0], file_list[1:])
		cv.imwrite("stacked_image.jpg", stacked_image)



if __name__ == "__main__":
	main()