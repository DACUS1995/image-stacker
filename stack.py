import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
import argparse

from utils import resize_image, draw_matches

def simple_stack_images_ECC(base_image_path, image_list, scale_percent, draw=False):
	M = np.eye(3, 3, dtype=np.float32)

	base_image = cv.imread(base_image_path, 1).astype(np.float32) / 255

	print('Original Dimensions : ', base_image.shape)
	width = int(base_image.shape[1] * scale_percent / 100)
	height = int(base_image.shape[0] * scale_percent / 100)
	dim = (width, height)
	resized_base_image = cv.resize(base_image, dim, interpolation = cv.INTER_AREA)
	print('Resized Dimensions : ', resized_base_image.shape)

	resized_stacked_image = resized_base_image
	base_image = cv.cvtColor(base_image, cv.COLOR_BGR2GRAY)

	for image_path in tqdm(image_list):
		image = cv.imread(image_path, 1).astype(np.float32) / 255

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


def simple_stack_images_orb(base_image_path, image_list, scale_percent, draw=False):
	orb = cv.ORB_create()

	base_image = cv.imread(base_image_path, 1)

	width = int(base_image.shape[1] * scale_percent / 100)
	height = int(base_image.shape[0] * scale_percent / 100)
	dim = (width, height)

	print('Original Dimensions : ', base_image.shape)
	resized_base_image = resize_image(base_image, scale_percent)
	print('Resized Dimensions : ', resized_base_image.shape)

	resized_stacked_image = resized_base_image.astype(np.float32)
	base_image_keypoints, base_image_des = orb.detectAndCompute(base_image, None)

	base_image_edges = cv.Canny(base_image, 50, 150)
	base_image_edges_keypoints, base_image_edges_des = orb.detectAndCompute(base_image_edges, None)


	for image_path in tqdm(image_list):
		image = cv.imread(image_path, 1)
		image_edges = cv.Canny(image, 50, 150)
		keypoints, des = orb.detectAndCompute(image_edges, None)

		matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
		matches = matcher.match(base_image_edges_des, des)
		matches = sorted(matches, key=lambda x: x.distance)
		good_matches = matches[:int(len(matches) * 0.10)]

		ref_matched_kpts = np.float32([base_image_edges_keypoints[m.queryIdx].pt for m in good_matches])
		sensed_matched_kpts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

		H, status = cv.findHomography(sensed_matched_kpts, ref_matched_kpts, cv.RANSAC, 9.0)
		image = cv.warpPerspective(image, H, (image.shape[1], image.shape[0]))

		resized_image = cv.resize(image, dim, interpolation = cv.INTER_AREA)
		resized_stacked_image += resized_image

		if draw:
			draw_matches(base_image, image, method="orb")

	resized_stacked_image /= len(image_list)
	return resized_stacked_image


def simple_stack_images_akaze(base_image_path, image_list, scale_percent, draw=False):
	orb = cv.AKAZE_create()

	base_image = cv.imread(base_image_path, 1)

	width = int(base_image.shape[1] * scale_percent / 100)
	height = int(base_image.shape[0] * scale_percent / 100)
	dim = (width, height)

	print('Original Dimensions : ', base_image.shape)
	resized_base_image = resize_image(base_image, scale_percent)
	print('Resized Dimensions : ', resized_base_image.shape)

	resized_stacked_image = resized_base_image.astype(np.float32)
	base_image_keypoints, base_image_des = orb.detectAndCompute(base_image, None)

	base_image_edges = cv.Canny(base_image, 50, 150)
	base_image_edges_keypoints, base_image_edges_des = orb.detectAndCompute(base_image_edges, None)


	for image_path in tqdm(image_list):
		image = cv.imread(image_path, 1)
		image_edges = cv.Canny(image, 50, 150)
		keypoints, des = orb.detectAndCompute(image_edges, None)

		matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
		matches = matcher.match(base_image_edges_des, des)
		matches = sorted(matches, key=lambda x: x.distance)
		good_matches = matches[:int(len(matches) * 0.10)]

		ref_matched_kpts = np.float32([base_image_edges_keypoints[m.queryIdx].pt for m in good_matches])
		sensed_matched_kpts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

		H, status = cv.findHomography(sensed_matched_kpts, ref_matched_kpts, cv.RANSAC, 9.0)
		image = cv.warpPerspective(image, H, (image.shape[1], image.shape[0]))

		resized_image = cv.resize(image, dim, interpolation = cv.INTER_AREA)
		resized_stacked_image += resized_image

		if draw:
			draw_matches(base_image, image, method="akaze")

	resized_stacked_image /= len(image_list)
	return resized_stacked_image



def main(args):
	method = args.method
	image_folder = args.directory

	print(args.draw_matches)

	file_list = os.listdir(image_folder)
	file_list = [os.path.join(image_folder, x) for x in file_list if x.endswith(('.jpg', '.png','.bmp'))]

	if method == "ecc":
		stacked_image = simple_stack_images_ECC(
			file_list[0], 
			file_list[1:],
			args.scale_percent,
			args.draw_matches
		)
		cv.imwrite("stacked_image_ecc.jpg", stacked_image)

	elif method == "orb":
		stacked_image = simple_stack_images_orb(
			file_list[0], 
			file_list[1:], 
			args.scale_percent,
			args.draw_matches
		)
		cv.imwrite("stacked_image_orb.jpg", stacked_image)
	elif method == "akaze":
		stacked_image = simple_stack_images_akaze(
			file_list[0], 
			file_list[1:], 
			args.scale_percent,
			args.draw_matches
		)
		cv.imwrite("stacked_image_akaze.jpg", stacked_image)
	else:
		raise Exception("Unhandled method.")



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--method", type=str, default="orb")
	parser.add_argument("--directory", type=str, default="./images/noisy_images")
	parser.add_argument("--scale-percent", type=int, default=200)
	parser.add_argument("--draw-matches", default=False, action="store_true")
	args = parser.parse_args()
	main(args)