import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

import calibration
from utils import threshold, perspective_transform

def thresholdImage(img):
	## threshold image
	# color threshold
	#yellow_mask_img = threshold.isolateYellowHSL(img)
	white_mask_img = threshold.isolateWhiteHLS(img)
	#color_thresholded_img = threshold.colorThresholdImage(yellow_mask_img, white_mask_img)
	#cv2.imshow('color_thresholded_img', color_thresholded_img)

	# gradient threshold
	# use canny edge detection
	edge_thresholded_img = threshold.cannyEdgeDetection(img)
	#cv2.imshow('edge_thresholded_img', edge_thresholded_img)

	## combine two images
	thresholded_img = cv2.bitwise_or(white_mask_img, edge_thresholded_img)

	return thresholded_img


def getTopviewImage(img):
	# img.shape : height, width
	img_shape = img.shape
	points = np.array([[100,img_shape[0]],[100,280],[540,280], [540, img_shape[0]]], np.int32)

	src_points = points.astype(np.float32)
	dst_points = np.array([[100,img_shape[0]], [100,0], [540,0], [540,img_shape[0]]], np.float32)
	birds_eye_view = perspective_transform.perspectiveTransform(img, src_points, dst_points)

	return birds_eye_view


def drawTopviewBoundary(img):
	img_shape = img.shape
	points = np.array([[100,img_shape[0]],[100,280],[540,280], [540, img_shape[0]]], np.int32)

	lined_img = np.copy(img)
	cv2.polylines(lined_img, [points], True, (0,0,255), 3)
	return lined_img


def binarization(img):
	_, bin_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
	return bin_img


def drawHistogram(img):
	histogram = np.sum(img[img.shape[0]//2:,:], axis=0)/255

	fig, ax = plt.subplots(1, 2, figsize=(15,4))
	ax[0].imshow(img, cmap='gray')
	ax[0].axis('off')
	ax[0].set_title('Binary Thresholded Perspective Transform Image')

	ax[1].plot(histogram)
	ax[1].set_title('Histogram Of Pixel Intensities (Image Bottom Half)')

	plt.show()


def findStartLine(img, start_left_x, start_right_x):
	histogram = np.sum(img[img.shape[0]//2:,:], axis=0)/255

	#midpoint = np.int(midpoint)
	midpoint = np.int(histogram.shape[0]//2)
	#start_left_x = np.argmax(histogram[:midpoint])
	#start_right_x = np.argmax(histogram[midpoint:]) + midpoint

	if np.argmax(histogram[midpoint:]) + midpoint - np.argmax(histogram[:midpoint]) < 50:
		pass
	else:
		start_left_x = np.argmax(histogram[:midpoint])
		start_right_x = np.argmax(histogram[midpoint:]) + midpoint

	#print(start_left_x, start_right_x)

	hist_value_left = np.int(np.max(histogram[:midpoint]))
	hist_value_right = np.int(np.max(histogram[midpoint:]))
	hist_value = [hist_value_left, hist_value_right]
	#print(hist_value_left, hist_value_right)

	return start_left_x, start_right_x, hist_value


if __name__ == '__main__':
	mtx, dist = calibration.calib()

	#input_video = 'data/input2.mp4'
	cap = cv2.VideoCapture(0)

	prev_time = 0
	start_left_x = 200
	start_right_x = 440

	while True:
		'''if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT)-3:
			cap.open(input_video)'''

		ret, frame = cap.read()
		frame = cv2.resize(frame, dsize=(640,360), interpolation=cv2.INTER_AREA)

		frame = calibration.undistort(frame, mtx, dist)
		thresholded_img = thresholdImage(frame)
		birds_eye_view = getTopviewImage(thresholded_img)
		bin_birds_eye_view = binarization(birds_eye_view)

		start_left_x, start_right_x, hist_value = findStartLine(bin_birds_eye_view, start_left_x, start_right_x)
		if hist_value[0] < 100:
			deviation = 50
		elif hist_value[1] < 100:
			deviation = -50
		else:
			lane_midpoint_x = (start_left_x + start_right_x)/2
			deviation = lane_midpoint_x - bin_birds_eye_view.shape[1]/2
		print(start_left_x, start_right_x, hist_value, deviation)

		frame = drawTopviewBoundary(frame)
		curr_time = time.time()
		sec = curr_time - prev_time
		prev_time = curr_time
		fps = 1 / sec
		str_fps = "FPS: %0.1f" %fps
		cv2.putText(frame, str_fps, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

		cv2.imshow('frame', frame)
		#cv2.imshow('thresholded_img', thresholded_img)
		cv2.imshow('bin_birds_eye_view', bin_birds_eye_view)
		cv2.imshow('topview', getTopviewImage(frame))

		if cv2.waitKey(30) > 0:
			break

	cap.release()
	cv2.destroyAllWindows()

