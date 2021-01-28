import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

import calibration
from utils import threshold_old, perspective_transform

def thresholdImage(img):
	## threshold image
	# color threshold
	yellow_mask_img = threshold_old.isolateYellowHSL(img)
	white_mask_img = threshold_old.isolateWhiteHSL(img)
	
	color_thresholded_img = threshold_old.colorThresholdImage(yellow_mask_img, white_mask_img)
	#cv2.imshow('color_thresholded_img', color_thresholded_img)

	# gradient threshold
	sobel_x = threshold_old.absSobel(img, kernel_size=15, thresh=(20,120))
	sobel_y = threshold_old.absSobel(img, x_dir=False, kernel_size=15, thresh=(20,120))
	#cv2.imshow('sobel_x', sobel_x)
	#cv2.imshow('sobel_y', sobel_y)

	sobel_xy_mag = threshold_old.magSobel(img, kernel_size=15, thresh=(80,200))
	#cv2.imshow('sobel_xy_mag', sobel_xy_mag)

	sobel_thresholded_img = threshold_old.combinedSobels(sobel_x, sobel_y, sobel_xy_mag, img, kernel_size=15, angle_thresh=(np.pi/3, np.pi/2))
	#cv2.imshow('sobel_thresholded_img', sobel_thresholded_img)

	# combining color and gradient(sobel) thresholds
	thresholded_img = np.zeros_like(color_thresholded_img)
	thresholded_img[(color_thresholded_img==255) | (sobel_thresholded_img==255)] = 255
	#cv2.imshow('thresholed_img', thresholded_img)

	return thresholded_img


def getTopviewImage(img):
	img_shape = img.shape
	points = np.array([[105,img_shape[0]],[298,220],[345,220], [575, img_shape[0]]], np.int32)
	#lined_img = np.copy(img)
	#cv2.polylines(lined_img, [points], True, (0,0,255), 10)
	#cv2.imshow('lined_img', lined_img)

	src_points = points.astype(np.float32)
	dst_points = np.array([[100,img_shape[0]], [100,0], [540,0], [540,img_shape[0]]], np.float32)
	birds_eye_view = perspective_transform.perspectiveTransform(img, src_points, dst_points)
	#cv2.imshow('birds_eye_view', birds_eye_view)

	return birds_eye_view


def drawTopviewBoundary(img):
	img_shape = img.shape
	points = np.array([[105,img_shape[0]],[298,220],[345,220], [575, img_shape[0]]], np.int32)
	lined_img = np.copy(img)
	cv2.polylines(lined_img, [points], True, (0,0,255), 1)
	return lined_img


def binarization(img):
	binary_img = np.zeros_like(img)
	binary_img[(img>=100)] = 255
	return binary_img


def drawHistogram(img):
	histogram = np.sum(img[img.shape[0]//2:,:], axis=0)/255

	fig, ax = plt.subplots(1, 2, figsize=(15,4))
	ax[0].imshow(img, cmap='gray')
	ax[0].axis('off')
	ax[0].set_title('Binary Thresholded Perspective Transform Image')

	ax[1].plot(histogram)
	ax[1].set_title('Histogram Of Pixel Intensities (Image Bottom Half)')

	plt.show()


def findStartLine(img):
	histogram = np.sum(img[img.shape[0]//2:,:], axis=0)/255

	midpoint = np.int(histogram.shape[0]//2)
	start_left_x = np.argmax(histogram[:midpoint])
	start_right_x = np.argmax(histogram[midpoint:]) + midpoint
	#print(start_left_x, start_right_x)

	hist_value_left = np.int(np.max(histogram[:midpoint]))
	hist_value_right = np.int(np.max(histogram[midpoint:]))
	#print(hist_value_left, hist_value_right)

	line_shape = []
	if (hist_value_left >= 80):
		line_shape.append('line')
	else:
		line_shape.append('dot')
	if (hist_value_right >= 80):
		line_shape.append('line')
	else:
		line_shape.append('dot')

	return start_left_x, start_right_x, line_shape


if __name__ == '__main__':
	mtx, dist = calibration.calib()

	'''input_image = 'data/test5.jpg'
	img = cv2.imread(input_image)

	## remove distortion
	undist_img = calibration.undistort(img, mtx, dist)
	#cv2.imshow('img', img)
	#cv2.imshow('undist_img', undist_img)

	thresholded_img = thresholdImage(undist_img)
	birds_eye_view = getTopviewImage(thresholded_img)
	birds_eye_view = binarization(birds_eye_view)

	cv2.imshow('thresholded_img', thresholded_img)
	cv2.imshow('birds_eye_view', birds_eye_view)

	#drawHistogram(birds_eye_view)
	start_left_x, start_right_x, line_shape = findStartLine(birds_eye_view)
	print(start_left_x, start_right_x, line_shape)

	cv2.waitKey()
	cv2.destroyAllWindows()'''


	input_video = 'data/input2.mp4'
	cap = cv2.VideoCapture(input_video)

	prev_time = 0

	while True:
		if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
			cap.open(input_video)

		ret, frame = cap.read()
		frame = cv2.resize(frame, dsize=(640,360), interpolation=cv2.INTER_AREA)

		## remove distortion
		undist_frame = calibration.undistort(frame, mtx, dist)
		#cv2.imshow('img', img)
		#cv2.imshow('undist_img', undist_img)

		thresholded_img = thresholdImage(undist_frame)
		birds_eye_view = getTopviewImage(thresholded_img)
		birds_eye_view = binarization(birds_eye_view)

		start_left_x, start_right_x, line_shape = findStartLine(birds_eye_view)
		print(start_left_x, start_right_x, line_shape)

		lined_img = drawTopviewBoundary(frame)

		curr_time = time.time()
		sec = curr_time - prev_time
		prev_time = curr_time
		fps = 1 / sec
		str_fps = "FPS: %0.1f" %fps
		cv2.putText(lined_img, str_fps, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

		cv2.imshow('frame', lined_img)
		#cv2.imshow('thresholded_img', thresholded_img)
		cv2.imshow('birds_eye_view', birds_eye_view)

		if cv2.waitKey(30) > 0:
			break

	cap.release()
	cv2.destroyAllWindows()

