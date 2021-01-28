import cv2
import numpy as np

## convert rgb to hls color space
def toHLS(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


'''def isolateYellowHSL(img):
	hsl_img = toHSL(img)

	low_threshold = np.array([90,38,115], dtype=np.uint8)
	high_threshold = np.array([110,204,255], dtype=np.uint8)

	yellow_mask = cv2.inRange(hsl_img, low_threshold, high_threshold)

	return yellow_mask'''


def isolateWhiteHLS(img):
	hls_img = toHLS(img)

	low_threshold = np.array([0,240,0], dtype=np.uint8)
	high_threshold = np.array([180,255,255], dtype=np.uint8)

	white_mask = cv2.inRange(hls_img, low_threshold, high_threshold)

	return white_mask


## Combine yellow mask imgae & white mask image
def colorThresholdImage(yellow_mask, white_mask):
	color_thresholded_img = cv2.bitwise_or(yellow_mask, white_mask)

	return color_thresholded_img


## Canny edge detection
def cannyEdgeDetection(img):
	gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	edge_img = cv2.Canny(gray_img, 200, 250)

	return edge_img

