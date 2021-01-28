import cv2
import numpy as np

## convert rgb to hsl color space
def toHSL(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


def isolateYellowHSL(img):
	hsl_img = toHSL(img)

	low_threshold = np.array([90,38,115], dtype=np.uint8)
	high_threshold = np.array([110,204,255], dtype=np.uint8)

	yellow_mask = cv2.inRange(hsl_img, low_threshold, high_threshold)

	return yellow_mask


def isolateWhiteHSL(img):
	hsl_img = toHSL(img)

	low_threshold = np.array([0,200,0], dtype=np.uint8)
	high_threshold = np.array([180,255,255], dtype=np.uint8)

	white_mask = cv2.inRange(hsl_img, low_threshold, high_threshold)

	return white_mask


## Combine yellow mask imgae & white mask image
def colorThresholdImage(yellow_mask, white_mask):
	color_thresholded_img = cv2.bitwise_or(yellow_mask, white_mask)

	return color_thresholded_img


## convert rgb to lab color space
def toLAB(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


## sobel in x or y directions
## x_dir=True -> sobel mask for x / x_dir=False -> sobel mask for y
def absSobel(img, x_dir=True, kernel_size=3, thresh=(0,255)):
	lab_l_img = toLAB(img)[:,:,0]

	if x_dir:
		sobel = cv2.Sobel(lab_l_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
	else:
		sobel = cv2.Sobel(lab_l_img, cv2.CV_64F, 0, 1, ksize=kernel_size)

	sobel_abs = np.absolute(sobel)
	sobel_scaled = np.uint8(255 * sobel / np.max(sobel_abs))

	sobel_binary = np.zeros_like(sobel_scaled)
	sobel_binary[(thresh[0] <= sobel_scaled) & (sobel_scaled <= thresh[1])] = 255
	
	return sobel_binary


## sobel magnitude in x and y directions
def magSobel(img, kernel_size=3, thresh=(0,255)):
	lab_l_img = toLAB(img)[:,:,0]

	sobel_x = cv2.Sobel(lab_l_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
	sobel_y = cv2.Sobel(lab_l_img, cv2.CV_64F, 0, 1, ksize=kernel_size)

	sobel_xy = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
	scaled_sobel_xy = np.uint8(255 * sobel_xy / np.max(sobel_xy))

	sobel_mag_binary = np.zeros_like(scaled_sobel_xy)
	sobel_mag_binary[(thresh[0] <= scaled_sobel_xy) & (scaled_sobel_xy <= thresh[1])] = 255

	return sobel_mag_binary


## sobel with gradient direction
def dirSobel(img, kernel_size=3, thresh=(0,np.pi/2)):
	lab_l_img = toLAB(img)[:,:,0]

	sobel_x_abs = np.absolute(cv2.Sobel(lab_l_img, cv2.CV_64F, 1, 0, ksize=kernel_size))
	sobel_y_abs = np.absolute(cv2.Sobel(lab_l_img, cv2.CV_64F, 1, 0, ksize=kernel_size))

	sobel_xy_dir = np.arctan2(sobel_x_abs, sobel_y_abs)

	sobel_dir_binary = np.zeros_like(sobel_xy_dir)
	sobel_dir_binary[(thresh[0] <= sobel_xy_dir) & (sobel_xy_dir <= thresh[1])] = 255

	return sobel_dir_binary


## combine sobel images
def combinedSobels(sobel_x, sobel_y, sobel_xy_mag, img, kernel_size=3, angle_thresh=(0,np.pi/2)):
	sobel_xy_dir = dirSobel(img, kernel_size=kernel_size, thresh=angle_thresh)

	combined = np.zeros_like(sobel_xy_dir)
	# sobel x returned the best output so we keep all of its results. perform a binary and on all the other sobels
	combined[(sobel_x==255) | ((sobel_y==255) & (sobel_xy_mag==255) & (sobel_xy_dir==255))] = 255

	return combined