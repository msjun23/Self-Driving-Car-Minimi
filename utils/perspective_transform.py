import cv2
import numpy as np

def compute_perspective_transform_matrices(src, dst):
	M = cv2.getPerspectiveTransform(src, dst)
	M_inv = cv2.getPerspectiveTransform(dst, src)

	return (M, M_inv)

def perspectiveTransform(img, src, dst):
	M = cv2.getPerspectiveTransform(src, dst)
	img_size = (img.shape[1], img.shape[0])
	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

	return warped