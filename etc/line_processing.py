import cv2
import numpy as np
import time

import calibration
import lane_detection

class Line:
	def __init__(self):
		# was the line detected in the last iteration?
		self.detected = False
		# Set the width of the windows +/- margin
		self.window_margin = 28
		# x values of the fitted line over the last n iterations
		self.prevx = []
		# polynomial coefficients for the most recent fit
		self.current_fit = [np.array([False])]
		#radius of curvature of the line in some units
		self.radius_of_curvature = None
		# starting x_value
		self.startx = None
		# ending x_value
		self.endx = None
		# x values for detected line pixels
		self.allx = None
		# y values for detected line pixels
		self.ally = None
		# road information
		self.road_inf = None
		self.curvature = None
		self.deviation = None


def smoothing(lines, pre_lines=3):
	# collect lines & print average line
	lines = np.squeeze(lines)
	avg_line = np.zeros((360))

	for i, line in enumerate(reversed(lines)):
		if i == pre_lines:
			break
		avg_line += line
	avg_line = avg_line / pre_lines

	return avg_line


def rad_of_curvature(left_line, right_line):
	## measure radius of curvature
	plot_y = left_line.ally
	left_x, right_x = left_line.allx, right_line.allx

	# reverse to match top to bottom in y
	left_x = left_x[::-1]
	right_x = right_x[::-1]

	# define conversions in x, y from pixels space to meters
	width_lanes = abs(right_line.startx - left_line.startx)
	y_meter_per_pixel = 30 / 360 # meter per pixel in y dimension
	x_meter_per_pixel = 3.7 * (360/640) / width_lanes # meter per pixel in x dimension

	# define y value where we want radius of cuvature
	# the maximum y value, corresponding to the bottom of the image
	y_eval = np.max(plot_y)

	# fit new polynomials to x, y in world space
	left_fit_cr = np.polyfit(plot_y * y_meter_per_pixel, left_x * x_meter_per_pixel, 2)
	right_fit_cr = np.polyfit(plot_y * y_meter_per_pixel, right_x * x_meter_per_pixel, 2)

	# calculate the new radius of curvature
	left_curvature = ((1 + (2*left_fit_cr[0]*y_eval*y_meter_per_pixel + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curvature = ((1 + (2*right_fit_cr[0]*y_eval*y_meter_per_pixel + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

	# radius of cuvature result
	left_line.radius_of_curvature = left_curvature
	right_line.radius_of_curvature = right_curvature
	#print('left_line_curvature: ', left_curvature, '\n', 'right_line_curvature: ', right_curvature)


def slidingWindowSearch(binary_img, left_line, right_line):
	histogram = np.sum(binary_img[binary_img.shape[0]//2:,:], axis=0)/255

	# create an output image to draw on and visualize the result
	# create 720 x 1280 x 3 image
	output_img = np.dstack((binary_img, binary_img, binary_img))
	#cv2.imshow('output_img', output_img)

	# find starting points of line
	midpoint = np.int(histogram.shape[0]//2)
	start_left_x = np.argmax(histogram[:midpoint])
	start_right_x = np.argmax(histogram[midpoint:]) + midpoint

	# windows
	num_windows = 9
	window_height = np.int(binary_img.shape[0] / num_windows)

	# identify the x and y positions of all nonzero pixels
	nonzero = binary_img.nonzero()
	nonzero_y = np.array(nonzero[0])
	nonzero_x = np.array(nonzero[1])
	#print(nonzero, '\n', nonzero_y, '\n', nonzero_x)

	# current positions to be updated for each window
	current_left_x = start_left_x
	current_right_x = start_right_x
	#print(current_left_x, current_right_x)

	# set minimum number of pixels found to recenter window
	min_num_pixel = 50

	# create empty lists to receive left and right line pixels
	win_left_line = []
	win_right_line = []

	window_margin = left_line.window_margin

	# step through the windows one by one
	for window in range(num_windows):
		# identify window boundaries in x and y (and left and right)
		win_y_low = binary_img.shape[0] - (window+1) * window_height
		win_y_high = binary_img.shape[0] - window * window_height
		win_leftx_min = current_left_x - window_margin
		win_leftx_max = current_left_x + window_margin
		win_rightx_min = current_right_x - window_margin
		win_rightx_max = current_right_x + window_margin

		# draw the windows on the visualization image
		cv2.rectangle(output_img, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0,255,0), 2)
		cv2.rectangle(output_img, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0,255,0), 2)

		# identify the nonzero pixels in x and y within the window
		left_window_idxs = ((win_y_low<=nonzero_y) & (nonzero_y<=win_y_high) & (win_leftx_min<=nonzero_x) & (nonzero_x<=win_leftx_max)).nonzero()[0]
		right_window_idxs = ((win_y_low<=nonzero_y) & (nonzero_y<=win_y_high) & (win_rightx_min<=nonzero_x) & (nonzero_x<=win_rightx_max)).nonzero()[0]
		# append these indices to the lists
		win_left_line.append(left_window_idxs)
		win_right_line.append(right_window_idxs)

		# if found pixels more than min_num_pixel, recenter next window on their mean position
		if len(left_window_idxs) > min_num_pixel:
			current_left_x = np.int(np.mean(nonzero_x[left_window_idxs]))
		if len(right_window_idxs) > min_num_pixel:
			current_right_x = np.int(np.mean(nonzero_x[right_window_idxs]))
	#cv2.imshow('output_img', output_img)

	# concatenate the arrays of indices
	win_left_line = np.concatenate(win_left_line)
	win_right_line = np.concatenate(win_right_line)
	#print('win_left_line: ', win_left_line, '\n', 'win_right_line: ', win_right_line, '\n\n')

	# extract left and right line pixel positions
	left_x, left_y = nonzero_x[win_left_line], nonzero_y[win_left_line]
	right_x, right_y = nonzero_x[win_right_line], nonzero_y[win_right_line]
	#print('left_x, left_y:', left_x, left_y)
	#print('right_x, right_y: ', right_x, right_y)

	output_img[left_y, left_x] = [255,0,0]
	output_img[right_y, right_x] = [0,0,255]
	#cv2.imshow('output_img', output_img)

	## fit a 2 degree polynomial to each line
	left_fit = np.polyfit(left_y, left_x, 2)
	right_fit = np.polyfit(right_y, right_x, 2)

	#left_line.current_fit = left_fit
	#right_line.current_fit = right_fit

	# generate x, y values for plotting
	plot_y = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])

	# ax^2 + bx + c
	left_plot_x = left_fit[0]*(plot_y**2) + left_fit[1]*plot_y + left_fit[2]
	right_plot_x = right_fit[0]*(plot_y**2) + right_fit[1]*plot_y + right_fit[2]

	left_line.prevx.append(left_plot_x)
	right_line.prevx.append(right_plot_x)

	if len(left_line.prevx) > 10:
		left_avg_line = smoothing(left_line.prevx, 10)
		left_avg_fit = np.polyfit(plot_y, left_avg_line, 2)
		left_fit_plotx = left_avg_fit[0]*(plot_y**2) + left_avg_fit[1]*plot_y + left_avg_fit[2]
		left_line.current_fit = left_avg_fit
		left_line.allx, left_line.ally = left_fit_plotx, plot_y
	else:
		left_line.current_fit = left_fit
		left_line.allx, left_line.ally = left_plot_x, plot_y

	if len(right_line.prevx) > 10:
		right_avg_line = smoothing(right_line.prevx, 10)
		right_avg_fit = np.polyfit(plot_y, right_avg_line, 2)
		right_fit_plotx = right_avg_fit[0]*(plot_y**2) + right_avg_fit[1]*plot_y + right_avg_fit[2]
		right_line.current_fit = right_avg_fit
		right_line.allx, right_line.ally = right_fit_plotx, plot_y
	else:
		right_line.current_fit = right_fit
		right_line.allx, right_line.ally = right_plot_x, plot_y

	left_line.startx, right_line.startx = left_line.allx[len(left_line.allx)-1], right_line.allx[len(right_line.allx)-1]
	left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

	left_line.detected, right_line.detected = True, True

	# print radius of curvature
	rad_of_curvature(left_line, right_line)
	return output_img


def prevWindowRefer(binary_img, left_line, right_line):
	"""
    refer to previous window info - after detecting lane lines in previous frame
    """
    # Create an output image to draw on and  visualize the result
	output_img = np.dstack((binary_img, binary_img, binary_img))

    # Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

    # Set margin of windows
	window_margin = left_line.window_margin

	left_line_fit = left_line.current_fit
	right_line_fit = right_line.current_fit
	leftx_min = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] - window_margin
	leftx_max = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] + window_margin
	rightx_min = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] - window_margin
	rightx_max = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] + window_margin

	# Identify the nonzero pixels in x and y within the window
	left_inds = ((nonzerox >= leftx_min) & (nonzerox <= leftx_max)).nonzero()[0]
	right_inds = ((nonzerox >= rightx_min) & (nonzerox <= rightx_max)).nonzero()[0]

    # Extract left and right line pixel positions
	leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
	rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

	output_img[lefty, leftx] = [255, 0, 0]
	output_img[righty, rightx] = [0, 0, 255]
    
    # Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
	ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])

    # ax^2 + bx + c
	left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
	right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

	leftx_avg = np.average(left_plotx)
	rightx_avg = np.average(right_plotx)

	left_line.prevx.append(left_plotx)
	right_line.prevx.append(right_plotx)

	if len(left_line.prevx) > 10:
		left_avg_line = smoothing(left_line.prevx, 10)
		left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
		left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
		left_line.current_fit = left_avg_fit
		left_line.allx, left_line.ally = left_fit_plotx, ploty
	else:
		left_line.current_fit = left_fit
		left_line.allx, left_line.ally = left_plotx, ploty

	if len(right_line.prevx) > 10:
		right_avg_line = smoothing(right_line.prevx, 10)
		right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
		right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
		right_line.current_fit = right_avg_fit
		right_line.allx, right_line.ally = right_fit_plotx, ploty
	else:
		right_line.current_fit = right_fit
		right_line.allx, right_line.ally = right_plotx, ploty

    # goto blind_search if the standard value of lane lines is high.
	standard = np.std(right_line.allx - left_line.allx)

	if (standard > 80):
		left_line.detected = False

	left_line.startx, right_line.startx = left_line.allx[len(left_line.allx) - 1], right_line.allx[len(right_line.allx) - 1]
	left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

	# print radius of curvature
	rad_of_curvature(left_line, right_line)
	return output_img


def findLines(binary_img, left_line, right_line):
	# if don't have lane lines info
	if left_line.detected == False:
		return slidingWindowSearch(binary_img, left_line, right_line)
	# if have lane lines info
	else:
		return prevWindowRefer(binary_img, left_line, right_line)


if __name__ == '__main__':
	mtx, dist = calibration.calib()

	left_line = Line()
	right_line = Line()

	input_video = 'data/input2.mp4'
	cap = cv2.VideoCapture(input_video)

	prev_time = 0

	while True:
		if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT)-3:
			cap.open(input_video)

		ret, frame = cap.read()
		frame = cv2.resize(frame, dsize=(640,360), interpolation=cv2.INTER_AREA)

		frame = calibration.undistort(frame, mtx, dist)
		thresholded_img = lane_detection.thresholdImage(frame)
		birds_eye_view = lane_detection.getTopviewImage(thresholded_img)
		res = findLines(birds_eye_view, left_line, right_line)

		frame = lane_detection.drawTopviewBoundary(frame)
		curr_time = time.time()
		sec = curr_time - prev_time
		prev_time = curr_time
		fps = 1 / sec
		str_fps = "FPS: %0.1f" %fps
		cv2.putText(frame, str_fps, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

		cv2.imshow('frame', frame)
		#cv2.imshow('thresholded_img', thresholded_img)
		cv2.imshow('birds_eye_view', birds_eye_view)
		cv2.imshow('res', res)

		if cv2.waitKey(30) > 0:
			break

	cap.release()
	cv2.destroyAllWindows()
