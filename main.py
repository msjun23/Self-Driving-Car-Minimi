import numpy as np
import tensorflow as tf
import cv2
import os
import time
import serial
from multiprocessing import Pool, Queue

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import calibration
import lane_detection

mtx, dist = calibration.calib()

def getBirdsEyeView(input_que, output_que):
	start_left_x = 200
	start_right_x = 440
	deviation = 0
	pre_deviation = 0

	while True:
		frame = input_que.get()
		if frame is not None:
			undist_frame = calibration.undistort(frame, mtx, dist)
			thresholded_img = lane_detection.thresholdImage(undist_frame)
			birds_eye_view = lane_detection.getTopviewImage(thresholded_img)
			bin_birds_eye_view = lane_detection.binarization(birds_eye_view)
			
			start_left_x, start_right_x, hist_value = lane_detection.findStartLine(bin_birds_eye_view, start_left_x, start_right_x)
			if hist_value[0] < 100 or hist_value[1] < 100:
				deviation = pre_deviation
			else:
				lane_midpoint_x = (start_left_x + start_right_x)/2
				deviation = lane_midpoint_x - bin_birds_eye_view.shape[1]/2
				pre_deviation = deviation
			#print(start_left_x, start_right_x, deviation)

			output_que.put([bin_birds_eye_view, deviation])

		else:
			output_que.put(frame)


if __name__ == '__main__':
	#fname = 'data/input2.mp4'
	#cap = cv2.VideoCapture(fname)
	cap = cv2.VideoCapture(0)

	input_que = Queue()
	output_que = Queue()

	# parallel process
	pool = Pool(2, getBirdsEyeView, (input_que, output_que))

	PATH_TO_CKPT = 'data/inference_graph/frozen_inference_graph.pb'
	PATH_TO_LABELS = os.path.join('data', 'label_map.pbtxt')
	NUM_CLASSES = 2

	# Load a (frozen) Tensorflow model into memory.
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.compat.v1.GraphDef()
		with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	# Loading label map
	# Label maps map indices to category names, 
	# so that when our convolution network predicts `5`, 
	# we know that this corresponds to `airplane`.  
	# Here we use internal utility functions, 
	# but anything that returns a dictionary mapping integers 
	# to appropriate string labels would be fine
	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(
		label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)

	prev_time = 0
	data_len = 10
	ser = serial.Serial('/dev/ttyACM0', 115200)

	# Save video-setting
	'''video_width = 360
	video_height = 640

	fourcc = cv2.VideoWriter_fourcc(*'XVID')

	video_writer = cv2.VideoWriter('output.avi', fourcc, 5.0, (video_width, video_height))'''

	# Detection
	with detection_graph.as_default():
		with tf.compat.v1.Session(graph=detection_graph) as sess:
			while True:
				ret, frame = cap.read()
				frame = cv2.resize(frame, dsize=(640,360), interpolation=cv2.INTER_AREA)
				info = [[],[]]

				input_que.put(frame)
				bin_birds_eye_view, deviation = output_que.get()
				info[0] = deviation
				info[1] = 0

				# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				image_np_expanded = np.expand_dims(frame, axis=0)
				# Extract image tensor
				image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
				# Extract detection boxes
				boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
				# Extract detection scores
				scores = detection_graph.get_tensor_by_name('detection_scores:0')
				# Extract detection classes
				classes = detection_graph.get_tensor_by_name('detection_classes:0')
				# Extract number of detectionsd
				num_detections = detection_graph.get_tensor_by_name('num_detections:0')
				# Actual detection.
				(boxes, scores, classes, num_detections) = sess.run(
					[boxes, scores, classes, num_detections],
					feed_dict={image_tensor: image_np_expanded})
				# Visualization of the results of a detection.
				vis_util.visualize_boxes_and_labels_on_image_array(
					frame,
					np.squeeze(boxes),
					np.squeeze(classes).astype(np.int32),
					np.squeeze(scores),
					category_index,
					use_normalized_coordinates=True,
					line_thickness=8)


				# If score of detected object is bigger than 0.5
				# means API draws bounding box on image
				# Print detected object class
				for i in range(len(boxes[0])):
					if np.squeeze(scores)[i] > 0.5:
						class_name = category_index[np.squeeze(classes).astype(np.int32)[i]]['name']
						display_str = str(class_name)
						if display_str == 'car':
							info[1] = 1
						elif display_str == 'pedestrian':
							info[1] = 2
						#print('Class: ', display_str)

				# calculate fps
				frame = lane_detection.drawTopviewBoundary(frame)
				curr_time = time.time()
				sec = curr_time - prev_time
				prev_time = curr_time
				fps = 1 / sec
				str_fps = "FPS: %0.1f" %fps
				cv2.putText(frame, str_fps, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

				# Display output
				cv2.imshow('object detection', frame)
				cv2.imshow('bin_birds_eye_view', bin_birds_eye_view)

				#print(info)
				data = ''
				for i in info:
					data += str(i) + ' '
				for i in range(data_len - len(data)):
					data += '*'
				print('deviation : ', info[0], 'object : ', info[1])
				try:
					ser.write(data.encode())
				except:
					print('USB disconnected!')

				# Save video
				#video_writer.write(frame)

				if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break
