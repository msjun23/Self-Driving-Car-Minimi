import numpy as np
import tensorflow as tf
import cv2
import os
import time
from multiprocessing import Pool, Queue

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import calibration
import lane_detection
import line_processing

mtx, dist = calibration.calib()
left_line = line_processing.Line()
right_line = line_processing.Line()

def getBirdsEyeView(input_que, output_que):
	while True:
		frame = input_que.get()
		if frame is not None:
			undist_frame = calibration.undistort(frame, mtx, dist)
			thresholded_img = lane_detection.thresholdImage(undist_frame)
			birds_eye_view = lane_detection.getTopviewImage(thresholded_img)
			
			start_left_x, start_right_x, line_shape = lane_detection.findStartLine(birds_eye_view)
			lane_midpoint_x = (start_right_x + start_left_x)/2
			deviation = lane_midpoint_x - birds_eye_view.shape[1]/2

			processed_bin = line_processing.findLines(birds_eye_view, left_line, right_line)
			output_que.put([processed_bin, line_shape, deviation, left_line.radius_of_curvature, right_line.radius_of_curvature])

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

	'''# Save video-setting
	video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	fourcc = cv2.VideoWriter_fourcc(*'XVID')

	video_writer = cv2.VideoWriter('output.avi', fourcc, 30.0, (video_width, video_height))'''

	# Detection
	with detection_graph.as_default():
		with tf.compat.v1.Session(graph=detection_graph) as sess:
			while True:
				# Read frame from camera
				ret, frame = cap.read()
				frame = cv2.resize(frame, dsize=(640,360), interpolation=cv2.INTER_AREA)
				info = [[],[],[],[],[],[]]

				input_que.put(frame)
				birds_eye_view, line_shape, deviation, left_line_curvature, right_line_curvature = output_que.get()
				info[0] = line_shape
				info[1] = deviation
				info[2] = left_line_curvature
				info[3] = right_line_curvature

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
				# Print detected object coordinates
				# Print detected object class
				height = frame.shape[0]
				width = frame.shape[1]

				for i in range(len(boxes[0])):
					if np.squeeze(scores)[i] > 0.5:
						#print('Accuracy score: ', np.squeeze(scores)[i])
						ymin = (int(boxes[0][i][0]*height))
						xmin = (int(boxes[0][i][1]*width))
						ymax = (int(boxes[0][i][2]*height))
						xmax = (int(boxes[0][i][3]*width))
						#print('Image boundary: ', ymin,xmin,ymax,xmax)
						info[4] = [(xmin+xmax)/2, (ymin+ymax)/2]
						#print('Center coordinate: ', (xmin+xmax)/2, (ymin+ymax)/2)

						class_name = category_index[np.squeeze(classes).astype(np.int32)[i]]['name']
						display_str = str(class_name)
						info[5] = display_str
						#print('Class: ', display_str)

				# calculate fps
				curr_time = time.time()
				sec = curr_time - prev_time
				prev_time = curr_time
				fps = 1 / sec
				str_fps = "FPS: %0.1f" %fps
				cv2.putText(frame, str_fps, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

				# Display output
				cv2.imshow('object detection', frame)
				cv2.imshow('birds_eye_view', birds_eye_view)
				print(info)

				# Save video
				#video_writer.write(frame)

				if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break
