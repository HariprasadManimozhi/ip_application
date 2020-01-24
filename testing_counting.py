import tensorflow as tf
import csv
import cv2
import numpy as np
from utils import visualization_utils as vis_util
from mail import mail
import pandas as pd
from datetime import datetime

from flask import Flask, request, render_template, Response
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps

import sys
import argparse
from libfaceid.detector import FaceDetectorModels, FaceDetector
from libfaceid.encoder  import FaceEncoderModels, FaceEncoder

import time

# Object detection imports
from utils import backbone
from api import object_counting_api


# Set the input directories
INPUT_DIR_DATASET               = "datasets"
INPUT_DIR_MODEL_DETECTION       = "models/detection/"
INPUT_DIR_MODEL_ENCODING        = "models/encoding/"
INPUT_DIR_MODEL_TRAINING        = "models/training/"
INPUT_DIR_MODEL_ESTIMATION      = "models/estimation/"

# Set width and height
RESOLUTION_QVGA   = (320, 240)
RESOLUTION_VGA    = (640, 480)
RESOLUTION_HD     = (1280, 720)
RESOLUTION_FULLHD = (1920, 1080)



def label_face(frame, face_rect, face_id, confidence):
    (x, y, w, h) = face_rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
    if face_id is not None:
        cv2.putText(frame, "{} {:.2f}%".format(face_id, confidence), 
            (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# Variables
total_passed_vehicle = 0  # using it to count vehicles

def cumucount():
	input_video = "tiv.mp4"
#	input_video = "http://admin:admin123@192.168.0.105:80/1"


	# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
	detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2017_11_17')

	targeted_objects = "person"
	fps = 24 # change it with your input video fps
	width = 640 # change it with your input video width
	height = 480 # change it with your input vide height
	is_color_recognition_enabled = 0 # set it to 1 for enabling the color prediction for the detected objects
	roi = 200 # roi line position
	deviation = 5 # the constant that represents the object counting area
	total_passed_vehicle = 0
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))
    # input video
	cap = cv2.VideoCapture(input_video)
	cam_resolution = RESOLUTION_VGA
	model_detector=FaceDetectorModels.HAARCASCADE
	model_recognizer=FaceEncoderModels.LBPH
	try:
	# Initialize face detection
		face_detector = FaceDetector(model=model_detector, path=INPUT_DIR_MODEL_DETECTION)
	# Initialize face recognizer
		face_encoder = FaceEncoder(model=model_recognizer, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)
	except:
		face_encoder = None
		print("Warning, check if models and trained dataset models exists!")
	face_id, confidence = (None, 0)
	total_passed_vehicle = 0
	counting_mode = "..."
	width_heigh_taken = True
	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			# Definite input and output Tensors for detection_graph
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')

			# for all the frames that are extracted from input video
			while(cap.isOpened()):
				ret, frame = cap.read()

				if not  ret:
					print("end of the video file...")
					break
				input_frame = frame
				# Detect and identify faces in the frame
				faces = face_detector.detect(input_frame)
				for (index, face) in enumerate(faces):
					(x, y, w, h) = face
					# Identify face based on trained dataset (note: should run facial_recognition_training.py)
					if face_encoder is not None:
						face_id, confidence = face_encoder.identify(input_frame, (x, y, w, h))
					# Set text and bounding box on face
					label_face(input_frame, (x, y, w, h), face_id, confidence)
					# Process 1 face only
					#break
					# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				image_np_expanded = np.expand_dims(input_frame, axis=0)
				# Actual detection
				(boxes, scores, classes, num) = sess.run(
					[detection_boxes, detection_scores, detection_classes, num_detections],
					feed_dict={image_tensor: image_np_expanded})
					# insert information text to video frame
				font = cv2.FONT_HERSHEY_SIMPLEX

				# Visualization of the results of a detection.
				counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_y_axis(cap.get(1),
																											input_frame,
																											2,
																											is_color_recognition_enabled,
																											np.squeeze(boxes),
																											np.squeeze(classes).astype(np.int32),
																											np.squeeze(scores),
																											category_index, 
																											targeted_objects="person",
																											y_reference = roi,
																											deviation = deviation,
																											use_normalized_coordinates=True,
																											line_thickness=4)
				# when the vehicle passed over line and counted, make the color of ROI line green
				if counter == 1:
					cv2.line(input_frame, (0, roi), (width, roi), (0, 0xFF, 0), 5)
				else:
					cv2.line(input_frame, (0, roi), (width, roi), (0, 0, 0xFF), 5)

				total_passed_vehicle = total_passed_vehicle + counter
				# insert information text to video frame
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(input_frame,'Detected: ' + str(total_passed_vehicle),(10, 35),
                    font,0.8,(0, 0xFF, 0xFF),2,cv2.FONT_HERSHEY_SIMPLEX,)
				cv2.putText(input_frame,'ROI Line',(545, roi-10),font,0.6,(0, 0, 0xFF),2,cv2.LINE_AA,)
				output_movie.write(input_frame)
				#print ("writing frame")
				#cv2.imshow('object counting',input_frame)
				#if cv2.waitKey(1) & 0xFF == ord('q'):
					#break
				# Display updated frame to web app
				yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n\r\n')
			cap.release()
			cv2.destroyAllWindows()

