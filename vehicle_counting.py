#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

# Imports
import tensorflow as tf

# Object detection imports
from utils import backbone
from api import object_counting_api

#input_video = "./input_images_and_videos/vehicle_survaillance.mp4"
#input_video = "http://192.168.113.194:8080/video"

def process_facerecog():
	input_video = 0

	# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
	detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2017_11_17')

	targeted_objects = "person"
	fps = 24 # change it with your input video fps
	width = 640 # change it with your input video width
	height = 480 # change it with your input vide height
	is_color_recognition_enabled = 0 # set it to 1 for enabling the color prediction for the detected objects
	roi = 350 # roi line position
	deviation = 3 # the constant that represents the object counting area

	object_counting_api.cumulative_object_counting_y_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects,fps, width, height, roi, deviation) # counting all the objects