import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
#import scikitimage

from inception_preprocessing.py import preprocess_image
from cnn_inception_resnet import *
slim = tf.contrib.slim

save_path = './tgif_dataset/'
train_path = './tgif_dataset/train/'
test_path = './tgif_dataset/test'

def create_gif_list(train_mode):
	# preprocesses each frame
	# downloads the frame and then transforms it into an array
	print("Preprocessing frames for CNN...")
	
	dataset_li = []
	data_path = 'string'
	if train_mode == True:
		data_path = train_path
	else: data_path = test_path
	
	# create a list of all frameset directories
	print("Searching for gifs within directory", data_path)
	if os.path.exists(data_path):
		print("Found directory! Saving gifs...")
		dataset_li = os.listdir(data_path)
		print(len(dataset), "gifs found!")
	else:
		print("Directory doesn't exist. Please run download_gifs.py to download dataset")
		return None

	return dataset_li

def create_frame_list(gif_path, train_mode):
	# for given gif_path, list all frames
	frame_set = []
	if os.path.exists(data_path):
		frame_set = os.listdir(gif_path)
		return frame_set
	else:
		return None		
	
def frame_preprocessing(train_mode=True)
	# feed each frame to cnn
	dataset_li = create_gif_list(train_mode)
	frame_list = []
	for gif_path in dataset_li:
		frame_list = create_frame_list(gif_path, train_mode)
		for frame in frame_list:
			preprocess_image(frame, height=227, width=227, )
