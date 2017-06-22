import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
#import ipdb
import pprint

from inception_preprocessing import preprocess_image, preprocess_for_eval
from cnn_inception_resnet import *
slim = tf.contrib.slim

save_path = '/dataset/tgif_dataset/'
train_path = '/dataset/tgif_dataset/train'
test_path = '/dataset/tgif_dataset/test'

def create_gif_list(train_mode):
	# preprocesses each frame
	# downloads the frame and then transforms it into an array
	print("Preprocessing frames for CNN...")
	
	if train_mode == True:
		data_path = train_path
	else: data_path = test_path
#	data_path = save_path
	# create a list of all frameset directories
	print("Searching for gifs within directory", data_path)
	if os.path.exists(data_path):
		print("Found directory! Saving gifs...")
		dataset_li = os.listdir(data_path)
		print(len(dataset_li), "gifs found!")
	else:
		print("Directory doesn't exist. Please run download_gifs.py to download dataset")
		return None

	return dataset_li

def create_frame_list(gif_path, train_mode):
	# for given gif_path, list all frames
	frame_set = []
	if train_mode==True:
		data_path = os.path.join(train_path, gif_path)
	else:
		data_path = os.path.join(test_path, gif_path)
	if os.path.exists(data_path):
		frame_set = os.listdir(data_path)
		frame_set.remove('description.txt')
		frame_set.sort()
		return frame_set
	else:
		return None		

def create_dataframe():
	train_list = []
	train_captions = []
	test_list = []
	test_captions = []

	if os.path.exists(train_path):
		train_list = create_gif_list(train_mode=True)
		for gif_path in train_list:
			gif_path = os.path.join(train_path, gif_path)
			desc_path = os.path.join(gif_path, 'description.txt')
			with open(desc_path, 'r') as desc_file:
				train_captions.append(desc_file.read())
	if os.path.exists(test_path):
		test_list = create_gif_list(train_mode=False)
		for gif_path in test_list:
			gif_path = os.path.join(test_path, gif_path)
			desc_path = os.path.join(gif_path, 'description.txt')
			with open(desc_path, 'r') as desc_file:
				test_captions.append(desc_file.read())
				
	train_set = pd.DataFrame({'gif_path':train_list,'description':train_captions})
	test_set = pd.DataFrame({'gif_path':test_list,'description':test_captions})

	return train_set, test_set

def preprocess_frame(frame_path, gif_path, train_mode):
	print("function: preprocess_frame()")
	if train_mode == True:
		gif_path = os.path.join(train_path, gif_path)
	else:
		gif_path = os.path.join(test_path, gif_path)
	frame_path = os.path.join(gif_path, frame_path)
	frame_contents = tf.read_file(frame_path)
	print(frame_contents)
	frame_tensor = tf.image.decode_png(frame_contents)
	print(frame_tensor)
	# print input tensor shape??
	print("frame tensor shape:",frame_tensor.get_shape())
	# print tensor everytime it's modified
#	resized_frame = tf.image.resize_images(frame_tensor,299,299)
	frame_tensor = tf.reshape(frame_tensor, (299,299,3)) 
#	frame_tensor = 2*(frame_tensor/tf.cast(255.0,tf.uint8))-tf.cast(1.0,tf.uint8)
	frame_tensor = preprocess_for_eval(frame_tensor, height=227, width=227)
#	frame_tensor = tf.cast(frame_tensor, uint8)
	
	return frame_tensor

def main():
	print("Hello! We're preprocessing the frames by extracting their features through CNN. :)")
	num_frames = 70
	cnn = CNN_inception(batch_size=20, height=227, width=227, channels=3)
	print("Initialized inception resnet v3 class")
	train_list = create_gif_list(train_mode=True)
	test_list = create_gif_list(train_mode=False)
	list_set = [train_list, test_list]
	train_set, test_set = create_dataframe()
	for thing in list_set:
		if thing == train_list:
			train_mode = True
		else:
			train_mode = False
		for gif_path in thing:
			frame_list = create_frame_list(gif_path, train_mode)
#			ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
			frame_count = len(frame_list)
#			ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
			frame_list = np.array(frame_list)
#			ipdb.set_trace(context=7) # BREAKPOINT (for debugging)

			if frame_count > 70:
				frame_indices = np.linspace(0, frame_count, num=num_frames, endpoint=False).astype(int)
				frame_list = frame_list[frame_indices]
#				ipdb.set_trace() # BREAKPOINT (for debugging)

#			ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
			processed_frame_list = []
			for frame_path in frame_list:
				processed_frame = preprocess_frame(frame_path, gif_path, train_mode)
				processed_frame_list.append(processed_frame)
#			processed_frame_list = preprocess_frames(gif_path, train_mode)
#			processed_frame_list = np.array(processed_frame_list)
#			processed_frame_list = np.array(map(lambda x: preprocess_frame(x),frame_list))
#			ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
			features = cnn.extract_features(processed_frame_list)

			if train_mode == True:
				gif_feats_path = os.path.join('/output/tgif_dataset/train/', gif_path)
				description_path = '/output/tgif_dataset/train/description.txt'
				with open(description_path, 'w') as desc_file:
					desc_file.write(train_set['description'])
			else:
				gif_feats_path = os.path.join('/output/tgif_dataset/test/', gif_path)
				description_path = '/output/tgif_dataset/test/description.txt'
				with open(description_path, 'w') as desc_file:
					desc_file.write(test_set['description'])
			gif_feats_path = os.path.join(gif_feats_path, 'feats.npy')
#			ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
			np.save(gif_feats_path, features)
	return

if __name__=="__main__":
#	ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
	main()
