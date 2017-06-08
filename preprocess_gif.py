import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
import ipdb
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
	# queue of all frames 
	frame_path = os.path.join(gif_path, frame_path)
#	ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
	frame_queue = tf.train.string_input_producer([frame_path])
#	ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
	frame_reader = tf.WholeFileReader()
#	ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
	_, frame_file = frame_reader.read(frame_queue)
#	ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
	frame = tf.image.decode_png(frame_file, channels=3)
#	ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
	with tf.Session() as sess:
#		ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
		tf.global_variables_initializer().run()
		image_tensor = sess.run([frame])
		print(image_tensor)
		# hello ajet end the code at this point :)
#		ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
	return image_tensor
	

#def preprocess_frame(frame_path, gif_path, train_mode):
#	if train_mode == True:
#		gif_path = os.path.join(train_path, gif_path)
#	else:
#		gif_path = os.path.join(test_path, gif_path)
#	frame_path = os.path.join(gif_path, frame_path)
#	image = Image.open(frame_path)
#	ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
#	image = tf.image.decode_png(image, channels=3)
#	ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
#	resized_frame = tf.image.resize_images(image,299,299)
#	ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
	
#	frame_path = os.path.join(gif_path, frame_file)
	
#	frame = Image.open(frame_path).resize((299,299))
#	ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
#	frame = np.array(frame)
#	ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
#	frame = frame.reshape(-1, 299, 299, 3) # reshape into 3d tensor
#	frame = 2*(frame/255.0)-1.0 # normalization???
#	ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
#	frame = preprocess_for_eval(frame,height=227,width=227)
#	ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
	# consider changing above function to preprocess_for_training
	# not really sure because preprocessing might be doubled
	# anyway it crops the frame i think lol
	return frame

def main():
	print("Hello! We're preprocessing the frames by extracting their features through CNN. :)")
	num_frames = 70
	cnn = CNN_inception(batch_size=20, height=227, width=227, channels=3)
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
	ipdb.set_trace(context=7) # BREAKPOINT (for debugging)
	main()
