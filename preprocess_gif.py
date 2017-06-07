import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
# import scikitimage

from inception_preprocessing.py import preprocess_for_eval
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
    else:
        data_path = test_path

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

def create_frame_list(gif_path):
    # for given gif_path, list all frames
    frame_set = []
    if os.path.exists(data_path):
        frame_set = os.listdir(gif_path)
        return frame_set
    else:
        return None

def preprocess_frame(frame_file):
    frame = Image.open(frame_file).resize((299,299))
    frame = np.array(frame)
    frame = frame.reshape(-1, 299, 299, 3) # reshape into 3d tensor
    frame = 2*(frame/255.0)-1.0 # normalization???
    frame = preprocess_for_eval(frame, height=227, width=227) # consider changing this to preprocess_for_training
    # not really sure about line 57 because the preprocessing might be doubled
    # anyway line 57 crops the frame i think lol
    return frame

def create_dataframe():
    train_list = []
    train_captions = []
    test_list = []
    test_captions = []

    if os.path.exists(train_path):
        train_list = create_gif_list(train_mode=True)
        for gif_path in train_list:
            desc_path = os.path.join(gif_path, 'description.txt')
            with open(desc_path, 'r') as desc_file:
                train_captions.append(desc_file.read())

    if os.path.exists(test_path):
        test_list = create_gif_list(test_mode=False)
        for gif_path in test_list:
            desc_path = os.path.join(gif_path, 'description.txt')
            with open(desc_path, 'r') as desc_file:
                test_captions.append(desc_file.read())

    train_set = pd.DataFrame({'gif_path': train_list, 'description': train_captions})
    test_set = pd.DataFrame({'gif_path': test_list, 'description': test_captions})

    return train_set, test_set
