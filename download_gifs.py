'''
Alyssa Joyce E. Telles
BS Computer Engineering
University of the Philippines

Running this code will:
- divide the dataset into training and test set
- create directories
- download the GIFs as a folder containing a set of frames along with
  their descriptions.

Dataset: TGIF (github.com/raingo/TGIF-Release)
Python 3.5.2
Tensorflow 1.0
'''

from __future__ import print_function
import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO

dataset_url = 'https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/tgif-v1.0.tsv'
save_path = './tgif_dataset/'
train_ratio = 0.9
debug = True # this will be changed to false once the real training begins
# During the creation of this code, I used a smaller dataset for debugging

def dl_process_data(save_path, row, debug):
    '''
	This function downloads and processes the GIFs
	Arguments:
	- save_path
	- row in dataset dataframe
	- debug to indicate if code is in debug mode (uses smaller dataset)
    '''
    # download in save_path
    print("In function dl_process_data()")
    gif_url = row['gif_url']
    print("Creating directory for", gif_url)
    gif_count = 0
    gif_path = 'gif_' + str(gif_count)
    full_path = os.path.join(save_path, gif_path)
    while os.path.exists(full_path):
        gif_count += 1
        gif_path = 'gif_' + str(gif_count)
        full_path = os.path.join(save_path, gif_path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    # if path already exists
    # download each gif as folders with frames and description
    if os.path.exists(full_path):
        print(full_path, "directory created!")
        
        # access gif url
        response = requests.get(gif_url)
        gif = Image.open(BytesIO(response.content))
        gif.seek(1)
        frame_count = 0
        print("Saving frame...")
    
        # download each frame into given directory
        try:
            while 1:
                gif.seek(gif.tell() + 1)
                frame_filename = 'frame_' + str(frame_count) + '.png'
                print("Saved", frame_filename)
                frame_path = os.path.join(full_path, frame_filename)
                gif.save(frame_path)
                frame_count += 1
                print("Saving next frame...")
        except EOFError:
            pass
    
        # download text file containing description
        text_path = os.path.join(full_path, 'description.txt')
        with open(text_path, "w") as desc:
            print("writing sentence", row['description'])
            desc.write(row['description'])
    
    else:
        print("failed to create directory", full_path)
        
    return

def main():
    # access dataset and save into a dataframe
    if debug == False:
        print("not in debug mode. Accessing original dataset...")
        response = requests.get(dataset_url)
        tgif_data = pd.read_table(BytesIO(response.content), sep='\t', header=None)
    else:
        print("in debug mode. Accessing mini dataset...")
        tgif_data = pd.read_table('mini-tgif.tsv', sep='\t', header=None)
    print("Finished reading dataset file!")
    
    tgif_data.columns = ['gif_url', 'description']
    
    train_num = int(len(tgif_data.index)*train_ratio)

    # Split dataset into training and testing data
    train_data = tgif_data.sample(frac=train_ratio)
    test_data = tgif_data.drop(train_data.index)
    
    train_path = os.path.join(save_path, 'train')
    test_path = os.path.join(save_path, 'test')
    
    train_data.apply(lambda row: dl_process_data(train_path, row, debug), axis=1)
    test_data.apply(lambda row: dl_process_data(test_path, row, debug), axis=1)
    
    print("Finished downloading entire dataset. Exiting...")
    return

if __name__ == "__main__":
    main()

