'''
Paper: Animated GIF Description using Recurrent Weighted Average
This paper aims to compare the performance of RWA with the more widely used LSTM
on the task of generating descriptions for animated GIFs.

Dataset: TGIF (github.com/raingo/TGIF-Release)
Python 3.5.2

Alyssa Joyce E. Telles
BS Computer Engineering
University of the Philippines
'''

#######################################################################
#	Libraries
#######################################################################

import tensorflow as tf 
import requests 
from PIL import Image 
from io import BytesIO

debug = True # if in debug mode, use mini dataset

# Parse URLs in dataset and store in array
datafile = open('mini-tgif.tsv', 'r')
datastring = datafile.read()
datafile.close()
rawset = datastring.split("\n")
dataset = []
for line in rawset:
	dataline = line.split("\t")
	dataset.append(dataline)
dataset.remove(dataset[-1])

#

# Access GIF from online

# Animated GIF Description using LSTM (for comparison)

# Animated GIF Description using RWA (for comparison)
