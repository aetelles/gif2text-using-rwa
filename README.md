# Generating Descriptions for Animated GIFs using Recurrent Weighted Average

this is under construction, go away for now :P

this uses the pre-trained inception resnet v2 model found in https://github.com/tensorflow/models/tree/master/slim/nets 
this uses the recurrent weighted average model propsed by Jared Ostmeyer in https://github.com/jostmey/rwa

hah wish me luck m8 

# instructions idk
log into my floydhub
dataset is in ID of gif2text:14
inception checkpoint is in ID of inception-resnet-v2

run preprocess_gif.py
running preprocess_gif.py is supposed to output something like this:

tgif_feats:
train
-| gif_0
----| feats.npy
----| description.txt
-| gif_1
----| feats.npy
----| description.txt
.
.
.
test
-| gif_0
----| feats.npy
----| description.txt
-| gif_1
----| feats.npy
----| description.txt
.
.
.
