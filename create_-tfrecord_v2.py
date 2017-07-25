# other version is in windows :3

import tensorflow as tf
import os
from __future__ import print_function
# import numpy as np

dataset_dir = '/dataset' # or change to whatever you wanna name it? tgif_data sounds good
filename = 'tgif'

# note in case of bugs maybe we should consider to make all the first numbers into 1 instead of 0 idk

def read_frames(gif_dir):
    '''
    Convert each gif into a set of matrix representations for each frame.
    :param gif_dir: directory of gif
    :return: 
    '''
#   frame_num = len(os.listdir(gif_dir)) - 1
    # the minus 1 there is for the description.txt in the directory
    gif_frameset = []
    gif_caption = ''
    for frame in os.listdir(gif_dir):
        if frame = 'description.txt':
            with open(frame, 'r') as f:
                gif_caption = f.read()
            continue
        with open(frame, 'rb') as f:
            frame_bytes = f.read()
        bytes = tf.placeholder(tf.string) # change to int or something if error occurs around here
        decode_frame = tf.image.decode_image(bytes, channels=3)
        with tf.Session() as sess:
            image = sess.run(decode_frame, feed_dict-{bytes:frame_bytes})
        gif_frameset.append(image)
    if len(gif_frameset) == 0:
        return None
    else:
        return gif_frameset, gif_caption

# di ko gets 'tong function na 'to besh pa-double check tnx
def create_seq_example(gif_frameset, caption):
    '''
    Creates a sequence example?
    :param gif_frames: 
    :param caption: 
    :return: 
    '''
    example_sequence = tf.train.SequenceExample()
    frame_num = len(gif_frameset)
    example_sequence.context.feature("length").int64_list.value.append(frame_num)
    gif_contents = example_sequence.feature_lists.feature_list["gif_contents"]
    caption_str = example_sequence.feature_lists.feature_list["caption_str"]
    # above code may be a bug

    for gif_content, caption in zip(gif_frameset):
        if gif_frameset is not None: # float ata yung sa baba
            gif_contents.feature.add().int64_list.value.append(gif_content)
#       if caption_chars is not None: # possible bug
#       if something goes wrong, replace the if and try to make it word vec
    caption_str.feature.add().bytes_list.value.append(caption)

    return example_sequence

def make_tfrecord(filename, tgif_gifset, tgif_capset):
    with open(filename, 'w') as fp:
        writer = tf.python_io.TFRecordWriter(fp.name)
        for frameset, caption in zip(tgif_gifset, tgif_capset):
            ex = create_seq_example(frameset, caption)
            writer.write(ex.SerializeToString())
        writer.close()

def read_and_decode_single_example(filename):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "gif_contents": tf.VarLenFeature(tf.int64),
        "caption": tf.FixedLenSequenceFeature([], dtype=tf.bytes)
    }
    gif_dense = tf.sparse_tensor_to_dense(sequence_parsed["gif_contents"])
    # ????? there may be a bug around here? cri
    return serialized_example, context_features, sequence_features

# i decided to do the main function first, before polishing the
# rest of the code because i suddenly realized something
# the input_sequences bit in your reference*
# is the set of all sequences, meaning, it's a list of all the gif frame sets
# what the fuck, that's just crazy, can that really be handled?
# oh well fuck it
# * https://stackoverflow.com/questions/39524323/tf-sequenceexample-with-multidimensional-arrays

# first im gonna have to make that giant list of all frame sets
# (i am still absolutely disgusted at this development)

def main():
    tgif_dataset = [] # huge af list containing all matrix reps of frames
    tgif_capset = [] # list of all captions
    tgif_list = os.listdir(dataset_dir)
    tgif_list.sort()
    for gif in tgif_list:
        gif_dir = os.path.join(dataset_dir, gif)
        gif_frameset, gif_caption = read_frames(gif_dir)
        tgif_dataset.append(gif_frameset)
        # i feel like this shit will be too huge :(
        tgif_capset.append(gif_caption)
    make_tfrecord(filename, tgif_dataset, tgif_capset)
    ex, context_features, sequence_features = read_and_decode_single_example(filename)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=ex,
        context_features=context_features,
        sequence_features=sequence_features
    )
    sequence = tf.contrib.learn.run_n(sequence_parsed, n=1, feed_dict=None)
    # check if saved data matches input
    print(sequences[0] in sequence[0]['gif_contents'])

if __name__ = "__main__":
    main()
