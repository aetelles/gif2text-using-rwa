from __future__ import print_function

import tensorflow as tf
import numpy as np
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope

slim = tf.contrib.slim

layer_size = 256  # if you wish to change this go to function CNN_inception.extract_features


class CNN_inception(object):
    def __init__(self, batch_size=20, width=227, height=227, channels=3):
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.channels = channels

    def load_cnn(self):
        print("Loading pre-trained Inception ResNet V2...")
        # function loads the pretrained inception resnet v2 model
        X = tf.placeholder(tf.float32, shape=[None, height, width, channels])
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(X, num_classes=1001, is_training=False)
        # initialize network with checkpoint file
        self.saver = tf.train.Saver()
        sess = tf.Session()
        self.saver.restore(sess, 'inception_resnet_v2_resnet_v2_2016_08_30.ckpt')
        print("Pre-trained Inception ResNet V2 is succesfully loaded!")

        return

    def extract_features(self, image_list, layer_sizes=[256]):
        print("Extracting features using CNN...")
        iter_until = len(image_list) + self.batch_size
        print("CNN will iterate until", iter_until)
        all_features = np.zeros([len(image_list)] + layer_sizes)

        for start, end in zip(range(0, iter_until, self.batch_size),
                              range(self.batch_size, iter_until, self.batch_size)):

            image_batch = image_list[start:end]

            cnn_in = np.zeros(np.array(image_batch.shape)[[0, 3, 1, 2]], dtype=np.float32)

            for idx in enumerate(image_batch):
                cnn_in[idx] = end_points['Logits']
                output[idx] = sess.run(cnn_in[idx], feed_dict={X: image_batch})

        return output
