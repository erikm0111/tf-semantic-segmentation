import io
import os

import tensorflow as tf
from network import Network
from conv_2d import Conv2d
from max_pool_2d import MaxPool2d
import numpy as np
import cv2
from termcolor import colored
from tensorflow.python.client import timeline

import matplotlib.pyplot as plt

if __name__ == '__main__':

    layers = []
    layers.append(Conv2d(kernel_size=3, strides=[1, 2, 2, 1], output_channels=64, name='conv_1_1'))
    layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_2'))
    layers.append(MaxPool2d(kernel_size=2, name='max_1', skip_connection=True))

    layers.append(Conv2d(kernel_size=3, strides=[1, 2, 2, 1], output_channels=128, name='conv_2_1'))
    layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=128, name='conv_2_2'))
    layers.append(MaxPool2d(kernel_size=2, name='max_2', skip_connection=True))

    layers.append(Conv2d(kernel_size=3, strides=[1, 2, 2, 1], output_channels=256, name='conv_3_1'))
    layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=256, name='conv_3_2'))
    layers.append(MaxPool2d(kernel_size=2, name='max_3'))

    layers.append(Conv2d(kernel_size=3, strides=[1, 2, 2, 1], output_channels=512, name='conv_4_1'))
    layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=512, name='conv_4_2'))
    layers.append(MaxPool2d(kernel_size=2, name='max_4'))


    network = Network(layers, per_image_standardization=False, shape=(570, 3260, 3))

    input_dir = '/home/ematosevic/CompSci/projects_repo/tf-semantic-segmentation/proba/'
    input_filename = '00002_1.jpg'
    input_image = input_dir + input_filename
    checkpoint = 'save/C3,64,2C3,64,1M2C3,128,2C3,128,1M2C3,256,2C3,256,1M2C3,512,2C3,512,1M2/2017-05-15_020348/'

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            print colored('Restoring model: {}'.format(ckpt.model_checkpoint_path), 'yellow')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise IOError('No model found in {}.'.format(checkpoint))
        
        image = cv2.imread(input_image)
        height, width = image.shape[:2]
        #image = np.array(cv2.imread(input_image, 0))  # load grayscale
        #image = cv2.resize(image, (int(width*2), int(height*2)), interpolation = cv2.INTER_CUBIC)
        image = np.array(image)
        image = np.multiply(image, 1.0/255)


        print colored("Image shape: {}".format(image.shape), 'yellow')

        segmentation = sess.run(network.segmentation_result, feed_dict={
            network.inputs: np.reshape(image, [1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS])})

        segmentation = np.reshape(segmentation[0], (570, 3260, 3))
        segmentation = np.multiply(segmentation, 255.)
        segmentation = cv2.threshold(segmentation, 127, 255, cv2.THRESH_BINARY)[1]
        
        cv2.imwrite('proba/prediction_' + input_filename, segmentation)

