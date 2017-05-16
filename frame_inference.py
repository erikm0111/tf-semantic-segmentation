import io
import os
import time

from network import Network
import tensorflow as tf
import train
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
    layers.append(MaxPool2d(kernel_size=2, strides=[1, 2, 2, 1], name='max_1', skip_connection=True))

    layers.append(Conv2d(kernel_size=3, strides=[1, 2, 2, 1], output_channels=128, name='conv_2_1'))
    layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=128, name='conv_2_2'))
    layers.append(MaxPool2d(kernel_size=2, strides=[1, 2, 2, 1], name='max_2', skip_connection=True))

    layers.append(Conv2d(kernel_size=3, strides=[1, 2, 2, 1], output_channels=256, name='conv_3_1'))
    layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=256, name='conv_3_2'))
    layers.append(MaxPool2d(kernel_size=2, strides=[1, 2, 2, 1], name='max_3'))


    network = Network(layers, per_image_standardization=False, shape=(570, 3260, 3))

    input_file = '/home/ematosevic/CompSci/projects_repo/tf-semantic-segmentation/video/t7.mp4'
    output_dir = '/home/ematosevic/CompSci/projects_repo/tf-semantic-segmentation/test_frames_output/'
    checkpoint = 'save/C3,64,2C3,64,1M2C3,128,2C3,128,1M2C3,256,2C3,256,1M2/2017-05-10_234424/'
    cap = cv2.VideoCapture(input_file)

    with tf.Session() as sess:
        # Needed for time metrics
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            print colored('Restoring model: {}'.format(ckpt.model_checkpoint_path), 'yellow')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise IOError('No model found in {}.'.format(checkpoint))
        
        #NUM_EPOCHS = 2000
        NUM_EPOCHS = 100
        NUM_FRAMES_IN_EPOCH = 1

        counter = 0
        for i in range(NUM_EPOCHS):
            inputs = []
            for k in range(NUM_FRAMES_IN_EPOCH):
                ret, real_frame = cap.read()
                inputs.append(np.array(real_frame))
            inputs = np.array(inputs, dtype=np.uint8)
            #inputs = np.reshape(inputs, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS))
            inputs = np.multiply(inputs, 1.0/255)


            #print colored("Image shape: {}".format(image.shape), 'yellow')
            start = time.time()

            segmentation = sess.run(network.segmentation_result, feed_dict={
                network.inputs: np.reshape(inputs, [NUM_FRAMES_IN_EPOCH, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS])},
                options=run_options, run_metadata=run_metadata)
            
            end = time.time()
            # time metrics
            #tl = timeline.Timeline(run_metadata.step_stats)
            #ctf = tl.generate_chrome_trace_format()
            #with open('proba/timeline.json', 'w') as f:
            #    f.write(ctf)

            for example_i in range(NUM_FRAMES_IN_EPOCH):
                frame_seg = np.reshape(segmentation[example_i], (570, 3260, 3))
                frame_seg = np.multiply(frame_seg, 255.)
                frame_seg = cv2.threshold(frame_seg, 127, 255, cv2.THRESH_BINARY)[1]
                output_name = "%05d" % counter
                output_name += '.jpg'
                counter += 1
                cv2.imwrite(output_dir + output_name, frame_seg)

            print colored("Done {}/{} in {}".format((i+1)*NUM_FRAMES_IN_EPOCH, NUM_FRAMES_IN_EPOCH*NUM_EPOCHS, end-start),'yellow')
        

