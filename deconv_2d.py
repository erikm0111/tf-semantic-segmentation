import tensorflow as tf
from termcolor import colored

class Deconv2d():

	def __init__(self, strides, name):
		self.strides = strides
		self.name = name


	def create_layer(self, input, input_shape, prev_layer=None):
		with tf.variable_scope('conv', reuse=True):
			W = tf.get_variable('W{}'.format(self.name[-3:]))
			b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
			#print colored('W -- {}'.format(W.get_shape()),'yellow')
			#print colored(' b - {}'.format(b.get_shape()),'yellow')

		output = tf.nn.conv2d_transpose(
			input,
			W,
			output_shape = tf.stack([tf.shape(input)[0], input_shape[1], input_shape[2], input_shape[3]]),
			strides=self.strides,
			padding="SAME")

		output.set_shape([None, input_shape[1], input_shape[2], input_shape[3]])
		output = tf.nn.elu(tf.add(tf.contrib.layers.batch_norm(output), b))
		return output