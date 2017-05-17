import tensorflow as tf
from termcolor import colored

class Conv2d():
	
	def __init__(self, kernel_size, strides, output_channels, name):
		self.kernel_size = kernel_size
		self.strides = strides
		self.output_channels = output_channels
		self.name = name


	def create_layer(self, input):
		num_input_channels = input.shape[3]

		with tf.variable_scope('conv', reuse=False):
			W = tf.get_variable('W{}'.format(self.name[-3:]),
				shape=(self.kernel_size, self.kernel_size, num_input_channels, self.output_channels))
			b = tf.Variable(tf.zeros([self.output_channels]))
			#print colored('W -- {}'.format(W.get_shape()),'blue')
			#print colored(' b - {}'.format(b.get_shape()),'blue')

		output = tf.nn.conv2d(input, W, strides=self.strides, padding="SAME")
		output = tf.nn.elu(tf.add(tf.contrib.layers.batch_norm(output), b))
		return output


	def get_description(self):
		return "C{},{},{}".format(self.kernel_size, self.output_channels, self.strides[1])

'''
def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
	return f1 * x + f2 * abs(x)
'''