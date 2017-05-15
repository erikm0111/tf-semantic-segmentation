import tensorflow as tf
from layer import Layer

class Conv2d(Layer):
	layer_index = 0
	
	def __init__(self, kernel_size, strides, output_channels, name):
		self.kernel_size = kernel_size
		self.strides = strides
		self.output_channels = output_channels
		self.name = name


	@staticmethod
	def reverse_global_variables():
		Conv2d.layer_index = 0


	def create_layer(self, input):
		num_input_channels = input.shape[3]
		self.input_shape = input.shape

		with tf.variable_scope('conv', reuse=False):
			W = tf.get_variable('W{}'.format(self.name[-3:]),
				shape=(self.kernel_size, self.kernel_size, num_input_channels, self.output_channels))
			b = tf.Variable(tf.zeros([self.output_channels]))

		Conv2d.layer_index += 1

		output = tf.nn.conv2d(input, W, strides=self.strides, padding="SAME")
		output = tf.nn.elu(tf.add(tf.contrib.layers.batch_norm(output), b))
		return output

	def create_layer_reversed(self, input, prev_layer=None):
		with tf.variable_scope('conv', reuse=True):
			W = tf.get_variable('W{}'.format(self.name[-3:]))
			b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))

		output = tf.nn.conv2d_transpose(
			input,
			W,
			output_shape = tf.stack([tf.shape(input)[0], self.input_shape[1], self.input_shape[2], self.input_shape[3]]),
			strides=self.strides,
			padding="SAME")

		Conv2d.layer_index += 1
		output.set_shape([None, self.input_shape[1], self.input_shape[2], self.input_shape[3]])

		output = tf.nn.elu(tf.add(tf.contrib.layers.batch_norm(output), b))
		return output


	def get_description(self):
		return "C{},{},{}".format(self.kernel_size, self.output_channels, self.strides[1])


def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
	return f1 * x + f2 * abs(x)