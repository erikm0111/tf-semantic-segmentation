import tensorflow as tf
from layer import Layer

class MaxPool2d(Layer):

	def __init__(self, kernel_size, name, skip_connection=False):
		self.kernel_size = kernel_size
		self.name = name
		self.skip_connection = skip_connection
		
	def create_layer(self, input):
		self.input_shape = input.shape
		with tf.name_scope("MaxPool2d") as scope:
			output = tf.nn.max_pool(
				input,
				ksize=[1, self.kernel_size, self.kernel_size, 1],
				strides=([1, self.kernel_size, self.kernel_size, 1]),
				padding="SAME")

		output.scope = scope
		return output


	def create_layer_reversed(self, input, prev_layer=None):
		if self.skip_connection==True:
			tf.add(input, prev_layer)

		with tf.name_scope("Upsample2D") as scope:
			output = tf.image.resize_nearest_neighbor(
				input,
				size=self.input_shape[1:3])
			output.set_shape((None, self.input_shape[1], self.input_shape[2], None))

		output.scope = scope
		return output


	def get_description(self):
		return "M{}".format(self.kernel_size)