import tensorflow as tf
from layer import Layer

class MaxPool2d(Layer):

	def __init__(self, kernel_size, strides, name):
		self.kernel_size = kernel_size
		self.strides = strides
		self.name = name
		

	def create_layer(self, input):
		self.input_shape = input.shape
		with tf.name_scope("MaxPool2d") as scope:
			output = tf.nn.max_pool(
				input,
				ksize=[1, self.kernel_size, self.kernel_size, 1],
				strides=self.strides,
				padding="SAME")

		output.scope = scope
		return output


	def get_description(self):
		return "M{}".format(self.kernel_size)
