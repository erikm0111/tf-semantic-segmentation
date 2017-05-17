import tensorflow as tf

class Upsample2d():

	def __init__(self, name, skip_connection=False):
		self.name = name
		self.skip_connection = skip_connection


	def create_layer(self, input, input_shape, prev_layer=None):
		if self.skip_connection==True:
			tf.add(input, prev_layer)

		with tf.name_scope("Upsample2D") as scope:
			output = tf.image.resize_nearest_neighbor(
				input,
				size=input_shape[1:3])
			output.set_shape((None, input_shape[1], input_shape[2], None))

		output.scope = scope
		return output