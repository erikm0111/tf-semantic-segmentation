import tensorflow as tf
from conv_2d import Conv2d
from max_pool_2d import MaxPool2d
from deconv_2d import Deconv2d
from upsample_2d import Upsample2d
from termcolor import colored


class Network():

	def __init__(self, encoderLayers=None, decoderLayers=None, per_image_standardization=True, shape=None):
		self.IMAGE_HEIGHT = shape[0]
		self.IMAGE_WIDTH = shape[1]
		self.IMAGE_CHANNELS = shape[2]

		# ENCODER
		if encoderLayers == None:
			encoderLayers = []
			encoderLayers.append(Conv2d(kernel_size=3, strides=[1,2,2,1], output_channels=64, name='conv_1_1'))
			encoderLayers.append(Conv2d(kernel_size=3, strides=[1,1,1,1], output_channels=64, name='conv_1_2'))
			encoderLayers.append(MaxPool2d(kernel_size=2, strides=[1,2,2,1], name='max_pool_1'))

			encoderLayers.append(Conv2d(kernel_size=3, strides=[1,2,2,1], output_channels=128, name='conv_2_1'))
			encoderLayers.append(Conv2d(kernel_size=3, strides=[1,1,1,1], output_channels=128, name='conv_2_2'))
			encoderLayers.append(MaxPool2d(kernel_size=2, strides=[1,2,2,1], name='max_pool_2'))

			encoderLayers.append(Conv2d(kernel_size=3, strides=[1,2,2,1], output_channels=512, name='conv_3_1'))
			encoderLayers.append(Conv2d(kernel_size=3, strides=[1,1,1,1], output_channels=512, name='conv_3_2'))
			encoderLayers.append(MaxPool2d(kernel_size=2, strides=[1,2,2,1], name='max_pool_3'))

		# DECODER
		if decoderLayers == None:
			decoderLayers = []
			decoderLayers.append(Upsample2d(name='max_pool_3'))
			decoderLayers.append(Deconv2d(strides=[1,1,1,1], name='conv_3_2'))
			decoderLayers.append(Deconv2d(strides=[1,2,2,1], name='conv_3_1'))

			decoderLayers.append(Upsample2d(name='max_pool_2', skip_connection=True))
			decoderLayers.append(Deconv2d(strides=[1,1,1,1], name='conv_2_2'))
			decoderLayers.append(Deconv2d(strides=[1,2,2,1], name='conv_2_1'))

			decoderLayers.append(Upsample2d(name='max_pool_1', skip_connection=True))
			decoderLayers.append(Deconv2d(strides=[1,1,1,1], name='conv_1_2'))
			decoderLayers.append(Deconv2d(strides=[1,2,2,1], name='conv_1_1'))


		self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS],
			name='inputs' )
		self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1],
			name='targets')
		self.description = ""

		self.layers = {}

		if per_image_standardization==True:
			list_of_images_norm = tf.map_fn(tf.image.per_image_standardization, self.inputs)
			net = tf.stack(list_of_images_norm)
		else:
			net = self.inputs

		counter = 0
		input_shapes = []
		for layer in encoderLayers:
			input_shapes.append(net)
			self.layers[layer.name] = net = layer.create_layer(net)
			self.description += "{}".format(layer.get_description())
			counter += 1

		for layer in decoderLayers:
			net = layer.create_layer(net, input_shape=input_shapes[counter-1].shape, prev_layer=self.layers[layer.name])
			counter -= 1


		self.segmentation_result = tf.sigmoid(net)
		#self.segmentation_result = tf.nn.elu(net)	

		self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.segmentation_result - self.targets)))
		self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

		with tf.name_scope('accuracy'):
			argmax_probs = tf.round(self.segmentation_result)
			correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
			self.accuracy = tf.reduce_mean(correct_pred)

			tf.summary.scalar('accuracy', self.accuracy)

		self.summaries = tf.summary.merge_all()