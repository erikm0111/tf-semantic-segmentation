import tensorflow as tf
from conv_2d import Conv2d
from max_pool_2d import MaxPool2d


class Network():

	def __init__(self, layers=None, per_image_standardization=True, shape=None):
		self.IMAGE_HEIGHT = shape[0]
		self.IMAGE_WIDTH = shape[1]
		self.IMAGE_CHANNELS = shape[2]

		if layers == None:
			layers = []
			layers.append(Conv2d(kernel_size=3, strides=[1,2,2,1], output_channels=64, name='conv_1_1'))
			layers.append(Conv2d(kernel_size=3, strides=[1,1,1,1], output_channels=64, name='conv_1_2'))
			layers.append(MaxPool2d(kernel_size=2, strides=[1,2,2,1], name='max_pool_1', skip_connection=True))

			layers.append(Conv2d(kernel_size=3, strides=[1,2,2,1], output_channels=128, name='conv_2_1'))
			layers.append(Conv2d(kernel_size=3, strides=[1,1,1,1], output_channels=128, name='conv_2_2'))
			layers.append(MaxPool2d(kernel_size=2, strides=[1,2,2,1], name='max_pool_2', skip_connection=True))

			layers.append(Conv2d(kernel_size=3, strides=[1,2,2,1], output_channels=256, name='conv_3_1'))
			layers.append(Conv2d(kernel_size=3, strides=[1,1,1,1], output_channels=256, name='conv_3_2'))
			layers.append(MaxPool2d(kernel_size=2, strides=[1,2,2,1], name='max_pool_3', skip_connection=True))

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

		for layer in layers:
			self.layers[layer.name] = net = layer.create_layer(net)
			self.description += "{}".format(layer.get_description())

		layers.reverse()
		Conv2d.reverse_global_variables()

		for layer in layers:
			net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])

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