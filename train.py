from network import Network
from dataset import Dataset
import datetime
import os
import time
import tensorflow as tf
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
import io
from imgaug import augmenters as iaa
import imgaug

# potrebno omoguciti:
# self.saver
# self.session
def restore_session(checkpoint_dir):
	global_step = 0
	if not os.path.exists(checkpoint_dir):
		raise IOError(checkpoint_dir + ' does not exist')
	else:
		path = tf.train.get_checkpoint_state(checkpoint_dir)
		if path is None:
			raise IOError('No checkpoint to restore in ' + checkpoint_dir)
		else:
			self.saver.restore(self.session, path.model_checkpoint_path)
			global_step = int(path.model_checkpoint_path.split('-')[-1])
	return global_step


def train(restore_session=False):
	BATCH_SIZE = 1
	IMAGE_HEIGHT = 570
	IMAGE_WIDTH = 3260
	IMAGE_CHANNELS = 3
	EVALUATE_EVERY_N_STEPS = 100

	network = Network(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

	timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
	os.makedirs(os.path.join('save', network.description, timestamp))

	dataset = Dataset(batch_size=BATCH_SIZE, folder="data{}_{}".format(network.IMAGE_HEIGHT, network.IMAGE_WIDTH))
	inputs, targets = dataset.next_batch()

	seq = iaa.Sequential([
		iaa.Crop(px=(0, 16), name="Crop"),
		iaa.Fliplr(0.5, name="Flip"),
		iaa.GaussianBlur(sigma=(0, 3.0), name="GaussianBlur")
	])

	def activator_heatmaps(images, augmenter, parents, default):
		if augmenter.name in ["GaussianBlur"]:
			return False
		else:
			return default

	hooks_heatmaps = ia.HooksImages(activator_heatmaps)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		summary_writer = tf.summary.FileWriter('{}/{}-{}'.format('logs', network.description, timestamp),
			graph=tf.get_default_graph())

		saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

		test_accuracies = []
		n_epochs = 60
		global_start = time.time()

		for epoch_i in range(n_epochs):
			dataset.reset_batch_pointer()

			for batch_i in range(dataset.num_batches_in_epoch()):
				batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1
				start = time.time()

				batch_inputs, batch_targets = dataset.next_batch()
				batch_inputs = np.reshape(batch_inputs, 
					(dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS))
				batch_targets = np.reshape(batch_targets,
					(dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))

				seq_det = seq.to_deterministic()
				batch_inputs = seq_det.augment_images(batch_inputs)
				batch_targets = seq_det.augment_images(batch_targets, hooks=hooks_heatmaps)

				batch_inputs = np.multiply(batch_inputs, 1.0 / 255)

				cost, _ = sess.run([network.cost, network.train_op], 
					feed_dict={network.inputs: batch_inputs, network.targets: batch_targets})

				end = time.time()
				print colored('{}/{}, epoch: {}, cost: {}, batch time: {}'.format(batch_num,
					n_epochs * dataset.num_batches_in_epoch(),
					epoch_i, cost, end - start), 'yellow')

				if batch_num % EVALUATE_EVERY_N_STEPS == 0 or batch_num == dataset.num_batches_in_epoch() * n_epochs:
					test_inputs, test_targets = dataset.get_test_set()
					test_inputs = test_inputs[:5]
					test_targets = test_targets[:5]

					test_inputs = np.reshape(test_inputs, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS))
					test_targets = np.reshape(test_targets, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))

					test_inputs = np.multiply(test_inputs, 1.0 / 255)

					summary, test_accuracy = sess.run([network.summaries, network.accuracy],
						feed_dict={network.inputs: test_inputs, network.targets: test_targets})

					summary_writer.add_summary(summary, batch_num)

					print colored('Step {}, test accuracy: {}'.format(batch_num, test_accuracy), 'blue')
					test_accuracies.append((test_accuracy, batch_num))
					print colored("Accuracies in time: {}".format([test_accuracies[x][0] for x in range(len(test_accuracies))]), 'blue')
					max_acc = max(test_accuracies)
					print colored("Best accuracy: {} in batch {}".format(max_acc[0], max_acc[1]), 'blue')
					print colored("Total time: {}".format(time.time() - global_start), 'blue')

					n_examples = 3
					test_inputs, test_targets = dataset.test_inputs[:n_examples], dataset.test_targets[:n_examples]
					test_inputs = np.multiply(test_inputs, 1.0 / 255)

					test_segmentation = sess.run(network.segmentation_result, feed_dict={
												network.inputs: np.reshape(test_inputs,
												[n_examples, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS])})

					test_plot_buf = draw_results(test_inputs, test_targets, test_segmentation, test_accuracy, network,
												batch_num, timestamp)

					image = tf.image.decode_png(test_plot_buf.getvalue(), channels=4)

					image = tf.expand_dims(image, 0)

					image_summary_op = tf.summary.image("plot", image)

					image_summary = sess.run(image_summary_op)
					summary_writer.add_summary(image_summary)

					if test_accuracy >= max_acc[0]:
						checkpoint_path = os.path.join('save', network.description, timestamp, 'model.ckpt')
						saver.save(sess, checkpoint_path, global_step=batch_num)


def draw_results(test_inputs, test_targets, test_segmentation, test_accuracy, network, batch_num, timestamp):
    n_examples_to_plot = 3
    fig, axs = plt.subplots(4, n_examples_to_plot, figsize=(n_examples_to_plot * 3, 10))
    fig.suptitle("Accuracy: {}, {}".format(test_accuracy, network.description), fontsize=20)
    for example_i in range(n_examples_to_plot):
        axs[0][example_i].imshow(test_inputs[example_i].astype(np.float32))
        axs[1][example_i].imshow(test_targets[example_i].astype(np.float32), cmap='gray')
        axs[2][example_i].imshow(
            np.reshape(test_segmentation[example_i].astype(np.float32), [network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS]),
            cmap='gray')

        test_image_thresholded = np.array(
            [0 if x < 0.5 else 255 for x in test_segmentation[example_i].flatten()])
        axs[3][example_i].imshow(
            np.reshape(test_image_thresholded.astype(np.float32), [network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS]),
            cmap='gray')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    IMAGE_PLOT_DIR = os.path.join('image_plots', timestamp)
    if not os.path.exists(IMAGE_PLOT_DIR):
        os.makedirs(IMAGE_PLOT_DIR)

    plt.savefig('{}/figure{}.jpg'.format(IMAGE_PLOT_DIR, batch_num))
    return buf


if __name__ == "__main__":
	train()