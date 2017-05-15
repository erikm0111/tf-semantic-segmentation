from math import ceil
import cv2
import os
import numpy as np
import math


class Dataset():
	
	def __init__(self, batch_size, folder="data128_128"):
		self.batch_size = batch_size
		files = os.listdir(os.path.join(folder, 'inputs'))
		num_files = len(files)

		train_files = files[:int(ceil(num_files * 0.7))]
		test_files = files[int(ceil(num_files * 0.3)):]

		self.train_inputs, self.train_targets = self.load_data(folder, train_files)
		self.test_inputs, self.test_targets = self.load_data(folder, test_files)

		self.pointer = 0


	def load_data(self, folder, files):
		inputs = []
		targets = []

		for file in files:
			input_file_path = os.path.join(folder, 'inputs', file)
			target_file_path = os.path.join(folder, 'targets', file)

			input_file = np.array(cv2.imread(input_file_path))
			inputs.append(input_file)

			target_file = cv2.imread(target_file_path, 0)
			target_file = cv2.threshold(target_file, 127, 1, cv2.THRESH_BINARY)[1]
			targets.append(target_file)

		return inputs, targets


	def next_batch(self):
		inputs = []
		targets = []

		for i in range(self.batch_size):
			inputs.append(np.array(self.train_inputs[self.pointer + i]))
			targets.append(np.array(self.train_targets[self.pointer + i]))

		self.pointer += self.batch_size

		return np.array(inputs, dtype=np.uint8), np.array(targets, dtype=np.uint8)


	def reset_batch_pointer(self):
		permutation = np.random.permutation(len(self.train_inputs))

		self.train_inputs = [self.train_inputs[i] for i in permutation]
		self.train_targets = [self.train_targets[i] for i in permutation]

		self.pointer = 0


	def get_test_set(self):
		return np.array(self.test_inputs, dtype=np.uint8), np.array(self.test_targets, dtype=np.uint8)


	def num_batches_in_epoch(self):
		return int(math.floor(len(self.train_inputs) / self.batch_size))