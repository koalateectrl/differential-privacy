import tensorflow as tf
import numpy as np
import time

from absl import flags
from absl import app
from absl import logging


def load_mnist():
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

	X_train = X_train / 255.0
	X_test = X_test / 255.0

	X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], -1).astype("float32")
	X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], -1).astype("float32")

	return X_train, y_train, X_test, y_test


def get_train_cycle(X_train, y_train, idx_start, idx_end, batch_size):
	cycle_X_train = X_train[idx_start:idx_end]
	cycle_y_train = y_train[idx_start:idx_end]
	train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(idx_end - idx_start).batch(batch_size)
	return train_ds


class AggModel(tf.keras.Model):
	def __init__(self):
		super(AggModel, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(16, 8, strides = 2, padding = 'same', activation = 'relu')
		self.pool1 = tf.keras.layers.MaxPool2D(2, 1)
		self.conv2 = tf.keras.layers.Conv2D(32, 4, strides = 2, padding = 'valid', activation = 'relu')
		self.pool2 = tf.keras.layers.MaxPool2D(2, 1)
		self.flat = tf.keras.layers.Flatten()
		self.d1 = tf.keras.layers.Dense(32, activation = 'relu')
		self.d2 = tf.keras.layers.Dense(10)

	def call(self, x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = self.flat(x)
		x = self.d1(x)
		x = self.d2(x)
		return x

	def model(self, input_shape = (28, 28, 1)):
		x = tf.keras.Input(shape = input_shape)
		return tf.keras.Model(inputs = x, outputs = self.call(x))

@tf.function
def loss_fn(y_true, y_pred):
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
	return loss(y_true, y_pred)

@tf.function
def train_step(images, labels, model, optimizer, train_loss, train_accuracy):
	with tf.GradientTape() as tape:
		predictions = model(images, training = True)
		loss = loss_fn(labels, predictions)

	grads = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))

	train_loss(loss)
	train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels, model, test_loss, test_accuracy):
	predictions = model(images, training = False)
	loss = loss_fn(labels, predictions)

	test_loss(loss)
	test_accuracy(labels, predictions)


X_train, y_train, X_test, y_test = load_mnist()

agg_model = AggModel()
optimizer = tf.keras.optimizers.Adam()
num_partitions = 10
num_epoch_list = [3, 2, 1, 1, 1, 2, 3, 1, 1, 1]
size_partition = len(X_train // num_partitions)
batch_size = 256


train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')


test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)


for i in range(num_partitions):
	print("Partition {} of {}:".format(i + 1, num_partitions))
	train_ds = get_train_cycle(X_train, y_train, size_partition * i, size_partition * (i + 1), batch_size)
	num_epochs = num_epoch_list[i]

	for epoch in range(1, num_epochs + 1):
		start_time = time.time()

		train_loss.reset_states()
		train_accuracy.reset_states()
		test_loss.reset_states()
		test_accuracy.reset_states()

		for images, labels in train_ds:
			train_step(images, labels, agg_model, optimizer, train_loss, train_accuracy)

		end_time = time.time()
		logging.info(f"Epoch {epoch} time in seconds: {end_time - start_time}")

		for test_images, test_labels in test_ds:
			test_step(test_images, test_labels, agg_model, test_loss, test_accuracy)

		template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss {}, Test Accuracy: {}'
		print(template.format(epoch,
			train_loss.result(),
			train_accuracy.result() * 100,
			test_loss.result(),
			test_accuracy.result() * 100))



























