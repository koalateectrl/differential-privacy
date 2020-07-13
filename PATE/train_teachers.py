import tensorflow as tf
import numpy as np
import time

from absl import flags
from absl import app
from absl import logging

flags.DEFINE_string('dataset', 'mnist', 'The name of the dataset to use')
flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')

flags.DEFINE_string('data_dir', '/tmp/', 'Temporary storage')
flags.DEFINE_string('train_dir', '/tmp/train_dir', 'Where model ckpt are saved')

flags.DEFINE_integer('max_steps', 3000, 'Number of training steps to run.')
flags.DEFINE_integer('nb_teachers', 50, 'Teachers in the ensemble.')
flags.DEFINE_integer('teacher_id', 0, 'ID of teacher being trained.')
flags.DEFINE_integer('batch_size', 128, 'Batch size')

flags.DEFINE_float('learning_rate', 0.05, 'Learning rate for training')
flags.DEFINE_integer('epochs', 50, 'Number of epochs')

FLAGS = flags.FLAGS


def partition_dataset(data, labels, nb_teachers, teacher_id):
	#Sanity check
	assert len(data) == len(labels)
	assert int(teacher_id) < int(nb_teachers)

	#This will floor the possible number of batches
	batch_len = int(len(data) / nb_teachers)

	#Compute start, end indices of partition
	start = teacher_id * batch_len
	end = (teacher_id + 1) * batch_len

	#Slice partition off
	partition_data = data[start:end]
	partition_labels = labels[start:end]

	return partition_data, partition_labels

def create_data_set():
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
	X_train = X_train / 255.0
	X_test = X_test / 255.0

	X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], -1).astype("float32")
	X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], -1).astype("float32")

	X_train, y_train = partition_dataset(X_train, y_train, FLAGS.nb_teachers, FLAGS.teacher_id)

	train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(60000).batch(FLAGS.batch_size)
	test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(FLAGS.batch_size)
	return train_ds, test_ds


#model subclassing API
class MyModel(tf.keras.Model):
	def __init__(self):
		super(MyModel, self).__init__()
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

	def model(self):
		x = tf.keras.Input(shape = (28, 28, 1))
		return tf.keras.Model(inputs = [x], outputs = self.call(x))

@tf.function
def loss_fn(y_true, y_pred):
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
	# vector_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = tf.keras.losses.Reduction.NONE)
	return loss(y_true, y_pred)

@tf.function
def train_step(images, labels, model, optimizer, train_loss, train_accuracy):
	with tf.GradientTape() as tape:
		predictions = model(images, training = True)
		loss = loss_fn(labels, predictions)
		# print(loss)
		# print(loss.shape)

		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		train_loss(loss)
		train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels, model, test_loss, test_accuracy):
	predictions = model(images, training = False)
	loss = loss_fn(labels, predictions)
	# scalar_loss = tf.reduce_mean(input_tensor=vector_loss)

	test_loss(loss)
	test_accuracy(labels, predictions)


def main(unused_argv):
	logging.set_verbosity(logging.INFO)
	train_ds, test_ds = create_data_set()

	filename = str(FLAGS.nb_teachers) + '_teachers_' + str(FLAGS.teacher_id) + '.ckpt'
	ckpt_path = FLAGS.train_dir + '/' + str(FLAGS.dataset) + '_' + filename

	#Instantiate the Model.
	model = MyModel()
	optimizer = tf.keras.optimizers.SGD(FLAGS.learning_rate)

	train_loss = tf.keras.metrics.Mean(name = 'train_loss')
	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

	test_loss = tf.keras.metrics.Mean(name = 'test_loss')
	test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')

	#Training loop.
	#****It appears they shuffle the training set each time they select a mini-batch. We go through the list sequentially
	#****I think they are picking a random group from the training set as val. We will do the same.
	for epoch in range(1, FLAGS.epochs + 1):
		start_time = time.time()
		#Train the model for one epoch.
		train_loss.reset_states()
		train_accuracy.reset_states()
		test_loss.reset_states()
		test_accuracy.reset_states()

		for images, labels in train_ds:
			train_step(images, labels, model, optimizer, train_loss, train_accuracy)

		end_time = time.time()
		logging.info(f"Epoch {epoch} time in seconds: {end_time - start_time}")

		model.save_weights(ckpt_path)

		for test_images, test_labels in test_ds:
			test_step(test_images, test_labels, model, test_loss, test_accuracy)

		template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss {}, Test Accuracy: {}'
		print(template.format(epoch,
			train_loss.result(),
			train_accuracy.result() * 100,
			test_loss.result(),
			test_accuracy.result() * 100))

if __name__ == '__main__':
	app.run(main)

# class MyModel(tf.keras.Model):
# 	def __init__(self):
# 		super(MyModel, self).__init__()
# 		#add weight decay?
# 		self.conv1 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (5, 5), padding = 'same', activation = 'relu')
# 		self.pool1 = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = (1, 1), padding = 'same')
# 		self.norm1 = tf.keras.layers.BatchNormalization()
# 		self.conv2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (5, 5), padding = 'same', activation = 'relu')
# 		self.norm2 = tf.keras.layers.BatchNormalization()
# 		self.pool2 = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = (1, 1), padding = 'same')
# 		self.flat = tf.keras.layers.Flatten()
# 		self.dense1 = tf.keras.layers.Dense(384, activation = 'relu')
# 		self.dense2 = tf.keras.layers.Dense(192, activation = 'relu')
# 		self.dense3 = tf.keras.layers.Dense(10)

# 	def call(self, x):
# 		x = self.conv1(x)
# 		x = self.pool1(x)
# 		x = self.norm1(x)
# 		x = self.conv2(x)
# 		x = self.norm2(x)
# 		x = self.pool2(x)
# 		x = self.flat(x)
# 		x = self.dense1(x)
# 		x = self.dense2(x)
# 		x = self.dense3(x)
# 		return x

# 	def model(self):
# 		x = tf.keras.Input(shape = (28, 28, 1))
# 		return tf.keras.Model(inputs = [x], outputs = self.call(x))






























