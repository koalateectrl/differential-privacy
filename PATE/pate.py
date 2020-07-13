import tensorflow as tf
import numpy as np
import time

from absl import flags
from absl import app
from absl import logging


flags.DEFINE_string('dataset', 'mnist', 'The name of the dataset to use')
flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')

flags.DEFINE_string('train_dir', '/tmp/train_dir', 'Where model ckpt are saved')

flags.DEFINE_integer('max_steps', 3000, 'Number of training steps to run.')
flags.DEFINE_integer('nb_teachers', 50, 'Teachers in the ensemble.')
flags.DEFINE_integer('batch_size', 128, 'Batch size')

flags.DEFINE_float('learning_rate', 0.05, 'Learning rate for training')
flags.DEFINE_integer('epochs', 50, 'Number of epochs')

flags.DEFINE_integer('stdnt_share', 1000, 'Student share (last index) of the test data')

flags.DEFINE_float('lap_location', 0.0, 'Location for Laplacian noise')
flags.DEFINE_float('lap_scale', 10.0, 'Scale of the Laplacian noise added for privacy')
flags.DEFINE_integer('depth', 10, 'Number of Classes')

FLAGS = flags.FLAGS



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
def predict_step(images, model):
	predictions = model(images, training = False)
	return predictions

def ensemble_preds(inputs, nb_teachers, train_dir):
	agg_preds = tf.zeros(shape = (inputs.shape[0], FLAGS.depth), dtype=tf.dtypes.float32, name=None)

	model = MyModel()

	for teacher_id in range(nb_teachers):
		checkpoint_path = train_dir + '/' + 'mnist' + '_' + str(nb_teachers) + '_teachers_' + str(teacher_id) + ".ckpt"
		model.load_weights(checkpoint_path)

		max_ind = tf.argmax(predict_step(inputs, model),1)
		agg_preds = tf.math.add(agg_preds, tf.one_hot(max_ind, depth = FLAGS.depth))

	agg_preds_numpy = agg_preds.numpy()

	for i in range(len(agg_preds_numpy)):
		for j in range(len(agg_preds_numpy[0])):
			agg_preds_numpy[i][j] += np.random.laplace(loc=FLAGS.lap_location, scale=float(FLAGS.lap_scale))

	max_ind_agg = tf.cast(tf.argmax(agg_preds_numpy, 1), 'uint8').numpy()
	return max_ind_agg


def create_data_set():
	(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
	X_test = X_test / 255.0
	X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], -1).astype("float32")
	stdnt_X_train = X_test[:FLAGS.stdnt_share]
	stdnt_y_train = y_test[:FLAGS.stdnt_share]

	stdnt_X_test = X_test[FLAGS.stdnt_share:]
	stdnt_y_test = y_test[FLAGS.stdnt_share:]

	stdnt_y_ensemb = ensemble_preds(stdnt_X_train, FLAGS.nb_teachers, FLAGS.train_dir)

	train_ds = tf.data.Dataset.from_tensor_slices((stdnt_X_train, stdnt_y_ensemb)).shuffle(60000).batch(FLAGS.batch_size)
	test_ds = tf.data.Dataset.from_tensor_slices((stdnt_X_test, stdnt_y_test)).batch(len(stdnt_y_test))

	return train_ds, test_ds


@tf.function
def loss_fn(y_true, y_pred):
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
	return loss(y_true, y_pred)

@tf.function
def train_step(images, labels, model, optimizer, train_loss, train_accuracy):
	with tf.GradientTape() as tape:
		predictions = model(images, training = True)
		loss = loss_fn(labels, predictions)

		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		train_loss(loss)
		train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels, model, test_loss, test_accuracy):
	predictions = model(images, training = False)
	loss = loss_fn(labels, predictions)

	test_loss(loss)
	test_accuracy(labels, predictions)

stdnt_share = 1000


def main(unused_argv):
	logging.set_verbosity(logging.INFO)
	train_ds, test_ds = create_data_set()

	student_model = MyModel()
	optimizer = tf.keras.optimizers.SGD(FLAGS.learning_rate)

	train_loss = tf.keras.metrics.Mean(name = 'train_loss')
	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

	test_loss = tf.keras.metrics.Mean(name = 'test_loss')
	test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')

	for epoch in range(1, FLAGS.epochs + 1):
		start_time = time.time()
		#Train the model for one epoch.
		train_loss.reset_states()
		train_accuracy.reset_states()
		test_loss.reset_states()
		test_accuracy.reset_states()

		for images, labels in train_ds:
			train_step(images, labels, student_model, optimizer, train_loss, train_accuracy)

		end_time = time.time()
		logging.info(f"Epoch {epoch} time in seconds: {end_time - start_time}")

		for test_images, test_labels in test_ds:
			test_step(test_images, test_labels, student_model, test_loss, test_accuracy)

		template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss {}, Test Accuracy: {}'
		print(template.format(epoch,
			train_loss.result(),
			train_accuracy.result() * 100,
			test_loss.result(),
			test_accuracy.result() * 100))

if __name__ == '__main__':
	app.run(main)




















