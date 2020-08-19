import tensorflow as tf
import numpy as np
import time
import math

from absl import flags
from absl import app
from absl import logging

# flags
flags.DEFINE_string('dataset', 'mnist', 'The name of the dataset to use')
flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')

flags.DEFINE_string('train_dir', '/tmp/train_dir', 'Where model ckpt are saved')
flags.DEFINE_string('teachers_dir', '/tmp/train_dir', 'Directory where teachers checkpoints are stored.')

flags.DEFINE_integer('max_steps', 3000, 'Number of training steps to run.')
flags.DEFINE_integer('nb_teachers', 50, 'Teachers in the ensemble.')
flags.DEFINE_integer('stdnt_share', 1000,
                     'Student share (last index) of the test data')

flags.DEFINE_integer('batch_size', 128, 'Batch size')

flags.DEFINE_float('learning_rate', 0.05, 'Learning rate for training')

flags.DEFINE_float('lap_location', 0.0, 'Location for Laplacian noise')
flags.DEFINE_float(
    'lap_scale', 10.0, 'Scale of the Laplacian noise added for privacy')

FLAGS = flags.FLAGS


# model subclassing API
class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(
        filters = 64, kernel_size = 5, strides = 1, padding='same', activation='relu')
    self.pool1 = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')
    self.conv2 = tf.keras.layers.Conv2D(
        filters = 128, kernel_size = 5, strides = 1, padding='same', activation='relu')
    self.pool2 = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')
    self.flat = tf.keras.layers.Flatten()
    self.d1 = tf.keras.layers.Dense(384, activation = 'relu')
    self.d2 = tf.keras.layers.Dense(192, activation = 'relu')
    self.d3 = tf.keras.layers.Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.pool1(x)
    x = tf.nn.local_response_normalization(x, depth_radius = 4, bias = 1, alpha = 0.001/9.0, beta = 0.75)
    x = self.conv2(x)
    x = tf.nn.local_response_normalization(x, depth_radius = 4, bias = 1, alpha = 0.001/9.0, beta = 0.75)
    x = self.pool2(x)
    x = self.flat(x)
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    return x

  def model(self):
    x = tf.keras.Input(shape=(28, 28, 1))
    return tf.keras.Model(inputs=[x], outputs=self.call(x))


@tf.function
def predict_step(images, model):
  predictions = model(images, training=False)
  return predictions


def ensemble_preds(inputs, nb_teachers, train_dir):
  agg_preds = tf.zeros(
      shape = (inputs.shape[0], FLAGS.nb_labels), dtype=tf.dtypes.float32, name=None)

  model = MyModel()

  # Load the weights from each trained teacher and make predictions
  for teacher_id in range(nb_teachers):
    checkpoint_path = train_dir + '/' + 'mnist' + '_' + \
        str(nb_teachers) + '_teachers_' + str(teacher_id) + ".ckpt"
    model.load_weights(checkpoint_path)

    max_ind = tf.argmax(predict_step(inputs, model), 1)
    agg_preds = tf.math.add(agg_preds, tf.one_hot(max_ind, depth=FLAGS.nb_labels))

  agg_preds_numpy = agg_preds.numpy()

  # Aggregate the predictions from the teachers and add Laplacian noise
  for i in range(len(agg_preds_numpy)):
    for j in range(len(agg_preds_numpy[0])):
      agg_preds_numpy[i][j] += np.random.laplace(
          loc=FLAGS.lap_location, scale=float(FLAGS.lap_scale))

  # Select the highest "voted" prediction as the label
  max_ind_agg = tf.cast(tf.argmax(agg_preds_numpy, 1), 'uint8').numpy()
  return max_ind_agg


def create_data_set():
  (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

  # Make sure there is data leftover to be used as a test set
  assert FLAGS.stdnt_share < len(X_test)

  # Preprocess data
  X_test = X_test / 255.0
  X_test = X_test.reshape(
      X_test.shape[0], X_test.shape[1], X_test.shape[2], -1).astype("float32")

  # Split data into data used for training student and for the test set
  stdnt_X_train = X_test[:FLAGS.stdnt_share]
  stdnt_y_train = y_test[:FLAGS.stdnt_share]

  stdnt_X_test = X_test[FLAGS.stdnt_share:]
  stdnt_y_test = y_test[FLAGS.stdnt_share:]

  # Creating labels for the training data using teacher ensembles with noise
  stdnt_y_ensemb = ensemble_preds(
      stdnt_X_train, FLAGS.nb_teachers, FLAGS.train_dir)

  train_ds = tf.data.Dataset.from_tensor_slices(
      (stdnt_X_train, stdnt_y_ensemb)).shuffle(FLAGS.stdnt_share).batch(FLAGS.batch_size)
  test_ds = tf.data.Dataset.from_tensor_slices(
      (stdnt_X_test, stdnt_y_test)).batch(len(stdnt_y_test))

  return train_ds, test_ds


@tf.function
def loss_fn(y_true, y_pred):
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  return loss(y_true, y_pred)


@tf.function
def train_step(images, labels, model, optimizer, train_loss, train_accuracy):
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_fn(labels, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels, model, test_loss, test_accuracy):
  predictions = model(images, training=False)
  loss = loss_fn(labels, predictions)

  test_loss(loss)
  test_accuracy(labels, predictions)



def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  train_ds, test_ds = create_data_set()

  student_model = MyModel()
  optimizer = tf.keras.optimizers.SGD(FLAGS.learning_rate)

  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')

  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')

  epochs = min(100, math.floor(FLAGS.max_steps / math.ceil(FLAGS.stdnt_share / FLAGS.batch_size)))
  for epoch in range(1, epochs + 1):
    start_time = time.time()
    # Train the model for one epoch.
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
      train_step(images, labels, student_model,
                 optimizer, train_loss, train_accuracy)

    end_time = time.time()
    logging.info(f"Epoch {epoch} time in seconds: {end_time - start_time}")

    for test_images, test_labels in test_ds:
      test_step(test_images, test_labels,
                student_model, test_loss, test_accuracy)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss {}, Test Accuracy: {}'
    print(template.format(epoch,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))


if __name__ == '__main__':
  app.run(main)
