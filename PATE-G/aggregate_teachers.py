import tensorflow as tf
import numpy as np

from absl import flags
from absl import app
from absl import logging

flags.DEFINE_string('train_dir', '/tmp/train_dir',
                    'Where model ckpt are saved')

flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')

flags.DEFINE_integer('nb_teachers', 50, 'Teachers in the ensemble.')
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


def accuracy_fn(y_true, y_pred):
  return len([i for i, j in zip(y_true, y_pred) if i == j]) / len(y_true)


@tf.function
def predict_step(images, model):
  predictions = model(images, training=False)
  return predictions


def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  model = MyModel()

  # Only need test data
  (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
  X_test = X_test / 255.0
  X_test = X_test.reshape(
      X_test.shape[0], X_test.shape[1], X_test.shape[2], -1).astype("float32")

  # Create a tensor of zeros
  agg_preds = tf.zeros(
      shape=(X_test.shape[0], FLAGS.nb_labels), dtype=tf.dtypes.float32, name=None)

  for teacher_id in range(FLAGS.nb_teachers):
    checkpoint_path = FLAGS.train_dir + '/' + 'mnist' + '_' + \
        str(FLAGS.nb_teachers) + '_teachers_' + str(teacher_id) + ".ckpt"
    model.load_weights(checkpoint_path)

    max_ind = tf.argmax(predict_step(X_test, model), 1)
    agg_preds = tf.math.add(agg_preds, tf.one_hot(max_ind, depth=FLAGS.nb_labels))

  agg_preds_numpy = agg_preds.numpy()

  for i in range(len(agg_preds_numpy)):
    for j in range(len(agg_preds_numpy[0])):
      agg_preds_numpy[i][j] += np.random.laplace(
          loc=FLAGS.lap_location, scale=float(FLAGS.lap_scale))

  max_ind_agg = tf.cast(tf.argmax(agg_preds_numpy, 1), 'uint8').numpy()

  test_acc = accuracy_fn(y_test, max_ind_agg)
  print(test_acc)


if __name__ == '__main__':
  app.run(main)
