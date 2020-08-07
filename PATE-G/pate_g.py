import tensorflow as tf
import numpy as np
import time

from absl import flags
from absl import app
from absl import logging

# flags
flags.DEFINE_string('dataset', 'mnist', 'The name of the dataset to use')
flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')

flags.DEFINE_string('train_dir', '/tmp/train_dir',
                    'Where model ckpt are saved')

flags.DEFINE_integer('max_steps', 3000, 'Number of training steps to run.')
flags.DEFINE_integer('nb_teachers', 50, 'Teachers in the ensemble.')
flags.DEFINE_integer('batch_size', 128, 'Batch size')

flags.DEFINE_float('learning_rate', 0.05, 'Learning rate for training')
flags.DEFINE_integer('epochs', 50, 'Number of epochs')

flags.DEFINE_integer('stdnt_share', 1000,
                     'Student share (last index) of the test data')

flags.DEFINE_float('lap_location', 0.0, 'Location for Laplacian noise')
flags.DEFINE_float(
    'lap_scale', 10.0, 'Scale of the Laplacian noise added for privacy')
flags.DEFINE_integer('depth', 10, 'Number of Classes')
flags.DEFINE_integer('gen_dim', 100, 'Dimension of Generative Input Vector')

FLAGS = flags.FLAGS


# generate input vector for generator
def generate_input_vector(nb_samples, input_dim = 100):
  norm_input = tf.random.normal([nb_samples, 100])
  return norm_input

#Functional API
def make_generator_model(latent_dim = 100):
  gen_inputs = tf.keras.layers.Input(shape = (latent_dim, ))
  
  dense1 = tf.keras.layers.Dense(128 * 7 * 7)(gen_inputs)
  leaky1 = tf.keras.layers.LeakyReLU(alpha = 0.2)(dense1)
  reshape = tf.keras.layers.Reshape((7, 7, 128))(leaky1)

  convt1 = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides = (2,2), padding = 'same', use_bias = False)(reshape)
  leaky2 = tf.keras.layers.LeakyReLU(alpha = 0.2)(convt1)

  convt2 = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides = (2,2), padding = 'same', use_bias = False)(leaky2)
  leaky3 = tf.keras.layers.LeakyReLU(alpha = 0.2)(convt2)

  g_out_layer = tf.keras.layers.Conv2D(1, (7, 7), padding = 'same', use_bias = False, activation = 'tanh')(leaky3)

  g_model = tf.keras.Model(gen_inputs, g_out_layer)
  return g_model


#Activation to go from softmax preds to logistic
def custom_activation(output):
  logexpsum = tf.keras.backend.sum(tf.keras.backend.exp(output), axis = -1, keepdims = True)
  result = logexpsum / (logexpsum + 1.0)
  return result


#Functional API
def make_discriminator_models(input_shape = (28, 28, 1), n_classes = 10):
  img_inputs = tf.keras.layers.Input(shape = input_shape)
  conv1 = tf.keras.layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same')(img_inputs)
  leaky1 = tf.keras.layers.LeakyReLU(alpha = 0.2)(conv1)

  conv2 = tf.keras.layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same')(leaky1)
  leaky2 = tf.keras.layers.LeakyReLU(alpha = 0.2)(conv2)

  conv3 = tf.keras.layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same')(leaky2)
  leaky3 = tf.keras.layers.LeakyReLU(alpha = 0.2)(conv3)

  conv4 = tf.keras.layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same')(leaky3)
  leaky4 = tf.keras.layers.LeakyReLU(alpha = 0.2)(conv4)

  flat = tf.keras.layers.Flatten()(leaky4)
  drop = tf.keras.layers.Dropout(0.4)(flat)
  dense = tf.keras.layers.Dense(n_classes)(drop)

  s_out_layer = tf.keras.layers.Activation('softmax')(dense)
  s_model = tf.keras.Model(img_inputs, s_out_layer)

  d_out_layer = tf.keras.layers.Lambda(custom_activation)(dense)
  d_model = tf.keras.Model(img_inputs, d_out_layer)

  return s_model, d_model


# model subclassing API
class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(
        16, 8, strides=2, padding='same', activation='relu')
    self.pool1 = tf.keras.layers.MaxPool2D(2, 1)
    self.conv2 = tf.keras.layers.Conv2D(
        32, 4, strides=2, padding='valid', activation='relu')
    self.pool2 = tf.keras.layers.MaxPool2D(2, 1)
    self.flat = tf.keras.layers.Flatten()
    self.d1 = tf.keras.layers.Dense(32, activation='relu')
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
    x = tf.keras.Input(shape=(28, 28, 1))
    return tf.keras.Model(inputs=[x], outputs=self.call(x))


@tf.function
def predict_step(images, model):
  predictions = model(images, training=False)
  return predictions


def ensemble_preds(inputs, nb_teachers, train_dir):
  agg_preds = tf.zeros(
      shape=(inputs.shape[0], FLAGS.depth), dtype=tf.dtypes.float32, name=None)

  model = MyModel()

  for teacher_id in range(nb_teachers):
    checkpoint_path = train_dir + '/' + 'mnist' + '_' + \
        str(nb_teachers) + '_teachers_' + str(teacher_id) + ".ckpt"
    model.load_weights(checkpoint_path)

    max_ind = tf.argmax(predict_step(inputs, model), 1)
    agg_preds = tf.math.add(agg_preds, tf.one_hot(max_ind, depth=FLAGS.depth))

  agg_preds_numpy = agg_preds.numpy()

  for i in range(len(agg_preds_numpy)):
    for j in range(len(agg_preds_numpy[0])):
      agg_preds_numpy[i][j] += np.random.laplace(
          loc=FLAGS.lap_location, scale=float(FLAGS.lap_scale))

  max_ind_agg = tf.cast(tf.argmax(agg_preds_numpy, 1), 'uint8').numpy()
  return max_ind_agg


def create_data_set():
  (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
  X_test = X_test / 255.0
  X_test = X_test.reshape(
      X_test.shape[0], X_test.shape[1], X_test.shape[2], -1).astype("float32")
  X_sup = X_test[:FLAGS.stdnt_share]
  y_sup = y_test[:FLAGS.stdnt_share]

  X_unsup = X_test[FLAGS.stdnt_share:9000]

  stdnt_X_test = X_test[9000:]
  stdnt_y_test = y_test[9000:]

  stdnt_y_ensemb = ensemble_preds(X_sup, FLAGS.nb_teachers, FLAGS.train_dir)

  supervised_ds = tf.data.Dataset.from_tensor_slices((X_sup, y_sup)).shuffle(FLAGS.stdnt_share).batch(FLAGS.batch_size)
  unsupervised_ds = tf.data.Dataset.from_tensor_slices(X_unsup).shuffle(9000 - FLAGS.stdnt_share).batch(FLAGS.batch_size)

  test_ds = tf.data.Dataset.from_tensor_slices((stdnt_X_test, stdnt_y_test)).batch(len(stdnt_y_test))

  return supervised_ds, unsupervised_ds, test_ds


def sup_loss(y_true, y_pred):
  cce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
  return cce_loss(y_true, y_pred)

def discriminator_loss(real_output, fake_output):
  cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

def generator_loss(fake_output):
  cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
  return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_sup_step(X, y, s_model, s_opt, class_loss_metric, class_acc_metric):
  with tf.GradientTape() as s_tape:
    sup_preds = s_model(X, training = True)
    supervised_loss = sup_loss(y, sup_preds)

  supervised_grads = s_tape.gradient(supervised_loss, s_model.trainable_variables)
  s_opt.apply_gradients(zip(supervised_grads, s_model.trainable_variables))

  class_loss_metric(supervised_loss)
  class_acc_metric(y, sup_preds)

@tf.function
def train_unsup_step(X_real, half_batch_size, gen_dim, d_model, d_opt, disc_loss_metric, g_model, g_opt, gen_loss_metric):
  X_gen = generate_input_vector(half_batch_size, gen_dim)

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_images = g_model(X_gen, training = True)

    real_output = d_model(X_real, training = True)
    fake_output = d_model(gen_images, training = True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

  gen_grads = gen_tape.gradient(gen_loss, g_model.trainable_variables)
  disc_grads = disc_tape.gradient(disc_loss, d_model.trainable_variables)

  g_opt.apply_gradients(zip(gen_grads, g_model.trainable_variables))
  d_opt.apply_gradients(zip(disc_grads, d_model.trainable_variables))

  disc_loss_metric(disc_loss)
  gen_loss_metric(gen_loss)


@tf.function
def test_step(X, y, s_model, test_loss_metric, test_acc_metric):
  test_preds = s_model(X, training = False)
  test_loss = sup_loss(y, test_preds)

  test_loss_metric(test_loss)
  test_acc_metric(y, test_preds)



def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  supervised_ds, unsupervised_ds, test_ds = create_data_set()

  s_model, d_model = make_discriminator_models()
  s_opt = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5)
  d_opt = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5)

  g_model = make_generator_model(FLAGS.gen_dim)
  g_opt = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5)
  
  class_loss_metric = tf.keras.metrics.Mean(name = 'classification_loss')
  class_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name = 'classification_accuracy')

  disc_loss_metric = tf.keras.metrics.Mean(name = 'discriminator_loss')
  gen_loss_metric = tf.keras.metrics.Mean(name = 'generator_loss')

  test_loss_metric = tf.keras.metrics.Mean(name = 'test_loss')
  test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')

  half_batch_size = FLAGS.batch_size // 2


  for epoch in range(1, FLAGS.epochs + 1):
    start_time = time.time()
    # Train the model for one epoch.
    print(epoch)

    class_loss_metric.reset_states()
    class_acc_metric.reset_states()
    disc_loss_metric.reset_states()
    gen_loss_metric.reset_states()
    test_loss_metric.reset_states()
    test_acc_metric.reset_states()

    for sup_images, sup_labels in supervised_ds:
      train_sup_step(sup_images, sup_labels, s_model, s_opt, class_loss_metric, class_acc_metric)


    # Train d_model/g_model (discriminator/generator)
    for unsup_images in unsupervised_ds:
      train_unsup_step(unsup_images, half_batch_size, FLAGS.gen_dim, d_model, d_opt, disc_loss_metric, g_model, g_opt, gen_loss_metric)

    end_time = time.time()
    logging.info(f"Epoch {epoch} time in seconds: {end_time - start_time}")

    for test_images, test_labels in test_ds:
      test_step(test_images, test_labels, s_model, test_loss_metric, test_acc_metric)

    template = 'Epoch {}, Classification Loss: {}, Classification Accuracy: {}, Discriminator Loss: {}, Generator Loss: {}, Test Loss {}, Test Accuracy: {}'
    print(template.format(epoch,
      class_loss_metric.result(),
      class_acc_metric.result() * 100,
      disc_loss_metric.result(),
      gen_loss_metric.result(),
      test_loss_metric.result(),
      test_acc_metric.result() * 100))


if __name__ == '__main__':
  app.run(main)
