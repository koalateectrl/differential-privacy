import tensorflow as tf
import time

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], -1).astype("float32")
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], -1).astype("float32")

total_train_size = X_train.shape[0]
num_partitions = 10
batch_size = 256
data_per_partition = total_train_size // num_partitions

partition_X_train = []
partition_y_train = []
for num_part in range(num_partitions):
	partition_X_train.append(X_train[num_part * data_per_partition: (num_part + 1) * data_per_partition])
	partition_y_train.append(y_train[num_part * data_per_partition: (num_part + 1) * data_per_partition])


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

	def model(self, input_shape = (28, 28, 1)):
		x = tf.keras.Input(shape = input_shape)
		return tf.keras.Model(inputs = [x], outputs = self.call(x))



def loss_fn(y_true, y_pred):
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
	return loss(y_true, y_pred)

#@tf.function
def train_step(images, labels, model, optimizer, train_loss, train_acc, first_step = False):
	with tf.GradientTape() as tape:
		predictions = model(images, training = True)
		loss = loss_fn(labels, predictions)

	if first_step:
		init_weights = model.get_weights()
	else:
		init_weights = None

	grads = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))

	train_loss(loss)
	train_acc(labels, predictions)

	return init_weights

@tf.function
def test_step(images, labels, model, test_loss, test_acc):
	predictions = model(images, training = False)
	loss = loss_fn(labels, predictions)

	test_loss(loss)
	test_acc(labels, predictions)



agg_epochs = 1
optimizer = tf.keras.optimizers.Adam(lr = 0.1)
agg_model = MyModel()
first_step = False

train_loss = tf.keras.metrics.Mean()
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()

test_loss = tf.keras.metrics.Mean()
test_acc = tf.keras.metrics.SparseCategoricalAccuracy()

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

for agg_epochs in range(1, agg_epochs + 1):

	weights_dict = {}

	start_time = time.time()

	train_loss.reset_states()
	train_acc.reset_states()
	test_loss.reset_states()
	test_acc.reset_states()

	#get the initial weights from the model
	init_weights = agg_model.get_weights()

	if init_weights == []:
		first_step = True
	#loop through each federated dataset
	for num_part in range(num_partitions):
		print(num_part)

		local_model = agg_model
		train_ds = tf.data.Dataset.from_tensor_slices((partition_X_train[num_part], partition_y_train[num_part])).shuffle(data_per_partition).batch(batch_size)
		for images, labels in train_ds:

			tmp_init_weights = train_step(images, labels, local_model, optimizer, train_loss, train_acc, first_step)
			first_step = False
			#first pass getting model weights is empty
			if tmp_init_weights:
				init_weights = tmp_init_weights
				local_model = agg_model

		#get the weights after training on one federated dataset
		end_weights = local_model.get_weights()

		#create the difference in weights for one full run of a federated dataset
		diff_weights = init_weights
		for layer in range(len(init_weights)):
			diff_weights[layer] = end_weights[layer] - init_weights[layer]

		n_c = partition_X_train[num_part].shape[0]

		weights_dict[num_part] = ([diff_weights, n_c])

	#TODO: Average out the weights with a weighted average by partition size



	#is it applying the new diff weights?
	optimizer.apply_gradients(zip(avg_weights, agg_model.trainable_variables))




	# end_time = time.time()
	# print(f"Epoch {epoch} time in seconds: {end_time - start_time}")

	# for test_images, test_labels in test_ds:
	# 	test_step(test_images, test_labels, model, test_loss, test_acc)

	# template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss {}, Test Accuracy: {}'
	# print(template.format(epoch,
	# 	train_loss.result(),
	# 	train_acc.result() * 100,
	# 	test_loss.result(),
	# 	test_acc.result() * 100))














