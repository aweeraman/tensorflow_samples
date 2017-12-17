import tensorflow as tf
import numpy as np

def model_fn(features, labels, mode):
	W = tf.get_variable("W", [1], dtype=tf.float64)
	b = tf.get_variable("b", [1], dtype=tf.float64)
	linear_model = W*features['X'] + b
	squared_error = tf.square(linear_model - labels)
	error_sum = tf.reduce_sum(squared_error)

	global_step = tf.train.get_global_step()
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train = tf.group(optimizer.minimize(error_sum),
		tf.assign_add(global_step, 1))

	return tf.estimator.EstimatorSpec(mode=mode, predictions=linear_model,
		loss=error_sum, train_op=train)

feature_columns = [tf.feature_column.numeric_column("X", shape=[1])]
estimator = tf.estimator.Estimator(model_fn=model_fn)

X_train = np.array([1., 2., 3., 4.])
y_train = np.array([-1., -2., -3., -4.])
X_val = np.array([3.1, 5., 7.01, 9.])
y_val = np.array([-3., -5., -7., -9.])

input_fn = tf.estimator.inputs.numpy_input_fn(
	{"X": X_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
	{"X": X_train}, y_train, batch_size=4, num_epochs=1000, shuffle=True)

val_input_fn = tf.estimator.inputs.numpy_input_fn(
	{"X": X_train}, y_train, batch_size=4, num_epochs=1000, shuffle=True)

estimator.train(input_fn=input_fn, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_fn)
validation_metrics = estimator.evaluate(input_fn=val_input_fn)
print("train metrics: %r" % train_metrics)
print("validation metrics: %r" % validation_metrics)
