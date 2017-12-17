import tensorflow as tf
import numpy as np

feature_columns = [tf.feature_column.numeric_column("X", shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

X_train = np.array([1, 2, 3, 4])
y_train = np.array([-1, -2, -3, -4])
X_val = np.array([3.1, 5, 7.01, 9])
y_val = np.array([-3, -5, -7, -9])

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
