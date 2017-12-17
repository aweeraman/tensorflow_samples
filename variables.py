import tensorflow as tf

W = tf.Variable([.2], dtype=tf.float32)
b = tf.Variable([-.2], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W*x + b

# Run the linear model
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# Calculate the loss
y = tf.placeholder(tf.float32)
squared_error = tf.square(linear_model - y)
error_sum = tf.reduce_sum(squared_error)
print(sess.run(error_sum, {x: [1, 2, 3, 4], y: [1.5, 2, 5, 6]}))

# Change the weights
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(error_sum, {x: [1, 2, 3, 4], y: [-1, -2, -3, -4]}))
