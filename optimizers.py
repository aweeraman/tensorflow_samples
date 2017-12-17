import tensorflow as tf

W = tf.Variable([.2], dtype=tf.float32)
b = tf.Variable([-.2], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W*x + b
squared_error = tf.square(linear_model - y)
error_sum = tf.reduce_sum(squared_error)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(error_sum)

for i in range(1000):
	sess.run(train, {x: [1, 2, 3, 4], y: [-1, -2, -3, -4]})

print(sess.run([W, b]))
