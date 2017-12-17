import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

MNIST_DIR = "/home/anuradha/wintermute/datasets/MNIST/"
LOG_DIR = "/tmp/summaries"

mnist = input_data.read_data_sets(MNIST_DIR, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)
l_h = tf.summary.scalar("loss", cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
summaries = tf.summary.merge_all()
sess.run(init)

for step in range(1000):
	batch_xs, batch_yx = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_yx})

	summ = sess.run(summaries, feed_dict={x: batch_xs, y_: batch_yx})
	writer.add_summary(summ, global_step=step)

correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
