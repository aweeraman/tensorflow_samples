import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MNIST_DIR = "/home/anuradha/wintermute/datasets/MNIST/"
LOG_DIR = "/tmp/summaries"

epochs = 1000
learning_rate = 0.5
batch_size = 100

mnist = input_data.read_data_sets(MNIST_DIR, one_hot=True)

with tf.name_scope("linear_model") as scope:
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.nn.softmax(tf.matmul(x, W) + b)
  w_summary = tf.summary.histogram("weights", W)
  b_summary = tf.summary.histogram("biases", b)

with tf.name_scope("cost_function") as scope:
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  l_summary = tf.summary.scalar("loss", cross_entropy)

with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
  summaries = tf.summary.merge_all()

  for step in range(epochs):
    batch_xs, batch_yx = mnist.train.next_batch(batch_size)
    loss = sess.run(train_step, feed_dict={x: batch_xs, y_: batch_yx})

    summ = sess.run(summaries, feed_dict={x: batch_xs, y_: batch_yx})
    writer.add_summary(summ, global_step=step)

  correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
