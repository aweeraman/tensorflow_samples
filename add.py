import tensorflow as tf
node1 = tf.constant(3.0)
node2 = tf.constant(5.0, dtype=tf.float32)
print(node1, node2)
sess = tf.Session()
node3 = tf.add(node1, node2)
print(sess.run(node3))
