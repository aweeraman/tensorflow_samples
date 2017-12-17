import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.add(a, b)
d = a + b # same as above

sess = tf.Session()
print(sess.run(c, {a: 3, b: 5}))
print(sess.run(d, {a: [2, 4], b: [1, 3]}))
