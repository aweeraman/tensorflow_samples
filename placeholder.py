import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.add(a, b)
d = a + b # same as above
e = d * 2

sess = tf.Session()
f = sess.run(c, {a: 3, b: 5})
g = sess.run(d, {a: [2, 4], b: [1, 3]})
h = sess.run(e, {a: 3, b: 3})

print(f)
print(g)
print(h)
