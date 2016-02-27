from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

yactual = tf.placeholder(tf.float32, [None, 10])
entropy = -tf.reduce_sum(yactual*tf.log(y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(entropy)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in xrange(1000):
	batch_x, batch_y = mnist.train.next_batch(100)
	sess.run(train, feed_dict={x:batch_x, yactual:batch_y})
#	if i%100 == 0:
#		print(i, sess.run(entropy))

pred = tf.equal(tf.argmax(y,1),tf.argmax(yactual,1))
accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

print sess.run(accuracy, feed_dict = {x:mnist.test.images, yactual:mnist.test.labels})
