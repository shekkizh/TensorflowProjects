from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_data
mnist = mnist_data.read_data_sets("MNIST_data/", one_hot = True)

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784], name = "X_input")

W = tf.Variable(tf.zeros([784, 10]), name = 'W')
b = tf.Variable(tf.zeros([10]), name = 'b')

W_hist = tf.histogram_summary("W", W)
b_hist = tf.histogram_summary("b", b)

y = tf.nn.softmax(tf.matmul(x,W) + b)

yactual = tf.placeholder(tf.float32, [None, 10])
entropy = -tf.reduce_sum(yactual*tf.log(y))
tf.scalar_summary("Entropy", entropy)
train = tf.train.GradientDescentOptimizer(0.01).minimize(entropy)

summary_op = tf.merge_all_summaries()
summaryWriter = tf.train.SummaryWriter("logs/mnist_logs/")

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in xrange(1000):
	batch_x, batch_y = mnist.train.next_batch(100)
        feed={x:batch_x, yactual:batch_y}	
	sess.run(train, feed_dict = feed)
	if i%10 == 0:
	        summary_str, entropy_val = sess.run([summary_op, entropy], feed_dict = feed)
	        summaryWriter.add_summary(summary_str, i)
		print (i, entropy_val)

pred = tf.equal(tf.argmax(y,1),tf.argmax(yactual,1))
accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

print sess.run(accuracy, feed_dict = {x:mnist.test.images, yactual:mnist.test.labels})
