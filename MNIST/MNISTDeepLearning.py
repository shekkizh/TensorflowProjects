from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

import tensorflow as tf

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d_basic(x, W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = "SAME")

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides = [1,2,2,1], padding = "SAME")


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_actual = tf.placeholder(tf.float32,shape= [None, 10])

#First Convolution Layer - RELU + pooling
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d_basic(x_image, W_conv1)+ b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

#Second Convolution layer - Relu + pooling
h_conv2 = tf.nn.relu(conv2d_basic(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Fully connected NN
h_pool2_flat = tf.reshape(h_pool2, [-1, (7*7*64)])

W_fc1 = weight_variable([(7*7*64),1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, prob)

#Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_pred = tf.nn.softmax(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)

#Train
entropy = -tf.reduce_sum(y_actual*tf.log(y_pred))
train_step = tf.train.AdamOptimizer(1e-4).minimize(entropy)

#Eval
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred,1), tf.argmax(y_actual, 1)), tf.float32))

#Session start
sess.run(tf.initialize_all_variables())

for i in xrange(20000):
	batch = mnist.train.next_batch(50)
	train_step.run(feed_dict = {x:batch[0], y_actual: batch[1], prob:0.5})

	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict = {x:batch[0], y_actual:batch[1], prob: 1.0})
		print("step: %d, training accuracy: %g"%(i,train_accuracy))

test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, prob: 1.0})
print("************* test accuracy: %g ************" % test_accuracy)


