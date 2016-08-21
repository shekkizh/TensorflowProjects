from tensorflow.examples.tutorials.mnist import input_data
import sys

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d_basic(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def leaky_relu(x, alpha=0.0):
    return tf.maximum(alpha * x, x)


def activation_function(x):
    activation_dict = {'relu': tf.nn.relu(x), 'elu': tf.nn.elu(x), 'lrelu': leaky_relu(x, 0.2), 'tanh': tf.nn.tanh(x),
                       'sigmoid': tf.nn.sigmoid(x)}
    return activation_dict[str(sys.argv[1])]


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_actual = tf.placeholder(tf.float32, shape=[None, 10])

# First Convolution Layer - RELU + pooling
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
tf.histogram_summary("W_conv1", W_conv1)
tf.histogram_summary("b_conv1", b_conv1)

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = activation_function(conv2d_basic(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
tf.histogram_summary("W_conv2", W_conv2)
tf.histogram_summary("b_conv2", b_conv2)

# Second Convolution layer - Relu + pooling
h_conv2 = activation_function(conv2d_basic(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected NN
h_pool2_flat = tf.reshape(h_pool2, [-1, (7 * 7 * 64)])

W_fc1 = weight_variable([(7 * 7 * 64), 1024])
b_fc1 = bias_variable([1024])
tf.histogram_summary("W_fc1", W_fc1)
tf.histogram_summary("b_fc1", b_fc1)

h_fc1 = activation_function(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, prob)

# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
tf.histogram_summary("W_fc2", W_fc2)
tf.histogram_summary("b_fc2", b_fc2)

y_pred = tf.nn.softmax(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)

# Train
entropy = -tf.reduce_sum(y_actual * tf.log(y_pred))
train_step = tf.train.AdamOptimizer(1e-4).minimize(entropy)

tf.scalar_summary("Cross_Entr", entropy)

# Eval
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_actual, 1)), tf.float32))

# Session start
sess.run(tf.initialize_all_variables())
summary_writer = tf.train.SummaryWriter("logs/MNIST_Activation/", sess.graph)
summary_op = tf.merge_all_summaries()

for i in xrange(12001):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_actual: batch[1], prob: 0.5})

    if i % 10 == 0:
        summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_actual: batch[1], prob: 0.5})
        summary_writer.add_summary(summary_str, i)

    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_actual: batch[1], prob: 1.0})
        print("step: %d, training accuracy: %g" % (i, train_accuracy))

test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, prob: 1.0})
print("************* test accuracy: %g ************" % test_accuracy)
