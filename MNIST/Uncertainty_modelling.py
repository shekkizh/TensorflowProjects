from tensorflow.examples.tutorials.mnist import input_data
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/MNIST_logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MNIST_data/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Optimizer")
tf.flags.DEFINE_string("mode", "train", "Model mode = train / test")
tf.flags.DEFINE_float("keep_prob", "0.5", "keep probability for test")
tf.flags.DEFINE_integer("label", "7", "image label to analyse for uncertainty during test")
MAX_ITERATION = int(1e4 + 1)


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
    return tf.nn.relu(x)


def inference(input, keep_prob=1.0):
    # First Convolution Layer - RELU + pooling
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    tf.histogram_summary("W_conv1", W_conv1)
    tf.histogram_summary("b_conv1", b_conv1)

    h_conv1 = activation_function(conv2d_basic(input, W_conv1) + b_conv1)
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
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    tf.histogram_summary("W_fc2", W_fc2)
    tf.histogram_summary("b_fc2", b_fc2)
    pred = tf.nn.softmax(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)
    return pred


def main(argv=None):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_actual = tf.placeholder(tf.float32, shape=[None, 10])
    prob = tf.placeholder(tf.float32)

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    y_pred = inference(x_image, prob)

    # Train
    entropy = -tf.reduce_sum(y_actual * tf.log(y_pred))
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(entropy)

    tf.scalar_summary("X_Entropy", entropy)

    # Eval
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_actual, 1)), tf.float32))

    # Session start
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)
    summary_op = tf.merge_all_summaries()

    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == 'train':
        for itr in xrange(MAX_ITERATION):
            batch = mnist.train.next_batch(FLAGS.batch_size)
            train_step.run(feed_dict={x: batch[0], y_actual: batch[1], prob: 0.9})

            if itr % 10 == 0:
                summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_actual: batch[1], prob: 0.9})
                summary_writer.add_summary(summary_str, itr)

            if itr % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_actual: batch[1], prob: 1.0})
                print("step: %d, training accuracy: %g" % (itr, train_accuracy))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
    else:
        # test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, prob: 1.0})
        # print("************* test accuracy: %g ************" % test_accuracy)
        # print (np.shape(mnist.test.labels))
        images_label = mnist.test.images[np.argmax(mnist.test.labels, 1) == FLAGS.label]
        # print(len(images_label))
        offset = np.random.randint(0, len(images_label) - FLAGS.batch_size)
        images = images_label[offset: offset + FLAGS.batch_size]

        feed_dict = {x: images, prob: FLAGS.keep_prob}
        predictions = [sess.run(y_pred, feed_dict=feed_dict) for i in range(100)]
        pred_nw = sess.run(y_pred, feed_dict={x: images, prob: 1.0})
        prediction_mean = np.mean(predictions, axis=0)
        prediction_var = np.var(predictions, axis=0) + (2 * FLAGS.batch_size * 0.0005 / FLAGS.keep_prob)
        # print (np.shape(prediction_mean))
        # print(np.shape(prediction_var))
        reshaped_images = np.reshape(images_label, [-1, 28, 28])
        fig = plt.figure(figsize=(20, 20))
        x_values = range(10)
        for i in range(FLAGS.batch_size):
            a = fig.add_subplot(FLAGS.batch_size, 2, (i + 1) * 2 - 1)
            plt.imshow(reshaped_images[i, :, :])
            a = fig.add_subplot(FLAGS.batch_size, 2, (i + 1) * 2)
            a.plot(x_values, pred_nw[i], 'r')
            a.errorbar(x_values, prediction_mean[i, :], yerr=prediction_var[i, :], linestyle=':')

        plt.show()


if __name__ == "__main__":
    tf.app.run()