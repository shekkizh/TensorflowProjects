__author__ = 'Charlie'
import numpy as np
import os, sys, inspect
import tensorflow as tf

utils_path = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
import TensorflowUtils as utils
import read_notMNIST

BATCH_SIZE = 128
TRAIN_DATA_URL = 'http://commondatastorage.googleapis.com/books1000/notMNIST_large.tar.gz'
TEST_DATA_URL = 'http://commondatastorage.googleapis.com/books1000/notMNIST_small.tar.gz'

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("mode", "train", "Mode train/ test")
tf.flags.DEFINE_string("logs_dir", "logs/notMNIST_logs/", "Path to save log and train checkpoints")
tf.flags.DEFINE_string("data_dir", "Data_zoo/notMNIST/", "Path to save/ load notMNIST data")

NUMBER_OF_CLASSES = 10
IMAGE_SIZE = 28
MAX_ITERATIONS = int(1 + 1e4)
LEARNING_RATE = 1e-3


def inference_fully_convolutional(dataset):
    '''
    Fully convolutional inference on notMNIST dataset
    :param datset: [batch_size, 28*28*1] tensor
    :return: logits
    '''
    dataset_reshaped = tf.reshape(dataset, [-1, 28, 28, 1])
    with tf.name_scope("conv1") as scope:
        W_conv1 = utils.weight_variable_xavier_initialized([3, 3, 1, 32], name="W_conv1")
        b_conv1 = utils.bias_variable([32], name="b_conv1")
        h_conv1 = tf.nn.relu(utils.conv2d_strided(dataset_reshaped, W_conv1, b_conv1))

    with tf.name_scope("conv2") as scope:
        W_conv2 = utils.weight_variable_xavier_initialized([3, 3, 32, 64], name="W_conv2")
        b_conv2 = utils.bias_variable([64], name="b_conv2")
        h_conv2 = tf.nn.relu(utils.conv2d_strided(h_conv1, W_conv2, b_conv2))

    with tf.name_scope("conv3") as scope:
        W_conv3 = utils.weight_variable_xavier_initialized([3, 3, 64, 128], name="W_conv3")
        b_conv3 = utils.bias_variable([128], name="b_conv3")
        h_conv3 = tf.nn.relu(utils.conv2d_strided(h_conv2, W_conv3, b_conv3))

    with tf.name_scope("conv4") as scope:
        W_conv4 = utils.weight_variable_xavier_initialized([3, 3, 128, 256], name="W_conv4")
        b_conv4 = utils.bias_variable([256], name="b_conv4")
        h_conv4 = tf.nn.relu(utils.conv2d_strided(h_conv3, W_conv4, b_conv4))

    with tf.name_scope("conv5") as scope:
        # W_conv5 = utils.weight_variable_xavier_initialized([2, 2, 256, 512], name="W_conv5")
        # b_conv5 = utils.bias_variable([512], name="b_conv5")
        # h_conv5 = tf.nn.relu(utils.conv2d_strided(h_conv4, W_conv5, b_conv5))
        h_conv5 = utils.avg_pool_2x2(h_conv4)

    with tf.name_scope("conv6") as scope:
        W_conv6 = utils.weight_variable_xavier_initialized([1, 1, 256, 10], name="W_conv6")
        b_conv6 = utils.bias_variable([10], name="b_conv6")
        logits = tf.nn.relu(utils.conv2d_basic(h_conv5, W_conv6, b_conv6))
        print logits.get_shape()
        logits = tf.reshape(logits, [-1, 10])
    return logits


def loss(logits, labels):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
    tf.scalar_summary("Entropy", cross_entropy)
    return cross_entropy


def train(loss_val, step):
    return tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_val, global_step=step)


def main(argv=None):
    print "Reading notMNIST data..."
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
        read_notMNIST.get_notMNISTData(FLAGS.data_dir)

    print "Setting up tf model..."
    dataset = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE * IMAGE_SIZE))

    labels = tf.placeholder(tf.float32, shape=(None, NUMBER_OF_CLASSES))

    global_step = tf.Variable(0, trainable=False)

    logits = inference_fully_convolutional(dataset)

    for var in tf.trainable_variables():
        utils.add_to_regularization_and_summary(var)

    loss_val = loss(logits, labels)
    train_op = train(loss_val, global_step)
    summary_op = tf.merge_all_summaries()
    with tf.Session() as sess:
        print "Setting up summary and saver..."
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Model restored!"

        if FLAGS.mode == "train":
            for step in xrange(MAX_ITERATIONS):
                offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)

                batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
                batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]

                feed_dict = {dataset: batch_data, labels: batch_labels}
                if step % 100 == 0:
                    l, summary_str = sess.run([loss_val, summary_op], feed_dict=feed_dict)
                    print "Step: %d Mini batch loss: %g"%(step, l)
                    summary_writer.add_summary(summary_str, step)

                if step % 1000 == 0:
                    valid_loss = sess.run(loss_val, feed_dict={dataset:valid_dataset, labels:valid_labels})
                    print "-- Validation loss %g" % valid_loss
                    saver.save(sess, FLAGS.logs_dir +"model.ckpt", global_step=step)

                sess.run(train_op, feed_dict=feed_dict)

        test_loss = sess.run(loss_val, feed_dict={dataset:test_dataset, labels:test_labels})
        print "Test loss: %g" % test_loss

if __name__ == "__main__":
    tf.app.run()