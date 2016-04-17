import tensorflow as tf
import numpy as np
import os, sys, inspect
from datetime import datetime

lib_path = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

import TensorflowUtils as utils
import EmotionDetectorUtils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "Data_zoo/EmotionDetector/", "Path to data files")
tf.flags.DEFINE_string("logs_dir", "logs/EmotionDetector_logs/", "Path to where log files are to be saved")
tf.flags.DEFINE_string("mode", "train", "mode: train (Default)/ test")

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_ITERATIONS = 10001


def inferece(dataset):
    with tf.name_scope("conv1") as scope:
        W_conv1 = utils.weight_variable([5, 5, 1, 32])
        b_conv1 = utils.bias_variable([32])
        tf.histogram_summary("W_conv1", W_conv1)
        tf.histogram_summary("b_conv1", b_conv1)
        h_conv1 = utils.conv2d_basic(dataset, W_conv1, b_conv1)
        h_1 = tf.nn.relu(h_conv1)
        h_pool1 = utils.max_pool_2x2(h_1)

    with tf.name_scope("conv2") as scope:
        W_conv2 = utils.weight_variable([3, 3, 32, 64])
        b_conv2 = utils.bias_variable([64])
        tf.histogram_summary("W_conv2", W_conv2)
        tf.histogram_summary("b_conv2", b_conv2)
        h_conv2 = utils.conv2d_basic(h_pool1, W_conv2, b_conv2)
        h_2 = tf.nn.relu(h_conv2)
        h_pool2 = utils.max_pool_2x2(h_2)

    with tf.name_scope("fc_1") as scope:
        image_size = IMAGE_SIZE / 4
        h_flat = tf.reshape(h_pool2, [-1, image_size * image_size * 64])
        W_fc1 = utils.weight_variable([image_size * image_size * 64, 256])
        b_fc1 = utils.bias_variable([256])
        tf.histogram_summary("W_fc1", W_fc1)
        tf.histogram_summary("b_fc1", b_fc1)
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

    with tf.name_scope("fc_2") as scope:
        W_fc2 = utils.weight_variable([256, NUM_LABELS])
        b_fc2 = utils.bias_variable([NUM_LABELS])
        tf.histogram_summary("W_fc2", W_fc2)
        tf.histogram_summary("b_fc2", b_fc2)
        pred = tf.matmul(h_fc1, W_fc2) + b_fc2

    return pred


def loss(pred, label):
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, label))
    tf.scalar_summary('Entropy', cross_entropy_loss)
    return cross_entropy_loss


def train(loss, step):
    return tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=step)


def get_next_batch(images, labels, step):
    offset = (step * BATCH_SIZE) % (images.shape[0] - BATCH_SIZE)
    batch_images = images[offset: offset + BATCH_SIZE]
    batch_labels = labels[offset:offset + BATCH_SIZE]
    return batch_images, batch_labels


def main(argv=None):
    train_images, train_labels, valid_images, valid_labels, test_images = EmotionDetectorUtils.read_data(FLAGS.data_dir)
    print "Train size: %s" % train_images.shape[0]
    print 'Validation size: %s' % valid_images.shape[0]
    print "Test size: %s" % test_images.shape[0]

    global_step = tf.Variable(0, trainable=False)
    input_dataset = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1])
    input_labels = tf.placeholder(tf.float32, [None, NUM_LABELS])
    pred = inferece(input_dataset)
    output_pred = tf.nn.softmax(pred)
    loss_val = loss(pred, input_labels)
    train_op = train(loss_val, global_step)

    summary_op = tf.merge_all_summaries()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph_def)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Model Restored!"

        for step in xrange(MAX_ITERATIONS):
            batch_image, batch_label = get_next_batch(train_images, train_labels, step)
            feed_dict = {input_dataset: batch_image, input_labels: batch_label}

            sess.run(train_op, feed_dict=feed_dict)
            if step % 10 == 0:
                train_loss, summary_str = sess.run([loss_val, summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=step)
                print "Training Loss: %f" % train_loss

            if step % 100 == 0:
                valid_loss = sess.run(loss_val, feed_dict={input_dataset: valid_images, input_labels: valid_labels})
                print "%s Validation Loss: %f" % (datetime.now(), valid_loss)
                saver.save(sess, FLAGS.logs_dir + 'model.ckpt', global_step=step)


if __name__ == "__main__":
    IMAGE_SIZE = EmotionDetectorUtils.IMAGE_SIZE
    NUM_LABELS = EmotionDetectorUtils.NUM_LABELS
    tf.app.run()
