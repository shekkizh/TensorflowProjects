"""Kaggle face detection competition solution based on deep net"""
__author__ = 'Charlie'
# Code follows some implementation from Daniel Nouri's Lasagne code.
import tensorflow as tf
import numpy as np
import os, sys, inspect
from datetime import datetime

lib_path = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

import TensorflowUtils as utils
import FaceDetectionDataUtils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "Data_zoo/FaceDetectionData/", "Path to data files")
tf.flags.DEFINE_string("logs_dir", "logs/FaceDetection_logs/", "Path to where log files are to be saved")
tf.flags.DEFINE_string("mode", "train", "mode: train (Default)/ test")

BATCH_SIZE = 128
LEARNING_RATE = 1e-2
MAX_ITERATIONS = 10000
MOMENTUM = 0.9


def inference(dataset):
    with tf.name_scope("conv1") as scope:
        W1 = utils.weight_variable([5, 5, 1, 32], name='W1')
        b1 = utils.bias_variable([32], name='b1')
        tf.histogram_summary("W1", W1)
        tf.histogram_summary("b1", b1)
        h_conv1 = utils.conv2d_basic(dataset, W1, b1)
        h_norm1 = utils.local_response_norm(h_conv1)
        h_1 = tf.nn.relu(h_norm1, name="conv1")
        h_pool1 = utils.max_pool_2x2(h_1)

    with tf.name_scope("conv2") as scope:
        W2 = utils.weight_variable([3, 3, 32, 64], name='W2')
        b2 = utils.bias_variable([64], name='b2')
        tf.histogram_summary("W2", W2)
        tf.histogram_summary("b2", b2)
        h_conv2 = utils.conv2d_basic(h_pool1, W2, b2)
        h_norm2 = utils.local_response_norm(h_conv2)
        h_2 = tf.nn.relu(h_norm2, name="conv2")
        h_pool2 = utils.max_pool_2x2(h_2)

    with tf.name_scope("conv3") as scope:
        W3 = utils.weight_variable([3, 3, 64, 128], name='W3')
        b3 = utils.bias_variable([128], name='b3')
        tf.histogram_summary("W3", W3)
        tf.histogram_summary("b3", b3)
        h_conv3 = utils.conv2d_basic(h_pool2, W3, b3)
        h_norm3 = utils.local_response_norm(h_conv3)
        h_3 = tf.nn.relu(h_norm3, name="conv3")
        h_pool3 = utils.max_pool_2x2(h_3)

    with tf.name_scope("conv4") as scope:
        W4 = utils.weight_variable([3, 3, 128, 256], name='W4')
        b4 = utils.bias_variable([256], name='b4')
        tf.histogram_summary("W4", W4)
        tf.histogram_summary("b4", b4)
        h_conv4 = utils.conv2d_basic(h_pool3, W4, b4)
        h_norm4 = utils.local_response_norm(h_conv4)
        h_4 = tf.nn.relu(h_norm4, name="conv4")

    with tf.name_scope("fc1") as scope:
        image_size = IMAGE_SIZE // 8
        h_flat = tf.reshape(h_4, [-1, image_size * image_size * 256])
        W_fc1 = utils.weight_variable([image_size * image_size * 256, 512], name="W_fc1")
        b_fc1 = utils.bias_variable([512], name="b_fc1")
        tf.histogram_summary("W_fc1", W_fc1)
        tf.histogram_summary("b_fc1", b_fc1)
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

    with tf.name_scope("fc2") as scope:
        W_fc2 = utils.weight_variable([512, NUM_LABELS], name="W_fc2")
        b_fc2 = utils.bias_variable([NUM_LABELS], name="b_fc2")
        tf.histogram_summary("W_fc2", W_fc2)
        tf.histogram_summary("b_fc2", b_fc2)
        pred = tf.matmul(h_fc1, W_fc2) + b_fc2

    return pred


def inference_simple(dataset):
    with tf.name_scope("conv1") as scope:
        W1 = utils.weight_variable([5, 5, 1, 32], name='W1')
        b1 = utils.bias_variable([32], name='b1')
        tf.histogram_summary("W1", W1)
        tf.histogram_summary("b1", b1)
        h_conv1 = tf.nn.relu(utils.conv2d_basic(dataset, W1, b1), name="h_conv1")
        h_pool1 = utils.max_pool_2x2(h_conv1)

    with tf.name_scope("conv2") as scope:
        W2 = utils.weight_variable([3, 3, 32, 64], name='W2')
        b2 = utils.bias_variable([64], name='b2')
        tf.histogram_summary("W2", W2)
        tf.histogram_summary("b2", b2)
        h_conv2 = tf.nn.relu(utils.conv2d_basic(h_pool1, W2, b2), name="h_conv2")
        h_pool2 = utils.max_pool_2x2(h_conv2)

    with tf.name_scope("fc") as scope:
        image_size = IMAGE_SIZE // 4
        h_flat = tf.reshape(h_pool2, [-1, image_size * image_size * 64])
        W_fc = utils.weight_variable([image_size * image_size * 64, NUM_LABELS], name="W_fc")
        b_fc = utils.bias_variable([NUM_LABELS], name="b_fc")
        tf.histogram_summary("W_fc", W_fc)
        tf.histogram_summary("b_fc", b_fc)
        pred = tf.matmul(h_flat, W_fc) + b_fc

    return pred


def loss(pred, labels):
    l2_loss = tf.reduce_mean(tf.nn.l2_loss(tf.sub(pred, labels)))
    tf.scalar_summary("Loss", l2_loss)
    return l2_loss


def train(loss_val, step):
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, step, MAX_ITERATIONS / 4, 0.95,
                                               staircase=True)
    return tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss_val)


def augment_data(data, label):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
    ]
    batch = data.shape[0]
    indices = np.random.choice(batch, batch / 2, replace=False)
    data[indices] = data[indices, :, :, ::-1]

    label[indices, ::2] = label[indices, ::2] * -1

    for a, b in flip_indices:
        label[indices, a], label[indices, b] = (label[indices, b], label[indices, a])

    return data, label


def main(argv=None):
    train_images, train_labels, validation_images, validation_labels, test_images = FaceDetectionDataUtils.read_data(
        FLAGS.data_dir)
    print "Training Set: %s" % train_images.shape[0]
    print "Validation Set: %s" % validation_images.shape[0]
    print "Test Set: %s" % test_images.shape[0]

    def get_next_batch(step):
        offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
        data = train_images[offset:offset + BATCH_SIZE]
        label = train_labels[offset:offset + BATCH_SIZE]
        return data, label

    global_step = tf.Variable(0.0, trainable=False)
    dataset = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1])
    labels = tf.placeholder(tf.float32, [None, NUM_LABELS])
    logits = inference(dataset)
    total_loss = loss(logits, labels)
    train_op = train(total_loss, global_step)

    print ("Model architecture built!")

    summary_op = tf.merge_all_summaries()

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph_def)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Model restored from previous checkpoint!"

        if FLAGS.mode == "train":
            print "Training..."

            for step in range(MAX_ITERATIONS):

                batch_data, batch_label = get_next_batch(step)
                feed_dict = {dataset: batch_data,
                             labels: batch_label}

                sess.run(train_op, feed_dict=feed_dict)

                if step % 10 == 0:
                    [err, summary_str] = sess.run([total_loss, summary_op], feed_dict=feed_dict)
                    print "%s : Step:%d, Training loss: %f" % (datetime.now(), step, err)
                    summary_writer.add_summary(summary_str, global_step=step)

                if step % 100 == 0:
                    valid_loss = sess.run(total_loss, feed_dict={dataset: validation_images, labels: validation_labels})
                    print ("======> Validation loss: %f" % valid_loss)
                    saver.save(sess, FLAGS.logs_dir + 'model.ckpt', global_step=step)
        print "Predicting test result..."
        test_labels = sess.run(logits, feed_dict={dataset: test_images})
        FaceDetectionDataUtils.kaggle_submission_format(test_images, test_labels, FLAGS.data_dir)

if __name__ == "__main__":
    IMAGE_SIZE = FaceDetectionDataUtils.IMAGE_SIZE
    NUM_LABELS = FaceDetectionDataUtils.NUM_LABELS
    tf.app.run()
