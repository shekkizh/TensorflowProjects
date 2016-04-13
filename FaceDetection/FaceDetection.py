"""Kaggle face detection competition solution based on deep net"""
__author__ = 'Charlie'

import tensorflow as tf
import numpy as np
import os, sys, inspect
from datetime import datetime

lib_path = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

import TensorflowUtils as utils
import readFaceDetectionData

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "Data_zoo/FaceDetectionData/", "Path to data files")
tf.flags.DEFINE_string("logs_dir", "logs/FaceDetection_logs", "Path to where log files are to be saved")

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_ITERATIONS = 10000


def inference(dataset):
    with tf.name_scope("conv1") as scope:
        W1 = utils.weight_variable([5, 5, 1, 32], name='W1')
        b1 = utils.bias_variable([32], name='b1')
        tf.histogram_summary("W1", W1)
        tf.histogram_summary("b1", b1)
        h_conv1 = tf.nn.relu(utils.conv2d_basic(dataset, W1, b1), name="h_conv1")
        h_norm1 = utils.batch_norm(h_conv1)
        h_pool1 = utils.max_pool_2x2(h_norm1)

    with tf.name_scope("conv2") as scope:
        W2 = utils.weight_variable([3, 3, 32, 64], name='W2')
        b2 = utils.bias_variable([64], name='b2')
        tf.histogram_summary("W2", W2)
        tf.histogram_summary("b2", b2)
        h_conv2 = tf.nn.relu(utils.conv2d_basic(h_pool1, W2, b2), name="h_conv2")
        h_norm2 = utils.batch_norm(h_conv2)
        h_pool2 = utils.max_pool_2x2(h_norm2)

    with tf.name_scope("conv3") as scope:
        W3 = utils.weight_variable([3, 3, 64, 64], name='W3')
        b3 = utils.bias_variable([64], name='b3')
        tf.histogram_summary("W3", W3)
        tf.histogram_summary("b3", b3)
        h_conv3 = tf.nn.relu(utils.conv2d_basic(h_pool2, W3, b3), name="h_conv3")
        h_norm3 = utils.batch_norm(h_conv3)
        h_pool3 = utils.max_pool_2x2(h_norm3)

    with tf.name_scope("fc") as scope:
        image_size = IMAGE_SIZE // 8
        h_flat = tf.reshape(h_pool3,[-1, image_size * image_size * 64])
        W_fc = utils.weight_variable([image_size * image_size * 64, NUM_LABELS], name="W_fc")
        b_fc = utils.bias_variable([NUM_LABELS], name="b_fc")
        tf.histogram_summary("W_fc", W_fc)
        tf.histogram_summary("b_fc", b_fc)
        pred = tf.matmul(h_flat, W_fc) + b_fc

    return pred


def loss(pred, labels):
    l2_loss = tf.reduce_mean(tf.nn.l2_loss(tf.sub(pred, labels))) / NUM_LABELS
    tf.scalar_summary("Loss", l2_loss)
    return l2_loss


def train(loss_val):
    return tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_val)


def main(argv=None):
    train_images, train_labels, validation_images, validation_labels, test_images = readFaceDetectionData.read_data(
        FLAGS.data_dir)
    print "Training Set: %s" % train_images.shape[0]
    print "Validation Set: %s" % validation_images.shape[0]
    print "Test Set: %s" % test_images.shape[0]

    dataset = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1])
    labels = tf.placeholder(tf.float32, [None, NUM_LABELS])
    logits = inference(dataset)
    total_loss = loss(logits, labels)
    train_op = train(total_loss)

    print ("Model architecture built!")

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_ckeckpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Model restored from previous checkpoint!"

        for step in range(MAX_ITERATIONS):
            offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
            feed_dict = {dataset: train_images[offset:offset + BATCH_SIZE],
                         labels: train_labels[offset:offset + BATCH_SIZE]}

            sess.run(train_op, feed_dict=feed_dict)

            if step % 10 == 0:
                [err, summary_str] = sess.run([total_loss, summary_op], feed_dict=feed_dict)
                print ("%s : Step:%d, Training loss: %f") % (datetime.now(), step, err)
                summary_writer.add_summary(summary_str, global_step=step)

            if step % 100 == 0:
                valid_loss = sess.run(total_loss, feed_dict={dataset: validation_images, labels: validation_labels})
                print ("======> Validation loss: %f" % valid_loss)
                saver.save(sess, FLAGS.logs_dir, global_step=step)


if __name__ == "__main__":
    IMAGE_SIZE = readFaceDetectionData.IMAGE_SIZE
    NUM_LABELS = readFaceDetectionData.NUM_LABELS
    tf.app.run()
