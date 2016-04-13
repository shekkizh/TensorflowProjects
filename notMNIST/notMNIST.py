__author__ = 'Charlie'
import read_notMNIST
import numpy as np
import os, sys, inspect
import tensorflow as tf
from six.moves import range

utils_path = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
import TensorflowUtils as utils

BATCH_SIZE = 128
TRAIN_DATA_URL = 'http://commondatastorage.googleapis.com/books1000/notMNIST_large.tar.gz'
TEST_DATA_URL = 'http://commondatastorage.googleapis.com/books1000/notMNIST_small.tar.gz'

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("mode", "train", "Mode train/ test")
tf.flags.DEFINE_string("logs_dir", "logs/notMNIST_logs", "Path to save log and train checkpoints")
tf.flags.DEFINE_string("data_dir", "Data_zoo/notMNIST", "Path to save/ load notMNIST data")

NUMBER_OF_CLASSES = 10
IMAGE_SIZE = 28
MAX_ITERATIONS = 10000


def accuracy(logits, labels):
    return 100.0 * np.sum(np.argmax(logits, 1) == np.argmax(labels, 1)) / logits.shape[0]


def inference_resnet(dataset):
    dataset_reshaped = tf.reshape(dataset, [-1, 28, 28, 1])
    with tf.name_scope("conv1") as scope:
        W_conv1 = utils.weight_variable([5, 5, 1, 32], name="W_conv1")
        bias1 = utils.bias_variable([32], name="bias1")
        tf.histogram_summary("W_conv1", W_conv1)
        tf.histogram_summary("bias1", bias1)
        h_conv1 = tf.nn.relu(utils.conv2d_basic(dataset_reshaped, W_conv1, bias1))
        h_norm1 = utils.batch_norm(h_conv1)

    bottleneck_1 = utils.bottleneck_unit(h_norm1, 32, 32, down_stride=True, name="res1")
    bottleneck_2 = utils.bottleneck_unit(bottleneck_1, 64, 64, down_stride=True, name="res2")

    with tf.name_scope("fc1") as scope:
        h_flat = tf.reshape(bottleneck_2, [-1, 7 * 7 * 64])
        W_fc1 = utils.weight_variable([7 * 7 * 64, 10], name="W_fc1")
        bias_fc1 = utils.bias_variable([10], name="bias_fc1")
        tf.histogram_summary("W_fc1", W_fc1)
        tf.histogram_summary("bias_fc1", bias_fc1)
        logits = tf.matmul(h_flat, W_fc1) + bias_fc1

    return logits


def inference_conv(dataset):
    dataset_reshaped = tf.reshape(dataset, [-1, 28, 28, 1])
    with tf.name_scope("conv1") as scope:
        W_conv1 = utils.weight_variable([5, 5, 1, 32], name="W_conv1")
        bias1 = utils.bias_variable([32], name="bias1")
        tf.histogram_summary("W_conv1", W_conv1)
        tf.histogram_summary("bias1", bias1)
        h_conv1 = tf.nn.relu(utils.conv2d_basic(dataset_reshaped, W_conv1, bias1))
        h_norm1 = utils.batch_norm(h_conv1)
        h_pool1 = utils.max_pool_2x2(h_norm1)

    with tf.name_scope("conv2") as scope:
        W_conv2 = utils.weight_variable([3, 3, 32, 64], name="W_conv2")
        bias2 = utils.bias_variable([64], name="bias2")
        tf.histogram_summary("W_conv2", W_conv2)
        tf.histogram_summary("bias2", bias2)
        h_conv2 = tf.nn.relu(utils.conv2d_basic(h_pool1, W_conv2, bias2))
        h_norm2 = utils.batch_norm(h_conv2)
        h_pool2 = utils.max_pool_2x2(h_norm2)

    with tf.name_scope("fc1") as scope:
        h_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        W_fc1 = utils.weight_variable([7 * 7 * 64, 10], name="W_fc1")
        bias_fc1 = utils.bias_variable([10], name="bias_fc1")
        tf.histogram_summary("W_fc1", W_fc1)
        tf.histogram_summary("bias_fc1", bias_fc1)
        logits = tf.matmul(h_flat, W_fc1) + bias_fc1

    return logits


def inference_nn(dataset):
    with tf.name_scope("fc1") as scope:
        W1 = utils.weight_variable([IMAGE_SIZE * IMAGE_SIZE, 512], name="W1")
        b1 = utils.bias_variable([512], name="b1")
        h_fc1 = tf.nn.relu(tf.matmul(dataset, W1) + b1)
        tf.histogram_summary("W1", W1)
        tf.histogram_summary("b1", b1)

    with tf.name_scope("fc2") as scope:
        W2 = utils.weight_variable([512, 512], name="W2")
        b2 = utils.bias_variable([512], name="b2")
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W2) + b2)
        tf.histogram_summary("W2", W2)
        tf.histogram_summary("b2", b2)

    with tf.name_scope("output") as scope:
        W3 = utils.weight_variable([512, NUMBER_OF_CLASSES], name="W3")
        b3 = utils.bias_variable([NUMBER_OF_CLASSES], name="b3")
        logits = tf.matmul(h_fc2, W3) + b3
        tf.histogram_summary("W3", W3)
        tf.histogram_summary("b3", b3)

    return logits


def main(argv=None):
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
        read_notMNIST.get_notMNISTData(FLAGS.data_dir)

    dataset = tf.placeholder(tf.float32,
                             shape=(None, IMAGE_SIZE * IMAGE_SIZE))
    labels = tf.placeholder(tf.float32, shape=(None, NUMBER_OF_CLASSES))

    # logits = inference_nn(dataset)
    logits = inference_conv(dataset)
    # logits = inference_resnet(dataset) #resnet implementation is not supported by tensorflow :(
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, labels))
    tf.scalar_summary("train_loss", loss)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir)

    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)

    with tf.Session() as session:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        print("Initialized")

        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            print "Reloaded variables from checkpoint"

        else:
            for step in range(1, MAX_ITERATIONS):
                offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)

                batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
                batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]

                feed_dict = {dataset: batch_data, labels: batch_labels}
                _, l, predictions = session.run(
                    [optimizer, loss, prediction], feed_dict=feed_dict)
                if step % 200 == 0:
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    summary_str = session.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, global_step=step)

                if step % 1000 == 0:
                    validation_pred = session.run(prediction, feed_dict={dataset: valid_dataset})
                    print("-----------> Validation accuracy: %.1f%%" % accuracy(
                        validation_pred, valid_labels))
                    saver.save(session, FLAGS.logs_dir + "model.ckpt", global_step=step)

        test_pred = session.run(prediction, feed_dict={dataset: test_dataset})
        print("Test accuracy: %.1f%%" % accuracy(test_pred, test_labels))


if __name__ == "__main__":
    tf.app.run()
