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


def accuracy(logits, labels):
    return 100.0 * np.sum(np.argmax(logits, 1) == np.argmax(labels, 1)) / logits.shape[0]


def main(argv=None):
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
        read_notMNIST.get_notMNISTData(FLAGS.data_dir)

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE))
        tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUMBER_OF_CLASSES))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, NUMBER_OF_CLASSES]))
        biases = tf.Variable(tf.zeros([NUMBER_OF_CLASSES]))

        # Training computation.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    num_steps = 3001

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
            offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)

            batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]

            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            l, predictions = session.run(
                [loss, train_prediction], feed_dict=feed_dict)
            if step % 500 == 0:
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


if __name__ == "__main__":
    tf.app.run()
