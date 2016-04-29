__author__ = 'Charlie'
"""Image coloring by fully convolutional networks - incomplete """
import tensorflow as tf
import numpy as np
import os, sys, inspect
from datetime import datetime
import scipy.misc as misc

lib_path = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

import TensorflowUtils as utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "Data_zoo/CIFAR10_data/", """Path to the CIFAR10 data""")
tf.flags.DEFINE_string("mode", "train", "Network mode train/ test")
tf.flags.DEFINE_string("test_image_path", "", "Path to test image - read only if mode is test")
tf.flags.DEFINE_integer("batch_size", "128", "train batch size")
tf.flags.DEFINE_string("logs_dir", "logs/ImageColoring_logs/", """Path to save logs and checkpoint if needed""")

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

LEARNING_RATE = 1e-3
MAX_ITERATIONS = 100001

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 20000

IMAGE_SIZE = 32


def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    label_bytes = 1
    result.height = IMAGE_SIZE
    result.width = IMAGE_SIZE
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    depth_major = tf.cast(tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                                     [result.depth, result.height, result.width]), tf.float32)

    image = tf.transpose(depth_major, [1, 2, 0])
    # extended_image = tf.reshape(image, (result.height, result.width, result.depth))
    result.color_image = image
    print result.color_image.get_shape()
    print "Converting image to gray scale"
    result.gray_image = 0.21 * result.color_image[ :, :, 2] + 0.72 * result.color_image[ :, :,
                                                                       1] + 0.07 * result.color_image[ :, :, 0]
    result.gray_image = tf.expand_dims(result.gray_image, 2)
    print result.gray_image.get_shape()

    return result


def get_image(image_dir):
    image = misc.imread(image_dir)
    image = np.ndarray.reshape(image.astype(np.float32), ((1,) + image.shape))
    return image


def inputs():
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(filename_queue)
    num_preprocess_threads = 8
    min_queue_examples = int(0.4 * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
    print "Shuffling"
    input_gray, input_colored = tf.train.shuffle_batch([read_input.gray_image, read_input.color_image],
                                                       batch_size=FLAGS.batch_size,
                                                       num_threads=num_preprocess_threads,
                                                       capacity=min_queue_examples + 3 * FLAGS.batch_size,
                                                       min_after_dequeue=min_queue_examples)
    input_gray = (input_gray - 128) / 255.0
    input_colored = (input_colored - 128) / 255.0
    return input_gray, input_colored


def inference(image):
    W1 = utils.weight_variable([9, 9, 1, 32])
    b1 = utils.bias_variable([32])
    tf.histogram_summary("W1", W1)
    tf.histogram_summary("b1", b1)
    h_conv1 = tf.nn.relu(utils.conv2d_basic(image, W1, b1))

    W2 = utils.weight_variable([3, 3, 32, 64])
    b2 = utils.bias_variable([64])
    tf.histogram_summary("W2", W2)
    tf.histogram_summary("b2", b2)
    h_conv2 = tf.nn.relu(utils.conv2d_strided(h_conv1, W2, b2))

    W3 = utils.weight_variable([3, 3, 64, 128])
    b3 = utils.bias_variable([128])
    tf.histogram_summary("W3", W3)
    tf.histogram_summary("b3", b3)
    h_conv3 = tf.nn.relu(utils.conv2d_strided(h_conv2, W3, b3))

    # upstrides
    W4 = utils.weight_variable([3, 3, 64, 128])
    b4 = utils.bias_variable([64])
    tf.histogram_summary("W4", W4)
    tf.histogram_summary("b4", b4)
    h_conv4 = tf.nn.relu(utils.conv2d_transpose_strided(h_conv3, W4, b4))

    W5 = utils.weight_variable([3, 3, 32, 64])
    b5 = utils.bias_variable([32])
    tf.histogram_summary("W5", W5)
    tf.histogram_summary("b5", b5)
    h_conv5 = tf.nn.relu(utils.conv2d_transpose_strided(h_conv4, W5, b5))

    W6 = utils.weight_variable([9, 9, 32, 3])
    b6 = utils.bias_variable([3])
    tf.histogram_summary("W6", W6)
    tf.histogram_summary("b6", b6)
    pred_image = tf.nn.tanh(utils.conv2d_basic(h_conv5, W6, b6))

    return pred_image


def loss(pred, colored):
    rmse = tf.sqrt(2 * tf.nn.l2_loss(tf.sub(colored, pred))) / FLAGS.batch_size
    tf.scalar_summary("RMSE", rmse)
    return rmse


def train(loss_val, step):
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, step, 0.4 * MAX_ITERATIONS, 0.99)
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_val, global_step=step)
    return train_op


def main(argv=None):
    utils.maybe_download_and_extract(FLAGS.data_dir, DATA_URL, is_tarfile=True)
    print "Setting up model..."
    global_step = tf.Variable(0,trainable=False)
    gray, color = inputs()
    pred = 255 * inference(gray) + 128
    tf.image_summary("Gray", gray, max_images=1)
    tf.image_summary("Ground_truth", color, max_images=1)
    tf.image_summary("Prediction", pred, max_images=1)

    image_loss = loss(pred, color)
    train_op = train(image_loss, global_step)

    summary_op = tf.merge_all_summaries()
    with tf.Session() as sess:
        print "Setting up summary writer, queue, saver..."
        sess.run(tf.initialize_all_variables())
        tf.train.start_queue_runners(sess)
        summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph_def)
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print "Restoring model from checkpoint..."
            saver.restore(sess, ckpt.model_checkpoint_path)

        for step in xrange(MAX_ITERATIONS):
            if step % 100 == 0:
                loss_val, summary_str = sess.run([image_loss, summary_op])
                print "Step %d, Loss: %g" % (step, loss_val)
                summary_writer.add_summary(summary_str, global_step=step)

            if step % 1000 == 0:
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", global_step=step)
                print "%s" % datetime.now()

            sess.run(train_op)

if __name__ == "__main__":
    tf.app.run()
