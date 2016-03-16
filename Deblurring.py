__author__ = 'Charlie'

import tensorflow as tf
import os, sys
from six.moves import urllib
import tarfile
import time
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'CIFAR10_Data/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")

tf.app.flags.DEFINE_string('train_dir', 'Deblurring_logs/Deblurring_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

IMAGE_SIZE = 32

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def conv2d_basic(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def activation_summary(x):
    """Helper to create summaries for activations."""
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = IMAGE_SIZE
    result.width = IMAGE_SIZE
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # # The first bytes represent the label, which we convert from uint8->int32.
    # result.label = tf.cast(
    #     tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].

    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    image4d = tf.cast(tf.reshape(result.uint8image, [-1, result.height, result.width, result.depth]), dtype=tf.float32)
    W = tf.truncated_normal((5, 5, 3, 3), stddev=tf.random_uniform([1]))
    result.noise_image = tf.reshape(conv2d_basic(image4d, W), [result.height, result.width, result.depth])
    return result


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                                 reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def inputs():
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    num_preprocess_threads = 16
    min_queue_examples = int(0.4 * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
    input_images, ref_images = tf.train.shuffle_batch([read_input.noise_image, read_input.uint8image],
                                                      batch_size=FLAGS.batch_size, num_threads=num_preprocess_threads,
                                                      capacity=min_queue_examples + 3 * FLAGS.batch_size,
                                                      min_after_dequeue=min_queue_examples)
    tf.image_summary("Input_Noise_images", input_images)
    tf.image_summary("Ref_images", ref_images)
    return input_images, ref_images


def inference(images):
    with tf.variable_scope("conv1") as scope:
        kernel = _variable_with_weight_decay("weights", [5, 5, 3, 64], stddev=1e-4, wd=0.0)
        conv = conv2d_basic(images, kernel)
        bias = _variable_on_cpu("bias", [64], tf.constant_initializer(0.0))
        h_conv1 = tf.nn.relu(conv + bias, name=scope.name)
        activation_summary(h_conv1)

    # norm1
    norm1 = tf.nn.lrn(h_conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    with tf.variable_scope("conv2") as scope:
        kernel = _variable_with_weight_decay("weights", [1, 1, 64, 32], stddev=1e-4, wd=0.0)
        conv = conv2d_basic(norm1, kernel)
        bias = _variable_on_cpu("bias", [32], tf.constant_initializer(0.0))
        h_conv2 = tf.nn.relu(conv + bias, name=scope.name)
        activation_summary(h_conv2)

    # norm2
    norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')

    with tf.variable_scope("output") as scope:
        kernel = _variable_with_weight_decay("weights", [5, 5, 32, 3], stddev=1e-4, wd=0.0)
        conv = conv2d_basic(norm2, kernel)
        bias = _variable_on_cpu("bias", [3], tf.constant_initializer(0.0))
        result = tf.nn.bias_add(conv, bias, name=scope.name)

    return result


def loss(pred, ref):
    square_error = tf.nn.l2_loss(tf.sub(pred, ref))
    l2_loss = tf.div(tf.cast(square_error, dtype=tf.float32), 3 * IMAGE_SIZE * IMAGE_SIZE, name="L2_Loss")
    tf.add_to_collection("losses", l2_loss)

    return tf.add_n(tf.get_collection("losses"), name="Total_loss")


def train(total_loss, global_step):
    tf.scalar_summary("Total_loss", total_loss)
    return tf.train.AdamOptimizer(1e-3).minimize(total_loss, global_step=global_step)


def main(argv=None):
    maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        input_images, ref_images = inputs()

        pred_images = inference(input_images)

        total_loss = loss(pred_images, tf.cast(ref_images, dtype= tf.float32))

        train_op = train(total_loss, global_step)

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, total_loss])
            duration = time.time() - start_time

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == "__main__":
    tf.app.run()

