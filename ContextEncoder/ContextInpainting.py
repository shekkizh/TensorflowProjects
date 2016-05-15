__author__ = "charlie"

import numpy as np
import tensorflow as tf
import os, sys, inspect
from datetime import datetime
import read_LabeledFacesWild

utils_path = os.path.abspath(
    os.path.realpath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
import TensorflowUtils as utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "Data_zoo/Faces_lfw/", "Path to dataset")
tf.flags.DEFINE_string("logs_dir", "logs/ContextInpainting_logs/", "path to logs")
tf.flags.DEFINE_integer("batch_size", "64", "batch size")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2000
MAX_ITERATIONS = int(1e5 + 1)
LEARNING_RATE = 1e-3

DATASET_IMAGE_SIZE = 250

IMAGE_SIZE = 128


def encoder(dataset, train_mode):
    with tf.variable_scope("Encoder"):
        with tf.name_scope("enc_conv1") as scope:
            W_conv1 = utils.weight_variable_xavier_initialized([3, 3, 3, 32], name="W_conv1")
            b_conv1 = utils.bias_variable([32], name="b_conv1")
            h_conv1 = utils.conv2d_strided(dataset, W_conv1, b_conv1)
            h_bn1 = utils.batch_norm(h_conv1, 32, train_mode, scope="conv1_bn")
            h_relu1 = tf.nn.relu(h_bn1)

        with tf.name_scope("enc_conv2") as scope:
            W_conv2 = utils.weight_variable_xavier_initialized([3, 3, 32, 64], name="W_conv2")
            b_conv2 = utils.bias_variable([64], name="b_conv2")
            h_conv2 = utils.conv2d_strided(h_relu1, W_conv2, b_conv2)
            h_bn2 = utils.batch_norm(h_conv2, 64, train_mode, scope="conv2_bn")
            h_relu2 = tf.nn.relu(h_bn2)

        with tf.name_scope("enc_conv3") as scope:
            W_conv3 = utils.weight_variable_xavier_initialized([3, 3, 64, 128], name="W_conv3")
            b_conv3 = utils.bias_variable([128], name="b_conv3")
            h_conv3 = utils.conv2d_strided(h_relu2, W_conv3, b_conv3)
            h_bn3 = utils.batch_norm(h_conv3, 128, train_mode, scope="conv3_bn")
            h_relu3 = tf.nn.relu(h_bn3)

        with tf.name_scope("enc_conv4") as scope:
            W_conv4 = utils.weight_variable_xavier_initialized([3, 3, 128, 256], name="W_conv4")
            b_conv4 = utils.bias_variable([256], name="b_conv4")
            h_conv4 = utils.conv2d_strided(h_relu3, W_conv4, b_conv4)
            h_bn4 = utils.batch_norm(h_conv4, 256, train_mode, scope="conv4_bn")
            h_relu4 = tf.nn.relu(h_bn4)

        with tf.name_scope("enc_conv5") as scope:
            W_conv5 = utils.weight_variable_xavier_initialized([3, 3, 256, 512], name="W_conv5")
            b_conv5 = utils.bias_variable([512], name="b_conv5")
            h_conv5 = utils.conv2d_strided(h_relu4, W_conv5, b_conv5)
            h_bn5 = utils.batch_norm(h_conv5, 512, train_mode, scope="conv5_bn")
            h_relu5 = tf.nn.relu(h_bn5)

        with tf.name_scope("enc_fc") as scope:
            image_size = IMAGE_SIZE // 32
            h_relu5_flatten = tf.reshape(h_relu5, [-1, image_size * image_size * 512])
            W_fc = utils.weight_variable([image_size * image_size * 512, 1024], name="W_fc")
            b_fc = utils.bias_variable([1024], name="b_fc")
            encoder_val = tf.matmul(h_relu5_flatten, W_fc) + b_fc

    return encoder_val


def inpainter(embedding, train_mode):
    with tf.variable_scope("context_inpainter"):
        image_size = IMAGE_SIZE // 32
        with tf.name_scope("dec_fc") as scope:
            W_fc = utils.weight_variable([1024, image_size * image_size * 512], name="W_fc")
            b_fc = utils.bias_variable([image_size * image_size * 512], name="b_fc")
            h_fc = tf.nn.relu(tf.matmul(embedding, W_fc) + b_fc)

        with tf.name_scope("dec_conv1") as scope:
            h_reshaped = tf.reshape(h_fc, tf.pack([tf.shape(h_fc)[0], image_size, image_size, 512]))
            W_conv_t1 = utils.weight_variable_xavier_initialized([3, 3, 256, 512], name="W_conv_t1")
            b_conv_t1 = utils.bias_variable([256], name="b_conv_t1")
            deconv_shape = tf.pack([tf.shape(h_reshaped)[0], 2 * image_size, 2 * image_size, 256])
            h_conv_t1 = utils.conv2d_transpose_strided(h_reshaped, W_conv_t1, b_conv_t1, output_shape=deconv_shape)
            h_bn_t1 = utils.batch_norm(h_conv_t1, 256, train_mode, scope="conv_t1_bn")
            h_relu_t1 = tf.nn.relu(h_bn_t1)

        with tf.name_scope("dec_conv2") as scope:
            W_conv_t2 = utils.weight_variable_xavier_initialized([3, 3, 128, 256], name="W_conv_t2")
            b_conv_t2 = utils.bias_variable([128], name="b_conv_t2")
            deconv_shape = tf.pack([tf.shape(h_relu_t1)[0], 4 * image_size, 4 * image_size, 128])
            h_conv_t2 = utils.conv2d_transpose_strided(h_relu_t1, W_conv_t2, b_conv_t2, output_shape=deconv_shape)
            h_bn_t2 = utils.batch_norm(h_conv_t2, 128, train_mode, scope="conv_t2_bn")
            h_relu_t2 = tf.nn.relu(h_bn_t2)

        with tf.name_scope("dec_conv3") as scope:
            W_conv_t3 = utils.weight_variable_xavier_initialized([3, 3, 64, 128], name="W_conv_t3")
            b_conv_t3 = utils.bias_variable([64], name="b_conv_t3")
            deconv_shape = tf.pack([tf.shape(h_relu_t2)[0], 8 * image_size, 8 * image_size, 64])
            h_conv_t3 = utils.conv2d_transpose_strided(h_relu_t2, W_conv_t3, b_conv_t3, output_shape=deconv_shape)
            h_bn_t3 = utils.batch_norm(h_conv_t3, 64, train_mode, scope="conv_t3_bn")
            h_relu_t3 = tf.nn.relu(h_bn_t3)

        with tf.name_scope("dec_conv4") as scope:
            W_conv_t4 = utils.weight_variable_xavier_initialized([3, 3, 3, 64], name="W_conv_t4")
            b_conv_t4 = utils.bias_variable([3], name="b_conv_t4")
            deconv_shape = tf.pack([tf.shape(h_relu_t3)[0], 16 * image_size, 16 * image_size, 3])
            pred_image = utils.conv2d_transpose_strided(h_relu_t3, W_conv_t4, b_conv_t4, output_shape=deconv_shape)
    return pred_image


def loss(pred, real):
    loss_val = tf.sqrt(2 * tf.nn.l2_loss(tf.sub(pred, real))) / FLAGS.batch_size
    tf.scalar_summary("Loss_objective", loss_val)
    return loss_val


def train(loss_val, step):
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, step, 0.4 * MAX_ITERATIONS, 0.99)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_val, global_step=step)
    return learning_rate, train_op


def get_next_batch(dataset, step):
    batch_image = []
    batch_clip = []
    return batch_image, batch_clip


def _read_input(filename_queue):
    class DataRecord(object):
        pass

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    record = DataRecord()
    decoded_image = tf.image.decode_jpeg(value, channels=3)
    decoded_image.set_shape([DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE, 3])
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    record.image = tf.image.resize_image_with_crop_or_pad(decoded_image_as_float, IMAGE_SIZE, IMAGE_SIZE)

    record.clipped_image = tf.slice(record.image, [IMAGE_SIZE / 4, IMAGE_SIZE / 4, 0],
                                    [IMAGE_SIZE / 2, IMAGE_SIZE / 2, 3])
    padded_image = tf.image.resize_image_with_crop_or_pad(record.clipped_image, IMAGE_SIZE, IMAGE_SIZE)
    record.input_image = tf.sub(record.image, padded_image)
    print record.input_image.get_shape()
    print record.clipped_image.get_shape()
    return record


def read_input_queue(filename_queue):
    read_input = _read_input(filename_queue)
    num_preprocess_threads = 8
    min_queue_examples = int(0.4 * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
    print "Shuffling"
    input_image, clipped_image = tf.train.shuffle_batch([read_input.input_image, read_input.clipped_image],
                                                        batch_size=FLAGS.batch_size,
                                                        num_threads=num_preprocess_threads,
                                                        capacity=min_queue_examples + 3 * FLAGS.batch_size,
                                                        min_after_dequeue=min_queue_examples)
    input_image = (input_image - 128) / 255.0
    clipped_image = (clipped_image - 128) / 255.0
    return input_image, clipped_image


def main(argv=None):
    print "Setting up dataset..."
    train_files, testing_files, validation_files = read_LabeledFacesWild.read_lfw_dataset(FLAGS.data_dir)
    filename_queue = tf.train.string_input_producer(train_files)
    input_image, clipped_image = read_input_queue(filename_queue)

    phase_train = tf.placeholder(tf.bool)
    global_step = tf.Variable(0, trainable=False)

    print "Setting up inference model..."
    embedding = encoder(input_image, phase_train)
    tf.image_summary("Input_image", input_image, max_images=1)
    pred_image = inpainter(embedding, phase_train)
    tf.image_summary("Ground_truth", clipped_image, max_images=1)
    tf.image_summary("Pred_image", pred_image, max_images=1)
    reconst_loss = loss(pred_image, clipped_image)
    learning_rate, train_op = train(reconst_loss, global_step)

    print "Setting up summary and saver..."
    summary_op = tf.merge_all_summaries()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Model restored!"
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            for step in xrange(MAX_ITERATIONS):
                while coord.should_stop():
                    print "Coordinator should stop!"
                    break

                feed_dict = {phase_train: True}
                if step % 100 == 0:
                    train_loss, summary_str = sess.run([reconst_loss, summary_op], feed_dict=feed_dict)
                    print "Step: %d, Train loss: %g" % (step, train_loss)
                    summary_writer.add_summary(summary_str, global_step=step)

                if step % 1000 == 0:
                    lr = sess.run(learning_rate)

                    print "%s ===> Learning Rate: %f" % (datetime.now(), lr)
                    saver.save(sess, FLAGS.logs_dir + "model.ckpt", global_step=step)

                sess.run(train_op, feed_dict=feed_dict)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
