__author__ = 'Charlie'
import numpy as np
import tensorflow as tf
import os, sys, inspect
import scipy.io
import scipy.misc as misc
from datetime import datetime

utils_folder = os.path.abspath(
    os.path.realpath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)

import TensorflowUtils as utils
import AnalogyDataLoader

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "Data_zoo/Analogy_data/", "Path to analogy data")
tf.flags.DEFINE_string("logs_dir", "logs/Analogy_logs/", "Path to logs")
tf.flags.DEFINE_string("mode", "train", "Model mode - train/test")
tf.flags.DEFINE_integer("batch_size", "64", "Batch size for training")

MAX_ITERATIONS = 1 + 1e5
IMAGE_SIZE = 48
LEARNING_RATE = 1e-3
ANALOGY_COEFF = 1e-2
REGULARIZER = 1e-4

DATA_URL = "http://www-personal.umich.edu/~reedscot/files/nips2015-analogy-data.tar.gz"


def add_to_regularization_and_summary(W, b):
    tf.histogram_summary(W.name, W)
    tf.histogram_summary(b.name, b)
    tf.add_to_collection("reg_loss", tf.nn.l2_loss(W))
    tf.add_to_collection("reg_loss", tf.nn.l2_loss(b))


def encoder_conv(image):
    with tf.name_scope("enc_conv1") as scope:
        W_conv1 = utils.weight_variable([3, 3, 1, 32], name="W_conv1")
        b_conv1 = utils.bias_variable([32], name="b_conv1")
        add_to_regularization_and_summary(W_conv1, b_conv1)
        h_conv1 = tf.nn.tanh(utils.conv2d_strided(image, W_conv1, b_conv1))

    with tf.name_scope("enc_conv2") as scope:
        W_conv2 = utils.weight_variable([3, 3, 32, 64], name="W_conv2")
        b_conv2 = utils.bias_variable([64], name="b_conv2")
        add_to_regularization_and_summary(W_conv2, b_conv2)
        h_conv2 = tf.nn.tanh(utils.conv2d_strided(h_conv1, W_conv2, b_conv2))

    with tf.name_scope("enc_conv3") as scope:
        W_conv3 = utils.weight_variable([3, 3, 64, 128], name="W_conv3")
        b_conv3 = utils.bias_variable([128], name="b_conv3")
        add_to_regularization_and_summary(W_conv3, b_conv3)
        h_conv3 = tf.nn.tanh(utils.conv2d_strided(h_conv2, W_conv3, b_conv3))

    with tf.name_scope("enc_conv4") as scope:
        W_conv4 = utils.weight_variable([3, 3, 128, 256], name="W_conv4")
        b_conv4 = utils.bias_variable([256], name="b_conv4")
        add_to_regularization_and_summary(W_conv4, b_conv4)
        h_conv4 = tf.nn.tanh(utils.conv2d_strided(h_conv3, W_conv4, b_conv4))

    with tf.name_scope("enc_fc") as scope:
        image_size = IMAGE_SIZE // 16
        h_conv4_flatten = tf.reshape(h_conv4, [-1, image_size * image_size * 256])
        W_fc5 = utils.weight_variable([image_size * image_size * 256, 512], name="W_fc5")
        b_fc5 = utils.bias_variable([512], name="b_fc5")
        add_to_regularization_and_summary(W_fc5, b_fc5)
        encoder_val = tf.matmul(h_conv4_flatten, W_fc5) + b_fc5

    return encoder_val


def decoder_conv(embedding):
    with tf.name_scope("dec_fc") as scope:
        image_size = IMAGE_SIZE // 16
        W_fc1 = utils.weight_variable([512, image_size * image_size * 256], name="W_fc1")
        b_fc1 = utils.bias_variable([image_size * image_size * 256], name="b_fc1")
        add_to_regularization_and_summary(W_fc1, b_fc1)
        h_fc1 = tf.nn.relu(tf.matmul(embedding, W_fc1) + b_fc1)

    with tf.name_scope("dec_conv1") as scope:
        h_reshaped = tf.reshape(h_fc1, [-1, image_size, image_size, 256])
        W_conv_t1 = utils.weight_variable([3, 3, 128, 256], name="W_conv_t1")
        b_conv_t1 = utils.bias_variable([128], name="b_conv_t1")
        add_to_regularization_and_summary(W_conv_t1, b_conv_t1)
        h_conv_t1 = tf.nn.relu(utils.conv2d_transpose_strided(h_reshaped, W_conv_t1, b_conv_t1))

    with tf.name_scope("dec_conv2") as scope:
        W_conv_t2 = utils.weight_variable([3, 3, 128, 64], name="W_conv_t2")
        b_conv_t2 = utils.bias_variable([64], name="b_conv_t2")
        add_to_regularization_and_summary(W_conv_t2, b_conv_t2)
        h_conv_t2 = tf.nn.relu(utils.conv2d_transpose_strided(h_conv_t1, W_conv_t2, b_conv_t2))

    with tf.name_scope("dec_conv3") as scope:
        W_conv_t3 = utils.weight_variable([3, 3, 64, 32], name="W_conv_t3")
        b_conv_t3 = utils.bias_variable([32], name="b_conv_t3")
        add_to_regularization_and_summary(W_conv_t3, b_conv_t3)
        h_conv_t3 = tf.nn.relu(utils.conv2d_transpose_strided(h_conv_t2, W_conv_t3, b_conv_t3))

    with tf.name_scope("dec_conv4") as scope:
        W_conv_t4 = utils.weight_variable([3, 3, 32, 3], name="W_conv_t4")
        b_conv_t4 = utils.bias_variable([3], name="b_conv_t4")
        add_to_regularization_and_summary(W_conv_t4, b_conv_t4)
        pred_image = tf.matmul(h_conv_t3, W_conv_t4, b_conv_t4)

    return pred_image


def read_train_inputs(loader):
    return loader.next()


def read_eval_inputs(loader):
    return loader.tests['rotate']


def train(loss, step):
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step=step, decay_steps=MAX_ITERATIONS / 4,
                                               decay_rate=0.99)
    return tf.train.RMSPropOptimizer(learning_rate, 0.99, momentum=0.9)


def main(argv=None):
    utils.maybe_download_and_extract(FLAGS.data_dir, DATA_URL, is_tarfile=True)
    print "Initializing Loader class..."
    loader = AnalogyDataLoader.Loader(FLAGS.data_dir, FLAGS.batch_size)
    global_step = tf.Variable(0, trainable=False)

    img_A = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
    img_B = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
    img_C = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
    img_D = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])

    tf.image_summary("Ground_truth", img_D, max_images=5)

    print "Setting up encoder.."
    enc_A = encoder_conv(img_A)
    enc_B = encoder_conv(img_B)
    enc_C = encoder_conv(img_C)
    enc_D = encoder_conv(img_D)

    print "Setting up analogy calc.."
    # analogy calculation
    analogy_input = tf.concat(1, [enc_B - enc_A, enc_C])
    W_analogy1 = utils.weight_variable([1024, 512], name="W_analogy1")
    b_analogy1 = utils.bias_variable([512], name="b_analogy1")
    analogy_fc1 = tf.nn.relu(tf.matmul(analogy_input, W_analogy1) + b_analogy1)

    W_analogy2 = utils.weight_variable([512, 512], name="W_analogy2")
    b_analogy2 = utils.bias_variable([512], name="b_analogy2")
    analogy_fc2 = tf.nn.relu(tf.matmul(analogy_fc1, W_analogy2) + b_analogy2)

    pred = decoder_conv(enc_C + analogy_fc2)
    tf.image_summary("Pred_image", pred, max_images=5)

    print "Loss and train setup..."
    loss1 = tf.nn.l2_loss(pred - enc_D) / FLAGS.batch_size
    tf.scalar_summary("image_loss", loss1)
    loss2 = tf.nn.l2_loss(enc_D - enc_C - analogy_fc2)
    tf.scalar_summary("analogy_loss", loss2)
    loss3 = tf.add_n(tf.get_collection("reg_loss"))
    tf.scalar_summary("regularization", loss3)

    total_loss = loss1 + ANALOGY_COEFF * loss2  # + REGULARIZER*loss3
    tf.scalar_summary("Total_loss", total_loss)
    train_op = train(total_loss, global_step)

    summary_op = tf.merge_all_summaries()
    eval_A, eval_B, eval_C, eval_D = read_eval_inputs(loader)
    eval_feed = {img_A: eval_A, img_B: eval_B, img_C: eval_C, img_D: eval_D}
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print "Setting up summary and saver..."
        summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph_def)
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Model restored!"

        for step in xrange(MAX_ITERATIONS):
            A, B, C, D = read_train_inputs(loader)
            feed_dict = {img_A: A, img_B: B, img_C: C, img_D: D}
            if step % 5000 == 0:
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", global_step=step)
                eval_loss = sess.run([loss1, loss2, loss3, total_loss], feed_dict=eval_feed)
                print "Evaluation: (Image loss %f, Variation loss %f, Reg loss %f) total loss %f"

            if step % 100 == 0:
                [loss_val, summary_str] = sess.run([total_loss, summary_op], feed_dict=feed_dict)
                print "%s Step %d: Training loss %f" % (datetime.now(), step, loss_val)
                summary_writer.add_summary(summary_str, global_step=step)

            sess.run(train_op, feed_dict=feed_dict)

if __name__ == "__main__":
    tf.app.run()
