import numpy as np
import tensorflow as tf
import os, sys, inspect
import tensorflow.examples.tutorials.mnist as mnist
import pandas as pd

utils_folder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)

import TensorflowUtils as utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("z_dim", "3", "Latent variable dimension")
tf.flags.DEFINE_integer("batch_size", "256", "Train batch size")
tf.flags.DEFINE_string("logs_dir", "logs/MNIST_VAE_logs/", "Path to logs dir")
tf.flags.DEFINE_string("activation", "relu", "Activation function to use in network")
tf.flags.DEFINE_float("regularization", "1e-5", "Regularization multiplier value")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate")

IMAGE_SIZE = 28
MAX_ITERATIONS = int(1 + 1e5)
LEARNING_RATE = FLAGS.learning_rate
NUM_LABELS = 10


def activation_function(x, name=""):
    activation_dict = {'relu': tf.nn.relu(x, name), 'elu': tf.nn.elu(x, name), 'lrelu': utils.leaky_relu(x, 0.2, name),
                       'tanh': tf.nn.tanh(x, name),
                       'sigmoid': tf.nn.sigmoid(x, name)}
    act = activation_dict[FLAGS.activation]
    utils.add_activation_summary(act)
    return act


def encoder_fc(images):
    with tf.variable_scope("encoder") as scope:
        W_fc1 = utils.weight_variable([IMAGE_SIZE * IMAGE_SIZE, 50], name="W_fc1")
        b_fc1 = utils.bias_variable([50], name="b_fc1")
        h_relu1 = activation_function(tf.matmul(images, W_fc1) + b_fc1, name="hfc_1")

        W_fc2 = utils.weight_variable([50, 50], name="W_fc2")
        b_fc2 = utils.bias_variable([50], name="b_fc2")
        h_relu2 = activation_function(tf.matmul(h_relu1, W_fc2) + b_fc2, name="hfc_2")

        W_fc3 = utils.weight_variable([50, FLAGS.z_dim], name="W_fc3")
        b_fc3 = utils.bias_variable([FLAGS.z_dim], name="b_fc3")
        mu = tf.add(tf.matmul(h_relu2, W_fc3), b_fc3, name="mu")
        utils.add_activation_summary(mu)

        W_fc4 = utils.weight_variable([50, FLAGS.z_dim], name="W_fc4")
        b_fc4 = utils.bias_variable([FLAGS.z_dim], name="b_fc4")
        log_var = tf.add(tf.matmul(h_relu2, W_fc4), b_fc4, name="log_var")
        utils.add_activation_summary(log_var)

    return mu, log_var


def decoder_fc(z):
    with tf.variable_scope("decoder") as scope:
        Wd_fc1 = utils.weight_variable([FLAGS.z_dim, 50], name="Wd_fc1")
        bd_fc1 = utils.bias_variable([50], name="bd_fc1")
        hd_relu1 = activation_function(tf.matmul(z, Wd_fc1) + bd_fc1, name="hdfc_1")

        Wd_fc2 = utils.weight_variable([50, 50], name="Wd_fc2")
        bd_fc2 = utils.bias_variable([50], name="bd_fc2")
        hd_relu2 = activation_function(tf.matmul(hd_relu1, Wd_fc2) + bd_fc2, name="hdfc_2")

        Wd_fc3 = utils.weight_variable([50, IMAGE_SIZE * IMAGE_SIZE], name="Wd_fc3")
        bd_fc3 = utils.bias_variable([IMAGE_SIZE * IMAGE_SIZE], name="bd_fc3")
        pred_image = tf.matmul(hd_relu2, Wd_fc3) + bd_fc3
    return pred_image


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    for grad, var in grads:
        utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    data = mnist.input_data.read_data_sets("MNIST_data", one_hot=False)
    images = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name="input_image")
    tf.image_summary("Input", tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, 1]), max_images=2)

    mu, log_var = encoder_fc(images)
    epsilon = tf.random_normal(tf.shape(mu), name="epsilon")
    z = mu + tf.mul(tf.exp(log_var * 0.5), epsilon)

    pred_image = decoder_fc(z)
    entropy_loss = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(pred_image, images, name="entropy_loss"), reduction_indices=1)
    tf.histogram_summary("Entropy_loss", entropy_loss)
    pred_image_sigmoid = tf.nn.sigmoid(pred_image)
    tf.image_summary("Output", tf.reshape(pred_image_sigmoid, [-1, IMAGE_SIZE, IMAGE_SIZE, 1]), max_images=2)

    KL_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var), reduction_indices=1)
    tf.histogram_summary("KL_Divergence", KL_loss)

    train_variables = tf.trainable_variables()
    for v in train_variables:
        utils.add_to_regularization_and_summary(var=v)

    reg_loss = tf.add_n(tf.get_collection("reg_loss"))
    tf.scalar_summary("Reg_loss", reg_loss)
    total_loss = tf.reduce_mean(KL_loss + entropy_loss) + FLAGS.regularization * reg_loss
    tf.scalar_summary("total_loss", total_loss)
    train_op = train(total_loss, train_variables)

    sess = tf.Session()
    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver()
    summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.initialize_all_variables())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print ("Model restored...")

    for itr in xrange(MAX_ITERATIONS):
        batch_images, batch_labels = data.train.next_batch(FLAGS.batch_size)
        sess.run(train_op, feed_dict={images: batch_images})

        if itr % 500 == 0:
            entr_loss, KL_div, tot_loss, summary_str = sess.run([entropy_loss, KL_loss, total_loss, summary_op],
                                                                feed_dict={images: batch_images})
            print (
                "Step: %d, Entropy loss: %g, KL Divergence: %g, Total loss: %g" % (itr, np.mean(entr_loss), np.mean(KL_div), tot_loss))
            summary_writer.add_summary(summary_str, itr)

        if itr % 1000 == 0:
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", global_step=itr)

    def test():
        z_vec = sess.run(z, feed_dict={images: data.test.images})
        write_array = np.hstack((z_vec, np.reshape(data.test.labels, (-1, 1))))
        df = pd.DataFrame(write_array)
        df.to_csv("z_vae_output.csv", header=False, index=False)

    test()


if __name__ == "__main__":
    tf.app.run()
