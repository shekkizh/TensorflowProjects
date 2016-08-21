import tensorflow as tf
import numpy as np
import os, sys, inspect
import tensorflow.examples.tutorials.mnist as mnist

utils_folder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)

import TensorflowUtils as utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "128", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/MNIST_GAN_logs/", "path to logs directory")
tf.flags.DEFINE_integer("z_dim", "100", "size of input vector to generator")
tf.flags.DEFINE_float("learning_rate", "2e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("beta1", "0.5", "beta1 for Adam optimizer")
MAX_ITERATIONS = int(1e5 + 1)
IMAGE_SIZE = 28


def generator(z, train_mode):
    with tf.variable_scope("generator") as scope:
        W_0 = utils.weight_variable([FLAGS.z_dim, 64 * 4 * 4 * 4], name="W_0")
        b_0 = utils.bias_variable([64 * 4 * 4 * 4], name="b_0")
        z_0 = tf.matmul(z, W_0) + b_0
        h_0 = tf.reshape(z_0, [-1, 4, 4, 64 * 4])
        h_bn0 = h_0  # utils.batch_norm(h_0, 64 * 4, train_mode, scope="gen_bn0")
        h_relu0 = tf.nn.relu(h_bn0)
        utils.add_activation_summary(h_relu0)

        W_1 = utils.weight_variable_xavier_initialized([5, 5, 64 * 2, 64 * 4], name="W_1")
        b_1 = utils.bias_variable([64 * 2], name="b_1")
        deconv_shape = tf.pack([tf.shape(h_relu0)[0], 7, 7, 64 * 2])
        h_conv_t1 = utils.conv2d_transpose_strided(h_relu0, W_1, b_1, output_shape=deconv_shape)
        h_bn1 = h_conv_t1  # utils.batch_norm(h_conv_t1, 64 * 2, train_mode, scope="gen_bn1")
        h_relu1 = tf.nn.relu(h_bn1)
        utils.add_activation_summary(h_relu1)

        W_2 = utils.weight_variable_xavier_initialized([5, 5, 64 * 1, 64 * 2], name="W_2")
        b_2 = utils.bias_variable([64 * 1], name="b_2")
        deconv_shape = tf.pack([tf.shape(h_relu1)[0], 14, 14, 64 * 1])
        h_conv_t2 = utils.conv2d_transpose_strided(h_relu1, W_2, b_2, output_shape=deconv_shape)
        h_bn2 = h_conv_t2  # utils.batch_norm(h_conv_t2, 64 * 1, train_mode, scope="gen_bn2")
        h_relu2 = tf.nn.relu(h_bn2)
        utils.add_activation_summary(h_relu2)

        W_3 = utils.weight_variable_xavier_initialized([5, 5, 1, 64 * 1], name="W_3")
        b_3 = utils.bias_variable([1], name="b_3")
        deconv_shape = tf.pack([tf.shape(h_relu2)[0], 28, 28, 1])
        h_conv_t3 = utils.conv2d_transpose_strided(h_relu2, W_3, b_3, output_shape=deconv_shape)
        pred_image = tf.nn.sigmoid(h_conv_t3)

    return pred_image


def discriminator(input_images, train_mode):
    # dropout_prob = 1.0
    # if train_mode:
    #     dropout_prob = 0.5
    W_conv0 = utils.weight_variable_xavier_initialized([5, 5, 1, 64 * 1], name="W_conv0")
    b_conv0 = utils.bias_variable([64 * 1], name="b_conv0")
    h_conv0 = utils.conv2d_strided(input_images, W_conv0, b_conv0)
    h_bn0 = utils.batch_norm(h_conv0, 64 * 1, train_mode, scope="disc_bn0")
    h_relu0 = utils.leaky_relu(h_bn0, 0.2)
    utils.add_activation_summary(h_relu0)

    W_conv1 = utils.weight_variable_xavier_initialized([5, 5, 64 * 1, 64 * 2], name="W_conv1")
    b_conv1 = utils.bias_variable([64 * 2], name="b_conv1")
    h_conv1 = utils.conv2d_strided(h_relu0, W_conv1, b_conv1)
    h_bn1 = utils.batch_norm(h_conv1, 64 * 2, train_mode, scope="disc_bn1")
    h_relu1 = utils.leaky_relu(h_bn1, 0.2)
    utils.add_activation_summary(h_relu1)

    W_conv2 = utils.weight_variable_xavier_initialized([5, 5, 64 * 2, 64 * 4], name="W_conv2")
    b_conv2 = utils.bias_variable([64 * 4], name="b_conv2")
    h_conv2 = utils.conv2d_strided(h_relu1, W_conv2, b_conv2)
    h_bn2 = utils.batch_norm(h_conv2, 64 * 4, train_mode, scope="disc_bn2")
    h_relu2 = utils.leaky_relu(h_bn2, 0.2)
    utils.add_activation_summary(h_relu2)

    shape = h_relu2.get_shape().as_list()
    h_2 = tf.reshape(h_relu2, [FLAGS.batch_size, 4 * 4 * shape[3]])
    W_3 = utils.weight_variable([h_2.get_shape().as_list()[1], 1], name="W_3")
    b_3 = utils.bias_variable([1], name="b_3")
    h_3 = tf.matmul(h_2, W_3) + b_3

    return tf.nn.sigmoid(h_3), h_3


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    for grad, var in grads:
        utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    train_phase = tf.placeholder(tf.bool)
    z_vec = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.z_dim], name="z")
    images = tf.placeholder(tf.float32, [FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 1], name="images_real")
    tf.histogram_summary("z", z_vec)
    tf.image_summary("image_real", 255 * images, max_images=2)
    gen_images = generator(z_vec, train_phase)
    tf.image_summary("image_generated", 255 * gen_images, max_images=2)
    with tf.variable_scope("discriminator") as scope:
        discriminator_real_prob, logits_real = discriminator(images, train_phase)
        scope.reuse_variables()
        discriminator_fake_prob, logits_fake = discriminator(gen_images, train_phase)

    discriminator_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits_real, tf.ones_like(logits_real)))
    discrimintator_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits_fake, tf.zeros_like(logits_fake)))
    discriminator_loss = discrimintator_loss_fake + discriminator_loss_real
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_fake, tf.ones_like(logits_fake)))

    tf.scalar_summary("Discriminator_loss_real", discriminator_loss_real)
    tf.scalar_summary("Discrimintator_loss_fake", discrimintator_loss_fake)
    tf.scalar_summary("Generator_loss", gen_loss)

    train_variables = tf.all_variables()
    generator_variables = [v for v in train_variables if v.name.startswith("generator")]
    discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]

    generator_train_op = train(gen_loss, generator_variables)
    discriminator_train_op = train(discriminator_loss / 2, discriminator_variables)

    for v in train_variables:
        utils.add_to_regularization_and_summary(var=v)

    sess = tf.Session()
    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver()
    summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.initialize_all_variables())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print "Model restored..."

    dataset = mnist.input_data.read_data_sets("MNIST_data", one_hot=True)
    for itr in xrange(MAX_ITERATIONS):
        batch_images, batch_labels = dataset.train.next_batch(FLAGS.batch_size)
        batch_z = np.random.uniform(-1.0, 1.0, size=[FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)

        sess.run(generator_train_op,
                 feed_dict={z_vec: batch_z, images: batch_images.reshape([-1, 28, 28, 1]) / 255.0, train_phase: True})
        if itr % 10 == 0:
            sess.run(discriminator_train_op,
                     feed_dict={z_vec: batch_z, images: batch_images.reshape([-1, 28, 28, 1]) / 255.0,
                                train_phase: True})
        if itr % 100 == 0:
            g_loss_val, d_loss_val, summary_str = sess.run([gen_loss, discriminator_loss, summary_op],
                                                           feed_dict={z_vec: batch_z,
                                                                      images: batch_images.reshape(
                                                                          [-1, 28, 28, 1]) / 255.0,
                                                                      train_phase: True})
            print ("Step: %d, generator loss: %g, discriminator_loss: %g" % (itr, g_loss_val, d_loss_val))
            summary_writer.add_summary(summary_str, itr)

        if itr % 1000 == 0:
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", global_step=itr)
            test_discriminator_loss = sess.run(discriminator_loss_real,
                                               feed_dict={images: dataset.test.next_batch(FLAGS.batch_size)[0].reshape(
                                                   [-1, 28, 28, 1]) / 255.0, train_phase: False})
            print("Test images discriminator loss: %g" % test_discriminator_loss)


if __name__ == "__main__":
    tf.app.run()
