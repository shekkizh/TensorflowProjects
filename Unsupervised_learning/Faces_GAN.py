from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, inspect

# import tensorflow.examples.tutorials.mnist as mnist

utils_folder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)

import TensorflowUtils as utils
import Dataset_Reader.read_celebADataset as celebA

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/CelebA_GAN_logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/CelebA_faces/", "path to dataset")
tf.flags.DEFINE_integer("z_dim", "100", "size of input vector to generator")
tf.flags.DEFINE_float("learning_rate", "2e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("beta1", "0.5", "beta1 for Adam optimizer")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 800
MAX_ITERATIONS = int(1e5 + 1)
MODEL_IMAGE_SIZE = 108
IMAGE_SIZE = 64
GEN_DIMENSION = 16


def _read_input(filename_queue):
    class DataRecord(object):
        pass

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    record = DataRecord()
    decoded_image = tf.image.decode_jpeg(value, channels=3)
    decoded_image_4d = tf.expand_dims(decoded_image, 0)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, [IMAGE_SIZE, IMAGE_SIZE])
    record.input_image = tf.squeeze(resized_image, squeeze_dims=[0])
    cropped_image = tf.cast(tf.image.crop_to_bounding_box(decoded_image, 55, 35, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
                            tf.float32)
    decoded_image_4d = tf.expand_dims(cropped_image, 0)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, [IMAGE_SIZE, IMAGE_SIZE])
    record.input_image = tf.squeeze(resized_image, squeeze_dims=[0])
    return record


def read_input_queue(filename_queue):
    read_input = _read_input(filename_queue)
    num_preprocess_threads = 4
    min_queue_examples = int(0.1 * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
    print("Shuffling")
    input_image = tf.train.batch([read_input.input_image],
                                 batch_size=FLAGS.batch_size,
                                 num_threads=num_preprocess_threads,
                                 capacity=min_queue_examples + 2 * FLAGS.batch_size
                                 )
    input_image = (input_image - 128) / 128.0
    return input_image


def generator(z, train_mode):
    with tf.variable_scope("generator") as scope:
        W_0 = utils.weight_variable([FLAGS.z_dim, 64 * GEN_DIMENSION / 2 * IMAGE_SIZE / 16 * IMAGE_SIZE / 16],
                                    name="W_0")
        b_0 = utils.bias_variable([64 * GEN_DIMENSION / 2 * IMAGE_SIZE / 16 * IMAGE_SIZE / 16], name="b_0")
        z_0 = tf.matmul(z, W_0) + b_0
        h_0 = tf.reshape(z_0, [-1, IMAGE_SIZE / 16, IMAGE_SIZE / 16, 64 * GEN_DIMENSION / 2])
        h_bn0 = utils.batch_norm(h_0, 64 * GEN_DIMENSION / 2, train_mode, scope="gen_bn0")
        h_relu0 = tf.nn.relu(h_bn0, name='relu0')
        utils.add_activation_summary(h_relu0)

        # W_1 = utils.weight_variable_xavier_initialized([5, 5, 64 * GEN_DIMENSION/2, 64 * GEN_DIMENSION], name="W_1")
        # b_1 = utils.bias_variable([64 * GEN_DIMENSION/2], name="b_1")
        # deconv_shape = tf.pack([tf.shape(h_relu0)[0], IMAGE_SIZE / 16, IMAGE_SIZE / 16, 64 * GEN_DIMENSION/2])
        # h_conv_t1 = utils.conv2d_transpose_strided(h_relu0, W_1, b_1, output_shape=deconv_shape)
        # h_bn1 = utils.batch_norm(h_conv_t1, 64 * GEN_DIMENSION/2, train_mode, scope="gen_bn1")
        # h_relu1 = tf.nn.relu(h_bn1, name='relu1')
        # utils.add_activation_summary(h_relu1)

        W_2 = utils.weight_variable_xavier_initialized([5, 5, 64 * GEN_DIMENSION / 4, 64 * GEN_DIMENSION / 2],
                                                       name="W_2")
        b_2 = utils.bias_variable([64 * GEN_DIMENSION / 4], name="b_2")
        deconv_shape = tf.pack([tf.shape(h_relu0)[0], IMAGE_SIZE / 8, IMAGE_SIZE / 8, 64 * GEN_DIMENSION / 4])
        h_conv_t2 = utils.conv2d_transpose_strided(h_relu0, W_2, b_2, output_shape=deconv_shape)
        h_bn2 = utils.batch_norm(h_conv_t2, 64 * GEN_DIMENSION / 4, train_mode, scope="gen_bn2")
        h_relu2 = tf.nn.relu(h_bn2, name='relu2')
        utils.add_activation_summary(h_relu2)

        W_3 = utils.weight_variable_xavier_initialized([5, 5, 64 * GEN_DIMENSION / 8, 64 * GEN_DIMENSION / 4],
                                                       name="W_3")
        b_3 = utils.bias_variable([64 * GEN_DIMENSION / 8], name="b_3")
        deconv_shape = tf.pack([tf.shape(h_relu2)[0], IMAGE_SIZE / 4, IMAGE_SIZE / 4, 64 * GEN_DIMENSION / 8])
        h_conv_t3 = utils.conv2d_transpose_strided(h_relu2, W_3, b_3, output_shape=deconv_shape)
        h_bn3 = utils.batch_norm(h_conv_t3, 64 * GEN_DIMENSION / 8, train_mode, scope="gen_bn3")
        h_relu3 = tf.nn.relu(h_bn3, name='relu3')
        utils.add_activation_summary(h_relu3)

        W_4 = utils.weight_variable_xavier_initialized([5, 5, 64 * GEN_DIMENSION / 16, 64 * GEN_DIMENSION / 8],
                                                       name="W_4")
        b_4 = utils.bias_variable([64 * GEN_DIMENSION / 16], name="b_4")
        deconv_shape = tf.pack([tf.shape(h_relu3)[0], IMAGE_SIZE / 2, IMAGE_SIZE / 2, 64 * GEN_DIMENSION / 16])
        h_conv_t4 = utils.conv2d_transpose_strided(h_relu3, W_4, b_4, output_shape=deconv_shape)
        h_bn4 = utils.batch_norm(h_conv_t4, 64 * GEN_DIMENSION / 16, train_mode, scope="gen_bn4")
        h_relu4 = tf.nn.relu(h_bn4, name='relu4')
        utils.add_activation_summary(h_relu4)

        W_5 = utils.weight_variable_xavier_initialized([5, 5, 3, 64 * GEN_DIMENSION / 16], name="W_5")
        b_5 = utils.bias_variable([3], name="b_5")
        deconv_shape = tf.pack([tf.shape(h_relu4)[0], IMAGE_SIZE, IMAGE_SIZE, 3])
        h_conv_t5 = utils.conv2d_transpose_strided(h_relu4, W_5, b_5, output_shape=deconv_shape)
        pred_image = tf.nn.tanh(h_conv_t5, name='pred_image')
        utils.add_activation_summary(pred_image)

    return pred_image


def discriminator(input_images, train_mode):
    # dropout_prob = 1.0
    # if train_mode:
    #     dropout_prob = 0.5
    W_conv0 = utils.weight_variable_xavier_initialized([5, 5, 3, 64 * 1], name="W_conv0")
    b_conv0 = utils.bias_variable([64 * 1], name="b_conv0")
    h_conv0 = utils.conv2d_strided(input_images, W_conv0, b_conv0)
    h_bn0 = h_conv0  # utils.batch_norm(h_conv0, 64 * 1, train_mode, scope="disc_bn0")
    h_relu0 = utils.leaky_relu(h_bn0, 0.2, name="h_relu0")
    utils.add_activation_summary(h_relu0)

    W_conv1 = utils.weight_variable_xavier_initialized([5, 5, 64 * 1, 64 * 2], name="W_conv1")
    b_conv1 = utils.bias_variable([64 * 2], name="b_conv1")
    h_conv1 = utils.conv2d_strided(h_relu0, W_conv1, b_conv1)
    h_bn1 = utils.batch_norm(h_conv1, 64 * 2, train_mode, scope="disc_bn1")
    h_relu1 = utils.leaky_relu(h_bn1, 0.2, name="h_relu1")
    utils.add_activation_summary(h_relu1)

    W_conv2 = utils.weight_variable_xavier_initialized([5, 5, 64 * 2, 64 * 4], name="W_conv2")
    b_conv2 = utils.bias_variable([64 * 4], name="b_conv2")
    h_conv2 = utils.conv2d_strided(h_relu1, W_conv2, b_conv2)
    h_bn2 = utils.batch_norm(h_conv2, 64 * 4, train_mode, scope="disc_bn2")
    h_relu2 = utils.leaky_relu(h_bn2, 0.2, name="h_relu2")
    utils.add_activation_summary(h_relu2)

    W_conv3 = utils.weight_variable_xavier_initialized([5, 5, 64 * 4, 64 * 8], name="W_conv3")
    b_conv3 = utils.bias_variable([64 * 8], name="b_conv3")
    h_conv3 = utils.conv2d_strided(h_relu2, W_conv3, b_conv3)
    h_bn3 = utils.batch_norm(h_conv3, 64 * 8, train_mode, scope="disc_bn3")
    h_relu3 = utils.leaky_relu(h_bn3, 0.2, name="h_relu3")
    utils.add_activation_summary(h_relu3)

    shape = h_relu3.get_shape().as_list()
    h_3 = tf.reshape(h_relu3, [FLAGS.batch_size, (IMAGE_SIZE // 16) * (IMAGE_SIZE // 16) * shape[3]])
    W_4 = utils.weight_variable([h_3.get_shape().as_list()[1], 1], name="W_4")
    b_4 = utils.bias_variable([1], name="b_4")
    h_4 = tf.matmul(h_3, W_4) + b_4

    return tf.nn.sigmoid(h_4), h_4


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    for grad, var in grads:
        utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    train_images, valid_images, test_images = celebA.read_dataset(FLAGS.data_dir)
    filename_queue = tf.train.string_input_producer(train_images)
    images = read_input_queue(filename_queue)

    train_phase = tf.placeholder(tf.bool)
    z_vec = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.z_dim], name="z")
    # images = tf.placeholder(tf.float32, [FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3], name="images_real")
    tf.histogram_summary("z", z_vec)
    tf.image_summary("image_real", images, max_images=2)
    gen_images = generator(z_vec, train_phase)
    tf.image_summary("image_generated", gen_images, max_images=2)

    with tf.variable_scope("discriminator") as scope:
        discriminator_real_prob, logits_real = discriminator(images, train_phase)
        utils.add_activation_summary(tf.identity(discriminator_real_prob, name='disc_real_prob'))
        scope.reuse_variables()
        discriminator_fake_prob, logits_fake = discriminator(gen_images, train_phase)
        utils.add_activation_summary(tf.identity(discriminator_fake_prob, name='disc_fake_prob'))

    discriminator_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits_real, tf.ones_like(logits_real)))
    discrimintator_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits_fake, tf.zeros_like(logits_fake)))
    discriminator_loss = discrimintator_loss_fake + discriminator_loss_real
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_fake, tf.ones_like(logits_fake)))

    tf.scalar_summary("Discriminator_loss_real", discriminator_loss_real)
    tf.scalar_summary("Discrimintator_loss_fake", discrimintator_loss_fake)
    tf.scalar_summary("Discriminator_loss", discriminator_loss)
    tf.scalar_summary("Generator_loss", gen_loss)

    train_variables = tf.all_variables()
    generator_variables = [v for v in train_variables if v.name.startswith("generator")]
    discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]

    generator_train_op = train(gen_loss, generator_variables)
    discriminator_train_op = train(discriminator_loss, discriminator_variables)

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
        print("Model restored...")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    try:
        for itr in xrange(MAX_ITERATIONS):
            batch_z = np.random.uniform(-1.0, 1.0, size=[FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)
            feed_dict = {z_vec: batch_z, train_phase: True}

            # if itr % 2 == 0:
            #     sess.run(discriminator_train_op, feed_dict=feed_dict)
            sess.run([discriminator_train_op, generator_train_op, generator_train_op], feed_dict=feed_dict)

            if itr % 10 == 0:
                g_loss_val, d_loss_val, summary_str = sess.run([gen_loss, discriminator_loss, summary_op],
                                                               feed_dict=feed_dict)
                print("Step: %d, generator loss: %g, discriminator_loss: %g" % (itr, g_loss_val, d_loss_val))
                summary_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", global_step=itr)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
