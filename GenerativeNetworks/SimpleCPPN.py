__author__ = 'Charlie'
# Implementation draws details from https://github.com/hardmaru/cppn-tensorflow
import numpy as np
import tensorflow as tf
import os, sys, inspect
# import scipy.misc as misc
import matplotlib.pyplot as plt

utils_folder = os.path.abspath(
    os.path.realpath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)

import TensorflowUtils as utils







FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("image_size", "256", "Image size")
tf.flags.DEFINE_integer("genome_length", "10", "Length of genome to be given as input")
tf.flags.DEFINE_integer("net_size", "32", "Breadth of hidden units")
tf.flags.DEFINE_string("mode", "color", "Color/ Gray image output")



def show_image(image):
    plt.subplot(1, 1, 1)
    if NUM_CHANNELS == 1:
        plt.imshow(255*image.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='Greysarr ', interpolation='nearest')
    else:
        plt.imshow(255 * image, interpolation='nearest')
    plt.axis('off')
    plt.show()


def mlp(inputs, output_dimension, scope=""):
    shape = inputs.get_shape().as_list()
    W_fc1 = utils.weight_variable([shape[1], output_dimension])
    b_fc1 = utils.bias_variable([output_dimension])
    linear = tf.matmul(inputs, W_fc1) + b_fc1
    return linear


def generate_image(x, y, r, z):
    with tf.name_scope("input_map") as scope:
        reshape_size = BATCH_SIZE * IMAGE_SIZE * IMAGE_SIZE
        x_reshpaed = tf.reshape(x, [reshape_size, 1])
        x_linear = mlp(x_reshpaed, FLAGS.net_size)
        # var_scope.reuse_variables()
        y_reshaped = tf.reshape(y, [reshape_size, 1])
        y_linear = mlp(y_reshaped, FLAGS.net_size)
        r_reshaped = tf.reshape(r, [reshape_size, 1])
        r_linear = mlp(r_reshaped, FLAGS.net_size)

    with tf.name_scope("z_input") as scope:
        z_reshaped = tf.reshape(z, (BATCH_SIZE, 1, Z_DIMENSION)) * tf.ones((IMAGE_SIZE * IMAGE_SIZE, 1),
                                                                           dtype=tf.float32)
        z_reshaped = tf.reshape(z_reshaped, [reshape_size, Z_DIMENSION])
        z_linear = mlp(z_reshaped, FLAGS.net_size)

    with tf.name_scope("hidden") as scope:
        h_1 = tf.nn.tanh(x_linear + y_linear + r_linear + z_linear)
        h_2 = tf.nn.tanh(mlp(h_1, FLAGS.net_size))
        h_3 = tf.nn.tanh(mlp(h_2, FLAGS.net_size))
        h_4 = tf.nn.tanh(mlp(h_3, FLAGS.net_size))

    with tf.name_scope("output") as scope:
        output = tf.sigmoid(mlp(h_4, NUM_CHANNELS))

    return tf.reshape(output, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])


def get_coordinates():
    x_range = -1 + np.arange(IMAGE_SIZE) * 2.0 / (IMAGE_SIZE - 1)  # uniform points from -1 to 1 mapped to image size
    y_range = -1 + np.arange(IMAGE_SIZE) * 2.0 / (IMAGE_SIZE - 1)
    x_mat = np.matmul(np.ones((IMAGE_SIZE, 1)), x_range.reshape((1, IMAGE_SIZE)))
    y_mat = np.matmul(y_range.reshape((IMAGE_SIZE, 1)), np.ones((1, IMAGE_SIZE)))
    r_mat = (x_mat ** 2 + y_mat ** 2) ** 0.5
    x_mat = np.expand_dims(np.tile(x_mat.flatten(), (BATCH_SIZE, 1)), axis=2)
    y_mat = np.expand_dims(np.tile(y_mat.flatten(), (BATCH_SIZE, 1)), axis=2)
    r_mat = np.expand_dims(np.tile(r_mat.flatten(), (BATCH_SIZE, 1)), axis=2)
    return x_mat, y_mat, r_mat


def main(argv=None):
    print "Setting up variables..."
    z = tf.placeholder(tf.float32, [BATCH_SIZE, Z_DIMENSION])
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE, 1])
    y = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE, 1])
    r = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE, 1])

    gen_image = generate_image(x, y, r, z)



    z_vec = np.random.normal(size=(BATCH_SIZE, Z_DIMENSION))
    # z_vec = np.random.uniform(-1.0, 1.0, size=(BATCH_SIZE, Z_DIMENSION)).astype(np.float32)

    # with open("genome.csv", "a") as f:
    #     np.savetxt(f, z_vec, delimiter=",")


    x_vec, y_vec, r_vec = get_coordinates()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        feed = {x: x_vec, y: y_vec, r: r_vec, z: z_vec}
        image = sess.run(gen_image, feed_dict=feed)
        show_image(image[0])


if __name__ == "__main__":
    IMAGE_SIZE = FLAGS.image_size
    NUM_CHANNELS = 3 if FLAGS.mode == "color" else 1
    BATCH_SIZE = 1
    Z_DIMENSION = FLAGS.genome_length

    tf.app.run()
