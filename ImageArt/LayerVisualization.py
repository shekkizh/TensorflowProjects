__author__ = 'Charlie'
# Attempt to learn alignment info given image and its reference

import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
from datetime import datetime
import os, sys, inspect

utils_path = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

import TensorflowUtils as utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("visualize_layer", "conv5_4", "Layer to visualize")
tf.flags.DEFINE_integer("visualize_filter", "0", """filter to visualize in a layer""")
tf.flags.DEFINE_string("model_dir", "Models_zoo/", """Path to the VGGNet model mat file""")
tf.flags.DEFINE_string("logs_dir", "logs/Visualization_logs/", """Path to save logs and checkpoint if needed""")

DATA_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

LEARNING_RATE = 1
MAX_ITERATIONS = 20
VISUALIZE_LAYER = FLAGS.visualize_layer  # Dream layers are usually conv layers
VISUALIZE_FEATURE = FLAGS.visualize_filter


def get_model_data():
    filename = DATA_URL.split("/")[-1]
    filepath = os.path.join(FLAGS.model_dir, filename)
    if not os.path.exists(filepath):
        raise IOError("VGGNet Model not found!")
    data = scipy.io.loadmat(filepath)
    return data


def get_image(image_dir):
    image = scipy.misc.imread(image_dir)
    image = np.ndarray.reshape(image.astype(np.float32), ((1,) + image.shape))
    return image


def save_image(filename, image, mean_pixel):
    output = utils.unprocess_image(image, mean_pixel)
    scipy.misc.imsave(filename, output)
    print "Image saved!"


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = utils.max_pool_2x2(current)
        elif kind == 'norm':
            current = tf.nn.lrn(current, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        net[name] = current

    assert len(net) == len(layers)
    return net


def main(argv=None):
    utils.maybe_download_and_extract(FLAGS.model_dir, DATA_URL)
    model_data = get_model_data()

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    image_shape = (1, 299, 299, 3)
    weights = np.squeeze(model_data['layers'])

    # dummy_image = tf.Variable(processed_image)
    input_image = tf.placeholder(tf.float32, shape=image_shape)
    dummy_image = np.random.uniform(size=image_shape) + 100.0
    tf.histogram_summary("Image_Output", dummy_image)
    image_net = vgg_net(weights, input_image)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        visualize_layer_feature = image_net[VISUALIZE_LAYER][:, :, :, VISUALIZE_FEATURE]
        feature_score = tf.reduce_mean(visualize_layer_feature)
        gradient = tf.gradients(feature_score, input_image)[0]

        for itr in xrange(MAX_ITERATIONS):
            grad, score = sess.run([gradient, feature_score], feed_dict={input_image: dummy_image})

            grad /= grad.std() + 1e-8
            dummy_image += LEARNING_RATE * grad
            print "Score %f" % score

        output = dummy_image.reshape(image_shape[1:])
        filename = "visualization_%s_%d.jpg" % (VISUALIZE_LAYER, VISUALIZE_FEATURE)
        save_image(os.path.join(FLAGS.logs_dir, filename), output, mean_pixel)


if __name__ == "__main__":
    tf.app.run()
