__author__ = 'Charlie'
# Implementation to deep dream with VGG net

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
tf.flags.DEFINE_string("image_path", "", """Path to image to be dreamed""")
tf.flags.DEFINE_string("model_dir", "Models_zoo/", """Path to the VGGNet model mat file""")
tf.flags.DEFINE_string("logs_dir", "logs/Deepdream_logs/", """Path to save logs and checkpoint if needed""")

DATA_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

LEARNING_RATE = 1.5
MAX_ITERATIONS = 50
DREAM_LAYER = "conv5_3"
DREAM_FEATURE = 11


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


def deepdream_image(model_params, image):
    filename = "%s_deepdream_%s.jpg" % os.path.splitext((FLAGS.image_path.split("/")[-1])[0],DREAM_LAYER)

    processed_image = utils.process_image(image, model_params["mean_pixel"]).astype(np.float32)
    input_image = tf.placeholder(tf.float32, processed_image.shape)
    dream_net = vgg_net(model_params["weights"], input_image)
    step = LEARNING_RATE

    with tf.Session() as sess:
        dream_layer_features = dream_net[DREAM_LAYER][:,:,:,DREAM_FEATURE]
        feature_score = tf.reduce_mean(dream_layer_features)
        grad_op = tf.gradients(feature_score, input_image)[0]

        for itr in range(1, MAX_ITERATIONS):
            grad, score = sess.run([grad_op, feature_score], feed_dict={input_image: processed_image})
            grad /= np.abs(grad).mean() + 1e-8
            processed_image += grad * step
            if itr % 10 == 0 or itr == MAX_ITERATIONS:
                step /= 2.0  # halfing step size every 10 iterations
                temp_file = "%d_%s" % (itr, filename)
                output = processed_image.reshape(image.shape[1:])
                save_image(os.path.join(FLAGS.logs_dir, "checkpoints", temp_file), output, model_params["mean_pixel"])
                print ("Step:%d Score:%f" % (itr, score))

    output = processed_image.reshape(image.shape[1:])
    save_image(os.path.join(FLAGS.logs_dir, filename), output, model_params["mean_pixel"])


def main(argv=None):
    utils.maybe_download_and_extract(FLAGS.model_dir, DATA_URL)
    model_data = get_model_data()
    dream_image = get_image(FLAGS.image_path)
    print dream_image.shape

    model_params = {}
    mean = model_data['normalization'][0][0][0]
    model_params["mean_pixel"] = np.mean(mean, axis=(0, 1))
    model_params["weights"] = np.squeeze(model_data['layers'])
    deepdream_image(model_params, dream_image)

if __name__ == "__main__":
    tf.app.run()
