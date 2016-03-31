__author__ = 'Charlie'
# Implementation based on neural style paper

import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
from datetime import datetime
import TensorflowUtils as utils
import os, sys

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("model_dir", "Models_zoo/", """Path to the VGG model mat file""")
tf.flags.DEFINE_string("content_path", "", """Path to content image to be drawn in different style""")
tf.flags.DEFINE_string("style_path", "", """Path to style image to use""")

tf.flags.DEFINE_string("log_dir", "Neural_style/", """Path to save logs and checkpoint if needed""")

DATA_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

CONTENT_WEIGHT = 5e0
CONTENT_LAYERS = ('conv1_1', 'conv2_1', 'conv4_1', 'conv4_2')

STYLE_WEIGHT = 1e4
STYLE_LAYERS = ('conv3_1', 'conv5_1')

VARIATION_WEIGHT = 1e3
LEARNING_RATE = 1e0
MAX_ITERATIONS = 10000


def get_model_data():
    filename = DATA_URL.split("/")[-1]
    filepath = os.path.join(FLAGS.model_dir, filename)
    if not os.path.exists(filepath):
        raise IOError("VGG Model not found!")
    data = scipy.io.loadmat(filepath)
    return data


def get_image(image_dir):
    image = scipy.misc.imread(image_dir)
    image = np.ndarray.reshape(image.astype(np.float32), (((1,) + image.shape)))
    return image


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
        net[name] = current

    assert len(net) == len(layers)
    return net


def main(argv=None):
    utils.maybe_download_and_extract(FLAGS.model_dir, DATA_URL)
    model_data = get_model_data()

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    content_image = get_image(FLAGS.content_path)
    print content_image.shape
    processed_content = utils.process_image(content_image, mean_pixel)
    style_image = get_image(FLAGS.style_path)
    processed_style = utils.process_image(style_image, mean_pixel)

    content_net = vgg_net(weights, processed_content)

    style_net = vgg_net(weights, processed_style)

    dummy_image = utils.weight_variable(content_image.shape, stddev=np.std(content_image) * 0.1)
    image_net = vgg_net(weights, dummy_image)

    with tf.Session() as sess:
        content_losses = []
        for layer in CONTENT_LAYERS:
            feature = content_net[layer].eval()
            content_losses.append(2 * tf.nn.l2_loss(
                image_net[layer] - feature) /
                                  feature.size)
        content_loss = CONTENT_WEIGHT * reduce(tf.add, content_losses)

        style_losses = []
        for layer in STYLE_LAYERS:
            features = style_net[layer].eval()
            features = np.reshape(features, (-1, features.shape[3]))
            style_gram = np.matmul(features.T, features) / features.size

            image_layer = image_net[layer]
            _, height, width, number = map(lambda i: i.value, image_layer.get_shape())
            size = height * width * number
            feats = tf.reshape(image_layer, (-1, number))
            image_gram = tf.matmul(tf.transpose(feats), feats) / size
            style_losses.append(2 * tf.nn.l2_loss(image_gram - style_gram) / style_gram.size)
        style_loss = STYLE_WEIGHT * reduce(tf.add, style_losses)

        tv_y_size = utils.get_tensor_size(dummy_image[:, 1:, :, :])
        tv_x_size = utils.get_tensor_size(dummy_image[:, :, 1:, :])
        tv_loss = VARIATION_WEIGHT * 2 * (
            (tf.nn.l2_loss(dummy_image[:, 1:, :, :] - dummy_image[:, :content_image.shape[1] - 1, :, :]) /
             tv_y_size) +
            (tf.nn.l2_loss(dummy_image[:, :, 1:, :] - dummy_image[:, :, :content_image.shape[2] - 1, :]) /
             tv_x_size))

        loss = content_loss + style_loss + tv_loss
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        best_loss = float('inf')
        best = None
        sess.run(tf.initialize_all_variables())

        for i in range(MAX_ITERATIONS):
            train_step.run()

            if i % 100 == 0 or i == MAX_ITERATIONS - 1:
                this_loss = loss.eval()
                if this_loss < best_loss:
                    best_loss = this_loss
                    best = dummy_image.eval()

            if i % 1000 == 0 or i == MAX_ITERATIONS - 1:
                print('Step %d' % (i + 1)),
                print('  content loss: %g' % content_loss.eval()),
                print('    style loss: %g' % style_loss.eval()),
                print('       tv loss: %g' % tv_loss.eval()),
                print('    total loss: %g' % loss.eval())

    output = utils.unprocess_image(best.reshape(content_image.shape[1:]), mean_pixel)
    scipy.misc.imsave("output.jpg", output)


if __name__ == "__main__":
    tf.app.run()
