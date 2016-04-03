__author__ = 'Charlie'
# Attempt to learn alignment info given image and its reference

import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
from datetime import datetime
import TensorflowUtils as utils
import os, sys

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("image_path", "", """Path to image to be dreamed""")
tf.flags.DEFINE_string("model_dir", "Models_zoo/", """Path to the AlexNet model mat file""")
tf.flags.DEFINE_string("log_dir", "logs/Deepdream_logs/", """Path to save logs and checkpoint if needed""")

DATA_URL = 'http://www.vlfeat.org/matconvnet/models/imagenet-caffe-alex.mat'

LEARNING_RATE = 1e0
MAX_ITERATIONS = 10000
DREAM_LAYER = "conv5"


def get_model_data():
    filename = DATA_URL.split("/")[-1]
    filepath = os.path.join(FLAGS.model_dir, filename)
    if not os.path.exists(filepath):
        raise IOError("AlexNet Model not found!")
    data = scipy.io.loadmat(filepath)
    return data


def get_image(image_dir):
    image = scipy.misc.imread(image_dir)
    image = np.ndarray.reshape(image.astype(np.float32), (((1,) + image.shape)))
    return image


def alex_net(weights, image):
    layers = (
        'conv1', 'relu1', 'norm1', 'pool1',
        'conv2', 'relu2', 'norm2', 'pool2',
        'conv3', 'relu3',
        'conv4', 'relu4',
        'conv5', 'relu5', 'pool5',
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
    dream_image = get_image(FLAGS.image_path)
    print dream_image.shape

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    processed_image = utils.process_image(dream_image, mean_pixel)
    weights = np.squeeze(model_data['layers'])

    dream_net = alex_net(weights, processed_image)

    dummy_image = utils.weight_variable(dream_image.shape, stddev=np.std(dream_image) * 0.1)
    tf.histogram_summary("Image Output", dummy_image)
    image_net = alex_net(weights, dummy_image)

    with tf.Session() as sess:
        dream_layer_features = dream_net[DREAM_LAYER].eval()
        loss = 2 * tf.nn.l2_loss(image_net[DREAM_LAYER] - dream_layer_features) / dream_layer_features.size
        tf.scalar_summary("Loss", loss)

        summary_op = tf.merge_all_summaries()
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        best_loss = float('inf')
        best = None
        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir)
        sess.run(tf.initialize_all_variables())

        for i in range(1, MAX_ITERATIONS):
            train_op.run()

            if i % 10 == 0 or i == MAX_ITERATIONS - 1:
                this_loss = loss.eval()
                print('Step %d' % (i)),
                print('    total loss: %g' % this_loss)
                summary_writer.add_summary(summary_op.eval(), global_step=i)
                if this_loss < best_loss:
                    best_loss = this_loss
                    best = dummy_image.eval()
                    output = utils.unprocess_image(best.reshape(dream_image.shape[1:]), mean_pixel)
                    scipy.misc.imsave("dream_check.jpg", output)

    output = utils.unprocess_image(best.reshape(dream_image.shape[1:]), mean_pixel)
    scipy.misc.imsave("output.jpg", output)


if __name__ == "__main__":
    tf.app.run()
