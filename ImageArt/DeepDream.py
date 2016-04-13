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
MAX_ITERATIONS = 10
DREAM_LAYER = "conv5_1"
DREAM_FEATURE = 0


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
    output = np.uint8(np.clip(output, 0, 255))
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


def resize_image(image, size):
    image_to_resize = image[0]
    # print image_to_resize.shape
    # print size
    resized = scipy.misc.imresize(image_to_resize, size).astype(np.float32)
    return np.expand_dims(resized, 0)


def deepdream_image(model_params, image, octave_scale=1.4, no_of_octave=4):
    filename = "%s_deepdream_%s.jpg" % (os.path.splitext((FLAGS.image_path.split("/")[-1]))[0], DREAM_LAYER)

    processed_image = utils.process_image(image, model_params["mean_pixel"]).astype(np.float32)
    input_image = tf.placeholder(tf.float32)
    dream_net = vgg_net(model_params["weights"], input_image)

    def calc_grad_tiled(img, gradient, tile_size=512):
        sz = tile_size
        h, w = img.shape[1:3]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 2), sy, 1)
        gradient_val = np.zeros_like(img)
        for y in xrange(0, max(h - sz // 2, sz), sz):
            for x in xrange(0, max(w - sz // 2, sz), sz):
                sub_img = img_shift[:, y:y + sz, x:x + sz]
                # print sub_img.shape
                g = sess.run(gradient, {input_image: sub_img})
                gradient_val[:, y:y + sz, x:x + sz] = g

        return np.roll(np.roll(gradient_val, -sx, 2), -sy, 1)

    step = LEARNING_RATE
    feature = DREAM_FEATURE
    with tf.Session() as sess:
        dream_layer_features = dream_net[DREAM_LAYER][:, :, :, feature]
        feature_score = tf.reduce_mean(dream_layer_features)
        grad_op = tf.gradients(feature_score, input_image)[0]

        dummy_image = processed_image.copy()
        for itr in xrange(5):
            octaves = []
            for i in xrange(no_of_octave - 1):
                hw = dummy_image.shape[1:3]
                lo = resize_image(dummy_image, np.int32(np.float32(hw) / octave_scale))
                hi = dummy_image - resize_image(dummy_image, hw)
                dummy_image = lo
                octaves.append(hi)

            for octave in xrange(no_of_octave):
                if octave > 0:
                    hi = octaves[-octave]
                    dummy_image = resize_image(dummy_image, hi.shape[1:3]) + hi
                for i in xrange(MAX_ITERATIONS):
                    grad = calc_grad_tiled(dummy_image, grad_op)
                    dummy_image += grad * (step / (np.abs(grad).mean() + 1e-8))
                    print '.',
                print "."

            step /= 2.0  # halfing step size every itr
            feature += 2
            temp_file = "%d_%s" % (itr, filename)
            # print dummy_image.shape
            output = dummy_image.reshape(processed_image.shape[1:])
            save_image(os.path.join(FLAGS.logs_dir, "checkpoints", temp_file), output, model_params["mean_pixel"])


def main(argv=None):
    utils.maybe_download_and_extract(FLAGS.model_dir, DATA_URL)
    model_data = get_model_data()
    dream_image = get_image(FLAGS.image_path)
    # dream_image = np.random.uniform(size=(1, 300, 300, 3)) + 100.0
    print dream_image.shape

    model_params = {}
    mean = model_data['normalization'][0][0][0]
    model_params["mean_pixel"] = np.mean(mean, axis=(0, 1))
    model_params["weights"] = np.squeeze(model_data['layers'])
    deepdream_image(model_params, dream_image, no_of_octave=1)


if __name__ == "__main__":
    tf.app.run()
