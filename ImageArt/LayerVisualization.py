__author__ = 'Charlie'
# Layer visualization based on deep dream code in tensorflow for VGG net

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
tf.flags.DEFINE_string("visualize_layer", "conv5_1", "Layer to visualize")
tf.flags.DEFINE_integer("visualize_filter", "0", """filter to visualize in a layer""")
tf.flags.DEFINE_string("model_dir", "Models_zoo/", """Path to the VGGNet model mat file""")
tf.flags.DEFINE_string("logs_dir", "logs/Visualization_logs/", """Path to save logs and checkpoint if needed""")

DATA_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

LEARNING_RATE = 1.5
MAX_ITERATIONS = 10
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


def visualize_layer(model_params):
    image_shape = (299, 299, 3)
    input_image = tf.placeholder(tf.float32)
    dummy_image = np.random.uniform(size=image_shape) + 100.0
    tf.histogram_summary("Image_Output", dummy_image)
    image_net = vgg_net(model_params["weights"], input_image)

    def resize_image(image, size):
        expanded_image = tf.expand_dims(image, 0)
        return tf.image.resize_bilinear(expanded_image, size)[0,:,:,:]

    def calc_grad_tiled(img, grad_op, tile_size=512):
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad_val = np.zeros_like(img)
        for y in xrange(0, max(h-sz//2, sz),sz):
            for x in xrange(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g = sess.run(grad_op, {input_image:sub})
                grad_val[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad_val, -sx, 1), -sy, 0)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        visualize_layer_feature = image_net[VISUALIZE_LAYER][:, :, :, VISUALIZE_FEATURE]
        feature_score = tf.reduce_mean(visualize_layer_feature)
        gradient = tf.gradients(feature_score, input_image)[0]

        # for itr in xrange(MAX_ITERATIONS):
        #     grad, score = sess.run([gradient, feature_score], feed_dict={input_image: dummy_image})
        #
        #     grad /= grad.std() + 1e-8
        #     dummy_image += LEARNING_RATE * grad
        #     print "Score %f" % score

        octave_n = 4
        octave_scale = 1.4
        octaves = []
        for i in xrange(octave_n - 1):
            hw = dummy_image.shape[:2]
            lo = resize_image(dummy_image, np.int32(np.float32(hw) / octave_scale))
            hi = dummy_image - resize_image(dummy_image, hw)
            dummy_image = lo
            dummy_image = dummy_image.eval()
            octaves.append(hi)

        for octave in xrange(octave_n):
            if octave > 0:
                hi = octaves[-octave].eval()
                dummy_image = resize_image(dummy_image, hi.shape[:2]) + hi
                dummy_image = dummy_image.eval()
            for i in xrange(MAX_ITERATIONS):
                expanded_image = np.expand_dims(dummy_image, 0)
                # grad = sess.run(gradient, {input_image: expanded_image})
                grad = calc_grad_tiled(expanded_image, gradient)[0]
                dummy_image += grad * (LEARNING_RATE / (np.abs(grad).mean() + 1e-7))
                print '.',

        output = dummy_image.reshape(image_shape)
        filename = "visualization_%s_%d.jpg" % (VISUALIZE_LAYER, VISUALIZE_FEATURE)
        save_image(os.path.join(FLAGS.logs_dir, filename), output, model_params["mean_pixel"])


def main(argv=None):
    utils.maybe_download_and_extract(FLAGS.model_dir, DATA_URL)
    model_data = get_model_data()
    model_params = {}
    mean = model_data['normalization'][0][0][0]
    model_params["mean_pixel"] = np.mean(mean, axis=(0, 1))
    model_params["weights"] = np.squeeze(model_data['layers'])
    visualize_layer(model_params)


if __name__ == "__main__":
    tf.app.run()
