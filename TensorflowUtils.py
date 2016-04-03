__author__ = 'Charlie'
# Utils used with tensorflow implemetation
import tensorflow as tf
import numpy as np
import os, sys
from six.moves import urllib
import tarfile


def maybe_download_and_extract(dir_path, url_name, tarfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)


def weight_variable(shape, stddev=0.1, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def get_tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def process_image(image, mean_pixel):
    return (image - mean_pixel).astype(np.float32)


def unprocess_image(image, mean_pixel):
    return (image + mean_pixel).astype(np.float32)
