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


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


def weight_variable(shape, stddev=0.1, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


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

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def process_image(image, mean_pixel):
    return (image - mean_pixel).astype(np.float32)


def unprocess_image(image, mean_pixel):
    return (image + mean_pixel).astype(np.float32)


def deep_residual_block(incoming, nb_blocks, bottleneck_size, out_channels,
                        downsample=False, downsample_strides=2,
                        activation='relu', batch_norm=True, bias=False,
                        weights_init='uniform_scaling', bias_init='zeros',
                        regularizer=None, weight_decay=0.001, trainable=True,
                        restore=True, name="DeepResidualBlock"):
    """ Deep Residual Block.
    A deep residual block as described in MSRA's Deep Residual Network paper.
    Notice: Because TensorFlow doesn't support a strides > filter size,
    an average pooling is used as a fix, but decrease performances.
    Input:
        4-D Tensor [batch, height, width, in_channels].
    Output:
        4-D Tensor [batch, new height, new width, nb_filter].
    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer.
        nb_blocks: `int`. Number of layer blocks.
        bottleneck_size: `int`. The number of convolutional filter of the
            bottleneck convolutional layer.
        out_channels: `int`. The number of convolutional filters of the
            layers surrounding the bottleneck layer.
        downsample:
        downsample_strides:
        activation: `str` (name) or `Tensor`. Activation applied to this layer.
             Default: 'linear'.
        batch_norm: `bool`. If True, apply batch normalization.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
           Default: 'uniform_scaling'.
        bias_init: `str` (name) or `tf.Tensor`. Bias initialization.
             Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights. Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model
        name: A name for this layer (optional). Default: 'DeepBottleneck'.
    References:
        Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu
        Zhang, Shaoqing Ren, Jian Sun. 2015.
    Links:
        [http://arxiv.org/pdf/1512.03385v1.pdf]
        (http://arxiv.org/pdf/1512.03385v1.pdf)
    """
    resnet = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    with tf.name_scope(name):
        for i in range(nb_blocks):
            with tf.name_scope('ResidualBlock'):

                identity = resnet

                if downsample:
                    # Use average pooling, because TensorFlow conv_2d can't
                    # accept kernel size < strides.
                    resnet = avg_pool_2d(resnet, downsample_strides,
                                         downsample_strides)
                    resnet = conv_2d(resnet, bottleneck_size, 1, 1, 'valid',
                                     activation, bias, weights_init,
                                     bias_init, regularizer, weight_decay,
                                     trainable, restore)
                else:
                    resnet = conv_2d(resnet, bottleneck_size, 1, 1, 'valid',
                                     activation, bias, weights_init,
                                     bias_init, regularizer, weight_decay,
                                     trainable, restore)
                if batch_norm:
                    resnet = tflearn.batch_normalization(resnet)

                resnet = conv_2d(resnet, bottleneck_size, 3, 1, 'same',
                                 activation, bias, weights_init,
                                 bias_init, regularizer, weight_decay,
                                 trainable, restore)
                if batch_norm:
                    resnet = tflearn.batch_normalization(resnet)

                resnet = conv_2d(resnet, out_channels, 1, 1, 'valid',
                                 activation, bias, weights_init,
                                 bias_init, regularizer, weight_decay,
                                 trainable, restore)
                if batch_norm:
                    resnet = tflearn.batch_normalization(resnet)

                if downsample:
                    # Use average pooling, because TensorFlow conv_2d can't
                    # accept kernel size < strides.
                    identity = avg_pool_2d(identity, downsample_strides,
                                           downsample_strides)

                # Projection to new dimension
                if in_channels != out_channels:
                    in_channels = out_channels
                    identity = conv_2d(identity, out_channels, 1, 1, 'valid',
                                       'linear', bias, weights_init,
                                       bias_init, regularizer, weight_decay,
                                       trainable, restore)

                resnet = resnet + identity
                resnet = tflearn.activation(resnet, activation)

    return resnet


# def shallow_residual_block(incoming, nb_blocks, out_channels,
#                            downsample=False, downsample_strides=2,
#                            activation='relu', batch_norm=True, bias=False,
#                            weights_init='uniform_scaling', bias_init='zeros',
#                            regularizer=None, weight_decay=0.0001,
#                            trainable=True, restore=True,
#                            name="ShallowResidualBlock"):
#     """ Shallow Residual Block.
#     A shallow residual block as described in MSRA's Deep Residual Network
#     paper.
#     Notice: Because TensorFlow doesn't support a strides > filter size,
#     an average pooling is used as a fix, but decrease performances.
#     Input:
#         4-D Tensor [batch, height, width, in_channels].
#     Output:
#         4-D Tensor [batch, new height, new width, nb_filter].
#     Arguments:
#         incoming: `Tensor`. Incoming 4-D Layer.
#         nb_blocks: `int`. Number of layer blocks.
#         out_channels: `int`. The number of convolutional filters of the
#             convolution layers.
#         downsample: `bool`. If True, apply downsampling using
#             'downsample_strides' for strides.
#         downsample_strides: `int`. The strides to use when downsampling.
#         activation: `str` (name) or `Tensor`. Activation applied to this layer.
#             (see tflearn.activations). Default: 'linear'.
#         batch_norm: `bool`. If True, apply batch normalization.
#         bias: `bool`. If True, a bias is used.
#         weights_init: `str` (name) or `Tensor`. Weights initialization.
#             (see tflearn.initializations) Default: 'uniform_scaling'.
#         bias_init: `str` (name) or `tf.Tensor`. Bias initialization.
#             (see tflearn.initializations) Default: 'zeros'.
#         regularizer: `str` (name) or `Tensor`. Add a regularizer to this
#             layer weights (see tflearn.regularizers). Default: None.
#         weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
#         trainable: `bool`. If True, weights will be trainable.
#         restore: `bool`. If True, this layer weights will be restored when
#             loading a model
#         name: A name for this layer (optional). Default: 'ShallowBottleneck'.
#     References:
#         Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu
#         Zhang, Shaoqing Ren, Jian Sun. 2015.
#     Links:
#         [http://arxiv.org/pdf/1512.03385v1.pdf]
#         (http://arxiv.org/pdf/1512.03385v1.pdf)
#     """
#     resnet = incoming
#     in_channels = incoming.get_shape().as_list()[-1]
#
#     with tf.name_scope(name):
#         for i in range(nb_blocks):
#             with tf.name_scope('ResidualBlock'):
#
#                 identity = resnet
#
#                 if downsample:
#                     resnet = conv_2d(resnet, out_channels, 3,
#                                      downsample_strides, 'same', 'linear',
#                                      bias, weights_init, bias_init,
#                                      regularizer, weight_decay, trainable,
#                                      restore)
#                 else:
#                     resnet = conv_2d(resnet, out_channels, 3, 1, 'same',
#                                      'linear', bias, weights_init,
#                                      bias_init, regularizer, weight_decay,
#                                      trainable, restore)
#                 if batch_norm:
#                     resnet = tflearn.batch_normalization(resnet)
#                 resnet = tflearn.activation(resnet, activation)
#
#                 resnet = conv_2d(resnet, out_channels, 3, 1, 'same',
#                                  'linear', bias, weights_init,
#                                  bias_init, regularizer, weight_decay,
#                                  trainable, restore)
#                 if batch_norm:
#                     resnet = tflearn.batch_normalization(resnet)
#
#                 # TensorFlow can't accept kernel size < strides, so using a
#                 # average pooling or resizing for downsampling.
#
#                 # Downsampling
#                 if downsample:
#                     #identity = avg_pool_2d(identity, downsample_strides,
#                     #                       downsample_strides)
#                     size = resnet.get_shape().as_list()
#                     identity = tf.image.resize_nearest_neighbor(identity,
#                                                                 [size[1],
#                                                                  size[2]])
#
#                 # Projection to new dimension
#                 if in_channels != out_channels:
#                     in_channels = out_channels
#                     identity = conv_2d(identity, out_channels, 1, 1, 'same',
#                                        'linear', bias, weights_init,
#                                        bias_init, regularizer, weight_decay,
#                                        trainable, restore)
#
#                 resnet = resnet + identity
#                 resnet = tflearn.activation(resnet, activation)
#
#     return resnet