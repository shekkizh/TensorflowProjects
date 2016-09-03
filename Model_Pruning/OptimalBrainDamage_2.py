from __future__ import print_function

__author__ = 'shekkizh'
import numpy as np
import tensorflow as tf
import os, sys, inspect
import tensorflow.examples.tutorials.mnist as mnist

utils_folder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)

import TensorflowUtils as utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/MNIST_Pruning_logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MNIST_data/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-5", "Learning rate for Adam Optimizer")

MAX_ITERATIONS = int(5e4 + 1)
IMAGE_SIZE = 28


def inference(data, keep_prob):
    with tf.variable_scope("inference") as scope:
        weight_variable_size = IMAGE_SIZE * IMAGE_SIZE * 50 + 50 * 50 * 3 + 50 * 10
        bias_variable_size = 4 * 50 + 10
        print (weight_variable_size + bias_variable_size)
        variable = utils.weight_variable([weight_variable_size + bias_variable_size], name="variables")
        weight_variable = tf.slice(variable, [0], [weight_variable_size], name="weights")
        bias_variable = tf.slice(variable, [weight_variable_size], [bias_variable_size], name="biases")
        weight_offset = 0
        bias_offset = 0
        W_1 = tf.slice(weight_variable, [weight_offset], [IMAGE_SIZE * IMAGE_SIZE * 50], name="W_1")
        b_1 = tf.slice(bias_variable, [bias_offset], [50], name="b_1")
        h_1_relu = tf.nn.relu(tf.matmul(data, tf.reshape(W_1, [IMAGE_SIZE * IMAGE_SIZE, 50])) + b_1, name='h_1')
        h_1 = tf.nn.dropout(h_1_relu, keep_prob)
        utils.add_activation_summary(h_1)

        weight_offset += IMAGE_SIZE * IMAGE_SIZE * 50
        bias_offset += 50

        W_2 = tf.slice(weight_variable, [weight_offset], [50 * 50], name="W_2")
        b_2 = tf.slice(bias_variable, [bias_offset], [50], name="b_2")
        h_2_relu = tf.nn.relu(tf.matmul(h_1, tf.reshape(W_2, [50, 50])) + b_2, name='h_2')
        h_2 = tf.nn.dropout(h_2_relu, keep_prob)
        utils.add_activation_summary(h_2)

        weight_offset += 50 * 50
        bias_offset += 50

        W_3 = tf.slice(weight_variable, [weight_offset], [50 * 50], name="W_3")
        b_3 = tf.slice(bias_variable, [bias_offset], [50], name="b_3")
        h_3_relu = tf.nn.relu(tf.matmul(h_2, tf.reshape(W_3, [50, 50])) + b_3, name='h_3')
        h_3 = tf.nn.dropout(h_3_relu, keep_prob)
        utils.add_activation_summary(h_3)

        weight_offset += 50 * 50
        bias_offset += 50

        W_4 = tf.slice(weight_variable, [weight_offset], [50 * 50], name="W_4")
        b_4 = tf.slice(bias_variable, [bias_offset], [50], name="b_4")
        h_4_relu = tf.nn.relu(tf.matmul(h_3, tf.reshape(W_4, [50, 50])) + b_4, name='h_4')
        h_4 = tf.nn.dropout(h_4_relu, keep_prob)
        utils.add_activation_summary(h_4)

        weight_offset += 50 * 50
        bias_offset += 50

        W_final = tf.slice(weight_variable, [weight_offset], [50 * 10], name="W_final")
        b_final = tf.slice(bias_variable, [bias_offset], [10], name="b_final")
        pred = tf.nn.softmax(tf.matmul(h_4, tf.reshape(W_final, [50, 10])) + b_final, name='h_final')
        # utils.add_activation_summary(pred)
    return pred


def train(loss, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss, var_list=var_list)
    hessian = []
    for grad, var in grads:
        # utils.add_gradient_summary(grad, var)
        grad = 0 if None else grad
        grad2 = tf.gradients(grad, var)
        grad2 = 0 if None else grad2
        # utils.add_gradient_summary(grad2, var)
        hessian.append(tf.pack(grad2))
    return optimizer.apply_gradients(grads), hessian


def main(argv=None):
    input_data = tf.placeholder(tf.float32, [None, 784])
    truth_labels = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    dataset = mnist.input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    pred_labels = inference(input_data, keep_prob)
    entropy = -tf.reduce_sum(truth_labels * tf.log(pred_labels))
    tf.scalar_summary('Cross_entropy', entropy)
    train_vars = tf.trainable_variables()
    # for v in train_vars:
    #     utils.add_to_regularization_and_summary(v)
    train_op, hess = train(entropy, train_vars)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred_labels, 1), tf.argmax(truth_labels, 1)), tf.float32))

    # Session start
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)
    summary_op = tf.merge_all_summaries()

    def test():
        test_accuracy = accuracy.eval(
            feed_dict={input_data: dataset.test.images, truth_labels: dataset.test.labels, keep_prob: 1.0})
        return test_accuracy

    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        for i in xrange(MAX_ITERATIONS):
            batch = dataset.train.next_batch(FLAGS.batch_size)
            feed_dict = {input_data: batch[0], truth_labels: batch[1], keep_prob: 0.8}
            train_op.run(feed_dict=feed_dict)

            if i % 10 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, i)

            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict=feed_dict)
                print("step: %d, training accuracy: %g" % (i, train_accuracy))

            if i % 5000 == 0:
                saver.save(sess, FLAGS.logs_dir + 'model.ckpt', i)

    train_vars_copy = sess.run([tf.identity(var) for var in train_vars])
    print('Variables Perecent: %d, Test accuracy: %g' % (100, test()))

    k = tf.placeholder(tf.int32)

    def scatter_update(saliency, variables):
        shape = utils.get_tensor_size(variables)
        values, indices = tf.nn.top_k(-1 * saliency, tf.cast(k * shape / 10, tf.int32))
        return tf.scatter_update(variables, indices, tf.zeros_like(values))

    def scatter_restore(saliency, variables1, variables2):
        shape = utils.get_tensor_size(variables2)
        values, indices = tf.nn.top_k(-1 * saliency, tf.cast(k * shape / 10, tf.int32))
        values = tf.gather(variables1, indices)
        return tf.scatter_update(variables2, indices, values)

    scatter_update_op = [scatter_update(sal, var) for sal, var in zip(hess, train_vars)]
    scatter_restore_op = [scatter_restore(sal, var1, var2) for sal, var1, var2 in
                          zip(hess, train_vars_copy, train_vars)]

    for count in range(1, 8):
        batch = dataset.train.next_batch(FLAGS.batch_size)
        feed_dict = {input_data: batch[0], truth_labels: batch[1], k: count, keep_prob: 1.0}
        sess.run(scatter_update_op, feed_dict=feed_dict)
        print('Variables Perecent: %d, Test accuracy: %g' % (10*(10 - count), test()))
        sess.run(scatter_restore_op, feed_dict=feed_dict)


if __name__ == "__main__":
    tf.app.run()
