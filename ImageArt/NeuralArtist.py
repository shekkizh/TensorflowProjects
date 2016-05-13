__author__ = 'Charlie'
import numpy as np
import tensorflow as tf
import scipy.misc as misc
import os, sys, argparse, inspect
import random

utils_path = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

import TensorflowUtils as utils

ap = argparse.ArgumentParser("Train a network to guess RGB of image")
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = vars(ap.parse_args())

LOG_DIR = "logs/NeuralArtist_logs/"

NEURONS_PER_LAYER = 20
LEARNING_RATE = 1e-4
MOMENTUM_RATE = 0.9
MAX_ITERATIONS = 100000

current_index = 0


def get_next_batch(batch_size):
    global current_index
    batch_inputs = []
    batch_preds = []
    for i in range(batch_size):
        # index = random.randrange(image_size)
        index = current_index
        batch_inputs.append(input_value[index, :])
        batch_preds.append(image_reshape[index, :])

        current_index += 1
        if current_index == image_size:
            current_index = 0

    return batch_inputs, batch_preds


def inference(inputs):
    with tf.name_scope("input"):
        W1 = utils.weight_variable([2, NEURONS_PER_LAYER], name="weights_1")
        b1 = utils.bias_variable([NEURONS_PER_LAYER], name="bias_1")
        tf.histogram_summary("W1", W1)
        tf.histogram_summary("b1", b1)
        h1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, W1), b1))

    with tf.name_scope("hidden2"):
        W2 = utils.weight_variable([NEURONS_PER_LAYER, NEURONS_PER_LAYER], name="weights_2")
        b2 = utils.bias_variable([NEURONS_PER_LAYER], name="bias_2")
        tf.histogram_summary("W2", W2)
        tf.histogram_summary("b2", b2)
        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    with tf.name_scope("hidden3"):
        W3 = utils.weight_variable([NEURONS_PER_LAYER, NEURONS_PER_LAYER], name="weights_3")
        b3 = utils.bias_variable([NEURONS_PER_LAYER], name="bias_3")
        tf.histogram_summary("W3", W3)
        tf.histogram_summary("b3", b3)
        h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
    with tf.name_scope("hidden4"):
        W4 = utils.weight_variable([NEURONS_PER_LAYER, NEURONS_PER_LAYER], name="weights_4")
        b4 = utils.bias_variable([NEURONS_PER_LAYER], name="bias_4")
        tf.histogram_summary("W4", W4)
        tf.histogram_summary("b4", b4)
        h4 = tf.nn.relu(tf.matmul(h3, W4) + b4)
    with tf.name_scope("hidden5"):
        W5 = utils.weight_variable([NEURONS_PER_LAYER, NEURONS_PER_LAYER], name="weights_5")
        b5 = utils.bias_variable([NEURONS_PER_LAYER], name="bias_5")
        tf.histogram_summary("W5", W5)
        tf.histogram_summary("b5", b5)
        h5 = tf.nn.relu(tf.matmul(h4, W5) + b5)
    with tf.name_scope("hidden6"):
        W6 = utils.weight_variable([NEURONS_PER_LAYER, NEURONS_PER_LAYER], name="weights_6")
        b6 = utils.bias_variable([NEURONS_PER_LAYER], name="bias_6")
        tf.histogram_summary("W6", W6)
        tf.histogram_summary("b6", b6)
        h6 = tf.nn.relu(tf.matmul(h5, W6) + b6)
    with tf.name_scope("hidden7"):
        W7 = utils.weight_variable([NEURONS_PER_LAYER, NEURONS_PER_LAYER], name="weights_7")
        b7 = utils.bias_variable([NEURONS_PER_LAYER], name="bias_7")
        tf.histogram_summary("W7", W6)
        tf.histogram_summary("b7", b6)
        h7 = tf.nn.relu(tf.matmul(h6, W7) + b7)
    with tf.name_scope("hidden8"):
        W8 = utils.weight_variable([NEURONS_PER_LAYER, NEURONS_PER_LAYER], name="weights_8")
        b8 = utils.bias_variable([NEURONS_PER_LAYER], name="bias_8")
        tf.histogram_summary("W8", W6)
        tf.histogram_summary("b8", b6)
        h8 = tf.nn.relu(tf.matmul(h7, W8) + b8)
    with tf.name_scope("output"):
        W9 = utils.weight_variable([NEURONS_PER_LAYER, channels], name="weights_9")
        b9 = utils.bias_variable([channels], name="bias_9")
        tf.histogram_summary("W9", W9)
        tf.histogram_summary("b9", b9)
        pred = tf.matmul(h8, W9) + b9

    return pred


def loss(pred, actual):
    loss_val =  tf.sqrt(2 *tf.nn.l2_loss(tf.sub(pred, actual))) / image_size
    tf.scalar_summary("loss", loss_val)
    return loss_val


def train(loss_val):
    return tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_val)


def main(argv=None):
    best_loss = float('Inf')
    with tf.Graph().as_default():
        inputs = tf.placeholder(tf.float32, shape=(None, 2))
        preds = tf.placeholder(tf.float32, shape=(None, 3))

        pred_val = inference(inputs)
        loss_val = loss(pred_val, preds)
        train_op = train(loss_val)
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(LOG_DIR)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(LOG_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            for step in xrange(MAX_ITERATIONS):
                batch_input, batch_pred = get_next_batch(BATCH_SIZE)
                feed_dict = {inputs: batch_input, preds: batch_pred}
                sess.run(train_op, feed_dict=feed_dict)

                if (step % 100 == 0):
                    this_loss, summary_str = sess.run([loss_val, summary_op], feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)

                    print("Step %d,  Loss %g" % (step, this_loss))
                    if (this_loss < best_loss):
                        pred_image = sess.run(pred_val, feed_dict={inputs: input_value})
                        pred_image = np.reshape(pred_image,
                                                image.shape)  # utils.unprocess_image(np.reshape(pred_image, image.shape), mean_pixel)
                        misc.imsave("neural_artist_check.png", pred_image)
                        best_loss = this_loss

                if step%1000 == 0:
                    saver.save(sess, LOG_DIR + "model.ckpt", global_step=step)

            best_image = sess.run(pred_val, feed_dict={inputs: input_value})
            best_image = np.reshape(best_image,
                                    image.shape)  # utils.unprocess_image(np.reshape(best_image, image.shape), mean_pixel)
            misc.imsave("neural_artist.png", best_image)


if __name__ == "__main__":
    image = misc.imread(args["image"])
    image = misc.imresize(image, (225,225))
    # image = np.array([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]])
    height, width, channels = image.shape
    image_size = height * width
    # mean_pixel = np.mean(image, axis=(0, 1))
    # processed_image = utils.process_image(image, mean_pixel)
    image_reshape = np.reshape(image, (-1, channels)).astype(np.float32)
    BATCH_SIZE = image_size/5
    print image_reshape.shape

    input_list = []
    for i in range(height):
        for j in range(width):
            input_list.append([i, j])

    input_value = np.array(input_list)
    print input_value.shape
    input_value = input_value.astype(np.float32)
    main()
