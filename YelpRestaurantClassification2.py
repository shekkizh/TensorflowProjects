__author__ = 'Charlie'

import random
import os, sys
import tensorflow as tf
from datetime import datetime
import numpy as np
from six.moves import urllib
import tarfile
import csv
import hashlib

from tensorflow.python.client import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_image_dir', "Yelp_Data/train_photos", """ Path to train image directory""")
tf.app.flags.DEFINE_string('train_data_dir', "Yelp_Data/train_data", """ Path to other train data""")
tf.app.flags.DEFINE_string('test_image_dir', "Yelp_Data/test_photos", """ Path to test image directory""")
tf.app.flags.DEFINE_string('test_data_dir', "Yelp_Data/test_data", """ Path to other test data""")

tf.app.flags.DEFINE_string('train_dir', 'Yelp_logs/',
                           """Where to save the trained graph's labels.""")
tf.app.flags.DEFINE_integer('batch_size', 512,
                            """How many images to train on at a time.""")
tf.app.flags.DEFINE_integer('validation_percentage', 10,
                            """What percentage of images to use as a validation set.""")
tf.app.flags.DEFINE_integer('train_steps', 100000, """No. of training steps """)

# File-system cache locations.
tf.app.flags.DEFINE_string('model_dir', 'Models_zoo/imagenet',
                           """Path to classify_image_graph_def.pb, """)

tf.app.flags.DEFINE_string(
    'bottleneck_dir', 'Yelp_Data/train_bottleneck',
    """Path to cache bottleneck layer values as files.""")

tf.app.flags.DEFINE_string('mode', "train", """Mode: train / test""")

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
IMAGE_SIZE = 299
IMAGE_DEPTH = 3

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape'
BOTTLENECK_TENSOR_SIZE = 2048
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents'

NUM_CLASSES = 9


def maybe_download_and_extract():
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def create_inception_graph():
    with tf.gfile.FastGFile(os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def read_csv_files():
    def create_labels(index_string):
        label = np.zeros((1, NUM_CLASSES), dtype=np.float32)
        index_split = index_string.split(" ")
        if index_string and index_split[0]:
            indexes = map(int, index_split)
            label[:, indexes] = 1
        return label

    with open(os.path.join(FLAGS.train_data_dir, "train_photo_to_biz_ids.csv"), 'r') as photo_map_file:
        photo_map_file.next()
        photo_dict = dict((x[0], x[1]) for x in csv.reader(photo_map_file))

    with open(os.path.join(FLAGS.train_data_dir, "train.csv"), 'r') as biz_map_file:
        biz_map_file.next()
        biz_dict = dict((x[0], create_labels(x[1])) for x in csv.reader(biz_map_file))

    return photo_dict, biz_dict


def ensure_name_has_port(tensor_name):
    if ':' not in tensor_name:
        name_with_port = tensor_name + ':0'
    else:
        name_with_port = tensor_name
    return name_with_port


def get_image_records():
    class ImageRecord:
        pass

    train_records = []
    eval_records = []
    for _, _, files in os.walk(FLAGS.train_image_dir):
        filenames = [x for x in files]
        
    filenames_length = len(filenames)
    eval_percentage_threshold = (FLAGS.validation_percentage/100.0) * filenames_length
    
    for file in filenames:
        # ImageRecord contains filename and imagename
        record = ImageRecord()
        record.filename = file
        record.image_name = os.path.splitext(file)[0]
        rand_val = random.randrange(filenames_length)
        if rand_val < eval_percentage_threshold:
            eval_records.append(record)
        else:
            train_records.append(record)

    return train_records, eval_records


def get_train_image_path(image_record):
    return os.path.join(FLAGS.train_image_dir, image_record.filename)


def get_bottleneck_path(image_record):
    return os.path.join(FLAGS.bottleneck_dir, image_record.image_name + '.txt')


def get_train_image_data(image_record):
    image_data_str = gfile.FastGFile(get_train_image_path(image_record), 'rb').read()
    # image_data = tf.image.decode_jpeg(image_data_str)
    # resized_image = tf.image.resize_images(image_data, IMAGE_SIZE, IMAGE_SIZE)
    # decoded_image_as_float = tf.cast(resized_image, dtype=tf.float32)
    # decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    # return decoded_image_4d
    return image_data_str


def get_or_create_bottleneck_value(sess, image_record):
    bottleneck_path = get_bottleneck_path(image_record)

    if not os.path.exists(bottleneck_path):
        bottleneck_tensor = sess.graph.get_tensor_by_name(ensure_name_has_port(BOTTLENECK_TENSOR_NAME))
        # image_data = sess.run(get_train_image_data(image_record))
        image_data = get_train_image_data(image_record)
        bottleneck_values = np.squeeze(
            sess.run(bottleneck_tensor, feed_dict={ensure_name_has_port(JPEG_DATA_TENSOR_NAME): image_data}))
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
        bottleneck_value = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_value


def create_bottleneck_cache(sess, records_list):
    bottlenecks_created = 0
    for image_record in records_list:
        try:
            get_or_create_bottleneck_value(sess, image_record)
            bottlenecks_created += 1

            if bottlenecks_created % 5000 == 0:
                print "%s bottlenecks created: %d" % (datetime.now(), bottlenecks_created)
        except:
            print "Unable to use image: %s" % image_record.image_name
            records_list.remove(image_record)


def get_random_bottlenecks(sess, records_list, batch_size, photo_biz_dict, biz_label_dict):
    bottlenecks = []
    labels = []
    #records = []
    for i in range(batch_size):
        record_index = random.randrange(len(records_list))
        #records.append(record_index)
        image_record = records_list[record_index]
        bottleneck_value = get_or_create_bottleneck_value(sess, image_record)
        label = biz_label_dict[photo_biz_dict[image_record.image_name]]
        bottlenecks.append(bottleneck_value)
        labels.append(label)
    #print records
    return bottlenecks, np.squeeze(labels)


def inference(graph):
    bottleneck_tensor = graph.get_tensor_by_name(ensure_name_has_port(BOTTLENECK_TENSOR_NAME))
    layer_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, BOTTLENECK_TENSOR_SIZE], stddev=0.001),
                                name='pre_final_layer_weights')
    tf.histogram_summary(layer_weights.name, layer_weights)

    layer_biases = tf.Variable(tf.zeros([BOTTLENECK_TENSOR_SIZE]), name='pre_final_layer_biases')
    tf.histogram_summary(layer_biases.name, layer_biases)

    h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(bottleneck_tensor, layer_weights), layer_biases))

    layer_weights2 = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, NUM_CLASSES], stddev=0.001),
                                name='final_layer_weights')
    tf.histogram_summary(layer_weights2.name, layer_weights2)

    layer_biases2 = tf.Variable(tf.zeros([NUM_CLASSES]), name='final_layer_biases')
    tf.histogram_summary(layer_biases2.name, layer_biases2)

    logits = tf.nn.bias_add(tf.matmul(h_fc1, layer_weights2), layer_biases2)
    
    return logits


def losses(logits_linear, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits_linear, labels, name="cross_entropy")
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.scalar_summary("Loss", cross_entropy_mean)
    return cross_entropy_mean


def train(loss, global_step):
    return tf.train.AdamOptimizer(1e-6).minimize(loss, global_step=global_step)


def evaluation(logits_linear, ground_truth):
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits_linear, ground_truth, name="eval_entropy")
    eval_step = tf.reduce_mean(entropy)
    tf.scalar_summary("Eval_entropy", eval_step)
    return eval_step


def main(argv=None):
    maybe_download_and_extract()
    photo_biz_dict, biz_label_dict = read_csv_files()
    
    train_image_records, eval_image_records = get_image_records()
    with tf.Session() as sess:
        create_inception_graph()

        global_step = tf.Variable(0, trainable=False)
        print "Creating bottleneck cache for training images..."
        create_bottleneck_cache(sess, train_image_records)
        print "Creating bottleneck cache for eval images..."
        create_bottleneck_cache(sess, eval_image_records)

        logits_linear = inference(sess.graph)
        print "Inference"

        label_placeholder = tf.placeholder(tf.float32, (None, NUM_CLASSES), name="ground_truth_tensor")
        loss = losses(logits_linear, label_placeholder)
        print "Loss"

        train_op = train(loss, global_step)
        print "Train"

        eval_op = evaluation(logits_linear, label_placeholder)

        bottleneck_tensor = sess.graph.get_tensor_by_name(ensure_name_has_port(BOTTLENECK_TENSOR_NAME))

        summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver(tf.all_variables())

        init = tf.initialize_all_variables()

        sess.run(init)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)

        for step in xrange(FLAGS.train_steps):
            train_bottlenecks, train_labels = get_random_bottlenecks(sess, train_image_records, FLAGS.batch_size,
                                                                     photo_biz_dict, biz_label_dict)
            train_feed = {bottleneck_tensor: train_bottlenecks, label_placeholder: train_labels}
            sess.run(train_op, feed_dict=train_feed)

            if step % 100 == 0:
                cross_entropy, summary_str = sess.run([loss, summary_op], feed_dict=train_feed)
                summary_writer.add_summary(summary_str, step)
                str_log = '%s step:%d, Train Cross Entropy: %0.2f' % (datetime.now(), step, cross_entropy)
                print str_log

            if step % 5000 == 0:
                eval_bottlenecks, eval_labels = get_random_bottlenecks(sess, eval_image_records, len(eval_image_records),
                                                                       photo_biz_dict, biz_label_dict)
                eval_entropy = sess.run(eval_op,
                                        feed_dict={bottleneck_tensor: eval_bottlenecks, label_placeholder: eval_labels})
                print "Eval Entropy %0.2f" % eval_entropy

                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == "__main__":
    tf.app.run()
