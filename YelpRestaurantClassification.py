__author__ = 'Charlie'

import os, sys
import tensorflow as tf
from datetime import datetime
import numpy as np
from six.moves import urllib
import tarfile
import csv

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
tf.app.flags.DEFINE_integer('batch_size', 128,
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
IMAGE_SIZE = 229
IMAGE_DEPTH = 3

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape'
BOTTLENECK_TENSOR_SIZE = 2048
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents'

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def maybe_download_and_extract():
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
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
        label = np.zeros((1,10),dtype=np.float32)
        index_split = index_string.split(" ")
        if index_string and index_split[0]:
            indexes = map(int, index_split)
            label[:,indexes] = 1
        return label

    with open(os.path.join(FLAGS.train_data_dir, "train_photo_to_biz_ids.csv"),'r') as photo_map_file:
        photo_map_file.next()
        photo_dict = dict((x[0],x[1]) for x in csv.reader(photo_map_file))

    with open(os.path.join(FLAGS.train_data_dir, "train.csv"),'r') as biz_map_file:
        biz_map_file.next()
        biz_dict =dict((x[0],create_labels(x[1])) for x in csv.reader(biz_map_file))

    return photo_dict, biz_dict

def process_image(image):
    # image_shape = tf.shape(image)
    # height = image_shape[0]
    # width = image_shape[1]
    # if (height < width):
    #     new_height = IMAGE_SIZE
    #     new_width = tf.cast(width * new_height/height, dtype=tf.int32)
    # else:
    #     new_width = IMAGE_SIZE
    #     new_height = tf.cast(height*new_width/width, dtype=tf.int32)
    # resized_image = tf.image.resize_images(image, new_height, new_width)
    cropped_image = tf.image.resize_images(image, IMAGE_SIZE, IMAGE_SIZE)
    return cropped_image

def generate_batch(image_record):
    min_queue_examples = int(0.4 * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
    image, name = tf.train.batch([image_record.image, image_record.image_name], batch_size=FLAGS.batch_size, num_threads=16, capacity=min_queue_examples + 3 * FLAGS.batch_size)
    return image, name

def get_inputs(sess, path, name, photo_biz_map, biz_label_map):
    class ImageRecord(object):
        pass
    result = ImageRecord()
    file_path, filename = sess.run([path, name])
    result.image = sess.run(process_image(tf.image.decode_jpeg(gfile.FastGFile(file_path, 'rb').read())))
    result.image_name = os.path.splitext(filename)[0]
    result.bizId = photo_biz_map[result.image_name]
    result.labels = biz_label_map[result.bizId]
    result.bottleneck_value = None
    tf.image_summary("Input_image", result.image)
    return generate_batch(result)

def ensure_name_has_port(tensor_name):
    if ':' not in tensor_name:
        name_with_port = tensor_name + ':0'
    else:
        name_with_port = tensor_name
    return name_with_port

def get_or_create_bottleneck_value(sess, image, name):
    bottleneck_path = os.path.join(FLAGS.bottleneck_dir, sess.run(name), '.txt')
    print bottleneck_path

    if not os.path.exists(bottleneck_path):
        bottleneck_tensor = sess.graph.get_tensor_by_name(ensure_name_has_port(BOTTLENECK_TENSOR_NAME))
        bottleneck_values = np.squeeze(sess.run(bottleneck_tensor, feed_dict={ensure_name_has_port(JPEG_DATA_TENSOR_NAME):image}))
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
        bottleneck_value = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_value


def inference(sess, image, name):
    layer_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, NUM_CLASSES], stddev=0.001),name='final_weights')
    tf.histogram_summary(layer_weights.name, layer_weights)
    layer_biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='final_biases')
    tf.histogram_summary(layer_biases.name, layer_biases)
    bottleneck_value = get_or_create_bottleneck_value(sess, image, name)
    logits = tf.nn.bias_add(tf.matmul(bottleneck_value, layer_weights, name='final_matmul'), layer_biases)
    return sess.run(logits)

def losses(logits_linear, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits_linear, labels, name="cross_entropy")
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.scalar_summary("Loss", cross_entropy_mean)
    return cross_entropy_mean

def train(loss, global_step):
    return tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)

def main(argv=None):
    maybe_download_and_extract()
    photo_biz_dict, biz_label_dict = read_csv_files()
    create_inception_graph()

    for _,_,files in os.walk(FLAGS.train_image_dir):
        filenames = [x for x in files]
        filenames_path = [os.path.join(FLAGS.train_image_dir, x) for x in filenames]
        break

    if len(filenames) == 0:
        print "No image file found"
        return

    min_queue_examples = int(0.4*NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
    rsq = tf.RandomShuffleQueue(len(filenames), min_queue_examples,[tf.string, tf.string], shapes = [[],[]])
    enqueue_op = rsq.enqueue_many([filenames_path, filenames])
    path_op, name_op = rsq.dequeue()

    sess = tf.Session()

    sess.run(enqueue_op)

    tf.train.start_queue_runners(sess=sess, coord=tf.train.Coordinator())

    global_step = tf.Variable(0, trainable=False)

    image, name = get_inputs(sess, path_op, name_op, photo_biz_dict, biz_label_dict)

    logits_linear = inference(sess, image, name)
    print "Inference"

    label = biz_label_dict[photo_biz_dict[sess.run(name)]]

    loss = losses(logits_linear, label)
    print "loss"

    train_op = train(loss, global_step)
    print "Train"

    summary_op = tf.merge_all_summaries()

    saver = tf.train.Saver(tf.all_variables())

    init = tf.initialize_all_variables()

    sess.run(init)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)

    for step in xrange(FLAGS.train_steps):
        _, cross_entropy = sess.run([train_op, loss])

        if step % 10 == 0:
            str_log = '%s step:%d, entropy: %0.2f' % datetime.now(), step, cross_entropy
            print str_log

        if step% 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

        if step%1000 == 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


if __name__ == "__main__":
    tf.app.run()