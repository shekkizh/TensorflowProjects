__author__ = 'Charlie'
import os, sys
import tarfile
import tensorflow as tf
from tensorflow.python.platform import gfile
from six.moves import urllib

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir', 'Models_zoo/imagenet',
                           """Path to classify_image_graph_def.pb, """)
tf.app.flags.DEFINE_string('image1', '',
                           """Path to image 1.""")
tf.app.flags.DEFINE_string('image2', '',
                           """Path to image 2.""")

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
IMAGE_SIZE = 229
IMAGE_DEPTH = 3

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape'
BOTTLENECK_TENSOR_SIZE = 2048
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents'

def ensure_name_has_port(tensor_name):
    if ':' not in tensor_name:
        name_with_port = tensor_name + ':0'
    else:
        name_with_port = tensor_name
    return name_with_port

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

def find_similarity(sess, image1, image2):
    if not gfile.Exists(image1):
        print ("Cannot find image 1 %s" % image1)
        return -1
    if not gfile.Exists(image2):
        print ("Cannot find image 2 %s" % image2)
        return -1

    image_data1 = gfile.FastGFile(image1, 'rb').read()
    image_data2 = gfile.FastGFile(image2, 'rb').read()

    bottleneck_tensor = sess.graph.get_tensor_by_name(ensure_name_has_port(BOTTLENECK_TENSOR_NAME))
    feature1 = sess.run(bottleneck_tensor,
                           {ensure_name_has_port(JPEG_DATA_TENSOR_NAME): image_data1})

    feature2 = sess.run(bottleneck_tensor,
                           {ensure_name_has_port(JPEG_DATA_TENSOR_NAME): image_data2})

    float_similarity = tf.cast(tf.sub(feature1,feature2), dtype=tf.float32)
    print float_similarity
    l2_dist = tf.mul(2.0, tf.nn.l2_loss(float_similarity))
    print l2_dist
    return tf.sqrt(l2_dist)/BOTTLENECK_TENSOR_SIZE

def main(argv=None):
    maybe_download_and_extract()
    create_inception_graph()
    with tf.Session() as sess:
        print find_similarity(sess, FLAGS.image1, FLAGS.image2)

if __name__ == "__main__":
    tf.app.run()
