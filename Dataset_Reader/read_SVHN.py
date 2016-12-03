import numpy as np
import scipy.io
from six.moves import urllib
import os, inspect, sys

utils_path = os.path.abspath(
    os.path.realpath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
import TensorflowUtils as utils

DATA_URL = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"


class Dataset_svhn(object):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.epochs_completed = 0
        self.batch_offset = 0

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end]


def read_dataset(data_dir):
    filename = DATA_URL.split('/')[-1]
    if not os.path.exists(data_dir):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, data_dir, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    file = os.path.join(data_dir, filename)
    mat = scipy.io.loadmat(file)
    # images are of shape [w,h,d,n]
    input_images = np.transpose(mat['X'], (3, 1, 0, 2))
    input_labels = mat['y']
    return Dataset_svhn(input_images, input_labels)
