__author__ = 'charlie'
import numpy as np
import os, sys, inspect
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

utils_path = os.path.abspath(
    os.path.realpath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
import TensorflowUtils as utils

DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'


def read_dataset(data_dir):
    pickle_filename = "flowers_data.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        utils.maybe_download_and_extract(data_dir, DATA_URL, is_tarfile=True)
        flower_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
        result = create_image_lists(os.path.join(data_dir, flower_folder))
        print "Training set: %d" % len(result['train'])
        print "Test set: %d" % len(result['test'])
        print "Validation set: %d" % len(result['validation'])
        print "Pickling ..."
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print "Found pickle file!"

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_images = result['train']
        testing_images = result['test']
        validation_images = result['validation']

        del result

    print ("Training: %d, Validation: %d, Test: %d" % (
        len(training_images), len(validation_images), len(testing_images)))
    return training_images, testing_images, validation_images


def create_image_lists(image_dir, testing_percentage=0.0, validation_percentage=0.0):
    """
    Code modified from tensorflow/tensorflow/examples/image_retraining
    """
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    training_images = []
    sub_dirs = [x[0] for x in os.walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
            continue
        print "No. of files found: %d" % len(file_list)
        training_images.extend([f for f in file_list])

    random.shuffle(training_images)
    no_of_images = len(training_images)
    validation_offset = int(validation_percentage * no_of_images)
    validation_images = training_images[:validation_offset]
    test_offset = int(testing_percentage * no_of_images)
    testing_images = training_images[validation_offset:validation_offset + test_offset]
    training_images = training_images[validation_offset + test_offset:]

    result = {
        'train': training_images,
        'test': testing_images,
        'validation': validation_images,
    }
    return result
