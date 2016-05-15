__author__ = 'charlie'
# Incomplete
import numpy as np
import os, sys, inspect
from six.moves import cPickle as pickle

utils_path = os.path.abspath(
    os.path.realpath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
import TensorflowUtils as utils

DATA_URL = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"


def read_caltech(data_dir):
    pickle_filename = "caltech.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        utils.maybe_download_and_extract(data_dir)
