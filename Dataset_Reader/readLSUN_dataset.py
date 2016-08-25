"""
Modification of https://github.com/fyu/lsun/blob/master/download.py
Downloads the following:
- LSUN dataset
"""

from __future__ import print_function
import os, sys, inspect
import subprocess
from six.moves import urllib
from six.moves import cPickle as pickle

utils_path = os.path.abspath(
    os.path.realpath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
import TensorflowUtils as utils


def read_LSUN(data_dir, category, tag="latest"):
    data_dir = os.path.join(data_dir, category)
    if os.path.exists(data_dir):
        print('Found dataset for %s - Skipping download' % category)
    else:
        download(data_dir, category, 'train', tag)
        download(data_dir, category, 'val', tag)
        # download(data_dir, '', 'test', tag)

    for _, _, files in os.walk(os.path.join(data_dir, 'train')):
        train_fns = [x for x in files]
    for _, _, files in os.walk(os.path.join(data_dir, 'val')):
        validation_fns = [x for x in files]

    return train_fns, validation_fns


def download(out_dir, category, set_name, tag):
    url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
          '\&category={category}\&set={set_name}'.format(**locals())
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = os.path.join(out_dir, out_name)
    print (url, out_path)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)
