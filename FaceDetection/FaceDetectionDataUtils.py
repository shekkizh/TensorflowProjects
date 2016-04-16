__author__ = 'Charlie'
import pandas as pd
import numpy as np
import os, sys, inspect
from six.moves import cPickle as pickle

IMAGE_SIZE = 96
NUM_LABELS = 30
VALIDATION_PERCENT = 0.1  # use 10 percent of training images for validation

np.random.seed(0)


def read_data(data_dir, force=False):
    pickle_file = os.path.join(data_dir, "FaceDetectionData.pickle")
    if force or not os.path.exists(pickle_file):
        train_filename = os.path.join(data_dir, "training.csv")
        data_frame = pd.read_csv(train_filename)
        cols = data_frame.columns[:-1]
        np.savetxt(os.path.join(data_dir, "column_labels.txt"))
        data_frame['Image'] = data_frame['Image'].apply(lambda x: np.fromstring(x, sep=" ") / 255.0)
        data_frame = data_frame.dropna()
        print "Reading training.csv ..."

        # scale data to a 1x1 image with pixel values 0-1
        train_images = np.vstack(data_frame['Image']).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        train_labels = data_frame[cols].values / float(IMAGE_SIZE)
        permutations = np.random.permutation(train_images.shape[0])
        train_images = train_images[permutations]
        train_labels = train_labels[permutations]
        validation_percent = int(train_images.shape[0] * VALIDATION_PERCENT)
        validation_images = train_images[:validation_percent]
        validation_labels = train_labels[:validation_percent]
        train_images = train_images[validation_percent:]
        train_labels = train_labels[validation_percent:]

        print "Reading test.csv ..."
        test_filename = os.path.join(data_dir, "test.csv")
        data_frame = pd.read_csv(test_filename)
        data_frame['Image'] = data_frame['Image'].apply(lambda x: np.fromstring(x, sep=" ") / 255.0)
        data_frame = data_frame.dropna()
        test_images = np.vstack(data_frame['Image']).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

        with open(pickle_file, "wb") as file:
            try:
                print 'Picking ...'
                save = {
                    "train_images": train_images,
                    "train_labels": train_labels,
                    "validation_images": validation_images,
                    "validation_labels": validation_labels,
                    "test_images": test_images,
                }
                pickle.dump(save, file, pickle.HIGHEST_PROTOCOL)

            except:
                print("Unable to pickle file :/")

    with open(pickle_file, "rb") as file:
        save = pickle.load(file)
        train_images = save["train_images"]
        train_labels = save["train_labels"]
        validation_images = save["validation_images"]
        validation_labels = save["validation_labels"]
        test_images = save["test_images"]

    return train_images, train_labels, validation_images, validation_labels, test_images


def make_submission(test_labels, data_dir):
    test_labels *= 96.0
    test_labels = test_labels.clip(0, 96)

    lookup_table = pd.read_csv("IdLookupTable.csv")
    values = []

    cols = np.genfromtxt(os.path.join(data_dir, "column_labels.txt"))

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            test_labels[row.ImageId - 1][np.where(cols == row.FeatureName)[0][0]],
        ))
    submission = pd.DataFrame(values, columns=('RowId', 'Location'))
    submission.to_csv(os.path.join(data_dir, 'submission.csv'), index=False)