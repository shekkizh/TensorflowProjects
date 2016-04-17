__author__ = 'Charlie'
import pandas as pd
import numpy as np
import os, sys, inspect
from six.moves import cPickle as pickle
import scipy.misc as misc

IMAGE_SIZE = 96
NUM_LABELS = 30
VALIDATION_PERCENT = 0.1  # use 10 percent of training images for validation

IMAGE_LOCATION_NORM = IMAGE_SIZE / 2

np.random.seed(0)


def read_data(data_dir, force=False):
    pickle_file = os.path.join(data_dir, "FaceDetectionData.pickle")
    if force or not os.path.exists(pickle_file):
        train_filename = os.path.join(data_dir, "training.csv")
        data_frame = pd.read_csv(train_filename)
        cols = data_frame.columns[:-1]
        np.savetxt(os.path.join(data_dir, "column_labels.txt"), cols.values, fmt="%s")
        data_frame['Image'] = data_frame['Image'].apply(lambda x: np.fromstring(x, sep=" ") / 255.0)
        data_frame = data_frame.dropna()
        print "Reading training.csv ..."

        # scale data to a 1x1 image with pixel values 0-1
        train_images = np.vstack(data_frame['Image']).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        train_labels = (data_frame[cols].values - IMAGE_LOCATION_NORM) / float(IMAGE_LOCATION_NORM)

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


def save_sample_result(X, y, save_dir):
    for i in range(X.shape[0]):
        fn = os.path.join(save_dir, "checkpoints", "%d.jpg" % i)
        for j in range(0, y.shape[1], 2):
            pt1 = y[i, j + 1]
            pt2 = y[i, j]
            X[i, pt1 - 1:pt1 + 1, pt2 - 1:pt2 + 1] = 0

        misc.imsave(fn, X[i, :, :, 0])


def kaggle_submission_format(test_images, test_labels, data_dir):
    test_labels *= IMAGE_LOCATION_NORM
    test_labels += IMAGE_LOCATION_NORM
    test_labels = test_labels.clip(0, 96)

    save_sample_result(test_images[0:16], test_labels[0:16], data_dir)
    lookup_filename = os.path.join(data_dir, "IdLookupTable.csv")
    lookup_table = pd.read_csv(lookup_filename)
    values = []

    cols = np.genfromtxt(os.path.join(data_dir, "column_labels.txt"), dtype=str)

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            test_labels[row.ImageId - 1][np.where(cols == row.FeatureName)[0][0]],
        ))
    submission = pd.DataFrame(values, columns=('RowId', 'Location'))
    submission.to_csv(os.path.join(data_dir, 'submission.csv'), index=False)
    print "Submission created!"
