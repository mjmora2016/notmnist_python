# Some parts of this code are heavily inspired by
# https://github.com/rndbrtrnd/udacity-deep-learning/blob/master/1_notmnist.ipynb

import os
from six.moves import cPickle as pickle
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import random

PREPROCESSED_FOLDER = "preprocessed"
INPUT_FOLDER = "input"
IMAGE_SIZE = 28  # Pixel width and height.
PIXEL_DEPTH = 255.0  # Number of levels per pixel.


def preprocess(folder, cache_if_possible=True):
    x, y = [], []
    for letter_folder in os.listdir(_get_input_folder(folder)):
        dataset = _load_letter(folder, letter_folder, cache_if_possible)
        x.extend(list(map(lambda x: x.flatten(), dataset)))
        y.extend([letter_folder] * len(dataset))

    return x, y


def _load_letter(folder, letter_folder, cache_if_possible=True):

    if cache_if_possible and os.path.isfile(_get_pickle_filename(letter_folder)):
        with open(_get_pickle_filename(letter_folder), "rb") as out:
            return pickle.load(out)
    else:
        image_files = os.listdir(_get_letter_input_folder(folder, letter_folder))
        dataset = np.ndarray(shape=(len(image_files), IMAGE_SIZE, IMAGE_SIZE),
                             dtype=np.float32)

        image_index = 0
        for image_file in os.listdir(_get_letter_input_folder(folder, letter_folder)):
            image_data = (ndimage.imread(__get_letter_input_file(folder, letter_folder, image_file))
                          .astype(float) - PIXEL_DEPTH / 2) / PIXEL_DEPTH

            if image_data.shape != (IMAGE_SIZE, IMAGE_SIZE):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))

            dataset[image_index, :, :] = image_data
            image_index += 1

        with open(_get_pickle_filename(letter_folder), "wb") as out:
            pickle.dump(dataset, out, pickle.HIGHEST_PROTOCOL)

        return dataset


def show_preprocessed_sample(letter, num_samples):
    with open(_get_pickle_filename(letter), "rb") as pickle_in:
        dataset = pickle.load(pickle_in)

    for i, img in enumerate(random.sample(list(dataset), num_samples)):
        plt.subplot(2, num_samples // 2, i + 1)
        plt.axis('off')
        plt.imshow(img)

    plt.show()


def write_as_csv(X, Y, filename):
    print (len(X), len(Y))
    with open(filename, "w") as out:
        out.write("letter," + ",".join([str(pixel+1) for pixel in range(IMAGE_SIZE * IMAGE_SIZE)]))
        out.write("\n")
        for x, y in zip(X, Y):
            out.write(y+",")
            out.write(",".join([str(val) for val in x]))
            out.write("\n")


def _get_input_folder(folder):
    return INPUT_FOLDER + os.sep + folder


def _get_letter_input_folder(folder, letter_folder):
   return _get_input_folder(folder) + os.sep + letter_folder


def __get_letter_input_file(folder, letter_folder, file):
    return _get_letter_input_folder(folder, letter_folder) + os.sep + file


def _get_pickle_filename(file):
    return PREPROCESSED_FOLDER + os.sep + file + ".pickle"


def _load_preprocessed(picke_file):
    return pickle.load(picke_file)
