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
        # Get letter dataset
        dataset = _load_letter(folder, letter_folder, cache_if_possible)

        # Append the letter dataset to the global dataset
        x.extend(list(map(lambda x: x.flatten(), dataset)))
        y.extend([letter_folder] * len(dataset))

    return x, y


def _load_letter(folder, letter_folder, cache_if_possible=True):
    if cache_if_possible:
        # Try to use cached dataset
        dataset = _get_cached_letter(letter_folder)

    if not dataset:
        # Get the image names
        image_files = os.listdir(_get_letter_input_folder(folder, letter_folder))

        # Preallocate dataset for the images. The size of the images is IMAGE_SIZE * IMAGE_SIZE
        dataset = np.ndarray(shape=(len(image_files), IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        image_index = 0
        for image_file in image_files:
            # Get image file (name)
            image_filepath = __get_letter_input_file(folder, letter_folder, image_file)

            # Read image
            image_data = (ndimage.imread(image_filepath).astype(float) - PIXEL_DEPTH / 2) / PIXEL_DEPTH

            # Store the image in the dataset
            dataset[image_index, :, :] = image_data
            image_index += 1

        # Cache the dataset
        _cache_letter(letter_folder, dataset)

    return dataset


def _cache_letter(letter_folder, dataset):
    """
        Store letter dataset as a pickle file.
    """
    with open(_get_pickle_filename(letter_folder), "wb") as out:
        pickle.dump(dataset, out, pickle.HIGHEST_PROTOCOL)


def _get_cached_letter(letter_folder):
    """
        Try to read the pickle file for the given letter. If there is not such file, returns None.
    """
    if os.path.isfile(_get_pickle_filename(letter_folder)):
        with open(_get_pickle_filename(letter_folder), "rb") as out:
            return pickle.load(out)
    return None


def show_preprocessed_sample(letter, num_samples):
    """
        Display a sample of images for the given letter.
    """
    with open(_get_pickle_filename(letter), "rb") as pickle_in:
        dataset = pickle.load(pickle_in)

        for i, img in enumerate(random.sample(list(dataset), num_samples)):
            plt.subplot(2, num_samples // 2, i + 1)
            plt.axis('off')
            plt.imshow(img)

        plt.show()


def write_as_csv(X, Y, filename):
    """
        Write the dataset as a csv
    """
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
