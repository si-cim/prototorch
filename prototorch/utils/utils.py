"""Utilities that provide various small functionalities."""

import os
import pickle
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np


def progressbar(title, value, end, bar_width=20):
    percent = float(value) / end
    arrow = '=' * int(round(percent * bar_width) - 1) + '>'
    spaces = '.' * (bar_width - len(arrow))
    sys.stdout.write('\r{}: [{}] {}%'.format(title, arrow + spaces,
                                             int(round(percent * 100))))
    sys.stdout.flush()
    if percent == 1.0:
        print()


def prettify_string(inputs, start='', sep=' ', end='\n'):
    outputs = start + ' '.join(inputs.split()) + end
    return outputs


def pretty_print(inputs):
    print(prettify_string(inputs))


def writelog(self, *logs, logdir='./logs', logfile='run.txt'):
    f = os.path.join(logdir, logfile)
    with open(f, 'a+') as fh:
        for log in logs:
            fh.write(log)
            fh.write('\n')


def start_tensorboard(self, logdir='./logs'):
    cmd = f'tensorboard --logdir={logdir} --port=6006'
    os.system(cmd)


def make_directory(save_dir):
    if not os.path.exists(save_dir):
        print(f'Making directory {save_dir}.')
        os.mkdir(save_dir)


def make_gif(filenames, duration, output_file=None):
    try:
        import imageio
    except ModuleNotFoundError as e:
        print('Please install Protoflow with [other] extra requirements.')
        raise (e)

    images = list()
    for filename in filenames:
        images.append(imageio.imread(filename))
    if not output_file:
        output_file = f'makegif.gif'
    if images:
        imageio.mimwrite(output_file, images, duration=duration)


def gif_from_dir(directory,
                 duration,
                 prefix='',
                 output_file=None,
                 verbose=True):
    images = os.listdir(directory)
    if verbose:
        print(f'Making gif from {len(images)} images under {directory}.')
    filenames = list()
    # Sort images
    images = sorted(
        images,
        key=lambda img: int(os.path.splitext(img)[0].replace(prefix, '')))
    for image in images:
        fname = os.path.join(directory, image)
        filenames.append(fname)
    if not output_file:
        output_file = os.path.join(directory, 'makegif.gif')
    make_gif(filenames=filenames, duration=duration, output_file=output_file)


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)
    normalized_acc = accuracy / float(len(y_true))
    return normalized_acc


def predict_and_score(clf,
                      x_test,
                      y_test,
                      verbose=False,
                      title='Test accuracy'):
    y_pred = clf.predict(x_test)
    accuracy = np.sum(y_test == y_pred)
    normalized_acc = accuracy / float(len(y_test))
    if verbose:
        print(f'{title}: {normalized_acc * 100:06.04f}%')
    return normalized_acc


def remove_nan_rows(arr):
    """Remove all rows with `nan` values in `arr`."""
    mask = np.isnan(arr).any(axis=1)
    return arr[~mask]


def remove_nan_cols(arr):
    """Remove all columns with `nan` values in `arr`."""
    mask = np.isnan(arr).any(axis=0)
    return arr[~mask]


def replace_in(arr, replacement_dict, inplace=False):
    """Replace the keys found in `arr` with the values from
    the `replacement_dict`.
    """
    if inplace:
        new_arr = arr
    else:
        import copy
        new_arr = copy.deepcopy(arr)
    for k, v in replacement_dict.items():
        new_arr[arr == k] = v
    return new_arr


def train_test_split(data, train=0.7, val=0.15, shuffle=None, return_xy=False):
    """Split a classification dataset in such a way so as to
    preserve the class distribution in subsamples of the dataset.
    """
    if train + val > 1.0:
        raise ValueError('Invalid split values for train and val.')
    Y = data[:, -1]
    labels = set(Y)
    hist = dict()
    for l in labels:
        data_l = data[Y == l]
        nl = len(data_l)
        nl_train = int(nl * train)
        nl_val = int(nl * val)
        nl_test = nl - (nl_train + nl_val)
        hist[l] = (nl_train, nl_val, nl_test)

    train_data = list()
    val_data = list()
    test_data = list()
    for l, (nl_train, nl_val, nl_test) in hist.items():
        data_l = data[Y == l]
        if shuffle:
            np.random.shuffle(data_l)
        train_l = data_l[:nl_train]
        val_l = data_l[nl_train:nl_train + nl_val]
        test_l = data_l[nl_train + nl_val:nl_train + nl_val + nl_test]
        train_data.append(train_l)
        val_data.append(val_l)
        test_data.append(test_l)

    def _squash(data_list):
        data = np.array(data_list[0])
        for item in data_list[1:]:
            data = np.vstack((data, np.array(item)))
        return data

    train_data = _squash(train_data)
    if val_data:
        val_data = _squash(val_data)
    if test_data:
        test_data = _squash(test_data)
    if return_xy:
        x_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        x_val = val_data[:, :-1]
        y_val = val_data[:, -1]
        x_test = test_data[:, :-1]
        y_test = test_data[:, -1]
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    return train_data, val_data, test_data


def class_histogram(data, title='Untitled'):
    plt.figure(title)
    plt.clf()
    plt.title(title)
    dist, counts = np.unique(data[:, -1], return_counts=True)
    plt.bar(dist, counts)
    plt.xticks(dist)
    print('Call matplotlib.pyplot.show() to see the plot.')


def ntimer(n=10):
    """Wraps a function which wraps another function to time it."""
    if n < 1:
        raise (Exception(f'Invalid n = {n} given.'))

    def timer(func):
        """Wraps `func` with a timer and returns the wrapped `func`."""
        def wrapper(*args, **kwargs):
            rv = None
            before = time()
            for _ in range(n):
                rv = func(*args, **kwargs)
            after = time()
            elapsed = after - before
            print(f'Elapsed: {elapsed*1e3:02.02f} ms')
            return rv

        return wrapper

    return timer


def memoize(verbose=True):
    """Wraps a function which wraps another function that memoizes."""
    def memoizer(func):
        """Memoize (cache) return values of `func`.
        Wraps `func` and returns the wrapped `func` so that `func`
        is executed when the results are not available in the cache.
        """
        cache = {}

        def wrapper(*args, **kwargs):
            t = (pickle.dumps(args), pickle.dumps(kwargs))
            if t not in cache:
                if verbose:
                    print(f'Adding NEW rv {func.__name__}{args}{kwargs} '
                          'to cache.')
                cache[t] = func(*args, **kwargs)
            else:
                if verbose:
                    print(f'Using OLD rv {func.__name__}{args}{kwargs} '
                          'from cache.')
            return cache[t]

        return wrapper

    return memoizer