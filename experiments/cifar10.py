# The Global Variable `is_training` is Data()._is_training
# ==============================================================================
import os
import sys
import tarfile
from six.moves import xrange, urllib
import pickle

import numpy as np


class Cifar10:
    """
    load cifar10, provide mini-batch method.
    """

    def __init__(self):
        self.maybe_download_and_extract()
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.unpickle()

        self.iter = 0
        self.epoch = 0

    @classmethod
    def maybe_download_and_extract(
            cls, data_url="http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", dst_dir="./cifar10"):
        """Download and extract the tarball from Alex's website."""
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        filename = data_url.split('/')[-1]
        filepath = os.path.join(dst_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        extracted_dir_path = os.path.join(dst_dir, 'cifar-10-batches-py')
        if not os.path.exists(extracted_dir_path):
            tarfile.open(filepath, 'r:gz').extractall(dst_dir)

    @classmethod
    def unpickle(cls, data_dir="./cifar10/cifar-10-batches-py"):
        """:return: train images, train labels, test images, test labels
            http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
        """
        train_images, train_labels = [], []

        for file in [os.path.join(data_dir, 'data_batch_%d' % i) for i in xrange(1, 6)]:
            with open(file, 'rb') as fo:
                dic = pickle.load(fo, encoding='bytes')
                train_images.append(dic[b'data'].reshape(-1, 3, 32, 32))
                train_labels += dic[b'labels']
        train_images = np.vstack(train_images).transpose([0, 2, 3, 1])

        with open(os.path.join(data_dir, 'test_batch'), 'rb') as fo:
            dic = pickle.load(fo, encoding='bytes')
            test_images = dic[b'data'].reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])
            test_labels = dic[b'labels']

        assert train_images.shape == (50000, 32, 32, 3)
        assert train_labels.__len__() == 50000
        assert test_images.shape == (10000, 32, 32, 3)
        assert test_labels.__len__() == 10000

        return train_images, train_labels, test_images, test_labels

    def next_batch(self, batch_size):
        """
        :param batch_size: number of samples in a mini-batch
        :return: images and labels of samples
        """
        if (self.iter + 1) * batch_size > 50000:
            self.iter = 0
            self.epoch += 1

            indices = list(range(50000))
            np.random.shuffle(indices)
            self.train_images = self.train_images[indices]
            self.train_labels = np.reshape(np.reshape(self.train_labels, [-1, 1])[indices], -1)

        start = self.iter * batch_size
        end = start + batch_size
        self.iter += 1

        batch_images = self.train_images[start:end]
        batch_labels = self.train_labels[start:end]

        # ------------------- data augmentation ---------------------
        # pad
        pad_images = np.pad(
            batch_images,
            pad_width=((0, 0), (2, 2), (2, 2), (0, 0)),
            mode='constant',
            constant_values=0
        )

        # crop & flip
        batch_images = np.zeros((batch_size, 32, 32, 3), dtype=batch_images.dtype)
        for i, image in enumerate(pad_images):
            h_offset = np.random.randint(low=0, high=4, size=1)[0]
            w_offset = np.random.randint(low=0, high=4, size=1)[0]

            if np.random.randint(low=0, high=2):
                batch_images[i, ...] = image[h_offset:h_offset + 32, w_offset:w_offset + 32, :]
            else:
                batch_images[i, ...] = image[h_offset:h_offset + 32, w_offset + 32:w_offset:-1, :]

        # white
        means = np.mean(batch_images, axis=(1, 2, 3)).reshape((batch_size, 1, 1, 1))
        stds = np.maximum(np.std(batch_images, axis=(1, 2, 3)), 1.0 / np.sqrt(32 * 32 * 3))
        batch_images = (batch_images - means) / stds.reshape((batch_size, 1, 1, 1))
        # means = np.mean(batch_images, axis=(1, 2)).reshape((batch_size, 1, 1, 3))
        # stds = np.maximum(np.std(batch_images, axis=(1, 2)), 1.0 / np.sqrt(32 * 32))
        # batch_images = (batch_images - means) / stds.reshape((batch_size, 1, 1, 3))

        return batch_images, batch_labels

    def test_batch(self):
        means = np.mean(self.test_images, axis=(1, 2, 3)).reshape((-1, 1, 1, 1))
        stds = np.maximum(np.std(self.test_images, axis=(1, 2, 3)), 1.0 / np.sqrt(32 * 32 * 3))
        return (self.test_images - means) / stds.reshape((-1, 1, 1, 1)), self.test_labels
        # means = np.mean(self.test_images, axis=(1, 2)).reshape((-1, 1, 1, 3))
        # stds = np.maximum(np.std(self.test_images, axis=(1, 2)), 1.0 / np.sqrt(32 * 32))
        # return (self.test_images - means) / stds.reshape((-1, 1, 1, 3)), self.test_labels


# data = Cifar10()
# data.iter = 50000
# images, labels = data.next_batch(128)
#
# img = images[0]
# print(img)
# print(labels)
