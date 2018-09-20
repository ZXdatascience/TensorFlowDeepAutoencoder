"""Functions for downloading and reading MNIST data."""
from __future__ import division
from __future__ import print_function

import gzip

import numpy
from statsmodels.tsa.stattools import pacf
from six.moves import urllib
from six.moves import range  # pylint: disable=redefined-builtin
from TensorFlowDeepAutoencoder.code.ae.utils.flags import FLAGS
import tensorflow as tf
import pandas as pd
import numpy as np
import os
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


class DataSet(object):

  def __init__(self, data, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert data.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (data.shape,
                                                 labels.shape))
      self._num_examples = data.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert data.shape[3] == 1
      data = data.reshape(data.shape[0],
                              data.shape[1] * data.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].
      data = data.astype(numpy.float32)
    self._data = data
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def data(self):
    return self._data

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._data = self._data[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


class DataSetPreTraining(object):

  def __init__(self, data):
    self._num_examples = data.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    assert data.shape[3] == 1
    data = data.reshape(data.shape[0],
                        data.shape[1] * data.shape[2])
    # Convert from [0, 255] -> [0.0, 1.0].
    data = data.astype(numpy.float32)
    self._data = data
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def data(self):
    return self._data

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._data[start:end], self._data[start:end]


def add_lag(cons, thres=0.1):
  cons = np.array(cons)
  pa = pacf(cons, nlags=150)
  above_thres_indices = np.argwhere(pa > 0.1)
  above_thres_indices = np.delete(above_thres_indices, 0)
  max_lag = max(above_thres_indices)
  data = []
  labels = []
  for i in range(max_lag, len(cons) - 1):
    new_indices = i - above_thres_indices
    new_series = cons[new_indices]
    data.append(new_series)
    labels.append(cons[i])
  return np.array(data), np.array(labels)


def mean_filter(cons, window_size=2):
  res = []
  for i in range(len(cons) - window_size + 1):
    res.append(np.array(cons[i: i+window_size]).mean())
  return res


def to_30_min(cons):
  i = 0
  res = []
  while i + 1 < len(cons) :
    res.append(cons[i] + cons[i+1])
    i += 2
  return res

def read_data_sets(train_dir='', fake_data=False, one_hot=False):
  class DataSets(object):
    pass
  data_sets = DataSets()
  train_data_name = 'building1retail.csv'
  df = pd.read_csv(train_dir + train_data_name)

  electricity_cons = df['Power (kW)'].values
  mean_filtered_cons = mean_filter(electricity_cons)
  min30_filtered = to_30_min(mean_filtered_cons)
  data, labels = add_lag(min30_filtered)
  data = data.reshape(data.shape[0], data.shape[1], 1, 1)
  train_ratio = 0.7
  validation_ratio = 0.2
  test_ratio = 0.1
  train_data = data[:train_ratio * len(data)]
  train_labels = labels[:train_ratio * len(data)]
  validation_data = data[train_ratio * len(data): (train_ratio + validation_ratio) * len(data)]
  validation_labels = labels[train_ratio * len(data): (train_ratio + validation_ratio) * len(data)]
  test_data = data[(1 - test_ratio) * len(data):]
  test_labels = labels[(1 - test_ratio) * len(data):]

  data_sets.train = DataSet(train_data, train_labels)
  data_sets.validation = DataSet(validation_data, validation_labels)
  data_sets.test = DataSet(test_data, test_labels)

  return data_sets


def read_data_sets_pretraining(train_dir=''):
  class DataSets(object):
    pass

  data_sets = DataSets()
  train_data_name = 'building1retail.csv'
  df = pd.read_csv(train_dir + train_data_name)

  electricity_cons = df['Power (kW)'].values
  mean_filtered_cons = mean_filter(electricity_cons)
  min30_filtered = to_30_min(mean_filtered_cons)
  data, labels = add_lag(min30_filtered)
  data = data.reshape(data.shape[0], data.shape[1], 1, 1)
  train_ratio = 0.7
  validation_ratio = 0.2
  test_ratio = 0.1
  train_data = data[:int(train_ratio * data.shape[0])]
  validation_data = data[int(train_ratio * len(data)): int((train_ratio + validation_ratio) * len(data))]
  test_data = data[int((1 - test_ratio) * len(data)):]

  data_sets.train = DataSetPreTraining(train_data)
  data_sets.validation = DataSetPreTraining(validation_data)
  data_sets.test = DataSetPreTraining(test_data)

  return data_sets


def _add_noise(x, rate):
  x_cp = numpy.copy(x)
  pix_to_drop = numpy.random.rand(x_cp.shape[0],
                                  x_cp.shape[1]) < rate
  x_cp[pix_to_drop] = FLAGS.zero_bound
  return x_cp


def fill_feed_dict_ae(data_set, input_pl, target_pl, noise=None):
    input_feed, target_feed = data_set.next_batch(FLAGS.batch_size)
    if noise:
      input_feed = _add_noise(input_feed, noise)
    feed_dict = {
        input_pl: input_feed,
        target_pl: target_feed
    }
    return feed_dict


def fill_feed_dict(data_set, images_pl, labels_pl, noise=False):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
  if noise:
      images_feed = _add_noise(images_feed, FLAGS.drop_out_rate)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


if __name__ == '__main__':
    data = read_data_sets_pretraining(FLAGS.data_dir)
    input_ = tf.placeholder(dtype=tf.float32,
                            shape=(FLAGS.batch_size, FLAGS.input_dim),
                            name='ae_input_pl')
    target_ = tf.placeholder(dtype=tf.float32,
                             shape=(FLAGS.batch_size, FLAGS.input_dim),
                             name='ae_target_pl')
    noise = {j: getattr(FLAGS, "noise_{0}".format(j + 1))
             for j in range(1)}

    print(fill_feed_dict_ae(data.train, input_, target_, noise[0]))
    # print(fill_feed_dict_ae(data, input_, target_, noise[0]).values()[0].shape)