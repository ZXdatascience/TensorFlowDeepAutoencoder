from __future__ import division
import os
from os.path import join as pjoin
from pathlib import Path
import sys

import tensorflow as tf


def home_out(path):
  return pjoin(str(Path.home()), 'tmp', 'SAE', path)

flags = tf.app.flags
FLAGS = flags.FLAGS

units_each_hidden_layer = 50
units_each_dense_layer = 50
num_hidden_layers = 3
num_dense_layer = 3
# Autoencoder Architecture Specific Flags
flags.DEFINE_integer("num_hidden_layers", num_hidden_layers, "Number of hidden autoencoder layers")
flags.DEFINE_integer("num_dense_layers", num_dense_layer, "Number of dense layers")

for i in range(num_hidden_layers):
    flags.DEFINE_integer('hidden{}_units'.format(i+1), units_each_hidden_layer,
                         'Number of units in autoencoder layer {}.'.format(i+1))

for i in range(num_dense_layer):
    flags.DEFINE_integer('dense{}_units'.format(i+1), units_each_dense_layer,
                         'Number of units in dense layer {}.'.format(i+1))

for i in range(num_hidden_layers):
    flags.DEFINE_float('pre_layer{}_learning_rate'.format(i+1), 0.0001,
                       'Initial learning rate.')

for i in range(num_hidden_layers):
    flags.DEFINE_float('noise_{}'.format(i+1), 0.50, 'Rate at which to set data to 0')

# Constants
flags.DEFINE_integer('image_size', 50, 'Image square size')

flags.DEFINE_integer('input_dim', 50 * 50, 'Input dimension')

flags.DEFINE_integer('output_dim', 3, 'Output dimension')

flags.DEFINE_integer('seed', 1234, 'Random seed')

flags.DEFINE_integer('batch_size', 100,
                     'Batch size. Must divide evenly into the dataset sizes.')

flags.DEFINE_float('supervised_learning_rate', 0.1,
                   'Supervised initial learning rate.')

flags.DEFINE_integer('pretraining_epochs', 60,
                     "Number of training epochs for pretraining layers")
flags.DEFINE_integer('finetuning_epochs', 56,
                     "Number of training epochs for "
                     "fine tuning supervised step")

flags.DEFINE_float('zero_bound', 1.0e-9,
                   'Value to use as buffer to avoid '
                   'numerical issues at 0')
flags.DEFINE_float('one_bound', 1.0 - 1.0e-9,
                   'Value to use as buffer to avoid numerical issues at 1')

flags.DEFINE_float('flush_secs', 120, 'Number of seconds to flush summaries')

# Directories
flags.DEFINE_string('data_dir', home_out('data'),
                    'Directory to put the training data.')

flags.DEFINE_string('summary_dir', home_out('summaries'),
                    'Directory to put the summary data')

flags.DEFINE_string('chkpt_dir', home_out('chkpts'),
                    'Directory to put the model checkpoints')

# TensorBoard
flags.DEFINE_boolean('no_browser', True,
                     'Whether to start browser for TensorBoard')

# Python
flags.DEFINE_string('python', sys.executable,
                    'Path to python executable')
