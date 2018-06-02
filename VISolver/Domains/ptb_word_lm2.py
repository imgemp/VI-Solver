# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

# function approximation baseline
Epoch: 20 Train Perplexity: 117.068
Epoch: 20 Valid Perplexity: 211.004
Test Perplexity: 196.415

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm2.py --data_path=/nfs/nemo/u5/imgemp/Data/simple-examples/data/ --model small

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn.ptb import reader

from tensorflow.python.ops.rnn_field_cell import DNNField, BasicFieldCell, BasicFieldCell2, DNNKernelField, KernelFieldCell, QuantumFieldCell, DynamicFieldCell, DynamicFieldCell2, DeltaRNNCell, DeltaRNNCell2, BasicRNNCell, BasicLSTMCell, LSTMFieldCell, LSTMFieldCell2, LSTMFieldCell3, LSTMFieldCell4, LSTMFieldCell5, LSTMFieldKernelCell, LSTMKernelCell, LSTMKernelPathCell
from tensorflow.python.ops.rnn_field_cell import DropoutWrapper
# from IPython import embed
flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._input = input_
    
    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    

    # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    

    # lstm_cell = BasicRNNCell(size)


    # lstm_cell = DeltaRNNCell2(input_size=size,num_units=size)


    # lstm_cell = DeltaRNNCell(size)

    # Epoch: 20 Train Perplexity: 151.499
    # Epoch: 20 Valid Perplexity: 175.868
    # Test Perplexity: 169.027
    # lstm_cell = LSTMFieldKernelCell(input_size=size,num_units=size,hidden_size=size,n_inter=0,keep_prob=1.)  # 200 num_steps, 0.1 init, 1.0 lr


    # Small
    # Epoch: 10 Train Perplexity: 137.534
    # Epoch: 10 Valid Perplexity: 167.200
    # Medium
    # Epoch: 33 Train Perplexity: 96.921
    # Epoch: 33 Valid Perplexity: 132.503
    # fields = DNNField(input_size=size,num_units=50,hidden_units=[size,50])  # [50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = LSTMFieldCell(input_size=size,num_units=size,fields=fields.dnn_dynfield,hidden_size=fields._num_units,n_inter=0,keep_prob=1.)


    # Epoch: 20 Train Perplexity: 69.544
    # Epoch: 20 Valid Perplexity: 255.152
    # Test Perplexity: 177.773
    # fields = DNNField(input_size=size,num_units=50,hidden_units=[50])  # [50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = LSTMFieldCell(input_size=size,num_units=size,fields=fields.dnn_field2A,hidden_size=fields._num_units,n_inter=6,keep_prob=1.)


    # fields = DNNField(input_size=size,num_units=50,hidden_units=[50])  # [50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = LSTMFieldDynCell(input_size=size,num_units=size,fields=fields.dnn_dynfield,hidden_size=fields._num_units,n_inter=6,keep_prob=1.)

    # Epoch: 20 Train Perplexity: 64.321
    # Epoch: 20 Valid Perplexity: 154.736
    # Test Perplexity: 148.378
    # fields = DNNField(input_size=size,num_units=size,hidden_units=[])  # [50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = LSTMFieldCell2(input_size=size,num_units=size,fields=fields.dnn_field2AB,hidden_size=fields._num_units,n_inter=0,keep_prob=1.)


    # fields = DNNField(input_size=size,num_units=size,hidden_units=[])  # 2 layers, 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = LSTMFieldCell3(input_size=size,num_units=size,fields=fields.dnn_field2AB,hidden_size=fields._num_units,n_inter=0,keep_prob=1.)

    # Epoch: 8 Train Perplexity: 90.852
    # Epoch: 8 Valid Perplexity: 142.973
    # fields = DNNField(input_size=size,num_units=size,hidden_units=[size])  # [50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = LSTMFieldCell(input_size=size,num_units=size,fields=fields.dnn_field2A,hidden_size=fields._num_units,n_inter=4,keep_prob=1.)

    # 1 Layer
    # Epoch: 20 Train Perplexity: 66.916
    # Epoch: 20 Valid Perplexity: 157.994
    # Test Perplexity: 153.726
    # 2 Layers
    # Epoch: 20 Train Perplexity: 84.392
    # Epoch: 20 Valid Perplexity: 149.885
    # Test Perplexity: 144.370
    # fields = DNNField(input_size=size,num_units=50,hidden_units=[size,50])  # [50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = LSTMFieldCell(input_size=size,num_units=size,fields=fields.dnn_field2A,hidden_size=fields._num_units,n_inter=2,keep_prob=1.)

    # Epoch: 20 Train Perplexity: 67.232
    # Epoch: 20 Valid Perplexity: 161.711
    # Test Perplexity: 153.928
    # fields = DNNField(input_size=size,num_units=50,hidden_units=[size])  # [50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = LSTMFieldCell4(input_size=size,num_units=size,fields=fields.dnn_field2A,hidden_size=fields._num_units,n_inter=2,keep_prob=1.)

    # Epoch: 20 Train Perplexity: 101.710
    # Epoch: 20 Valid Perplexity: 169.422
    # Test Perplexity: 160.792
    # fields = DNNField(input_size=size,num_units=50,hidden_units=[size])  # size=50, 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = LSTMFieldCell5(input_size=size,num_units=size,fields=fields.dnn_field2A,hidden_size=fields._num_units,n_inter=2,keep_prob=1.)

    # Epoch: 20 Train Perplexity: 56.808
    # Epoch: 20 Valid Perplexity: 135.481
    # Test Perplexity: 130.065
    # fields = DNNField(input_size=size,num_units=size,hidden_units=[])  # size=50, 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = LSTMKernelCell(input_size=size,num_units=size,fields=fields.dnn_kernel_int,hidden_size=fields._num_units,n_inter=0,keep_prob=1.)

    # Epoch: 20 Train Perplexity: 57.210
    # Epoch: 20 Valid Perplexity: 136.945
    # Test Perplexity: 130.935
    # fields = DNNField(input_size=size,num_units=size,hidden_units=[])  # size=50, 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = LSTMKernelCell(input_size=size,num_units=size,fields=fields.dnn_kernel_int,hidden_size=fields._num_units,n_inter=4,keep_prob=1.)

    # Small
    # Epoch: 20 Train Perplexity: 59.664
    # Epoch: 20 Valid Perplexity: 132.561
    # Test Perplexity: 127.170
    # Small x 2 layers
    # Epoch: 20 Train Perplexity: 59.664
    # Epoch: 20 Valid Perplexity: 132.561
    # Test Perplexity: 127.170
    # Medium
    # print(size)
    # fields = DNNField(input_size=size,num_units=size,hidden_units=[])  # 1e-1 lr
    # if is_training:
    #   lstm_cell = LSTMKernelCell(input_size=size,num_units=size,fields=fields.dnn_kernel_int2,hidden_size=fields._num_units,n_inter=0,keep_prob=config.keep_prob)
    # else:
    #   lstm_cell = LSTMKernelCell(input_size=size,num_units=size,fields=fields.dnn_kernel_int2,hidden_size=fields._num_units,n_inter=0,keep_prob=1.)

    # ALL OF THIS WAS USING LEFT HAND INTEGRAL APPROX!!!
    # N_path = 10, keep_prob=0.9
    # Epoch: 20 Train Perplexity: 60.042
    # Epoch: 20 Valid Perplexity: 125.632
    # Test Perplexity: 121.541
    # N_path = 50, keep_prob=0.8
    # Epoch: 20 Train Perplexity: 61.398
    # Epoch: 20 Valid Perplexity: 124.660
    # Test Perplexity: 119.746
    # N_path = 100, keep_prob=0.75
    # Epoch: 20 Train Perplexity: 62.514
    # Epoch: 20 Valid Perplexity: 123.597
    # Test Perplexity: 119.908
    # N_path = 10, keep_prob=0.75, n_inter = 2
    # Epoch: 20 Train Perplexity: 65.966
    # Epoch: 20 Valid Perplexity: 125.377
    # Test Perplexity: 121.608
    # N_path = 200, keep_prob=0.75, n_inter = 0
    # Epoch: 20 Train Perplexity: 62.956
    # Epoch: 20 Valid Perplexity: 125.201
    # Test Perplexity: 120.120
    # N_path = 10, keep_prob=0.85, num_layers=2
    # Epoch: 20 Train Perplexity: 62.002
    # Epoch: 20 Valid Perplexity: 130.367
    # Test Perplexity: 125.605
    # N_path = 10, keep_prob=0.75, num_layers=2
    # Epoch: 20 Train Perplexity: 66.680
    # Epoch: 20 Valid Perplexity: 129.773
    # Test Perplexity: 124.572
    # N_path = 20, keep_prob=0.65, num_layers=2
    # Epoch: 20 Train Perplexity: 69.606
    # Epoch: 20 Valid Perplexity: 131.423
    # Test Perplexity: 126.134
    # N_path = 20, keep_prob=0.85, lr=5e-1
    # Epoch: 20 Train Perplexity: 20.611
    # Epoch: 20 Valid Perplexity: 363.019
    # Test Perplexity: 349.496
    # N_path = 20, keep_prob=0.85, lr=5e-2, 40 epochs
    # Epoch: 40 Train Perplexity: 59.937
    # Epoch: 40 Valid Perplexity: 123.395
    # Test Perplexity: 119.271
    # N_path = 20, keep_prob=0.8, lr=1e-1, lr_decay=0.5, max_epoch=20, max_max_epoch=40
    # Epoch: 40 Train Perplexity: 54.175
    # Epoch: 40 Valid Perplexity: 120.047
    # Test Perplexity: 114.030


    # BACK TO TRAPEZOID RULE!!!
    # N_path = 20, keep_prob=0.8, lr=1e-1, max_max_epoch=20
    # Epoch: 20 Train Perplexity: 64.989
    # Epoch: 20 Valid Perplexity: 125.532
    # Test Perplexity: 121.978
    # N_path = 20, keep_prob=0.8, lr=1e-1, max_max_epoch=20, n_inter=2
    # Epoch: 20 Train Perplexity: 66.193
    # Epoch: 20 Valid Perplexity: 127.504
    # Test Perplexity: 123.299
    # N_path = 20, keep_prob=0.8, lr=1e-1, max_epoch=20, lr_decay=0.5, max_max_epoch=60, num_layers=2
    # Epoch: 60 Train Perplexity: 56.397
    # Epoch: 60 Valid Perplexity: 123.371
    # Test Perplexity: 118.249
    # N_path = 50, max_grad_norm=30, keep_prob=0.8, lr=1e-2, max_epoch=20, lr_decay=0.2, max_max_epoch=60, num_layers=2


    # Define a function that spits out N (e.g., 4) intermediate points
    # xa,xb,xc,xd = f(x0,xf)
    N_path = 50
    print(size)
    fields = DNNField(input_size=size,num_units=size,hidden_units=[])  # 1e-1 lr
    if is_training:
      lstm_cell = LSTMKernelPathCell(input_size=size,num_units=size,fields=fields.dnn_kernel_int2,hidden_size=fields._num_units,n_inter=0,keep_prob=config.keep_prob,N_path=N_path)
    else:
      lstm_cell = LSTMKernelPathCell(input_size=size,num_units=size,fields=fields.dnn_kernel_int2,hidden_size=fields._num_units,n_inter=0,keep_prob=config.keep_prob,N_path=N_path)

    # Epoch: 20 Train Perplexity: 65.140
    # Epoch: 20 Valid Perplexity: 160.097
    # Test Perplexity: 152.112
    # fields = DNNField(input_size=size,num_units=50,hidden_units=[size,50])  # [50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = LSTMFieldCell(input_size=size,num_units=size,fields=fields.dnn_dynfield,hidden_size=fields._num_units,n_inter=2,keep_prob=1.)


    # midsize = 50
    # fields1 = DNNField(input_size=size,num_units=midsize,hidden_units=[midsize])  # [50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell1 = LSTMFieldCell(input_size=size,num_units=midsize,fields=fields1.dnn_dynfield,hidden_size=fields1._num_units,n_inter=0,keep_prob=1.)
    # fields2 = DNNField(input_size=midsize,num_units=size,hidden_units=[midsize])  # [size,50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell2 = LSTMFieldCell(input_size=midsize,num_units=size,fields=fields2.dnn_dynfield,hidden_size=fields2._num_units,n_inter=0,keep_prob=1.)
    # if is_training and config.keep_prob < 1:
    #   lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(
    #       lstm_cell1, output_keep_prob=config.keep_prob)
    #   lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(
    #       lstm_cell2, output_keep_prob=config.keep_prob)
    # cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1,lstm_cell2], state_is_tuple=True)


    # fields = DNNField(input_size=size,num_units=50,hidden_units=[size,50])  # [50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = DynamicFieldCell2(input_size=size,num_units=size,fields=fields.dnn_dynfield,hidden_size=fields._num_units,n_inter=0,keep_prob=1.,activation=None)


    # fields = DNNField(size,size,hidden_units=[2*size,50]).dnn_dynfield  # [50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = DynamicFieldCell(size,size,fields,n_inter=0,keep_prob=1.)

    # Epoch: 13 Train Perplexity: 700.869
    # Epoch: 13 Valid Perplexity: 687.249
    # Test Perplexity: 654.177
    # fields = DNNField(size,size,hidden_units=[size,50]).dnn_field2A  # [size,50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = BasicFieldCell(size,size,fields,n_inter=0,keep_prob=0.6)

    # fields = DNNField(size,size,hidden_units=[size,50]).dnn_field2A  # [50], 1e-1 lr, num_steps=40, no tanh on outputs (before softmax)
    # lstm_cell = QuantumFieldCell(size,size,fields,n_inter=0,gam=0.5)

    # Epoch: 13 Train Perplexity: 233.104
    # Epoch: 13 Valid Perplexity: 281.561
    # Test Perplexity: 252.535
    # fields = DNNField(size,size,hidden_units=[size,size,50]).dnn_field2A  # [50], 1e-1 lr, num_steps=40, no tanh on outputs (before softmax)
    # lstm_cell = BasicFieldCell(size,size,fields,n_inter=10)

    # Epoch: 13 Train Perplexity: 229.054
    # Epoch: 13 Valid Perplexity: 275.070
    # Test Perplexity: 249.124
    # fields = DNNField(size,size,hidden_units=[size,size,50]).dnn_field2A  # [50], 1e-1 lr, num_steps=40, no tanh on outputs (before softmax)
    # lstm_cell = BasicFieldCell(size,size,fields,n_inter=0)

    # Epoch: 13 Train Perplexity: 255.024
    # Epoch: 13 Valid Perplexity: 313.981
    # Test Perplexity: 281.714
    # fields = DNNField(size,size,hidden_units=[size,50]).dnn_field2A  # [size,50], 1e-1 lr, num_steps=40, no tanh on outputs (before softmax)
    # lstm_cell = BasicFieldCell(size,size,fields,n_inter=2)

    # fields = DNNField(input_size=size,num_units=50,hidden_units=[size,50])  # [size,50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = BasicFieldCell2(size,size,fields.dnn_field2A,hidden_size=fields._num_units,n_inter=0,activation=None)

    # Epoch: 13 Train Perplexity: 205.959
    # Epoch: 13 Valid Perplexity: 255.117
    # Test Perplexity: 238.763
    # fields = DNNField(size,size,hidden_units=[size,50]).dnn_field2A  # [size,50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = BasicFieldCell(size,size,fields,n_inter=0)

    # fields = DNNField(size,size,hidden_units=[size,50]).dnn_field3A  # [size,50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = BasicFieldCell(size,size,fields,n_inter=0)

    # Epoch: 30 Train Perplexity: 707.528
    # Epoch: 30 Valid Perplexity: 703.176
    # Test Perplexity: 679.491
    # fields = DNNField(size,size,hidden_units=[size,size,size,size,50]).dnn_field2A  # [size,50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell = BasicFieldCell(size,size,fields,n_inter=5)

    # fields_kernel = DNNKernelField(size,size,hidden_units=[10*size,5*size,size]).dnn_ker_field
    # lstm_cell = KernelFieldCell(size,size,fields_kernel)

    # if is_training and config.keep_prob < 1:
    #   lstm_cell = DropoutWrapper(
    #       lstm_cell, output_keep_prob=config.keep_prob)
      # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
      #     lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)


    # midsize = 50
    # fields1 = DNNField(input_size=size,num_units=midsize,hidden_units=[size,50])  # [size,50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell1 = BasicFieldCell2(input_size=size,num_units=midsize,fields=fields1.dnn_field2A,hidden_size=fields1._num_units,n_inter=0)
    # fields2 = DNNField(input_size=midsize,num_units=size,hidden_units=[midsize])  # [size,50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell2 = BasicFieldCell2(input_size=midsize,num_units=size,fields=fields2.dnn_field2A,hidden_size=fields2._num_units,n_inter=0)
    # if is_training and config.keep_prob < 1:
    #   lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(
    #       lstm_cell1, output_keep_prob=config.keep_prob)
    #   lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(
    #       lstm_cell2, output_keep_prob=config.keep_prob)
    # cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1,lstm_cell2], state_is_tuple=True) 

    # Epoch: 13 Train Perplexity: 700.899
    # Epoch: 13 Valid Perplexity: 687.240
    # Test Perplexity: 654.164
    # midsize = 50
    # fields1 = DNNField(input_size=size,num_units=midsize,hidden_units=[]).dnn_field2A  # [size,50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell1 = BasicFieldCell(input_size=size,num_units=midsize,fields=fields1,n_inter=3)
    # fields2 = DNNField(input_size=midsize,num_units=size,hidden_units=[]).dnn_field2A  # [size,50], 1e-1 lr, no tanh on outputs (before softmax)
    # lstm_cell2 = BasicFieldCell(input_size=midsize,num_units=size,fields=fields2,n_inter=3)
    # # fields_kernel = DNNKernelField(size,size,hidden_units=[10*size,5*size,size]).dnn_ker_field
    # # lstm_cell = KernelFieldCell(size,size,fields_kernel)
    # if is_training and config.keep_prob < 1:
    #   lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(
    #       lstm_cell1, output_keep_prob=config.keep_prob)
    #   lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(
    #       lstm_cell2, output_keep_prob=config.keep_prob)
    # cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1,lstm_cell2], state_is_tuple=True) 

    self._initial_state = cell.zero_state(batch_size*N_path, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
    
    # if is_training and config.keep_prob < 1:
    #   inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # print(inputs.get_shape())

    # print('tiled')
    inputs = tf.tile(inputs,[N_path,1,1], name='tiled_inputs')
    # print(inputs.get_shape())

    inputs = tf.unpack(inputs, num=num_steps, axis=1)
    outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
    # outputs = []
    # state = self._initial_state
    # with tf.variable_scope("RNN"):
    #   for time_step in range(num_steps):
    #     if time_step > 0: tf.get_variable_scope().reuse_variables()
    #     (cell_output, state) = cell(inputs[:, time_step, :], state)
    #     outputs.append(cell_output)

    # print('outputs before reshape')
    # print(len(outputs))
    # print(outputs[0].get_shape())
    #     25
    # (20, 200)
    # outputs = tf.pack(outputs)
    # print(outputs.get_shape())

    outputs = tf.reshape(outputs,(num_steps,N_path,batch_size,size))
    # print('reshape to num_steps,N_path,batch_size,hidden_size')
    # print(outputs.get_shape())

    # outputs = tf.reshape(outputs,(num_steps,batch_size,size))
    outputs = tf.reduce_mean(outputs,reduction_indices=1)
    # print('take mean')
    # print(outputs.get_shape())

    # print('unpack')
    outputs = tf.unpack(outputs, num=num_steps, axis=0)
    # print(len(outputs))
    # print(outputs[0].get_shape())

    # outputs = tf.tanh(outputs)


    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    # output = tf.reshape(outputs,[-1,size])
    # print('before softmax')
    # print(output.get_shape())
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    print(len(tvars))
    # embed()
    print([tvar.name for tvar in tvars])
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)


    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    # optimizer = tf.train.AdamOptimizer(self._lr)


    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1 # 0.1
  learning_rate = 1.0e-1 # 1.0
  max_grad_norm = 5 # 5
  num_layers = 2 # 2
  num_steps = 25 # 20
  hidden_size = 200
  max_epoch = 20
  max_max_epoch = 40
  keep_prob = 0.8 # 1.0
  lr_decay = 0.5 # 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.1 # 0.01
  learning_rate = 1.0e-1 # 1.
  max_grad_norm = 5
  num_layers = 1 # 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 20
  max_max_epoch = 60
  keep_prob = 0.8 # 0.5
  lr_decay = 0.1 # 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.01
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    # for i, (c, h) in enumerate(model.initial_state):
    #   feed_dict[c] = state[i].c
    #   feed_dict[h] = state[i].h
    # for i, (prev_inputs, h, q_inputs, gamk) in enumerate(model.initial_state):
    #     feed_dict[prev_inputs] = state[i].prev_inputs
    #     feed_dict[h] = state[i].h
    #     feed_dict[q_inputs] = state[i].q_inputs
    #     feed_dict[gamk] = state[i].gamk
    # for i, (prev_inputs, h) in enumerate(model.initial_state):
    #     feed_dict[prev_inputs] = state[i].prev_inputs
    #     feed_dict[h] = state[i].h
    # for i, (h) in enumerate(model.initial_state):
    #   feed_dict[h] = state[i]
    for i, (prev_inputs, h, prev_mask) in enumerate(model.initial_state):
        feed_dict[prev_inputs] = state[i].prev_inputs
        feed_dict[h] = state[i].h
        feed_dict[prev_mask] = state[i].prev_mask

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.scalar_summary("Training Loss", m.cost)
      tf.scalar_summary("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.scalar_summary("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()
