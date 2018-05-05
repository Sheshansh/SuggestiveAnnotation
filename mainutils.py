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

"""Builds the BBBC006 network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf
# from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import bottleneck
import numpy as np
import math
import data_input

from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.slim.python.slim.nets import resnet_utils

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 1,
							"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'data',
						   """Path to the BBBC006 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
							"""Train the model using fp16.""")
tf.app.flags.DEFINE_integer('num_layers', 6,
							"""Number of layers in model.""")
tf.app.flags.DEFINE_integer('num_classes', 2,
							"""Number of output classes.""")
tf.app.flags.DEFINE_integer('feat_root', 32,
							"""Feature root.""")
tf.app.flags.DEFINE_integer('deconv_root', 8,
							"""Transposed convolution upscaling factor.""")

# Global constants describing the BBBC006 data set.
IMAGE_WIDTH = data_input.IMAGE_WIDTH
IMAGE_HEIGHT = data_input.IMAGE_HEIGHT
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9995  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 100.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.01  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.05  # Initial learning rate.
DROPOUT_RATE = 0.5  # Probability for dropout layers.
S_CLASS_PROP = .2249  # Segments proportion of pixels in class 1.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
	"""Helper to create summaries for activations.
	Creates a summary that provides a histogram of activations.
	Creates a summary that measures the sparsity of activations.

	Args:
		x: Tensor
	"""
	# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
	# session. This helps the clarity of presentation on tensorboard.
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.
	Args:
		name: name of the variable
		shape: list of ints
		initializer: initializer for Variable
	Returns:
		Variable Tensor
	"""
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var


def _variable_with_weight_decay(name, shape, stddev, wd):
	"""Helper to create an initialized Variable with weight decay.
	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.
	Args:
		name: name of the variable
		shape: list of ints
		stddev: standard deviation of a truncated Gaussian
		wd: add L2Loss weight decay multiplied by this float. If None, weight
			decay is not added for this Variable.
	Returns:
		Variable Tensor
	"""
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = _variable_on_cpu(
		name,
		shape,
		tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None and not tf.get_variable_scope().reuse:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var


def distorted_inputs():
	"""Construct distorted input for BBBC006 training using the Reader ops.
	Returns:
		images: Images. 4D tensor of [batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, 1] size.
		labels: Labels. 4D tensor of [batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, 2] size.
	Raises:
		ValueError: If no data_dir
	"""
	if not FLAGS.data_dir:
		raise ValueError('Please supply a data_dir')
	images, labels = data_input.distorted_inputs(batch_size=FLAGS.batch_size)
	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return images, labels


def inputs(eval_data, sessid):
	"""Construct input for BBBC006 evaluation using the Reader ops.
	Args:
		eval_data: bool, indicating if one should use the train or eval data set.
	Returns:
		images: Images. 4D tensor of [batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, 1] size.
		labels: Labels. 4D tensor of [batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, 2] size.

	Raises:
		ValueError: If no data_dir
	"""
	if not FLAGS.data_dir:
		raise ValueError('Please supply a data_dir')
	images, labels, i_paths = data_input.inputs(eval_data=eval_data,
										  batch_size=FLAGS.batch_size, sessid = sessid)
	labels = tf.cast(tf.divide(labels,255),tf.int32)
	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return images, labels, i_paths


def get_deconv_filter(shape):
	"""Return deconvolution weight tensor w/bilinear interpolation.
	Args:
		shape: 4D list of weight tensor shape.
	Returns:
		Tensor containing weight variable.

	Source:
		https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn16_vgg.py#L245
	"""
	width = shape[0]
	height = shape[0]
	f = math.ceil(width / 2.0)
	c = (2.0 * f - 1 - f % 2) / (2.0 * f)

	bilinear = np.zeros([shape[0], shape[1]])
	for x in range(width):
		for y in range(height):
			bilinear[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))

	weights = np.zeros(shape)
	for i in range(shape[2]):
		weights[:, :, i, i] = bilinear

	init = tf.constant_initializer(value=weights, dtype=tf.float32)
	return tf.get_variable(name='up_filter', initializer=init, shape=weights.shape)


def _deconv_layer(in_layer, w, b, dc, ds, scope):
	"""Return deconvolution layer tensor.
	Args:
		in_layer: Input tensor layer.
		w: Weight tensor.
		b: Bias tensor.
		dc: Deconvolution constant.
		ds: Deconvolution layer output shape in list format.
		scope: Enclosing variable scope.
	Returns:
		Tensor for deconvolution layer.
	"""
	deconv = tf.nn.conv2d_transpose(in_layer, w, ds, strides=[1, dc, dc, 1],
									padding='SAME')
	deconv = tf.nn.bias_add(deconv, bias=b, name=scope.name)
	deconv = tf.nn.relu(deconv)
	_activation_summary(deconv)
	return deconv


def inference(images, train=True):
	"""Build the BBBC006 model.
	Args:
		images: Images returned from distorted_inputs() or inputs().
	Returns:
		s_fuse: List of fused segments 4D tensors of [batch_size, 696, 520, 2] size.
	"""
	# We instantiate all variables using tf.get_variable() instead of
	# tf.Variable() in order to share variables across multiple GPU training runs.
	feat_out = FLAGS.feat_root
	# in_layer = tf.layers.batch_normalization(images)
	in_layer = images

	# Deconvolution constant: kernel size = 2 * dc, stride = dc
	# Deconvolution output shape
	dc = FLAGS.deconv_root
	ds = [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, FLAGS.num_classes]

	# Up-sampled layers 4-6 output maps for contours and segments, respectively
	s_outputs = []

	for layer in range(FLAGS.num_layers):
		# CONVOLUTION
		with tf.variable_scope('conv{}'.format(layer + 1)) as scope:
			# Double the number of feat_out for all but convolution layer 4
			feat_out *= 2 if layer != 4 else 1
			conv = tf.layers.conv2d(in_layer, feat_out, (3, 3), padding='same',
									activation=tf.nn.relu, name=scope.name)

			if train and layer > 3:  # During training, add dropout to layers 5 and 6
				conv = tf.nn.dropout(conv, keep_prob=DROPOUT_RATE)

			_activation_summary(conv)

		# POOLING
		# First and convolution layers has no pooling afterwards
		if 0 < layer:
			pool = tf.layers.max_pooling2d(conv, 2, 2, padding='same')

			_activation_summary(pool)
			in_layer = pool
		else:
			in_layer = conv

		# Transposed convolution and output mapping for segments and contours
		if layer > 2:  # Only applies to layers 3-5
			# TRANSPOSED CONVOLUTION
			with tf.variable_scope('deconv{0}'.format(layer + 1)) as scope:
				feat_in = in_layer.get_shape().as_list()[-1]
				shape = [dc * 2, dc * 2, FLAGS.num_classes, feat_in]
				w = get_deconv_filter(shape)
				b = _variable_on_cpu('biases', [FLAGS.num_classes],
							tf.constant_initializer(0.1))

				deconv = _deconv_layer(in_layer, w, b, dc, ds, scope)

			with tf.variable_scope('output{0}'.format(layer + 1)) as scope:
				output = tf.layers.conv2d(deconv, FLAGS.num_classes, (1, 1),
										  padding='same', activation=tf.nn.relu,
										  name=scope.name)
				s_outputs.append(output)
			dc *= 2
	s_fuse = tf.add_n(s_outputs)

	return s_fuse

"""
	Copied from tf website, just tweaked a bit to include BN and ReLU
"""
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None):
  """Bottleneck residual unit variant with BN before convolutions.
  This is the full preactivation residual unit variant proposed in [2]. See
  Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
  variant which has an extra bottleneck layer.
  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.
  Returns:
    The ResNet unit's output.
  """
  with variable_scope.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = layers.batch_norm(
        inputs, activation_fn=nn_ops.relu, scope='preact')
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = layers_lib.conv2d(
          preact,
          depth, [1, 1],
          stride=stride,
          normalizer_fn=None,
          activation_fn=None,
          scope='shortcut')

    residual = preact
    residual = tf.layers.batch_normalization(residual)
    residual = tf.nn.relu(residual)
    residual = layers_lib.conv2d(
        residual, depth_bottleneck, [1, 1], stride=1, scope='conv1')
    residual = tf.layers.batch_normalization(residual)
    residual = tf.nn.relu(residual)
    residual = resnet_utils.conv2d_same(
        residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
    residual = tf.layers.batch_normalization(residual)
    residual = tf.nn.relu(residual)
    residual = layers_lib.conv2d(
        residual,
        depth, [1, 1],
        stride=1,
        normalizer_fn=None,
        activation_fn=None,
        scope='conv3')

    output = shortcut + residual

    return utils.collect_named_outputs(outputs_collections, sc.name, output)

def inference_bottleneck(images, train=True):
	in_layer = images
	feat_out = FLAGS.feat_root
	s_outputs = []
	
	with tf.variable_scope('bottleneck0-1') as scope:
		in_layer = tf.layers.max_pooling2d(in_layer, 2, 2, padding='same')
		in_layer = tf.layers.batch_normalization(in_layer)
		in_layer = tf.nn.relu(in_layer)
		in_layer = tf.layers.conv2d(in_layer, feat_out, (3, 3), padding='same', name=scope.name)
	with tf.variable_scope('bottleneck0-2') as scope:
		in_layer = tf.layers.batch_normalization(in_layer)
		in_layer = tf.nn.relu(in_layer)
		in_layer = tf.layers.conv2d(in_layer, feat_out, (3, 3), padding='same', name=scope.name)
		s_outputs.append(in_layer)
	for layer in range(5):
		with tf.variable_scope('bottleneck{0}-{1}'.format(layer + 1,1)) as scope:
			in_layer = tf.layers.max_pooling2d(in_layer, 2, 2, padding='same')
			in_layer = bottleneck(in_layer,min(feat_out*2,256),feat_out,1)
		with tf.variable_scope('bottleneck{0}-{1}'.format(layer + 1,2)) as scope:
			feat_out = min(feat_out*2,256)
			in_layer = bottleneck(in_layer,feat_out,feat_out,1)
			s_outputs.append(in_layer)
	# populate s_outputs
	encoding = in_layer
	dc = 2 
	ds = [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, FLAGS.num_classes]
	for layer in range(len(s_outputs)):
		with tf.variable_scope('deconv{0}'.format(layer + 1)) as scope:
			in_layer = s_outputs[layer]
			feat_in = in_layer.get_shape().as_list()[-1]
			shape = [dc * 2, dc * 2, FLAGS.num_classes, feat_in]
			w = get_deconv_filter(shape)
			b = _variable_on_cpu('biases', [FLAGS.num_classes],
								 tf.constant_initializer(0.1))
			deconv = _deconv_layer(in_layer, w, b, dc, ds, scope)
			s_outputs[layer] = deconv
		dc *= 2

	s_fuse = tf.concat(s_outputs,axis=-1)
	with tf.variable_scope('after_fusion-1') as scope:
		s_fuse = tf.layers.batch_normalization(s_fuse)
		s_fuse = tf.nn.relu(s_fuse)
		s_fuse = tf.layers.conv2d(s_fuse, 2, (3, 3), padding='same', name=scope.name)
	with tf.variable_scope('after_fusion-2') as scope:
		s_fuse = tf.layers.batch_normalization(s_fuse)
		s_fuse = tf.nn.relu(s_fuse)
		s_fuse = tf.layers.conv2d(s_fuse, 2, (1, 1), padding='same', name=scope.name)
	return s_fuse, encoding

def _add_loss_summaries(total_loss):
	"""Add summaries for losses in BBBC006 model.
	Generates moving average for all losses and associated summaries for
	visualizing the performance of the network.

	Args:
		total_loss: Total loss from loss().
	Returns:
		loss_averages_op: op for generating moving averages of losses.
	"""
	# Compute the moving average of all individual losses and the total loss.
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	# Attach a scalar summary to all individual losses and the total loss; do the
	# same for the averaged version of the losses.
	for l in losses + [total_loss]:
		# Name each loss as '(raw)' and name the moving average version of the loss
		# as the original loss name.
		tf.summary.scalar(l.op.name + '_raw', l)
		tf.summary.scalar(l.op.name, loss_averages.average(l))

	return loss_averages_op


def train(total_loss, global_step):
	"""Train BBBC006 model.
	Create an optimizer and apply to all trainable variables. Add moving
	average for all trainable variables.

	Args:
		total_loss: Total loss from loss().
		global_step: Integer Variable counting the number of training steps
		processed.
	Returns:
		train_op: op for training.
	"""
	# Variables that affect learning rate.
	num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
	# Decay the learning rate exponentially based on the number of steps.
	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
									global_step,
									decay_steps,
									LEARNING_RATE_DECAY_FACTOR,
									staircase=True)
	tf.summary.scalar('learning_rate', lr)

	# Generate moving averages of all losses and associated summaries.
	loss_averages_op = _add_loss_summaries(total_loss)

	# Compute gradients.
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.2)
		grads = opt.compute_gradients(total_loss, var_list=tf.trainable_variables())

	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	# Add histograms for trainable variables.
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)

	# Add histograms for gradients.
	# for grad, var in grads:
	#     if grad is not None:
	#         tf.summary.histogram(var.op.name + '/gradients', grad)

	# Track the moving averages of all trainable variables.
	variable_averages = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	return train_op


def get_show_preds(s_fuse):
	"""Compute and view logits.
	Args:
		s_fuse: Segments fuse layer.
	Returns:
		s_logits: Softmax applied to segments fuse layer.
	"""
	# Index 1 of fuse layers correspond to foreground, so discard index 0.
	_, s_logits = tf.split(tf.cast(tf.nn.softmax(s_fuse), tf.float32), 2, 3)

	tf.summary.image('s_logits', s_logits)
	return s_logits


def get_show_labels(labels):
	"""Get and view labels.
	Args:
		labels: Labels from distorted_inputs or inputs().
	Returns:
		s_labels: Segments labels.
	"""
	s_labels = labels
	s_labels = tf.cast(s_labels, tf.float32)

	tf.summary.image('s_labels', s_labels)
	return s_labels


def get_dice_coef(logits, labels):
	"""Compute dice coefficient.
	Args:
		logits: Softmax probability applied to fuse layers.
		labels: Correct annotations (0 or 1).
	Returns:
		Mean dice coefficient over full tensor.

	Source:
		https://github.com/zsdonghao/tensorlayer/blob/master/tensorlayer/cost.py#L125
	"""
	smooth = 1e-5
	inter = tf.reduce_sum(tf.multiply(logits, labels))
	l = tf.reduce_sum(logits)
	r = tf.reduce_sum(labels)
	return tf.reduce_mean((2.0 * inter + smooth) / (l + r + smooth))


def dice_op(s_fuse, labels):
	"""Compute mean dice coefficient for both contours and segments outputs.
	Args:
		s_fuse: Segments output map from inference().
		labels: Labels from distorted_inputs or inputs().
	Returns:
		s_dice: Float representing average dice coefficient for segments.
	"""
	s_logits = get_show_preds(s_fuse)
	s_labels = get_show_labels(labels)

	s_dice = get_dice_coef(s_logits, s_labels)

	return s_dice
