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

"""A binary to train BBBC006 using a single GPU."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import logging
import tensorflow as tf

import mainutils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/warwick_train',
						   """Directory where to write event logs """
						   """and checkpoint.""")
tf.app.flags.DEFINE_string('eval_data', 'train',
						   """Either 'test' or 'train'.""")
tf.app.flags.DEFINE_integer('max_steps', 4000,
							"""Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
							"""Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 200,
							"""How often to log results to the console.""")
tf.logging.set_verbosity(tf.logging.DEBUG)

def train(sessid):
	"""Train BBBC006 for a number of steps."""
	with tf.Graph().as_default():
		ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
		global_step_init = -1
		global_step = tf.contrib.framework.get_or_create_global_step()
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			global_step_init = int(ckpt.model_checkpoint_path.split('/')[-1]
								   .split('-')[-1])

		# Get images and labels for BBBC006.
		# Force input pipeline to CPU:0 to avoid operations sometimes ending up
		# on GPU and resulting in a slow down.
		images, labels, i_paths = mainutils.inputs(eval_data=FLAGS.eval_data,sessid=sessid)
		# Build a Graph that computes the logits predictions from the
		# inference model.
		# s_fuse = mainutils.inference(images)
		s_fuse, encoding = mainutils.inference_bottleneck(images)
		mainutils.get_show_preds(s_fuse)

		# # Code for finding out the total number of trainable parameters
		# total_parameters = 0
		# for variable in tf.trainable_variables():
		# 	# shape is an array of tf.Dimension
		# 	shape = variable.get_shape()
		# 	print(shape)
		# 	print(len(shape))
		# 	variable_parameters = 1
		# 	for dim in shape:
		# 		print(dim)
		# 	variable_parameters *= dim.value
		# 	print(variable_parameters)
		# 	total_parameters += variable_parameters
		# print(total_parameters)
		# exit(0)

		# Calculate loss.
		segments_labels = labels
		with tf.variable_scope('{}_cross_entropy'.format('s')) as scope:
			class_prop = mainutils.S_CLASS_PROP
			weight_per_label = tf.scalar_mul(class_prop, tf.cast(tf.equal(segments_labels, 0),
																 tf.float32)) + \
							   tf.scalar_mul(1.0 - class_prop, tf.cast(tf.equal(segments_labels, 1),
																	   tf.float32))
			cross_entropy = tf.losses.sparse_softmax_cross_entropy(
				labels=tf.squeeze(segments_labels, squeeze_dims=[3]), logits=s_fuse)
			cross_entropy_weighted = tf.multiply(weight_per_label, cross_entropy)
			cross_entropy_mean = tf.reduce_mean(cross_entropy_weighted, name=scope.name)

		loss = cross_entropy_mean

		# Build a Graph that trains the model with one batch of examples and
		# updates the model parameters.
		train_op = mainutils.train(loss, global_step)

		class _LoggerHook(tf.train.SessionRunHook):
			"""Logs loss and runtime."""

			def begin(self):
				self._step = global_step_init
				self._start_time = time.time()

			def before_run(self, run_context):
				self._step += 1
				return tf.train.SessionRunArgs(loss)  # Asks for loss value.

			def after_run(self, run_context, run_values):
				if self._step % FLAGS.log_frequency == 0:
					current_time = time.time()
					duration = current_time - self._start_time
					self._start_time = current_time

					loss_value = run_values.results
					examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
					sec_per_batch = float(duration / FLAGS.log_frequency)

					format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
								  'sec/batch)')
					print(format_str % (datetime.now(), self._step, loss_value,
										examples_per_sec, sec_per_batch))

		saver = tf.train.Saver()
		with tf.train.MonitoredTrainingSession(
				checkpoint_dir=FLAGS.train_dir,
				hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
					   tf.train.NanTensorHook(loss),
					   _LoggerHook()],
				config=tf.ConfigProto(
					log_device_placement=FLAGS.log_device_placement)
				) as mon_sess:
			ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

			if ckpt:
				saver.restore(mon_sess, ckpt.model_checkpoint_path)
				logging.info("Model restored from file: %s" % ckpt.model_checkpoint_path)
			while not mon_sess.should_stop():
				_,losseval = mon_sess.run([train_op,loss])

def main(argv=None):  # pylint: disable=unused-argument
	FLAGS.train_dir = '/tmp/warwick_train_0'
	train(0)
	FLAGS.train_dir = '/tmp/warwick_train_1'
	train(1)
	FLAGS.train_dir = '/tmp/warwick_train_2'
	train(2)
	FLAGS.train_dir = '/tmp/warwick_train_3'
	train(3)

if __name__ == '__main__':
	tf.app.run()
