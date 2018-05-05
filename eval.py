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

"""Evaluation for BBBC006."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from PIL import Image
import numpy as np
import tensorflow as tf

import mainutils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/warwick_eval',
						   """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'train_eval',
						   """Either 'test' or 'train' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/warwick_train',
						   """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
							"""How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 100,
							"""Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
							"""Whether to run eval only once.""")


def eval_once(saver, dice_op, summary_writer, summary_op, s_fuse, images, labels, i_paths, encoding, sessid):
	"""Run Eval once.
	Args:
		saver: Saver.
		summary_writer: Summary writer.
		summary_op: Summary op.
	"""

	FLAGS.checkpoint_dir = '/tmp/warwick_train_'+str(sessid)
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("Model restored from file: %s" % ckpt.model_checkpoint_path)
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		else:
			print('No checkpoint file found')
			return

		# Start the queue runners.
		coord = tf.train.Coordinator()
		predictions = {}
		encodings = {}
		dice_scores = {}
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
												 start=True))

			num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
			avg_s_dice = 0
			step = 0
			while step < num_iter and not coord.should_stop():
				s_dice,i_path,s_fuse_out,encoded_image = sess.run([dice_op,i_paths,s_fuse,encoding])
				predictions[i_path[0]] = s_fuse_out
				encodings[i_path[0]] = encoded_image
				dice_scores[i_path[0]] = s_dice
				im0 = s_fuse_out[0,:,:,0]
				im1 = s_fuse_out[0,:,:,1]
				image = (im1>im0)*128
				im = Image.fromarray(image.astype(np.uint8))
				im.save('results/'+i_path[0].split('/')[2]+'.bmp')
				avg_s_dice += s_dice
				step += 1

			avg_s_dice /= step
			print('%s: s_dice avg = %.3f' % (datetime.now(), avg_s_dice))

			summary = tf.Summary()
			summary.ParseFromString(sess.run(summary_op))
			summary.value.add(tag='dice_s', simple_value=avg_s_dice)
			summary_writer.add_summary(summary, global_step)
		except Exception as e:  # pylint: disable=broad-except
			coord.request_stop(e)
		if FLAGS.eval_data == 'train_eval':
			np.save('train_eval_data_'+str(sessid)+'.npy',[predictions, encodings, dice_scores]) # otherwise don't save
		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=20)


def evaluate():
	"""Eval BBBC006 for a number of steps."""
	for sessid in range(4):
		FLAGS.eval_dir = '/tmp/warwick_eval_' + str(sessid)
		with tf.Graph().as_default() as g:
			# Get images and labels for BBBC006.
			images, labels, i_paths = mainutils.inputs(eval_data=FLAGS.eval_data, sessid=sessid)
			
			# Build a Graph that computes the logits predictions from the
			# inference model.
			# s_fuse = mainutils.inference(images, train=False)
			s_fuse, encoding = mainutils.inference_bottleneck(images, train=False)

			dice_op = mainutils.dice_op(s_fuse, labels)

			# Restore the moving average version of the learned variables for eval.
			variable_averages = tf.train.ExponentialMovingAverage(
				mainutils.MOVING_AVERAGE_DECAY)
			variables_to_restore = variable_averages.variables_to_restore()
			saver = tf.train.Saver(variables_to_restore)

			# Build the summary operation based on the TF collection of Summaries.
			summary_op = tf.summary.merge_all()
			summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
			s_fuse_softmax = tf.nn.softmax(s_fuse)

			while True:
				eval_once(saver, dice_op, summary_writer, summary_op, s_fuse_softmax, images, labels, i_paths, encoding, sessid)
				if FLAGS.run_once:
					break
				time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
	if tf.gfile.Exists(FLAGS.eval_dir):
		tf.gfile.DeleteRecursively(FLAGS.eval_dir)
	tf.gfile.MakeDirs(FLAGS.eval_dir)
	evaluate()


if __name__ == '__main__':
	tf.app.run()
