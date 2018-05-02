from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import logging
import tensorflow as tf

import mainutils
import data_input
read_input, reshaped_image = data_input.get_read_input('train')
sess = tf.Session()
i_path = read_input.i_path
s_path = read_input.s_path
print(sess.run([i_path,s_path]))