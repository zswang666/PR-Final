from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf

import mnist_input


# testing
dataset = '~/Desktop/test_on_mnist/mnist_data/mnist_test.tfrecords'
image_size = 28
batch_size = 100
n_epochs = 2
n_preprocessing_thread = 2
random_flip = False

summary_log_dir = '/tmp/tf_log'

# define graph
images, labels = mnist_input._input(dataset, image_size, batch_size, n_epochs, n_preprocessing_thread, random_flip)

merge_summary_op = tf.merge_all_summaries()

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

summary_writer = tf.train.SummaryWriter(summary_log_dir, graph=tf.get_default_graph())

# session run
sess = tf.Session()
with sess.as_default():
	sess.run(init_op)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	try:
		i = 0
		while not coord.should_stop():
			process = [images, labels]
			_, labels_ = sess.run(process)

			if (i%100==0):
				summary = sess.run(merge_summary_op)
				summary_writer.add_summary(summary)
                        
			i = i + 1
            
			if i==3:
			    print(len(labels_))
            
	except tf.errors.OutOfRangeError:
		print('epoch limit reached')
	finally:
		# When done, ask the threads to stop.
		coord.request_stop()

	coord.join(threads)
	sess.close()