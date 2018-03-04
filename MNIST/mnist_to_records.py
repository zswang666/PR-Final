from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import cv2

from tensorflow.contrib.learn.python.learn.datasets import mnist

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(dataset, name, directory):
	# specify and check arguments
	images = dataset.images
	labels = dataset.labels
	n_examples = dataset.num_examples
	if n_examples != len(labels):
		raise ValueError('Image size %d does not match label size %d' %(n_examples, len(labels)))
        
	# define output filename
	filename = os.path.join(os.path.expanduser(directory), name+'.tfrecords')

	# start writing file
	print('writing', filename)
	writer = tf.python_io.TFRecordWriter(filename)
	for i in range(5000):
		image_raw = images[i].tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
						'label': _int64_feature(int(labels[i])),
						'image_raw': _bytes_feature(image_raw)}))
		writer.write(example.SerializeToString())
	writer.close()

mnist_data_dir = os.path.expanduser('~/Desktop/test_on_mnist/mnist_data')
data_sets = mnist.read_data_sets(mnist_data_dir,
								 dtype=tf.uint8,
								 reshape=False,
								 validation_size=5000)
mnist_tfrecord_dir = '~/Desktop/test_on_mnist/mnist_data'
# convert_to(data_sets.train, 'mnist_train', mnist_tfrecord_dir)
# convert_to(data_sets.validation, 'mnist_valid', mnist_tfrecord_dir)
# convert_to(data_sets.test, 'mnist_test', mnist_tfrecord_dir)
convert_to(data_sets.train, 'mnist_GAN', mnist_tfrecord_dir)
