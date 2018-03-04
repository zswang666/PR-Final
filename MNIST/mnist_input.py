from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf

def parse_example_proto(serialized_example):
	'''
	Parse one example protobuf into encoded image and corresponding label
	'''
	features = tf.parse_single_example(
		serialized_example,
		# Defaults are not specified since both keys are required.
		features={
			'image_raw': tf.FixedLenFeature([], tf.string),
			'label': tf.FixedLenFeature([], tf.int64),
		})
	
	label = features['label']
	# label = tf.one_hot(label,10)

	return features['image_raw'], label

def image_preprocessing(image_buffer, image_size, random_flip):
	'''
	Decode images and perform preprocessing
	'''
	# decoding
	image = tf.decode_raw(image_buffer, tf.uint8)
	image = tf.cast(image, tf.float32)/127.5 - 1 # normalized to -1, +1
	image = tf.reshape(image, (28, 28, 1))
    
    # preprocessing
	image = tf.image.resize_images(image, [image_size, image_size])
    # image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
    
	if random_flip:
		image = tf.image.random_flip_left_right(image)
    
	# image = tf.image.per_image_whitening(image)

	return image

def read_and_decode(filename_queue, image_size, n_preprocessing_thread, random_flip):
	# only using 1 reader
	# define reader for the dataset (.tfrecord)
	reader = tf.TFRecordReader()
	# read files dequeued from filename_queue
	_, serialized_example = reader.read(filename_queue)
    
	# parse and preprocess files with multiple threads
	images_and_labels = []
	for thread_id in range(n_preprocessing_thread):
		# parse example protobuf. Image_buffer is not decoded yet
		image_buffer, label = parse_example_proto(serialized_example)
		# image preprocessing
		image = image_preprocessing(image_buffer, image_size, random_flip)

		images_and_labels.append([image, label])

	return images_and_labels

def _input(dataset, image_size, batch_size, n_epochs, n_preprocessing_thread, random_flip):
	'''
	read dataset n_epochs times
	Arguments:
	    dataset: a .tfrecord file containing data
	    image_size: size of an "square" image
	    batch_size: a scalar specifying batch size
	    n_epochs: number of epoch
	    n_preprocess_threads: integer, total number of preprocessing threads
	    random_flip: a bool, determine whether perform image flipping in preprocessing
	Returns:
	    image batch
	    labels batch
	'''
	with tf.name_scope('input'):
		# construct filename queue
		dataset = os.path.expanduser(dataset)
		filename_queue = tf.train.string_input_producer([dataset], num_epochs=n_epochs)

		# read and decode (and preprocessing) files to form a example queue
		images_and_labels = read_and_decode(filename_queue, image_size, n_preprocessing_thread, random_flip)

		# form batch from dequeueing example queue
		# min_after_dequeue is minimum number of example to be maintained in the example queue
		min_after_dequeue = 0
		image_batch, label_batch = tf.train.batch_join(
			images_and_labels, 
			batch_size=batch_size,
			capacity=2 * n_preprocessing_thread * batch_size)
        
	#	tf.image_summary('image_batch', image_batch, max_images=10)
        
	return image_batch, label_batch
