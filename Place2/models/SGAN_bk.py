from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.ops import *

slim = tf.contrib.slim

# need to be changed
def encoder(images, scope='E', n_classes=10,
		    is_training=True, dropout_keep_prob=0.5, reuse=False):
	# Encoder is a fine-tuned 10-class VGG-16 model
	with tf.variable_scope(scope, reuse=reuse):
		endpoints = {}
		print(images.get_shape())
		net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
		print(net.get_shape())
		net = slim.max_pool2d(net, [2, 2], scope='pool1')
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		net = slim.max_pool2d(net, [2, 2], scope='pool2')
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
		net = slim.max_pool2d(net, [2, 2], scope='pool3')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
		net = slim.max_pool2d(net, [2, 2], scope='pool4')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
		net = slim.max_pool2d(net, [2, 2], scope='pool5')
		# Use conv2d instead of fully_connected layers.
		net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
		net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
						   scope='dropout6')
		net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
		net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
						   scope='dropout7')
		net = slim.conv2d(net, n_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
		endpoints['logits'] = net
		net = slim.softmax(net, scope='predictions')
		net = tf.squeeze(net) 
		endpoints['predictions'] = net

	return endpoints

def discriminator0(image, batch_size, scope, reuse=False):
	raise NotImplementedError

def discriminator1():
	raise NotImplementedError

def discriminator2():
	raise NotImplementedError

def generator0():
	raise NotImplementedError

def generator1():
	raise NotImplementedError

def generator2():
	raise NotImplementedError

def z_sampler(dim, batch_size):
	z_mean = 0
	z_std = 1
	z = tf.random_normal([batch_size, dim], 
						  z_mean,
						  z_std,
						  dtype=tf.float32,
						  name='latent_sampler')

	return z
