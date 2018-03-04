from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.ops import *

slim = tf.contrib.slim

def discriminator0(image, scope, reuse=False):
	endpoints = {}
	with tf.variable_scope(scope, reuse=reuse):	
		net = slim.conv2d(image, 32, [5,5], stride=2, padding='SAME', activation_fn=None, scope='conv1')
		net = lrelu(net)
		net = slim.conv2d(net, 64, [5,5], stride=2, padding='SAME', activation_fn=None, scope='conv2')
		net = lrelu(net)
		net = slim.conv2d(net, 128, [5,5], stride=2, padding='SAME', activation_fn=None, scope='conv3')
		net = lrelu(net)
		net = slim.flatten(net)
		net = slim.fully_connected(net, 256, scope='fc_shared')
		endpoints['fc_shared'] = net

		net = slim.fully_connected(endpoints['fc_shared'], 50, activation_fn=None, scope='fc_recon')
		net = tf.nn.sigmoid(net)
		endpoints['fc_recon'] = net

		net = slim.fully_connected(endpoints['fc_shared'], 1, activation_fn=None, scope='fc_adv')
		#net = linear(endpoints['fc_shared'], 1, 'fc_adv')
		endpoints['fc_adv_logits'] = net
		net = tf.nn.sigmoid(net)
		endpoints['fc_adv'] = net

	return endpoints['fc_adv_logits'], endpoints['fc_recon'], endpoints['fc_shared']

def discriminator1(D1_in, scope, reuse=False):
	endpoints = {}
	with tf.variable_scope(scope, reuse=reuse):
		# make sure D1_in is output of FC256
		print('D1 in {}'.format(D1_in.get_shape()[:]))
		net = slim.fully_connected(D1_in, 256, scope='fc1')
		net = slim.fully_connected(net, 256, scope='fc2')
		endpoints['fc_shared'] = net

		net = slim.fully_connected(endpoints['fc_shared'], 50, activation_fn=None, scope='fc_recon')
		net = tf.nn.sigmoid(net)
		endpoints['fc_recon'] = net

		net = slim.fully_connected(endpoints['fc_shared'], 1, activation_fn=None, scope='fc_adv')
		#net = linear(endpoints['fc_shared'], 1, 'fc_adv')
		endpoints['fc_adv_logits'] = net
		net = tf.nn.sigmoid(net)
		endpoints['fc_adv'] = net
		
	return endpoints['fc_adv_logits'], endpoints['fc_recon'], endpoints['fc_shared']


def generator0(G0_in, z, scope, reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		bn2 = batch_norm(name='bn2')
		bn3 = batch_norm(name='bn3')
		bn4 = batch_norm(name='bn4')
		bn5 = batch_norm(name='bn5')

		net = tf.concat(1, [G0_in, z])
		net = slim.fully_connected(net, 4*4*128, scope='fc1')
		net = tf.reshape(net, [-1,4,4,128])
		print(net.get_shape()[:])
		net = slim.conv2d_transpose(net, 128, [5,5], 2, padding='SAME', scope='deconv2')
		net = bn2(net)
		print(net.get_shape()[:])
		net = slim.conv2d_transpose(net, 64, [5,5], 1, padding='VALID', scope='deconv3')
		net = bn3(net)
		print(net.get_shape()[:])
		net = slim.conv2d_transpose(net, 64, [5,5], 2, padding='SAME', scope='deconv4')
		net = bn4(net)
		print(net.get_shape()[:])
		net = slim.conv2d_transpose(net, 1, [5,5], 1, padding='VALID', scope='deconv5')
		net = bn5(net)
		net = tf.nn.tanh(net)
		# make sure this output is 28x28

	return net

def generator1(y, z, scope, reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		bn1 = batch_norm(name='bn1')
		bn2 = batch_norm(name='bn2')

		net = tf.concat(1, [y, z])
		net = slim.fully_connected(net, 512, scope='fc1')
		net = bn1(net)
		net = slim.fully_connected(net, 512,  scope='fc2')
		net = bn2(net)
		net = slim.fully_connected(net, 256, scope='fc3')
		
	return net

def E0(reuse_scope, image):
	with tf.variable_scope(reuse_scope, reuse=True):
		net = slim.conv2d(image, 32, [5,5], padding='VALID', scope='conv1')
		net = slim.max_pool2d(net, [2,2], 2, scope='pool1')
		net = slim.conv2d(net, 32, [5,5], padding='VALID', scope='conv2')
		net = slim.max_pool2d(net, [2,2], 2, scope='pool2')
		net = slim.flatten(net)
		net = slim.fully_connected(net, 256, scope='fc3')
	
	return net
	
def E1(reuse_scope, fc3):
	with tf.variable_scope(reuse_scope, reuse=True):
		logits = slim.fully_connected(fc3, 10, activation_fn=None, scope='fc4')
		#pred = slim.softmax(logits, scope='predictions')
	
	return logits

def encoder(image, scope, reuse=False, n_classes=10,
			dropout_keep_prob=0.5, is_training=False):
	endpoints = {}
	with tf.variable_scope(scope, regularizer=slim.l2_regularizer(0.0), reuse=reuse):
		net = slim.conv2d(image, 32, [5,5], padding='VALID', scope='conv1')
		net = slim.max_pool2d(net, [2,2], 2, scope='pool1')
		net = slim.conv2d(net, 32, [5,5], padding='VALID', scope='conv2')
		net = slim.max_pool2d(net, [2,2], 2, scope='pool2')
		net = slim.flatten(net)
		net = slim.fully_connected(net, 256, scope='fc3')
		endpoints['fc3'] = net
		net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout3')

		logits = slim.fully_connected(net, n_classes, activation_fn=None, scope='fc4')
		endpoints['logits'] = logits
		pred = slim.softmax(logits, scope='predictions')

	return logits, endpoints	

def z0_sampler(dim, batch_size):
	z_mean = 0
	z_std = 1
	z = tf.random_normal([batch_size, dim], 
						  z_mean,
						  z_std,
						  dtype=tf.float32,
						  name='latent_sampler')

	return z

def z1_sampler(dim, batch_size):
	z_mean = 0
	z_std = 1
	z = tf.random_normal([batch_size, dim], 
						  z_mean,
						  z_std,
						  dtype=tf.float32,
						  name='latent_sampler')

	return z

def z2_sampler(dim, batch_size):
	z_mean = 0
	z_std = 1
	z = tf.random_normal([batch_size, dim], 
						  z_mean,
						  z_std,
						  dtype=tf.float32,
						  name='latent_sampler')

	return z
