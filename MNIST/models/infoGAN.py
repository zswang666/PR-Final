from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.ops import *

def discriminator(image, batch_size, scope, reuse=False):
	endpoints = {}

	with tf.variable_scope(scope, reuse=reuse):
		# d_bn1 = batch_norm(name='d_bn1')
		d_bn2 = batch_norm(name='d_bn2')

		# image shape=[batch_size, 28, 28, 1]

		# conv 1, lrelu, shape=[batch_size, 14, 14, 64]
		h0 = conv2d(image, 64, name='d_h0_conv')
		h0 = lrelu(h0)

		# conv 2, bn-lrelu, shape=[batch_size, 7, 7, 128]
		h1 = conv2d(h0, 128, name='d_h1_conv')
		h1 = lrelu(d_bn2(h1))

		# flatten
		h2 = tf.reshape(h1, [batch_size, -1]) # shape=[batch_size, 7*7*128]
		endpoints['flatten1'] = h2

		# FC, for discriminate fake or real
		h2 = linear(endpoints['flatten1'], 1, 'd_h2_lin') # shape=[batch_size, 1]
		endpoints['logits_fake_or_real'] = h2
		h2 = tf.nn.sigmoid(h2)
		endpoints['scores_fake_or_real'] = h2

		### FORK
		# FC 128, bn-lrelu
		h2_2 = linear(endpoints['flatten1'], 128, 'd_h2_2_lin')
		h2_2 = lrelu(h2_2)

		# FC, output for discrete c
		h3_2 = linear(h2_2, 10, 'd_h3_2_lin')
		endpoints['logits_Q_disc'] = h3_2
		h3_2 = tf.nn.softmax(h3_2)
		endpoints['scores_Q_disc'] = h3_2

		# flatten, FC, output for continuous c
		h3_3 = linear(h2_2, 2, 'd_h3_3_lin')
		endpoints['logits_Q_cont'] = h3_3
		h3_3 = tf.nn.softmax(h3_3)
		endpoints['scores_Q_cont'] = h3_3

		#################### EXTRA ####################
		# # flatten, FC, bn-lrelu, shape=[batch_size, 1024]
		# h2 = tf.reshape(h1, [batch_size, -1])
		# h2 = linear(h2, 1024, 'd_h2_lin')
		# h2 = lrelu(d_bn3(h2))

		# # output for discriminate fake or real
		# h3 = linear(h2, 1, 'd_h3_lin')
		# endpoints['logits_fake_or_real'] = h3
		# h3 = tf.nn.sigmoid(h3)
		# endpoints['scores_fake_or_real'] = h3

		## FORK
		# # FC 128, bn-lrelu 
		# h3_2 = linear(h2, 128, 'd_h3_2_lin')
		# h3_2 = lrelu(d_bn4(h3_2))

		# # output for classification
		# h4_2 = linear(h3_2, 10, 'd_h4_2_lin')
		# endpoints['logits_which_num'] = h4_2
		# h4_2 = tf.nn.softmax(h4_2)
		# endpoints['scores_which_num'] = h4_2

	return endpoints

# inverse of LeNet
def generator(z, output_size, batch_size, scope, reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		g_bn0 = batch_norm(name='g_bn0')
		g_bn1 = batch_norm(name='g_bn1')
		g_bn2 = batch_norm(name='g_bn2')

		# FC, bn-relu
		z_ = linear(z, 128*7*7, 'g_h0_lin')
		h0 = tf.reshape(z_, [-1, 7, 7, 128])
		h0 = tf.nn.relu(g_bn0(h0))

		# deconv 1, bn-relu
		h1 = deconv2d(h0, [batch_size, 14, 14, 64], name='g_h1')
		h1 = tf.nn.relu(g_bn1(h1))

		# deconv 2, bn-tanh
		h2 = deconv2d(h1, [batch_size, 28, 28, 1], name='g_h2')
		h2 = tf.nn.tanh(g_bn2(h2))

	return h2

def latent_sampler(dim, batch_size):
	z_mean = 0
	z_std = 1
	z = tf.random_normal([batch_size, dim], 
						  z_mean,
						  z_std,
						  dtype=tf.float32,
						  name='latent_sampler')

	c_disc = tf.random_uniform([batch_size],
							   minval=1,
							   maxval=10,
							   dtype=tf.int32,
							   name='discrete_c_sampler')
	c_disc = tf.one_hot(c_disc, 10, axis=-1)
	# c_disc = tf.reshape(c_disc, [batch_size, ])

	c_cont = tf.random_uniform([batch_size, 2],
							   minval=-1,
							   maxval=1,
							   dtype=tf.float32,
							   name='continuous_c_sampler')

	return z, c_disc, c_cont