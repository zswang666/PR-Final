from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.ops import *

def discriminator(image, batch_size, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        df_dim = 64 # Dimension of discrim filters in first conv layer

        d_bn1 = batch_norm(name='d_bn1')
        d_bn2 = batch_norm(name='d_bn2')
        d_bn3 = batch_norm(name='d_bn3')

        h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
        h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv')))
        h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv')))
        h3 = lrelu(d_bn3(conv2d(h2, df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h3_lin')

    return tf.nn.sigmoid(h4), h4

def generator(z, output_size, batch_size, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        gf_dim = 64 # Dimension of gen filters in first conv layer
        c_dim = 3 # depth of images

        g_bn0 = batch_norm(name='g_bn0')
        g_bn1 = batch_norm(name='g_bn1')
        g_bn2 = batch_norm(name='g_bn2')
        g_bn3 = batch_norm(name='g_bn3')

        s = output_size
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        # project `z` and reshape
        z_, h0_w, h0_b = linear(z, gf_dim*8*s16*s16, 'g_h0_lin', with_w=True)

        h0 = tf.reshape(z_, [-1, s16, s16, gf_dim * 8])
        h0 = tf.nn.relu(g_bn0(h0))

        h1, h1_w, h1_b = deconv2d(h0, [batch_size, s8, s8, gf_dim*4],
                                  name='g_h1', with_w=True)
        h1 = tf.nn.relu(g_bn1(h1))

        h2, h2_w, h2_b = deconv2d(h1, [batch_size, s4, s4, gf_dim*2],
                                  name='g_h2', with_w=True)
        h2 = tf.nn.relu(g_bn2(h2))

        h3, h3_w, h3_b = deconv2d(h2, [batch_size, s2, s2, gf_dim*1],
                                  name='g_h3', with_w=True)
        h3 = tf.nn.relu(g_bn3(h3))

        h4, h4_w, h4_b = deconv2d(h3, [batch_size, s, s, c_dim],
                                  name='g_h4', with_w=True)

    return tf.nn.tanh(h4)

def discriminator2(image, batch_size, scope, reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		# d_bn1 = batch_norm(name='d_bn1')
		d_bn2 = batch_norm(name='d_bn2')

		# conv 1, lrelu
		h0 = conv2d(image, 64, name='d_h0_conv')
		h0 = lrelu(h0)

		# conv 2, bn-lrelu
		h1 = conv2d(h0, 128, name='d_h1_conv')
		h1 = lrelu(d_bn2(h1))

		# flatten, FC
		h2 = tf.reshape(h1, [batch_size, -1])
		h2 = linear(h2, 1, 'd_h2_lin')

	return tf.nn.sigmoid(h2), h2


def generator2(z, output_size, batch_size, scope, reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		g_bn0 = batch_norm(name='g_bn0')
		g_bn1 = batch_norm(name='g_bn1')
		g_bn2 = batch_norm(name='g_bn2')

		# FC, bn-relu and reshape
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

	return z
