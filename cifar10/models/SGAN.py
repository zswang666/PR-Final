from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function
import pdb
import tensorflow as tf
from models.ops import *

slim = tf.contrib.slim

def discriminator0(image, scope, reuse=False):
	endpoints = {}
	bn1_conv = batch_norm(name='bn1_conv')
	bn2_conv = batch_norm(name='bn2_conv')
	print('D0')
	with tf.variable_scope(scope, reuse=reuse):	
		print('in:{}'.format(image.get_shape()))
		net = slim.conv2d(image, 96, [5,5], stride=2, padding='SAME', activation_fn=None, scope='conv1')
		net = lrelu(net)
		print(net.get_shape())
		net = slim.conv2d(net, 192, [5,5], stride=2, padding='SAME', activation_fn=None, scope='conv2')
		net = bn1_conv(net)
		net = lrelu(net)
		print(net.get_shape())
		net = slim.conv2d(net, 192, [5,5], stride=2, padding='SAME', activation_fn=None, scope='conv3')
		net = bn2_conv(net)
		net = lrelu(net)
		print(net.get_shape())
		net = slim.flatten(net)
		net = slim.fully_connected(net, 256, scope='fc_shared')
		endpoints['fc_shared'] = net
		print('fc_shared:{}'.format(net.get_shape()))

		net = slim.fully_connected(endpoints['fc_shared'], 50, activation_fn=None, scope='fc_recon')
		net = tf.nn.sigmoid(net)
		endpoints['fc_recon'] = net
		print('fc_recon:{}'.format(net.get_shape()))

		net = slim.fully_connected(endpoints['fc_shared'], 1, activation_fn=None, scope='fc_adv')
		#net = linear(endpoints['fc_shared'], 1, 'fc_adv')
		endpoints['fc_adv_logits'] = net
		net = tf.nn.sigmoid(net)
		endpoints['fc_adv'] = net
		print('fc_adv:{}'.format(net.get_shape()))

	return endpoints['fc_adv_logits'], endpoints['fc_recon'], endpoints['fc_shared']

def discriminator1(D1_in, scope, reuse=False):
	endpoints = {}
	bn1_ = batch_norm(name='bn1_')
	print('D1:')
	with tf.variable_scope(scope, reuse=reuse):
		# make sure D1_in is output of FC256
		print('in:{}'.format(D1_in.get_shape()))
		net = slim.fully_connected(D1_in, 512, activation_fn=None, scope='fc1')
		net = lrelu(net)
		print(net.get_shape())
		net = slim.fully_connected(net, 512, activation_fn=None, scope='fc2')
		net = bn1_(net)
		net = lrelu(net)
		endpoints['fc_shared'] = net
		print('fc_shared:{}'.format(net.get_shape()))

		net = slim.fully_connected(endpoints['fc_shared'], 50, activation_fn=None, scope='fc_recon')
		net = tf.nn.sigmoid(net)
		endpoints['fc_recon'] = net
		print('fc_recon:{}'.format(net.get_shape()))
		
		net = slim.fully_connected(endpoints['fc_shared'], 1, activation_fn=None, scope='fc_adv')
		#net = linear(endpoints['fc_shared'], 1, 'fc_adv')
		endpoints['fc_adv_logits'] = net
		net = tf.nn.sigmoid(net)
		endpoints['fc_adv'] = net
		print('fc_adv:{}'.format(net.get_shape()))
		
	return endpoints['fc_adv_logits'], endpoints['fc_recon'], endpoints['fc_shared']

def generator0(G0_in, z, scope, reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		bn1 = batch_norm(name='bn1')
		bn2 = batch_norm(name='bn2')
		bn3 = batch_norm(name='bn3')
		bn4 = batch_norm(name='bn4')
		#bn5 = batch_norm(name='bn5')
		bn1_z = batch_norm(name='bn1_z')		
		bn2_z = batch_norm(name='bn2_z')
		
		print('G0:')
		
		print('z_emb')
		print('in:{}'.format(z.get_shape()))
		z_emb = slim.fully_connected(z, 128, activation_fn=None, scope='z_emb_fc1')
		z_emb = bn1_z(z_emb)
		z_emb = tf.nn.relu(z_emb)
		print(z_emb.get_shape())
		z_emb = slim.fully_connected(z_emb, 128, activation_fn=None, scope='z_emb_fc2')
		z_emb = bn2_z(z_emb)
		z_emb = tf.nn.relu(z_emb)
		print(z_emb.get_shape())
		
		net = tf.concat(1, [G0_in, z_emb])
		print('concat:{}'.format(net))
		net = slim.fully_connected(net, 5*5*256, activation_fn=None, scope='fc1')
		net = bn1(net)
		net = tf.nn.relu(net)
		print(net.get_shape())
		net = tf.reshape(net, [-1,5,5,256])
		print('reshape:{}'.format(net.get_shape()))
		net = slim.conv2d_transpose(net, 256, [5,5], 2, padding='SAME', activation_fn=None, scope='deconv2')
		net = bn2(net)
		net = tf.nn.relu(net)
		print(net.get_shape())
		net = slim.conv2d_transpose(net, 128, [5,5], 1, padding='VALID', activation_fn=None, scope='deconv3')
		net = bn3(net)
		net = tf.nn.relu(net)
		print(net.get_shape())
		net = slim.conv2d_transpose(net, 128, [5,5], 2, padding='SAME', activation_fn=None, scope='deconv4')
		net = bn4(net)
		net = tf.nn.relu(net)
		print(net.get_shape())
		net = slim.conv2d_transpose(net, 3, [5,5], 1, padding='VALID', activation_fn=None, scope='deconv5')
		#net = bn5(net)
		net = tf.nn.tanh(net)
		print(net.get_shape())
		# make sure this output is 32x32
	return net

def generator1(y, z, scope, reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		bn1 = batch_norm(name='bn1')
		bn2 = batch_norm(name='bn2')
		bn3 = batch_norm(name='bn3')
		bn4 = batch_norm(name='bn4')

		print('G1:')
		net = tf.concat(1, [y, z])
		print('concat:{}'.format(net.get_shape()))
		#net = slim.fully_connected(net, 256, scope='fc1')
		#net = bn1(net)
		#print(net.get_shape())
		#net = slim.fully_connected(net, 512, scope='fc2')
		#net = bn2(net)
		#print(net.get_shape())
		net = slim.fully_connected(net, 512, activation_fn=None, scope='fc3')
		net = bn3(net)
		net = tf.nn.relu(net)
		print(net.get_shape())
		net = slim.fully_connected(net, 512, activation_fn=None,  scope='fc4')
		net = bn4(net)
		net = tf.nn.relu(net)
		print(net.get_shape())
		net = slim.fully_connected(net, 256, scope='fc5')
		print(net.get_shape())		

	return net

#def generator2(y, z, scope, reuse= False):
	

def E0(reuse_scope, image):
	print('E0:')
	with tf.variable_scope(reuse_scope, reuse=True):
		print('in:{}'.format(image.get_shape()))
		net = slim.conv2d(image, 64, [5,5], padding='VALID', scope='conv1')
		net = slim.max_pool2d(net, [2,2], 2, scope='pool1')
		print(net.get_shape())
		net = slim.conv2d(net, 128, [5,5], padding='VALID', scope='conv2')
		net = slim.max_pool2d(net, [2,2], 2, scope='pool2')
		print(net.get_shape())
		net = slim.flatten(net)
		net = slim.fully_connected(net, 256, scope='fc3')
		print(net.get_shape())
	return net
	
def E1(reuse_scope, fc3):
	print('E1:')
	with tf.variable_scope(reuse_scope, reuse=True):
		print('in:{}'.format(fc3.get_shape()))
		logits = slim.fully_connected(fc3, 10, activation_fn=None, scope='fc4')
		#pred = slim.softmax(logits, scope='predictions')
		print(logits.get_shape())
	return logits

def encoder(image, scope, reuse=False, n_classes=10,
			dropout_keep_prob=0.5, is_training=False):
	endpoints = {}
	print('E:')
	with tf.variable_scope(scope, regularizer=slim.l2_regularizer(0.0), reuse=reuse):
		print('in:{}'.format(image.get_shape()))
		net = slim.conv2d(image, 64, [5,5], padding='VALID', scope='conv1')
		net = slim.max_pool2d(net, [2,2], 2, scope='pool1')
		print(net.get_shape())
		net = slim.conv2d(net, 128, [5,5], padding='VALID', scope='conv2')
		net = slim.max_pool2d(net, [2,2], 2, scope='pool2')
		print(net.get_shape())
		net = slim.flatten(net)
		net = slim.fully_connected(net, 256, scope='fc3')
		endpoints['fc3'] = net
		print(net.get_shape())
		net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout3')

		logits = slim.fully_connected(net, n_classes, activation_fn=None, scope='fc4')
		endpoints['logits'] = logits
		print(logits.get_shape())
		pred = slim.softmax(logits, scope='predictions')
		pdb.set_trace()

	return logits, endpoints	

def z0_sampler(dim, batch_size):
	z_mean = 0
	z_std = 1.
	z = tf.random_normal([batch_size, dim], 
						  z_mean,
						  z_std,
						  dtype=tf.float32,
						  name='latent_sampler0')

	return z

def z1_sampler(dim, batch_size):
	z_mean = 0
	z_std = 1.
	z = tf.random_normal([batch_size, dim], 
						  z_mean,
						  z_std,
						  dtype=tf.float32,
						  name='latent_sampler1')

	return z

def z2_sampler(dim, batch_size):
	z_mean = 0
	z_std = 1
	z = tf.random_normal([batch_size, dim], 
						  z_mean,
						  z_std,
						  dtype=tf.float32,
						  name='latent_sampler2')

	return z
