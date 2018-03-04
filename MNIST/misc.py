from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf


# define center loss
def center_loss_def(prelogits, center_alpha, labels, n_class):
	'''
	Arguments:
	    prelogits: [batch_size, deep_feature_dim] tensor
	    center_alpha: a scalar, determining center update rate (kind of like learning rate)
	    labels: [batch_size, 1] tensor, true labels
	    n_class: a scalar, number of class in our classification problem
	Returns:
	    center_loss: operation defined for center loss
	    centers: [n_class, deep_feature_dim] tensor, centers being tracked
	'''
	with tf.variable_scope('center_loss'):
		deep_feat_dim = prelogits.get_shape()[1]

		# define centers, non-trainable
		centers = tf.get_variable('centers', [n_class, deep_feat_dim], initializer=tf.constant_initializer(0.0), trainable=False)

		# define center loss
		chosen_c = tf.nn.embedding_lookup(centers, labels);
		center_loss = tf.nn.l2_loss(prelogits-chosen_c)

		# compute center update value, delta_c
		one_hot = tf.one_hot(labels, n_class, axis=-1) # shape=[batch_size, n_class]    
		delta_c = tf.matmul(one_hot, centers) - prelogits # shape=[batch, deep_feat_dim]
		# update centers    
		updated_centers_op = tf.scatter_add(centers, labels, -center_alpha*delta_c)
		with tf.control_dependencies([updated_centers_op]):
			centers = tf.identity(centers)

	return center_loss, centers
