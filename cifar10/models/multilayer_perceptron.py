from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def inference(images, num_classes=10, 
								prediction_fn=slim.softmax,
								scope='multilayer_perceptron',
								reuse=False):
	end_points = {}

	with tf.variable_scope(scope, [images, num_classes], regularizer=slim.l2_regularizer(0.0), reuse=reuse):
		net = slim.flatten(images)
		net = slim.fully_connected(net, 256, scope='fc1')
		net = slim.fully_connected(net, 256, scope='fc2')
		end_points['deep_feats'] = net

		logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc4')

		end_points['Logits'] = logits
		end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

	return logits, end_points