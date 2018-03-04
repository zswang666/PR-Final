from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def vgg16_finetune(images, scope='vgg16_finetune', n_classes=365,
                    is_training=True, dropout_keep_prob=0.5, reuse=False):
    # Encoder is a fine-tuned 10-class VGG-16 model
    #with tf.variable_scope(scope, reuse=reuse):
    endpoints = {}
    #net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.conv2d(images, 64, [3, 3], scope='conv1_1')#
    net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')#
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    #net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')#
    net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')#
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    #net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')#
    net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')#
    net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')#
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')#
    net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')#
    net = slim.conv2d(net, 512, [3, 3], scope='conv4_3')#
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')#
    net = slim.conv2d(net, 512, [3, 3], scope='conv5_2')#
    net = slim.conv2d(net, 512, [3, 3], scope='conv5_3')#
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    # Use conv2d instead of fully_connected layers.
    #net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
    net = slim.flatten(net)
    net = slim.fully_connected(net, 4096, scope='fc6')#
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope='dropout6')
    #net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
    net = slim.fully_connected(net, 4096, scope='fc7')#
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope='dropout7')
    #net = slim.conv2d(net, n_classes, [1, 1],
    #                  activation_fn=None,
    #                  normalizer_fn=None,
    #                  scope='fc8a')
    net = slim.fully_connected(net, n_classes, activation_fn=None, scope='fc8a' )
    #net = tf.squeeze(net)
    endpoints['logits'] = net
    net = slim.softmax(net, scope='predictions')
    #net = tf.squeeze(net) 
    endpoints['predictions'] = net

    return endpoints

