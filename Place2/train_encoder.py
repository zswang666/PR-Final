from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
from datetime import datetime
import tensorflow as tf
import argparse
import pickle

from data_loader import indoor_input as data_input
import utils
import models.encoder as encoder

IMAGE_SIZE = 224
N_CLASSES = 10

def parse_args(argv):
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument('--trainset_dir', type=str, 
                        help='Root directory of training dataset')
    parser.add_argument('--batch_size', type=int,
                        help='Size of a mini-batch')
    parser.add_argument('--n_epochs', type=int,
                        help='Number of epochs to be trained')
    parser.add_argument('--optimizer', type=str,
                        help='Optimizer used for training', default='ADAM')
    parser.add_argument('--base_lr', type=float,
                        help='Base learnging rate')
    parser.add_argument('--lr_decay_step', type=float,
                        help='Decay step of learning rate')
    parser.add_argument('--lr_decay_rate', type=float,
                        help='Decay rate of learning rate')
    # training auxiliary
    parser.add_argument('--pretrained_model', type=str,
                        help='Path to pretrained model', default='./vgg16_pretrained/vgg_16.ckpt')
    parser.add_argument('--checkpoint_step', type=int,
                        help='Interval of saving checkpoints')
    parser.add_argument('--checkpoint_save_path', type=str,
                        help='Checkpoint save path')
    parser.add_argument('--keep_checkpoint_num', type=int,
                        help='Maximum number of checkpoint files to be kept')
    parser.add_argument('--print_step', type=int,
                        help='Interval of printing out per-train-step messages', default=10)
    # evaluating parameters
    parser.add_argument('--evalset_dir', type=str,
                        help='Root directory of evaluation dataset')
    parser.add_argument('--eval_step', type=int,
                        help='Interval of evaluation')
    parser.add_argument('--eval_batch_size', type=int,
                        help='Batch size for evaluation')
    parser.add_argument('--eval_save_path', type=str,
                        help='Path to save evaluation results')
    # summaries
    parser.add_argument('--summary_step', type=int,
                        help='Interval of adding summaries')
    parser.add_argument('--summary_train_log_dir', type=str,
                        help='Directory to save summaries of training', default='./train_log')
    parser.add_argument('--summary_eval_log_dir', type=str,
                        help='Directory to save summaries of evaluation', default='./eval_log')
    # session
    parser.add_argument('--use_gpu', type=bool,
                        help='If True, use GPU, otherwise, don\'t use GPU', default=True)
    parser.add_argument('--gpu_fraction', type=float,
                        help='Fraction of GPU allowed to be used')

    return parser.parse_args(argv)

def train(args):
    # correct and validate path
    args.trainset_dir = utils.check_dir(args.trainset_dir)
    if args.pretrained_model: # there can be no pretrained_model
        args.pretrained_model = utils.check_path(args.pretrained_model)
    args.evalset_dir = utils.check_dir(args.evalset_dir)
    args.summary_train_log_dir = utils.check_dir(args.summary_train_log_dir)
    args.summary_eval_log_dir = utils.check_dir(args.summary_eval_log_dir)

    # append datetime to .ckpt file saved --> DO NOT check directory valid or not 
    args.checkpoint_save_path = args.checkpoint_save_path + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S') \
                                + '.ckpt'
    args.eval_save_path = args.eval_save_path + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S') \
                          + '.pkl'

    # training data loader
    train_DI = data_input(args.trainset_dir,
                          args.batch_size,
                          IMAGE_SIZE,
                          do_shuffle=True,
                          verbose=True)
    train_epoch_size = train_DI.n_samples // args.batch_size

    # evaluation data loader
    eval_DI = data_input(args.evalset_dir,
                         args.eval_batch_size,
                         IMAGE_SIZE,
                         True,
                         True)
    n_eval_samples = eval_DI.n_samples
    eval_epoch_size = n_eval_samples // args.eval_batch_size

    # eval_step default to epoch size
    if not args.eval_step:
        args.eval_step = epoch_size

    eval_results = []

    # define graph to be built on
    g = tf.Graph()
    with g.as_default():
        # define global step
        global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                      initializer=tf.constant_initializer(0), trainable=False)

        # learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, [], 'learning_rate_placeholder')
        dynamic_learning_rate = tf.train.exponential_decay(learning_rate_placeholder,
                                                           global_step,
                                                           decay_steps=args.lr_decay_step,
                                                           decay_rate=args.lr_decay_rate,
                                                           staircase=True)

        ###################### DATA LAYER #######################
        with tf.variable_scope('DATA_LAYER'):
            image_placeholder = tf.placeholder(tf.float32, [args.batch_size,IMAGE_SIZE,IMAGE_SIZE,3], 
                                               'image_placeholder')
            label_placeholder = tf.placeholder(tf.int32, [args.batch_size],
                                               'label_placeholder')

        ###################### NETWORK #######################
        epts = encoder.vgg16_finetune(image_placeholder, 
                                          scope='vgg_16',
                                          n_classes=N_CLASSES,
                                          is_training=True, 
                                          dropout_keep_prob=1.0,
                                          reuse=False)
        logits = epts['logits']
            
        ###################### LOSS & OPT #######################
        with tf.variable_scope('LOSS_N_OPT'):
            # loss
            one_hot_labels = tf.one_hot(label_placeholder, N_CLASSES, axis=-1)
            logits = tf.squeeze(logits)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_labels),
                                  name='loss')
            tf.scalar_summary('loss', loss)
            # gradients and optimization
            trainables = tf.trainable_variables()
            train_op = train_op_fn(loss, trainables, args.optimizer, dynamic_learning_rate, global_step)
        
        ###################### EVAL ##################
        with tf.variable_scope('EVAL'):
            predictions = epts['predictions']
            eval_op = tf.equal(tf.cast(tf.argmax(predictions,1), tf.int32), label_placeholder)
            eval_op = tf.reduce_sum(tf.cast(eval_op, tf.float32))

        ###################### AUG #######################
        # define initialization op
        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

        # ops about summary
        merge_summary_op = tf.merge_all_summaries()
        train_summary_writer = tf.train.SummaryWriter(args.summary_train_log_dir, graph=tf.get_default_graph())

        # define saver
        saver = tf.train.Saver(max_to_keep=args.keep_checkpoint_num)

        # define restorer
        restore_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        excluded_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vgg_16/fc8') + \
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vgg_16/predictions')
        for i in range(len(excluded_vars)):
            try:
                restore_vars.remove(excluded_vars[i]) # size of fc8 has changed, DO NOT LOAD
            except ValueError:
                print('ValueError: excluded_vars not in restore_vars.')
        restorer = tf.train.Saver(restore_vars)

        # define session
        if args.use_gpu:
            sess_conf = tf.ConfigProto()
            sess_conf.gpu_options.allow_growth = True
            sess_conf.gpu_options.per_process_gpu_memory_fraction = args.gpu_fraction
        else:
            sess_conf = tf.ConfigProto(device_count={'GPU':0})
        sess = tf.Session(config=sess_conf)

        ###################### SESSION RUN #######################                      
        with sess.as_default():
            # initialize variables
            sess.run(init_op)

            # start input enqueue threads
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # load pretrained model if any 
            if args.pretrained_model:
                print('Use pretrained model {}'.format(args.pretrained_model))
                restorer.restore(sess, args.pretrained_model)

            for epoch_idx in xrange(args.n_epochs):
                for iter_idx in xrange(train_epoch_size):
                    # current step number
                    global_step_ = sess.run(global_step)

                    # Run one training step
                    image_in, label_in = train_DI.next_batch()
                    feed_dict = {image_placeholder: image_in,
                                 label_placeholder: label_in,
                                 learning_rate_placeholder: args.base_lr}
                    process = [train_op, loss, merge_summary_op]
                    _, loss_, summary = sess.run(process, feed_dict=feed_dict)
            
                    # print out loss every x steps
                    if iter_idx%args.print_step==0:
                        print('epoch {} iter {}: loss={}'.format(epoch_idx, iter_idx, loss_))

                    # run training summary every x steps
                    if (global_step_%args.summary_step==0):
                        train_summary_writer.add_summary(summary, global_step=global_step_)

                    # save checkpoint every x steps
                    if (global_step_%args.checkpoint_step==0):
                        save_path = saver.save(sess, args.checkpoint_save_path, global_step=global_step_)
                        print('checkpoint at global step {} saved at {}'.format(global_step_, save_path))

                    # do evaluation every x steps
                    if global_step_%args.eval_step==0:
                        eval_acc = 0.
                        for eval_iter_idx in xrange(eval_epoch_size):
                            image_in, label_in = eval_DI.next_batch()
                            feed_dict = {image_placeholder: image_in,
                                         label_placeholder: label_in}
                            eval_add_sum = sess.run(eval_op, feed_dict=feed_dict)
                            eval_acc += eval_add_sum
                        eval_acc = eval_acc / n_eval_samples
                        eval_results.append([global_step_,eval_acc])
                        print('epoch {} iter {}: eval_acc={}'.format(epoch_idx, iter_idx, eval_acc))
                        # save evalutation results
                        with open(args.eval_save_path, 'wb') as output:
                            pickle.dump(eval_results, output)

def train_op_fn(loss, trainables, which_opt, learning_rate, global_step):
    # determine which optimizer to use
    if which_opt=='ADAGRAD':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif which_opt=='ADADELTA':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
    elif which_opt=='ADAM':
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5) #, beta2=0.999, epsilon=0.1)
    elif which_opt=='RMSPROP':
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    elif which_opt=='MOM':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    else:
        raise ValueError('Invalid optimization algorithm')

    # define backpropagation flow
    grads = tf.gradients(loss, trainables)
    grad_and_vars = zip(grads, trainables)
    train = optimizer.apply_gradients(grad_and_vars, global_step=global_step)
    # add summaries of gradients
    for grad, var in list(grad_and_vars):
        tf.histogram_summary(var.op.name+'/gradient', grad)

    return train

if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    train(args)

