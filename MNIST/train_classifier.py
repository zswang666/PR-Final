from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
from datetime import datetime
import tensorflow as tf
import argparse

import mnist_input as mnist_input
import models.SGAN as network

# z_dim, batch_size, lr_decay_step, lr_decay_rate, optimizer, summary_log_dir, pretrained_model, lr
# summary_step, checkpoint_step, print_step, model_save_path, trainset, image_size, n_epochs, n_preprocessing_thread
# X(save_fkimg_step, save_fkimg_dir)
# n_class
def main(args):
	# append now date & time to checkpoint filename 
	args.model_save_path = args.model_save_path + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

	# specify graph to be built on
	with tf.Graph().as_default():
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
		with tf.device('/cpu:0'):
			with tf.variable_scope('DataLayer'):
				# data layer (real image and latent variable), labels are not needed
				real_image, label = mnist_input._input(args.trainset, 
												   args.image_size, 
												   args.batch_size, 
												   args.n_epochs, 
												   args.n_preprocessing_thread, 
												   False)

		###################### NETWORK #######################
		logits, _ = network.encoder(real_image,
									scope='E', is_training=True) # name=E
			
		###################### LOSS & OPT #######################
		one_hot_labels = tf.one_hot(label, args.n_class, axis=-1)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_labels), name='loss')
		trainables = tf.trainable_variables()
		train = train_op(loss, trainables, args.optimizer, dynamic_learning_rate, global_step)

		###################### AUG #######################

		# define initialization op
		# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

		# ops about summary
		merge_summary_op = tf.merge_all_summaries()
		summary_writer = tf.train.SummaryWriter(args.summary_log_dir, graph=tf.get_default_graph())

		# define saver
		saver = tf.train.Saver(max_to_keep=3)

		# define session
		sess_conf = tf.ConfigProto()
		sess_conf.gpu_options.allow_growth = True
		sess_conf.gpu_options.per_process_gpu_memory_fraction = 0.4
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
				pretrained_model = os.path.expanduser(args.pretrained_model)
				saver.restore(sess, pretrained_model)

			try:
				while not coord.should_stop():
					# current step number
					step = sess.run(global_step)

					# Run training steps
					feed_dict = {learning_rate_placeholder: args.lr}
					_, loss_ = sess.run([train, loss], feed_dict=feed_dict)

					# print out loss every x steps
					if step%args.print_step==0:
						print('step {}: loss={:.6}'.format(step, loss_))

					# run summary every x steps
					if (step%args.summary_step==0):
						summary = sess.run(merge_summary_op)
						summary_writer.add_summary(summary, global_step=step)

					# save checkpoint every x steps
					if (step%args.checkpoint_step==0) and (step>=args.checkpoint_step):
						save_path = saver.save(sess, args.model_save_path, global_step=step)
						print('checkpoint at {} saved at {}'.format(step, save_path))

			except tf.errors.OutOfRangeError:
				print('Done training -- epoch limit reached')
            
			finally:
				# When done, ask the threads to stop.
				coord.request_stop()

			coord.join(threads)
			sess.close()

def train_op(loss, trainables, which_opt, learning_rate, global_step):
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

def parse_args(argv):
	parser = argparse.ArgumentParser()

	# training data flow parameters
	parser.add_argument('--trainset', type=str, 
						help='Path to .tfrecords file of training set.', default='~/test_on_mnist/mnist_data/mnist_train.tfrecords')
	parser.add_argument('--image_size', type=int, 
						help='Size of (square) images in training set.', default=28)
	parser.add_argument('--batch_size', type=int, 
						help='Batch size of data flow during training.', default=100)
	parser.add_argument('--n_epochs', type=int, 
						help='Number of epoch run in training', default=10)
	parser.add_argument('--n_preprocessing_thread', type=int, 
						help='Number of thread used in preprocessing.', default=2)
	parser.add_argument('--n_class', type=int, 
						help='Number of class', default=10)

	# training parameters
	parser.add_argument('--optimizer', type=str, 
						help='Optimizer used in backpropagation.', default='ADAM')
	parser.add_argument('--lr_decay_step', type=int, 
						help='Learning rate decay step.', default=10000)
	parser.add_argument('--lr_decay_rate', type=float, 
						help='Learning rate decay rate.', default=1.0)
	parser.add_argument('--lr', type=float, 
						help='Feed to learning rate placeholder.', default=0.001)

	# auxilary training parameters
	parser.add_argument('--summary_log_dir', type=str, 
						help='Directory to store summary logs.', default='./log/')
	parser.add_argument('--pretrained_model', type=str, 
						help='Path to pretrained model.')
	parser.add_argument('--model_save_path', type=str, 
						help='Path to save checkpoint models.', default='./checkpoints/SGAN_encoder')
	parser.add_argument('--summary_step', type=int, 
						help='Obtain summary every n steps.', default=100)
	parser.add_argument('--checkpoint_step', type=int, 
						help='Save checkpoints every n steps.', default=1000)
	parser.add_argument('--print_step', type=int, 
						help='Print losses every n steps.', default=10)
	# parser.add_argument('--save_fkimg_step', type=int, 
	# 					help='Save generated images every n steps.', default=100)

	return parser.parse_args(argv)

if __name__=='__main__':
	args = parse_args(sys.argv[1:])
	main(args)
