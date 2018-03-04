from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
from datetime import datetime
import tensorflow as tf
import argparse

import mnist_input
import misc
import models.infoGAN as network
from utils import *

# z_dim, batch_size, lr_decay_step, lr_decay_rate, optimizer, summary_log_dir, pretrained_model, lr
# summary_step, checkpoint_step, print_step, model_save_path, trainset, image_size, n_epochs, n_preprocessing_thread
# X(save_fkimg_step, save_fkimg_dir)
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
		tf.scalar_summary('dynamic_learning_rate', dynamic_learning_rate)

		###################### DATA LAYER #######################
		with tf.device('/cpu:0'):
			with tf.variable_scope('DataLayer'):
				# data layer (real image and latent variable), labels are not needed
				real_image, labels = mnist_input._input(args.trainset, 
												   args.image_size, 
												   args.batch_size, 
												   args.n_epochs, 
												   args.n_preprocessing_thread, 
												   False)
				z, c_disc, c_cont = network.latent_sampler(args.z_dim, args.batch_size)
				concat_z = tf.concat(1, [z, c_disc, c_cont])

		###################### NETWORK #######################
		with tf.variable_scope('Network'):
			# sample from generator
			fake_image = network.generator(concat_z, args.image_size, args.batch_size, scope='G')
			tf.image_summary('generated_image', fake_image, max_images=10)

			# define discriminator over real image, D being trained
			D_real_ep = network.discriminator(real_image, args.batch_size, scope='D')
			D_real_logits = D_real_ep['logits_fake_or_real']

			# define discriminator over fake image with G network fixed, D being trained
			D_fake_ep = network.discriminator(fake_image, args.batch_size, scope='D', reuse=True)
			D_fake_logits = D_fake_ep['logits_fake_or_real']

			# define discriminator over fake image with D network fixed, G being trained
			D_on_G_ep = network.discriminator(fake_image, args.batch_size, scope='D', reuse=True)
			D_on_G_logits = D_on_G_ep['logits_fake_or_real']

			# define Q network, only different from D in the final layer
			Q_ep = network.discriminator(fake_image, args.batch_size, scope='D', reuse=True)
			Q_logits_disc = Q_ep['logits_Q_disc']
			Q_logits_cont = Q_ep['logits_Q_cont']

		###################### LOSS & OPT #######################
		with tf.variable_scope('Loss'):
			####################### D #######################

			# define discriminator loss over real image --> D_real_score goal is 1
			all_ones_real = tf.ones_like(D_real_logits)
			D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_real_logits, all_ones_real), 
										 name='D_real_loss')
			tf.scalar_summary('D_real_loss', D_real_loss)

			# define discriminator loss over fake image --> D_fake_score goal is 0
			all_zeros_fake = tf.zeros_like(D_fake_logits)
			D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake_logits, all_zeros_fake), 
										 name='D_fake_loss')
			tf.scalar_summary('D_fake_loss', D_fake_loss)

			# total loss for discriminator
			D_loss = D_fake_loss + D_real_loss
			tf.scalar_summary('D_loss', D_loss)

			####################### G #######################

			# define generator loss --> D_fake_score goal is 1
			all_ones_fake = tf.ones_like(D_on_G_logits)
			G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_on_G_logits, all_ones_fake), 
									name='G_loss')
			tf.scalar_summary('G_loss', G_loss)

			####################### Q #######################

			# loss for discrete c
			Q_disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Q_logits_disc, c_disc), 
										 name='Q_discrete_loss')
			tf.scalar_summary('Q_discrete_loss', Q_disc_loss)

			# loss for continuous c
			Q_cont_loss = tf.reduce_mean(tf.nn.l2_loss(Q_logits_cont-c_cont), 
										 name='Q_continuous_loss')
			tf.scalar_summary('Q_continuous_loss', Q_cont_loss)

			Q_loss = Q_disc_loss + Q_cont_loss
			tf.scalar_summary('Q_loss', Q_loss)

		with tf.variable_scope('Optimizer'):
			# define train operation over G network
			G_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Network/G')
			train_G = train_op(G_loss, G_trainables, args.optimizer, 
							   learning_rate_placeholder, global_step, scope='G')

			# define train operation over D network, exclusive of augmented final layer
			D_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Network/D')
			D_aug_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Network/D/d_h2_2_lin') + \
							   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Network/D/d_h3_2_lin') + \
							   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Network/D/d_h3_3_lin')
			for i in range(len(D_aug_trainables)):
				try:
					D_trainables.remove(D_aug_trainables[i])
				except ValueError:
					print('ValueError: D_aug_trainables not in D_trainables')
			train_D = train_op(D_loss, D_trainables, args.optimizer, 
							   learning_rate_placeholder/10, global_step, scope='D')

			# define train operation over Q network --> train over all trainables, 
			# of course the branch final layer excluded
			Q_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
			Q_aug_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Network/D/d_h2_lin')
			for i in range(len(Q_aug_trainables)):
				try:
					Q_trainables.remove(Q_aug_trainables[i])
				except ValueError:
					print('ValueError: Q_aug_trainables not in Q_trainables')
			train_Q = train_op(Q_loss, Q_trainables, args.optimizer, 
							   learning_rate_placeholder/10, global_step, scope='Q')

		###################### AUG #######################

		# define initialization op
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

		# ops about summary
		merge_summary_op = tf.merge_all_summaries()
		summary_writer = tf.train.SummaryWriter(args.summary_log_dir, graph=tf.get_default_graph())

		# define saver
		saver = tf.train.Saver(max_to_keep=3)

		# define restorer
		restore_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		restorer = tf.train.Saver(restore_vars)

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
				restorer.restore(sess, pretrained_model)

			try:
				while not coord.should_stop():
					# current step number
					step = sess.run(global_step)

					# Run training steps
					feed_dict = {learning_rate_placeholder: args.lr}

					_, D_loss_ = sess.run([train_D, D_loss], feed_dict=feed_dict)

					_, G_loss_ = sess.run([train_G, G_loss], feed_dict=feed_dict)

					_, Q_loss_ = sess.run([train_Q, Q_loss], feed_dict=feed_dict)

					# print out loss every x steps
					if step%args.print_step==0:
						print('step {}: G_loss={:.6}, D_loss={:.6}, Q_loss={:.6}'.format\
							  (step, G_loss_, D_loss_, Q_loss_))

					# run summary every x steps
					if (step%args.summary_step==0):
						summary = sess.run(merge_summary_op, feed_dict=feed_dict)
						summary_writer.add_summary(summary, global_step=step)

					# save checkpoint every x steps
					if (step%args.checkpoint_step==0) and (step>=args.checkpoint_step):
						save_path = saver.save(sess, args.model_save_path, global_step=step)
						print('checkpoint at {} saved at {}'.format(step, save_path))

					# # save fake image every x steps
					# if step%args.save_fkimg_step==0:
					# 	fake_image_ = sess.run([fake_image])
					# 	## save fake image

			except tf.errors.OutOfRangeError:
				print('Done training -- epoch limit reached')
            
			finally:
				# When done, ask the threads to stop.
				coord.request_stop()

			coord.join(threads)
			sess.close()

def train_op(loss, trainables, which_opt, learning_rate, global_step, scope):
	# determine which optimizer to use
	if which_opt=='ADAGRAD':
		optimizer = tf.train.AdagradOptimizer(learning_rate)
	elif which_opt=='ADADELTA':
		optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
	elif which_opt=='ADAM':
		optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999, epsilon=0.1)
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
		tf.histogram_summary(scope+var.op.name+'/gradient', grad)

	return train

def parse_args(argv):
	parser = argparse.ArgumentParser()

	# training data flow parameters
	parser.add_argument('--trainset', type=str, 
						help='Path to .tfrecords file of training set.', default='~/Desktop/test_on_mnist/mnist_data/mnist_train.tfrecords')
	parser.add_argument('--image_size', type=int, 
						help='Size of (square) images in training set.', default=28)
	parser.add_argument('--batch_size', type=int, 
						help='Batch size of data flow during training.', default=64)
	parser.add_argument('--n_epochs', type=int, 
						help='Number of epoch run in training', default=25)
	parser.add_argument('--n_preprocessing_thread', type=int, 
						help='Number of thread used in preprocessing.', default=2)
	parser.add_argument('--z_dim', type=int, 
						help='Dimension of latent variable.', default=62)
	
	# training parameters
	parser.add_argument('--optimizer', type=str, 
						help='Optimizer used in backpropagation.', default='ADAM')
	parser.add_argument('--lr_decay_step', type=int, 
						help='Learning rate decay step.', default=1000)
	parser.add_argument('--lr_decay_rate', type=float, 
						help='Learning rate decay rate.', default=0.9)
	parser.add_argument('--lr', type=float, 
						help='Feed to learning rate placeholder.', default=0.002)

	# auxilary training parameters
	parser.add_argument('--summary_log_dir', type=str, 
						help='Directory to store summary logs.', default='/tmp/tf_log')
	parser.add_argument('--pretrained_model', type=str, 
						help='Path to pretrained model.')
	parser.add_argument('--model_save_path', type=str, 
						help='Path to save checkpoint models.', default='./checkpoints/infoGAN')
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