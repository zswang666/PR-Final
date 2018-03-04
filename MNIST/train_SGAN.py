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
import utils

# z_dim, batch_size, lr_decay_step, lr_decay_rate, optimizer, summary_log_dir, pretrained_model, lr
# summary_step, checkpoint_step, print_step, model_save_path, trainset, image_size, n_epochs, n_preprocessing_thread
# X(save_fkimg_step, save_fkimg_dir)
# ? advloss_weight, condloss_weight, entloss_weight, G_lr, D_lr
def main(args):
	
	# check all path
	args.summary_log_dir = utils.check_dir(args.summary_log_dir)
	#args.model_save_path = utils.check_path(args.model_save_path)
	args.img_dir = utils.check_dir(args.img_dir)

	args.model_save_path = args.model_save_path + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

	# specify graph to be built on
	with tf.Graph().as_default():
		# learning rate
		G_lr_placeholder = tf.placeholder(tf.float32, [], 'generator_learning_rate_placeholder')
		D_lr_placeholder = tf.placeholder(tf.float32, [], 'discriminator_learning_rate_placeholder')

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
				one_hot_label=  tf.one_hot(label, args.n_class, axis=-1)
				# create concatenated image summary
				#image_sz = real_image.get_shape().as_list()
				#real_image_show = tf.reshape(real_image, [1,image_sz[1]*8,image_sz[2]*8,image_sz[3]])
				tf.image_summary('real_images', real_image, max_images=10)

				z0 = network.z0_sampler(args.z_dim, args.batch_size)
				z1 = network.z1_sampler(args.z_dim, args.batch_size)

		###################### NETWORK #######################
		_, E_real_ep = network.encoder(real_image, scope='E')
		E0_real = E_real_ep['fc3']
		E1_real = E_real_ep['logits']

		# input: E(independent training) or G(joint training)
		G1 = network.generator1(one_hot_label, z1, 'G1')
		if args.train_phase=='INDEPENDENT':
			G0 = network.generator0(E0_real, z0, 'G0')
		elif args.train_phase=='JOINT':
			G0 = network.generator0(G1, z0, 'G0') 
		gen_image = G0
		print(gen_image.get_shape()[:])
		#gen_image_show = tf.reshape(gen_image, [1,image_sz[1]*8,image_sz[2]*8,image_sz[3]])
		tf.image_summary('generated_images', gen_image, max_images=10)

		# input: G(independent training) or D_shared(joint training)
		gen_D0, recon_z0, D0_fake_shared = network.discriminator0(gen_image, 'D0')
		if args.train_phase=='INDEPENDENT':
			gen_D1, recon_z1, _ = network.discriminator1(G1, 'D1')
		elif args.train_phase=='JOINT':
			gen_D1, recon_z1, _ = network.discriminator1(G1, 'D1')
			#gen_D1, recon_z1, _ = network.discriminator1(D0_fake_shared, 'D1')
		else:
			raise ValueError('No such training phase')

		# input: E(independent training) or D_shared(joint training)		
		real_D0, _, D0_real_shared = network.discriminator0(real_image, 'D0', reuse=True)
		if args.train_phase=='INDEPENDENT':
			real_D1, _, _ = network.discriminator1(E0_real, 'D1', reuse=True)
		elif args.train_phase=='JOINT':
			real_D1, _, _ = network.discriminator1(E0_real, 'D1', reuse=True)
			#real_D1, _, _ = network.discriminator1(D0_real_shared, 'D1', reuse=True)
		else:
			raise ValueError('No such training phase')
		_, E_gen_ep = network.encoder(gen_image, scope='E', reuse=True)
		if args.train_phase=='INDEPENDENT':
			E0_recon = network.E0('E', gen_image)
			E1_recon = network.E1('E', G1) #E_gen_ep['logits']
		elif args.train_phase=='JOINT':
			E0_recon = E_gen_ep['fc3']
			E1_recon = E_gen_ep['logits']
		else:
			raise ValueError('No such training phase')

		###################### LOSS & OPT #######################
		all_ones_D0 = tf.ones_like(real_D0)
		all_zeros_D0 = tf.zeros_like(gen_D0)
		all_ones_D1 = tf.ones_like(real_D1)
		all_zeros_D1 = tf.zeros_like(gen_D1)
		all_ones_G1 = tf.ones_like(gen_D1)
		all_ones_G0 = tf.ones_like(gen_D0)
		# Q network entropy loss
		Q0_loss = tf.nn.l2_loss(recon_z0-z0, name='Q0_loss') / args.z_dim / args.batch_size
		Q1_loss = tf.nn.l2_loss(recon_z1-z1, name='Q1_loss') / args.z_dim / args.batch_size
		tf.scalar_summary('Q0_loss', Q0_loss)
		tf.scalar_summary('Q1_loss', Q1_loss)

		# D0 adversarial loss
		D0_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_D0, all_ones_D0),
									  name='D0_real_loss')
		D0_gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gen_D0, all_zeros_D0),
									 name='D0_gen_loss')
		D0_adv_loss = 0.5*D0_real_loss + 0.5*D0_gen_loss
		tf.scalar_summary('D0_adv_loss', D0_adv_loss)
		# D0 total loss
		D0_loss = args.advloss_weight*D0_adv_loss + args.entloss_weight*Q0_loss
		tf.scalar_summary('D0_loss', D0_loss)
		
		# D1 adversarial loss
		D1_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_D1, all_ones_D1),
									  name='D1_real_loss')
		D1_gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gen_D1, all_zeros_D1),
									  name='D1_gen_loss')
		D1_adv_loss = 0.5*D1_real_loss + 0.5*D1_gen_loss
		tf.scalar_summary('D1_adv_loss', D1_adv_loss)
		# D1 total loss
		D1_loss = args.advloss_weight*D1_adv_loss + args.entloss_weight*Q1_loss
		tf.scalar_summary('D1_loss', D1_loss)
	
		# G1 adversarial loss
		G1_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gen_D1, all_ones_G1),
									 name='G1_adv_loss')
		tf.scalar_summary('G1_adv_loss', G1_adv_loss)
		# G1 conditional loss
		G1_cond_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(E1_recon, one_hot_label),
									  name='G1_cond_loss')
		tf.scalar_summary('G1_cond_loss', G1_cond_loss)
		# G1 total loss
		G1_loss = args.advloss_weight*G1_adv_loss + args.condloss_weight*G1_cond_loss + args.entloss_weight*Q1_loss
		tf.scalar_summary('G1_loss', G1_loss)

		# G0 adversarial loss
		G0_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gen_D0, all_ones_G0),
									 name='G0_adv_loss')
		tf.scalar_summary('G0_adv_loss', G0_adv_loss)
		# G0 conditional loss
		G0_cond_loss = tf.nn.l2_loss(E0_recon-E0_real, name='G0_cond_loss') / 256.0 / args.batch_size
		tf.scalar_summary('G0_cond_loss', G0_cond_loss)
		# G0 total loss
		G0_loss = args.advloss_weight*G0_adv_loss + args.condloss_weight*G0_cond_loss + args.entloss_weight*Q0_loss
		tf.scalar_summary('G0_loss', G0_loss)

		# opt
		if args.train_phase=='INDEPENDENT':
			G0_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G0/')
			G1_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G1/')

			D0_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D0/')
			D1_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D1/')

			train_D0 = train_op(D0_loss, D0_trainables, args.optimizer, D_lr_placeholder)
			train_D1 = train_op(D1_loss, D1_trainables, args.optimizer, D_lr_placeholder)

			train_G0 = train_op(G0_loss, G0_trainables, args.optimizer, G_lr_placeholder)
			train_G1 = train_op(G1_loss, G1_trainables, args.optimizer, G_lr_placeholder)
		elif args.train_phase=='JOINT':
			G_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G0/') + \
						   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G1/')
			D_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D0/') + \
						   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D1/')
			G_loss = G0_loss + G1_loss
			tf.scalar_summary('G_loss', G_loss)
			D_loss = D0_loss + D1_loss
			tf.scalar_summary('D_loss', D_loss)
		
			train_D = train_op(D_loss, D_trainables, args.optimizer, D_lr_placeholder)
			train_G = train_op(G_loss, G_trainables, args.optimizer, G_lr_placeholder)
		else:
			raise ValueError('No such training phase')

		###################### AUG #######################

		# define initialization op
		# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

		# ops about summary
		merge_summary_op = tf.merge_all_summaries()
		summary_writer = tf.train.SummaryWriter(args.summary_log_dir, graph=tf.get_default_graph())

		# define saver
		saver = tf.train.Saver(max_to_keep=3)

		# define restorer
		E_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'E/')
		E_restorer = tf.train.Saver(E_vars)
		restore_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G0/') + \
						   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G1/') + \
						   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D0/') + \
						   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D1/')
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

			# load pretrained encoder
			if args.pretrained_E:
				pretrained_E = os.path.expanduser(args.pretrained_E)
				E_restorer.restore(sess, pretrained_E)
			else:
				raise ValueError('pretrained model of Encoder is necessary')
			# load pretrained model if any
			if args.pretrained_model:
				print('Use pretrained model {}'.format(args.pretrained_model))
				pretrained_model = os.path.expanduser(args.pretrained_model)
				restorer.restore(sess, pretrained_model)

			step = 0
			try:
				while not coord.should_stop():
					step += 1

					# Run training steps
					feed_dict = {G_lr_placeholder: args.G_lr,
								 D_lr_placeholder: args.D_lr}
					if args.train_phase=='INDEPENDENT':
						# train one step
						process_D = [train_D0, train_D1]
						process_G = [train_G0, train_G1]
						fetch = [G0_cond_loss, G0_adv_loss, Q0_loss,
								 D0_real_loss, D0_gen_loss,
								 G1_cond_loss, G1_adv_loss, Q1_loss,
								 D1_real_loss, D1_gen_loss]
						_, _ = sess.run(process_D, feed_dict=feed_dict)
						_, _ = sess.run(process_G, feed_dict=feed_dict)
						_, _ = sess.run(process_G, feed_dict=feed_dict)
						# fetch some infos
						fetch_ = sess.run(fetch, feed_dict=feed_dict)
					elif args.train_phase=='JOINT':
						_, G_loss_ = sess.run([train_G, G_loss], feed_dict=feed_dict)
						_, D_loss_ = sess.run([train_D, D_loss], feed_dict=feed_dict)
					else:
						raise ValueError('No such training phase')

					# print out loss every x steps
					if step%args.print_step==0:
						if args.train_phase=='INDEPENDENT':
							print(fetch_)
						else:
							print(G_loss_, D_loss_)
						#print('step {}: G0_loss={:.6}, D0_loss={:.6}, G1_loss={:.6}, D1_loss={:.6}'\
						#	.format(step, G0_loss_, D0_loss_, G1_loss_, D1_loss_))

					# run summary every x steps
					if (step%args.summary_step==0):
						summary = sess.run(merge_summary_op)
						summary_writer.add_summary(summary, global_step=step)

					# save checkpoint every x steps
					if (step%args.checkpoint_step==0) and (step>=args.checkpoint_step):
						save_path = saver.save(sess, args.model_save_path, global_step=step)
						print('checkpoint at {} saved at {}'.format(step, save_path))

					# save fake image every x steps
					if step%args.save_img_step==0:
					 	real_image_, gen_image_ = sess.run([real_image, gen_image])
						gen_image_ = utils.denormalize_images(gen_image_)
						real_image_ = utils.denormalize_images(real_image_) 
						filename = args.img_dir + '/real_' + str(step) + '.jpg'
						utils.images_on_grid(real_image_,8,8,filename)
						filename = args.img_dir + '/generated_' + str(step) + '.jpg'
						utils.images_on_grid(gen_image_,8,8,filename)		
							
			except tf.errors.OutOfRangeError:
				print('Done training -- epoch limit reached')
            
			finally:
				# When done, ask the threads to stop.
				coord.request_stop()

			coord.join(threads)
			sess.close()

def train_op(loss, trainables, which_opt, learning_rate):
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
	train = optimizer.apply_gradients(grad_and_vars)
	# add summaries of gradients
	for grad, var in list(grad_and_vars):
		tf.histogram_summary(var.op.name+'/gradient', grad)

	return train

def parse_args(argv):
	parser = argparse.ArgumentParser()

	# training data flow parameters
	parser.add_argument('--trainset', type=str, 
						help='Path to .tfrecords file of training set.', 
						default='~/test_on_mnist/mnist_data/mnist_train.tfrecords')
	parser.add_argument('--image_size', type=int, 
						help='Size of (square) images in training set.', default=28)
	parser.add_argument('--batch_size', type=int, 
						help='Batch size of data flow during training.', default=64)
	parser.add_argument('--n_class', type=int, 
						help='Number of class', default=10)
	parser.add_argument('--n_epochs', type=int, 
						help='Number of epoch run in training', default=100)
	parser.add_argument('--n_preprocessing_thread', type=int, 
						help='Number of thread used in preprocessing.', default=2)
	parser.add_argument('--z_dim', type=int, 
						help='Dimension of latent variable.', default=50)
	
	# training parameters
	parser.add_argument('--optimizer', type=str, 
						help='Optimizer used in backpropagation.', default='ADAM')
	parser.add_argument('--G_lr', type=float, 
						help='learning rate for generator', default=0.0002)
	parser.add_argument('--D_lr', type=float, 
						help='learning rate for discriminator', default=0.0002)
	parser.add_argument('--advloss_weight', type=float, 
						help='weight for adverarial loss', default=1.0)
	parser.add_argument('--condloss_weight', type=float, 
						help='weight for conditional loss', default=1.0)
	parser.add_argument('--entloss_weight', type=float, 
						help='weight for entropy loss', default=1.0)
	parser.add_argument('--train_phase', type=str,
						help='training phase, INDEPENDENT or JOINT', default='INDEPENDENT')

	# auxilary training parameters
	parser.add_argument('--summary_log_dir', type=str, 
						help='Directory to store summary logs.', default='./log/')
	parser.add_argument('--pretrained_E', type=str, 
						help='Path to pretrained encoder (must have).', 
						default='~/test_on_mnist/checkpoints/SGAN_encoder20161223-203121-5000')
	parser.add_argument('--pretrained_model', type=str, 
						help='Path to pretrained model.')
	parser.add_argument('--model_save_path', type=str, 
						help='Path to save checkpoint models.', default='./checkpoints/SGAN')
	parser.add_argument('--summary_step', type=int, 
						help='Obtain summary every n steps.', default=100)
	parser.add_argument('--checkpoint_step', type=int, 
						help='Save checkpoints every n steps.', default=1000)
	parser.add_argument('--print_step', type=int, 
						help='Print losses every n steps.', default=10)
	parser.add_argument('--save_img_step', type=int, 	
 						help='Save images every n steps.', default=500)
	parser.add_argument('--img_dir', type=str, 
						help='directory to save batches of real and generated images', default='./grid_img')


	return parser.parse_args(argv)

if __name__=='__main__':
	args = parse_args(sys.argv[1:])
	main(args)
