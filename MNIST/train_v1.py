from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf

import mnist_input
import misc
import models.lenet as network
# import models.multilayer_perceptron as network

slim = tf.contrib.slim

def train(): 
	# train input params
	trainset = '~/Desktop/test_on_mnist/mnist_data/mnist_train.tfrecords'
	image_size = 28
	batch_size = 100
	n_epochs = 2
	n_preprocessing_thread = 2
	random_flip = False

	# train params
	summary_log_dir = '/tmp/tf_log'
	n_class = 10
	which_opt = 'ADAM'
	ema_decay = 0.1
	lr_decay_step = 1000
	lr_decay_rate = 0.9
	center_alpha = 0.1
	pretrained_model = None
	model_save_path = './checkpoints/lenet_model'
	reg_constant = 0.1
	center_loss_factor = 0.001
	summary_step = 100
	checkpoint_step = 10
	eval_step = 100
	lr = 0.001

	# test input param
	testset = '~/Desktop/test_on_mnist/mnist_data/mnist_test.tfrecords'
	test_image_size = 28
	test_batch_size = 100
	test_n_epochs = 1
	test_n_preprocessing_thread = 2
	test_random_flip = False

	# specify graph to be built on
	with tf.Graph().as_default():
		# define global step op, non-trainable
		global_step = tf.get_variable('global_step', [], dtype=tf.int32,
									initializer=tf.constant_initializer(0), trainable=False)

		# build input/output batch flow as ops in graph, data preprocessing also done!
		# trainset is a .tfrecord file --> process on CPU, reserve place in GPU for training
		with tf.device('/cpu:0'):
			image_batch, label_batch = mnist_input._input(trainset, 
													 	  image_size, 
														  batch_size, 
														  n_epochs, 
														  n_preprocessing_thread, 
														  random_flip)

		# inference
		logits, endpoints = network.inference(image_batch, num_classes=n_class, is_training=True,
										  dropout_keep_prob=0.5,
										  prediction_fn=slim.softmax,
										  scope='LeNet')
		prelogits = endpoints['deep_feats']
		# define objective loss op
		one_hot_labels = tf.one_hot(label_batch, n_class, axis=-1)
		obj_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_labels), name='obj_loss')

		# define center loss op as an additional regulizer
		#center_loss, centers = misc.center_loss_def(prelogits, center_alpha, label_batch, n_class)
		#tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, center_loss_factor*center_loss)

		# define decov loss op as an additional regulizer
		## decov_loss = fn(prelogits) ????
		## add to decov_loss*decov_loss_factor regularization loss collection

		# make learning rate determinable and dynamically adaptable
		learning_rate_placeholder = tf.placeholder(tf.float32, [], 'learning_rate_placeholder') # simply the initial value of dynamic learning rate
		dynamic_learning_rate = tf.train.exponential_decay(learning_rate_placeholder, 
															global_step, 
															decay_steps=lr_decay_step,
															decay_rate=lr_decay_rate,
															staircase=True)
    
		# get regularization loss, defined in trainable variables regulizer, also center loss and decov loss
		reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

		# define train op
		train_op, moving_average_loss = train_def(which_opt, 
							 obj_loss, 
							 reg_loss, 
							 reg_constant, 
							 dynamic_learning_rate, 
							 ema_decay, global_step)

		# merge all summaries op
		merge_summary_op = tf.merge_all_summaries()

		# define evaluation graph
		with tf.device('/cpu:0'):
			test_image_batch, test_label_batch = mnist_input._input(testset, 
													 	  test_image_size, 
														  test_batch_size, 
														  test_n_epochs, 
														  test_n_preprocessing_thread, 
														  test_random_flip)
		test_logits, _ = network.inference(test_image_batch, num_classes=n_class, is_training=False,
											   dropout_keep_prob=0.5,
											   prediction_fn=slim.softmax,
											   scope='LeNet',
											   reuse=True)		
		eval_op = tf.equal(tf.argmax(test_logits, 1), test_label_batch)
		eval_op = tf.reduce_mean(tf.cast(eval_op, tf.float32))
    
		# add op to save and restore all the variables
		saver = tf.train.Saver(max_to_keep=3)

		# initialization op
		#init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

		# define session
		sess_conf = tf.ConfigProto()
		sess_conf.gpu_options.allow_growth = True
		sess_conf.gpu_options.per_process_gpu_memory_fraction = 0.4
		sess = tf.Session(config=sess_conf)

		summary_writer = tf.train.SummaryWriter(summary_log_dir, graph=tf.get_default_graph())
    
		with sess.as_default():
			# can have learning rate schedule
			## ???

			# initialize variables
			sess.run(init_op)

			# start input enqueue threads
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			# load pretrained model if any 
			if pretrained_model is not None:
				saver.restore(sess, pretrained_model)

			try:
				while not coord.should_stop():
					step = sess.run(global_step)

					# Run training steps or whatever
					feed_dict = {learning_rate_placeholder: lr}
					process = [train_op, obj_loss, moving_average_loss]
					_, obj_loss_, moving_average_loss_ = sess.run(process, feed_dict=feed_dict)

					if step%10==0:
						print('step {:>5}: {:.6}, {:.6}'.format(step, obj_loss_, moving_average_loss_))

					# run summary every n? steps
					if (step%summary_step==0):
						summary = sess.run(merge_summary_op)
						summary_writer.add_summary(summary, global_step=step)

					# evaluate performance every n? steps
					if (step%eval_step==0):
						acc = sess.run(eval_op)
						print('Accuracy evaluating mnist_test: %f' %(acc))

					# save checkpoint every n? steps
					if (step%checkpoint_step==0) and (step>=checkpoint_step):
						save_path = saver.save(sess, model_save_path, global_step=step)
						print('checkpoint at {} saved at {}'.format(step, save_path))

			except tf.errors.OutOfRangeError:
				print('Done training -- epoch limit reached')
            
			finally:
				# When done, ask the threads to stop.
				coord.request_stop()

			coord.join(threads)
			sess.close()

# define train op 
def train_def(which_opt, obj_loss, reg_loss, reg_constant, learning_rate, ema_decay, global_step):
	'''
	Arguments:
	    which_opt: a string to specify which optimizer to be used
	    obj_loss: objective loss, inclusive of regularization loss
	    reg_loss: regularization loss (a list of tensor)
	    reg_constant: a scalar, weight of regularization loss
	                  --> total_loss = obj_loss + reg_constant*reg_loss
	    ema_decay: decay rate of moving average
	    global_step: number of steps done in entire training
	Returns:
	    train_op: training operation of entire network
	'''
	# track moving average loss (not including regularization loss)
	loss_ema = tf.train.ExponentialMovingAverage(decay=ema_decay) # add global_step???
	loss_ema_op = loss_ema.apply([obj_loss])
	moving_average_loss = loss_ema.average(obj_loss)
	# build summaries for raw loss and moving average loss and regularization loss
	for l in [obj_loss]:
		tf.scalar_summary(l.op.name+' (raw)', l)
		tf.scalar_summary(l.op.name, moving_average_loss)
    
	# determine optimizer to be used according to which_opt
	if which_opt == 'ADAM':
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)#, beta1=0.9, beta2=0.999, epsilon=1e-8)
	else:
		raise ValueError('Invalid optimizer to be used.')
    
	# define backpropagation flow
	trainables = tf.trainable_variables()
	if reg_loss!=[]:
		total_loss = obj_loss + reg_constant * tf.add_n(reg_loss)
	else:
		total_loss = obj_loss
	grad = tf.gradients(total_loss, trainables)
	grad_and_vars = zip(grad, trainables)
	apply_gradient_op = optimizer.apply_gradients(grad_and_vars, global_step=global_step)
	# add summaries of gradients
	for grad, var in list(grad_and_vars):
		tf.histogram_summary(var.op.name+'/gradient', grad)
    
	# apply moving average to all trainable variables
	vars_ema = tf.train.ExponentialMovingAverage(decay=ema_decay) # add global_step???
	vars_ema_op = vars_ema.apply(trainables)
    
	# define train op, both backpropagation and maintaining moving average must be done
	with tf.control_dependencies([apply_gradient_op, vars_ema_op, loss_ema_op]):
		train_op = tf.no_op(name='train')

	return train_op, moving_average_loss

if __name__=='__main__':
	train()
