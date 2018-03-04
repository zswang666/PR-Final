from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
from datetime import datetime
import tensorflow as tf
import argparse

from data_loader import indoor_input as data_input
import utils
import models.SGAN_new as network

IMAGE_SIZE = 224
N_CLASSES = 10

# unmodified
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
    parser.add_argument('--G_base_lr', type=float,
                        help='Base learning rate of generator')
    parser.add_argument('--D_base_lr', type=float,
                        help='Base learning rate of discriminator')
    parser.add_argument('--lr_decay_step', type=float,
                        help='Decay step of learning rate')
    parser.add_argument('--lr_decay_rate', type=float,
                        help='Decay rate of learning rate')
    parser.add_argument('--advloss_weight', type=float,
                        help='Weight of adversarial loss')
    parser.add_argument('--condloss_weight', type=float,
                        help='Weight of condition loss')
    parser.add_argument('--entloss_weight', type=float,
                        help='Weight of entropy loss')
    parser.add_argument('--z0_dim', type=int,
                        help='Dimension of z0')
    parser.add_argument('--z1_dim', type=int,
                        help='Dimension of z1')
    parser.add_argument('--z2_dim', type=int,
                        help='Dimension of z2')
    # training auxiliary
    parser.add_argument('--G2_pretrained', type=str,
                        help='Path to pretrained G2')
    parser.add_argument('--G1_pretrained', type=str,
                        help='Path to pretrained G1')
    parser.add_argument('--G0_pretrained', type=str,
                        help='Path to pretrained G0')
    parser.add_argument('--E_pretrained', type=str,
                        help='Path to pretrained encoder')
    parser.add_argument('--pretrained_model', type=str,
                        help='Path to pretrained model')
    parser.add_argument('--checkpoint_step', type=int,
                        help='Interval of saving checkpoints')
    parser.add_argument('--checkpoint_save_path', type=str,
                        help='Checkpoint save path')
    parser.add_argument('--keep_checkpoint_num', type=int,
                        help='Maximum number of checkpoint files to be kept')
    parser.add_argument('--print_step', type=int,
                        help='Interval of printing out per-train-step messages', default=10)
    parser.add_argument('--real_img_dir', type=str,
                        help='Directory to save merged real images (which has correspondence to generated images)')
    parser.add_argument('--gen_img_dir', type=str,
                        help='Directory to save merged generated images')
    parser.add_argument('--merge_nh', type=int,
                        help='Number of images in height in merged images, merge_nh*merge_nw must = batch_size')
    parser.add_argument('--merge_nw', type=int,
                        help='Number of images in width in merged images, merge_nh*merge_nw must = batch_size')
    parser.add_argument('--save_img_step', type=int,
                        help='Interval of saving merged images')
    # summaries
    parser.add_argument('--summary_step', type=int,
                        help='Interval of adding summaries')
    parser.add_argument('--summary_train_log_dir', type=str,
                        help='Directory to save summaries of training', default='./train_log')
    # session
    parser.add_argument('--use_gpu', type=bool,
                        help='If True, use GPU, otherwise, don\'t use GPU', default=True)
    parser.add_argument('--gpu_fraction', type=float,
                        help='Fraction of GPU allowed to be used')

    return parser.parse_args(argv)

def main(args):
    # check n_h and n_w of merged image
    assert(args.merge_nh*args.merge_nw==args.batch_size)
    # E_pretrained must exist or be contained in pretrained_model
    if not (args.pretrained_model or args.E_pretrained):
        print('You must give either E_pretrained to train from scratch or pretrained_model to continue previous training.')
        sys.exit()
    # correct and validate path
    args.trainset_dir = utils.check_dir(args.trainset_dir)
    if args.pretrained_model: # there can be no pretrained_model
        args.pretrained_model = utils.check_path(args.pretrained_model)
    if args.G0_pretrained:
        args.G0_pretrained = utils.check_path(args.G0_pretrained)
    args.summary_train_log_dir = utils.check_dir(args.summary_train_log_dir)
    args.E_pretrained = utils.check_path(args.E_pretrained)
    args.real_img_dir = utils.check_dir(args.real_img_dir)
    args.gen_img_dir = utils.check_dir(args.gen_img_dir)
    # append datetime to .ckpt file saved --> DO NOT check directory valid or not 
    args.checkpoint_save_path = args.checkpoint_save_path + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S') \
                                + '.ckpt'
    # training data loader
    train_DI = data_input(args.trainset_dir,
                          args.batch_size,
                          IMAGE_SIZE,
                          do_shuffle=True,
                          verbose=True)
    train_epoch_size = train_DI.n_samples // args.batch_size
    # define graph to be built on
    g = tf.Graph()
    with g.as_default():
        # define global step
        global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                      initializer=tf.constant_initializer(0), trainable=False)

        # learning rate
        G_lr_placeholder = tf.placeholder(tf.float32, [], 'G_lr_placeholder')
        G_lr = tf.train.exponential_decay(G_lr_placeholder,
                                          global_step,
                                          decay_steps=args.lr_decay_step,
                                          decay_rate=args.lr_decay_rate,
                                          staircase=True)
        D_lr_placeholder = tf.placeholder(tf.float32, [], 'D_lr_placeholder')
        D_lr = tf.train.exponential_decay(D_lr_placeholder,
                                          global_step,
                                          decay_steps=args.lr_decay_step,
                                          decay_rate=args.lr_decay_rate,
                                          staircase=True)

        ###################### DATA LAYER #######################
        with tf.variable_scope('DATA_LAYER'):
            # real world sampler
            real_image = tf.placeholder(tf.float32, [args.batch_size,IMAGE_SIZE,IMAGE_SIZE,3], 
                                        'real_image')
            label = tf.placeholder(tf.int32, [args.batch_size], 'label')
            one_hot_label = tf.one_hot(label, N_CLASSES, axis=-1)
            tf.image_summary('real_image', real_image, max_images=10)
    
            # latent variables sampler
            z0 = network.z0_sampler(args.batch_size, args.z0_dim)
            z1 = network.z1_sampler(args.batch_size, args.z1_dim)
            z2 = network.z2_sampler(args.batch_size, args.z2_dim)

        ###################### NETWORK #######################
        # stacked encoder over real images
        real_E0 = network.E0(real_image, 'E')
        real_E1 = network.E1(real_E0, 'E')
        real_E2, real_E2_conv = network.E2(real_E1, 'E')

        # stacked generator (independent stage --> G's input from E)
        G2 = network.generator0(z2, one_hot_label, 'G2') # Horatio's generator0 is my G2
        G1 = network.generator1(z1, real_E1, 'G1') # Horatio's generator1 is my G1
        G0 = network.generator2(z0, real_E0, 'G0') # Horatio's generator2 is my G0
        gen_image = G0
        tf.image_summary('generated_images', gen_image, max_images=10)
        
        # joint G0, G1, visualization
        gen_image_G01 = network.generator2(z0, G1, 'G0', reuse=True)
        tf.image_summary('joint_G0G1_gen_images', gen_image_G01, max_images=10)

        # joint G0, G1, G2, visualization
        gen_G1 = network.generator1(z1, G2, 'G1', reuse=True)
        gen_image_G012 = network.generator2(z0, gen_G1, 'G0', reuse=True)
        tf.image_summary('joint_G0G1G2_gen_images', gen_image_G012, max_images=10)

        # stacked encoder over generated images
        gen_E0 = network.E0(gen_image, 'E', reuse=True)
        gen_E1 = network.E1(G1, 'E', reuse=True)
        gen_E2, gen_E2_conv = network.E2(G2, 'E', reuse=True)

        # stacked discriminator over generated images
        gen_D0, recon_z0 = network.discriminator2(gen_image, args.z0_dim, 'D0') # Horatio's discriminator2 is my D0
        gen_D1, recon_z1 = network.discriminator1(G1, args.z1_dim, 'D1') # Horatio's discriminator1 is my D1
        gen_D2, recon_z2 = network.discriminator0(G2, args.z2_dim, 'D2') # Horatio's discriminator0 is my D2

        # stacked discriminator over real images
        real_D0, _ = network.discriminator2(real_image, args.z0_dim, 'D0', reuse=True) # H's discriminator2 = my D0
        real_D1, _ = network.discriminator1(real_E0, args.z1_dim, 'D1', reuse=True) # H's discriminator1 = my D1
        real_D2, _ = network.discriminator0(real_E1, args.z2_dim, 'D2', reuse=True) # H's discriminator0 = my D2

        ###################### LOSS & OPT #######################
        all_ones_0 = tf.ones_like(real_D0) # real_D0 and gen_D0 have the same shape
        all_ones_1 = tf.ones_like(real_D1) # real_D1 and gen_D1 have the same shape
        all_ones_2 = tf.ones_like(real_D2) # real_D2 and gen_D2 have the same shape
        all_zeros_0 = tf.zeros_like(gen_D0) # the same assumption as above
        all_zeros_1 = tf.zeros_like(gen_D1)
        all_zeros_2 = tf.zeros_like(gen_D2)

        # Q network entropy loss (z reconstruction)
        Q0_loss = tf.reduce_mean(tf.square(recon_z0-z0), name='Q0_loss') # l2_loss
        Q1_loss = tf.reduce_mean(tf.square(recon_z1-z1), name='Q1_loss') # l2_loss
        Q2_loss = tf.reduce_mean(tf.square(recon_z2-z2), name='Q2_loss') # l2_loss
        tf.scalar_summary('Q0_loss', Q0_loss)
        tf.scalar_summary('Q1_loss', Q1_loss)
        tf.scalar_summary('Q2_loss', Q2_loss)

        # G0 loss
        G0_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gen_D0, all_ones_0), name='G0_adv_loss')
        G0_cond_loss = tf.reduce_mean(tf.square(gen_E0-real_E0), name='G0_cond_loss') # l2_loss, not sure???
        G0_loss = args.advloss_weight*G0_adv_loss + args.condloss_weight*G0_cond_loss + args.entloss_weight*Q0_loss
        tf.scalar_summary('G0_adv_loss', G0_adv_loss)
        tf.scalar_summary('G0_cond_loss', G0_cond_loss)
        tf.scalar_summary('G0_loss', G0_loss)
        # G1 loss
        G1_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gen_D1, all_ones_1), name='G1_adv_loss')
        G1_cond_loss = 0.05*tf.reduce_mean(tf.square(gen_E1-real_E1), name='G1_cond_loss') # l2_loss, not sure???
        G1_loss = args.advloss_weight*G1_adv_loss + args.condloss_weight*G1_cond_loss + args.entloss_weight*Q1_loss
        tf.scalar_summary('G1_adv_loss', G1_adv_loss)
        tf.scalar_summary('G1_cond_loss', G1_cond_loss)
        tf.scalar_summary('G1_loss', G1_loss)
        # G2 loss
        G2_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gen_D2, all_ones_2), name='G2_adv_loss')
        G2_cond_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(gen_E2, one_hot_label), name='G2_cond_loss')
        G2_cond_loss_conv = tf.reduce_mean(tf.square(gen_E2_conv-real_E2_conv), name='G2_cond_loss_conv')
	G2_loss = args.advloss_weight*G2_adv_loss + args.condloss_weight*(G2_cond_loss+G2_cond_loss_conv) + args.entloss_weight*Q2_loss
        tf.scalar_summary('G2_adv_loss', G2_adv_loss)
        tf.scalar_summary('G2_cond_loss', G2_cond_loss)
        tf.scalar_summary('G2_loss', G2_loss)
	tf.scalar_summary('G2_cond_loss_conv', G2_cond_loss_conv)

        # D0 loss
        D0_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_D0, all_ones_0), name='D0_real_loss')
        D0_gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gen_D0, all_zeros_0), name='D0_gen_loss')
        D0_adv_loss = 0.5*D0_real_loss + 0.5*D0_gen_loss
        D0_loss = args.advloss_weight*D0_adv_loss + args.entloss_weight*Q0_loss
        tf.scalar_summary('D0_adv_loss', D0_adv_loss)
        tf.scalar_summary('D0_loss', D0_loss)
        # D1 loss
        D1_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_D1, all_ones_1), name='D1_real_loss')
        D1_gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gen_D1, all_zeros_1), name='D1_gen_loss')
        D1_adv_loss = 0.5*D1_real_loss + 0.5*D1_gen_loss
        D1_loss = args.advloss_weight*D1_adv_loss + args.entloss_weight*Q1_loss
        tf.scalar_summary('D1_adv_loss', D1_adv_loss)
        tf.scalar_summary('D1_loss', D1_loss)
        # D2 loss
        D2_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_D2, all_ones_2), name='D2_real_loss')
        D2_gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gen_D2, all_zeros_2), name='D2_gen_loss')
        D2_adv_loss = 0.5*D2_real_loss + 0.5*D2_gen_loss
        D2_loss = args.advloss_weight*D2_adv_loss + args.entloss_weight*Q2_loss
        tf.scalar_summary('D2_adv_loss', D2_adv_loss)
        tf.scalar_summary('D2_loss', D2_loss)

        # G opt
        G0_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G0/')
        G1_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G1/')
        G2_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G2/')
        train_G0 = train_op_fn(G0_loss, G0_trainables, args.optimizer, G_lr, global_step)
        train_G1 = train_op_fn(G1_loss, G1_trainables, args.optimizer, G_lr, global_step)
        train_G2 = train_op_fn(G2_loss, G2_trainables, args.optimizer, G_lr, global_step)
        # D opt
        D0_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D0/')
        D1_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D1/')
        D2_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D2/')
        train_D0 = train_op_fn(D0_loss, D0_trainables, args.optimizer, D_lr, global_step)
        train_D1 = train_op_fn(D1_loss, D1_trainables, args.optimizer, D_lr, global_step)
        train_D2 = train_op_fn(D2_loss, D2_trainables, args.optimizer, D_lr, global_step)

        ###################### AUG #######################
        # define initialization op
        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        # ops about summary
        merge_summary_op = tf.merge_all_summaries()
        train_summary_writer = tf.train.SummaryWriter(args.summary_train_log_dir, graph=tf.get_default_graph())
        # define saver
        saver = tf.train.Saver(max_to_keep=args.keep_checkpoint_num)
        # define encoder restorer
        E_restore_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'E/')
        E_restorer = tf.train.Saver(E_restore_vars)
        # define G0 restorer
        G0_restorer = tf.train.Saver(G0_trainables)
        # define G1 restorer
        G1_restorer = tf.train.Saver(G1_trainables)
        # define G2 restorer
        G2_restorer = tf.train.Saver(G2_trainables)
        # define general restorer
        restorer = tf.train.Saver()
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

            # load pretrained model if any 
            if args.pretrained_model:
                print('Use pretrained model {}'.format(args.pretrained_model))
                restorer.restore(sess, args.pretrained_model)
            else: # pretrained encoder is a must-have
                print('Train using encoder {}'.format(args.E_pretrained))
                E_restorer.restore(sess, args.E_pretrained)            
            if args.G0_pretrained:
                print('Use pretrained G0 {}'.format(args.G0_pretrained))
                G0_restorer.restore(sess, args.G0_pretrained)
            if args.G1_pretrained:
                print('Use pretrained G1 {}'.format(args.G1_pretrained))
                G1_restorer.restore(sess, args.G1_pretrained)
            if args.G2_pretrained:
                print('Use pretrained G2 {}'.format(args.G2_pretrained))
                G2_restorer.restore(sess, args.G2_pretrained)

            for epoch_idx in xrange(args.n_epochs):
                for iter_idx in xrange(train_epoch_size):
                    # current step number
                    global_step_ = sess.run(global_step)

                    # Run one training step
                    image_in, label_in = train_DI.next_batch()
                    feed_dict = {real_image: image_in,
                                 label: label_in,
                                 D_lr_placeholder: args.D_base_lr,
                                 G_lr_placeholder: args.G_base_lr}
                    process_D = [train_D0, train_D1, train_D2]
                    process_G = [train_G0, train_G1, train_G2]
                    fetch = [G0_adv_loss, G0_cond_loss, D0_adv_loss, Q0_loss,
                             G1_adv_loss, G1_cond_loss, D1_adv_loss, Q1_loss,
                             G2_adv_loss, G2_cond_loss, D2_adv_loss, Q2_loss,
                             G0_loss, D0_loss,
                             G1_loss, D1_loss,
                             G2_loss, D2_loss]
                    
                    #process_D = [train_D0, train_D1]#
                    #process_G = [train_G0, train_D1]#
                    #fetch = [G0_adv_loss, G0_cond_loss, D0_adv_loss, Q0_loss,#
                    #         G1_adv_loss, G1_cond_loss, D1_adv_loss, Q1_loss,#
                    #         G0_loss, D0_loss,#
                    #         G1_loss, D1_loss]#
                    
                    #process_D = train_D2#
                    #process_G = train_G2#
                    #fetch = [G2_adv_loss, G2_cond_loss, D2_adv_loss, Q2_loss,#
                    #         G2_loss, D2_loss]#

                    sess.run(process_D, feed_dict=feed_dict)
                    sess.run(process_G, feed_dict=feed_dict)
                    sess.run(process_G, feed_dict=feed_dict) #train G twice???
                    fetch_ = sess.run(fetch, feed_dict=feed_dict)
                                
                    # print out loss every x steps
                    if iter_idx%args.print_step==0:
                        print('\n')
                        print('epoch {} iter {}: (G_adv, G_cond, D_adv, Q, 012)'.format(epoch_idx, iter_idx))
                        
                        #print('{}'.format(fetch_[0:4]))#
                        #print('{}'.format(fetch_[4:6]))
                        
                        #print('{}'.format(fetch_[0:4]))#
                        #print('{}'.format(fetch_[4:8]))#
                        #print('{}'.format(fetch_[8:10]))#
                        #print('{}'.format(fetch_[10:12]))#
                        
                        print('{}'.format(fetch_[0:4]))
                        print('{}'.format(fetch_[4:8]))
                        print('{}'.format(fetch_[8:12]))
                        print('G0/D0: {}'.format(fetch_[12:14]))
                        print('G1/D1: {}'.format(fetch_[14:16]))
                        print('G2/D2: {}'.format(fetch_[16:18]))

                    # run training summary every x steps
                    if iter_idx%args.summary_step==0:
                        # use feed_dict in current training step
                        summary = sess.run(merge_summary_op, feed_dict=feed_dict)
                        train_summary_writer.add_summary(summary, global_step=global_step_)
                        print('Write summary OK!')#

                    # save checkpoint every x steps
                    if iter_idx%args.checkpoint_step==0 and global_step_>10:
                        save_path = saver.save(sess, args.checkpoint_save_path, global_step=global_step_)
                        print('checkpoint at global step {} saved at {}'.format(global_step_, save_path))

                    # save generated image every x steps
                    if iter_idx%args.save_img_step==0:
                        try:
                            # fetch real and generated images, use feed_dict from current train step
                            real_image_, gen_image_G01_, gen_image_G012_ = sess.run([real_image, gen_image_G01, gen_image_G012], feed_dict=feed_dict)
                            # denormalize images from -1~+1 to uint8
                            gen_image_G01_ = utils.denormalize_images(gen_image_G01_)
                            gen_image_G012_ = utils.denormalize_images(gen_image_G012_)
                            real_image_ = utils.denormalize_images(real_image_)
                            # save real images
                            filename = os.path.join(args.real_img_dir, 'R'+str(global_step_)+'.jpg')
                            utils.images_on_grid(real_image_, args.merge_nh, args.merge_nw, filename)
                            # save generated images
                            filename = os.path.join(args.gen_img_dir, 'G01_'+str(global_step_)+'.jpg')
                            utils.images_on_grid(gen_image_G01_, args.merge_nh, args.merge_nw, filename)
                            filename = os.path.join(args.gen_img_dir, 'G012_'+str(global_step_)+'.jpg')
                            utils.images_on_grid(gen_image_G012_, args.merge_nh, args.merge_nw, filename)
                            print('Save merged image OK!')#debug
                        except:
                            pass
                        #gen_image_G012_ = sess.run(gen_image_G012, feed_dict=feed_dict)
                        #gen_image_G012_ = utils.denormalize_images(gen_image_G012_)
                        #filename = os.path.join(args.gen_img_dir, 'G012'+str(global_step_)+'.jpg')
                        #utils.images_on_grid(gen_image_G012_, args.merge_nh, args.merge_nw, filename)

def train_op_fn(loss, trainables, which_opt, learning_rate, global_step):
    # determine which optimizer to use
    if which_opt=='ADAGRAD':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif which_opt=='ADADELTA':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
    elif which_opt=='ADAM':
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.6) #, beta2=0.999, epsilon=0.1)
    elif which_opt=='RMSPROP':
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    elif which_opt=='MOM':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    else:
        raise ValueError('Invalid optimization algorithm')
    
    # define backpropagation flow
    grads = tf.gradients(loss, trainables)
    grad_and_vars = zip(grads, trainables)
    train_op = optimizer.apply_gradients(grad_and_vars, global_step=global_step)
    # add summaries of gradients
    for grad, var in list(grad_and_vars):
        tf.histogram_summary(var.op.name+'/gradient', grad)

    return train_op

if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    main(args)
    print('Training ends')
