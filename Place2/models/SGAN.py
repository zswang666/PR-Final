from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.ops import *

slim = tf.contrib.slim
 
def encoder(images, scope='E', n_classes=10,
            is_training=True, dropout_keep_prob=0.5, reuse=False):
    # Encoder is a fine-tuned 10-class VGG-16 model
    with tf.variable_scope(scope, reuse=reuse):
        endpoints = {}
        #### E0
        # 2 conv + 1 max_pool
        net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        #### E1
        # 2 conv + 1 max_pool
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # 3 conv + 1 max_pool
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        #### E2
        # 3 conv + 1 max_pool
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # 3 conv + 1 max_pool
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        # fc6
        net = slim.flatten(net)
        net = slim.fully_connected(net, 4096, scope='fc6')#
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        # fc7
        net = slim.fully_connected(net, 4096, scope='fc7')#
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        # fc8a (output layer)
        net = slim.fully_connected(net, n_classes, activation_fn=None, scope='fc8a')
        endpoints['logits'] = net
        # predictions
        net = slim.softmax(net, scope='predictions')
        #net = tf.squeeze(net)
        endpoints['predictions'] = net

    return endpoints

def E0(E0_in, scope='E', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        # 2 conv + 1 max_pool
        net = slim.repeat(E0_in, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
    return net

def E1(E1_in, scope='E', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        # 2 conv + 1 max_pool
        net = slim.repeat(E1_in, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # 3 conv + 1 max_pool
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
    return net

def E2(E2_in, scope='E', reuse=False):
    is_training = False
    dropout_keep_prob = 1.0
    n_classes = 10
    with tf.variable_scope(scope, reuse=reuse):
        # 3 conv + 1 max_pool
        net = slim.repeat(E2_in, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # 3 conv + 1 max_pool
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        # fc6
        net = slim.flatten(net)
        net = slim.fully_connected(net, 4096, scope='fc6')#
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        # fc7
        net = slim.fully_connected(net, 4096, scope='fc7')#
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        # fc8a (output layer)
        net = slim.fully_connected(net, n_classes, activation_fn=None, scope='fc8a')
    return net

def discriminator0(input_d0, z0_dim, scope='Discriminator0', reuse=False):
    endpoints = {}
    #Input will be 28*28*256
    print("~~~~~~~~~~~~Discriminator0~~~~~~~~~~~~~~~~~~")
    with tf.variable_scope(scope,reuse=reuse):
        #Convolve to 14*14*512
        net = slim.conv2d(input_d0, 512,[3,3],stride=2,padding = 'SAME', activation_fn=None, scope='conv1')
        net = lrelu(net)
        print('discriminator0/conv1',net.get_shape())

        #Convolve to 7*7*512
        net = slim.conv2d(net, 512, [3,3], stride=2, padding='SAME', activation_fn=None, scope='conv2')
        net = lrelu(net)
        print('discriminator0/conv2',net.get_shape())

        #The channel of fc layer is analogy to the final fc layer of vgg16 (4096)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 4096, activation_fn=None, scope='fc_shared')
        net= lrelu(net)
        print('discriminator0/fc_shared',net.get_shape())

        endpoints['fc_shared']=net

        net = slim.fully_connected(endpoints['fc_shared'], z0_dim, activation_fn=None, scope='fc_recon')
        print('discriminator0/fc_recon',net.get_shape())
        net = tf.nn.sigmoid(net)
        endpoints['fc_recon']=net

        net = slim.fully_connected(endpoints['fc_shared'], 1, activation_fn=None, scope='fc_adv')
        print('discriminator0/fc_adv',net.get_shape())
        endpoints['fc_adv_logits'] = net
        net = tf.nn.sigmoid(net)
        endpoints['fc_adv'] = net

    return endpoints['fc_adv_logits'], endpoints['fc_recon']

def discriminator1(input_d1, z1_dim, scope='Discriminator1', reuse=False):
    endpoints = {}
    #Input will be 112*112*64
    print("~~~~~~~~~~~Discriminator1~~~~~~~~~~~~~~~~~~~~~")
    with tf.variable_scope(scope,reuse=reuse):
        #Convolve to 56*56*128
        net = slim.conv2d(input_d1, 128, [3,3], stride=2, padding = 'SAME', activation_fn=None, scope='conv1')
        net = lrelu(net)
        print("Discriminator1/conv1",net.get_shape())
        
        #Convolve to 28*28*256
        net = slim.conv2d(net, 256, [3,3], stride=2, padding = 'SAME', activation_fn=None, scope='conv2')
        net = lrelu(net)
        print("Discriminator1/conv2",net.get_shape())

        #Convolve to 14*14*512
        net = slim.conv2d(net, 512, [3,3], stride=2, padding = 'SAME', activation_fn=None, scope='conv3')
        net = lrelu(net)
        print("Discriminator1/conv3",net.get_shape())

        #Convolve to 7*7*512
        net = slim.conv2d(net, 512, [3,3], stride=2, padding = 'SAME', activation_fn=None, scope='conv4')
        net = lrelu(net)
        print("Discriminator1/conv4",net.get_shape())

        #Fully-connected to 4096
        net = slim.flatten(net)
        net = slim.fully_connected(net, 4096, activation_fn=None, scope='fc_shared')
        net = lrelu(net)
        print("Discriminator1/fc_shared",net.get_shape())

        endpoints['fc_shared']=net

        net = slim.fully_connected(endpoints['fc_shared'], z1_dim, activation_fn=None, scope='fc_recon')
        print("Discriminator1/fc_recon",net.get_shape())
        net = tf.nn.sigmoid(net)
        endpoints['fc_recon']=net

        net = slim.fully_connected(endpoints['fc_shared'], 1, activation_fn=None, scope='fc_adv')
        print("Discriminator1/fc_adv",net.get_shape())
        endpoints['fc_adv_logits'] = net
        net = tf.nn.sigmoid(net)
        endpoints['fc_adv'] = net

    return endpoints['fc_adv_logits'], endpoints['fc_recon']

def discriminator2(input_d2, z2_dim, scope='Discriminator2', reuse=False):
    endpoints = {}
    #Input will be 224*224*3
    print("~~~~~~~~~~Discriminator2~~~~~~~~~~~~~~~~")
    with tf.variable_scope(scope, reuse=reuse):
        #Convolution to 112*112*64
        net = slim.conv2d(input_d2, 64, [3,3], stride=2, padding = 'SAME', activation_fn=None, scope='conv1')
        net = lrelu(net)
        print("Discriminator2/conv1",net.get_shape())

        #Convolution to 56*56*128
        net = slim.conv2d(net, 128, [3,3], stride=2, padding = 'SAME', activation_fn=None, scope='conv2')
        net = lrelu(net)
        print("Discrimninator2/conv2",net.get_shape())

        #Convolution to 28*28*256
        net = slim.conv2d(net, 256, [3,3], stride=2, padding = 'SAME', activation_fn=None, scope='conv3')
        net = lrelu(net)
        print("Discriminator2/conv3",net.get_shape())

        #Convolution to 14*14*512
        net = slim.conv2d(net, 512, [3,3], stride=2, padding = 'SAME', activation_fn=None, scope='conv4')
        net = lrelu(net)
        print("Discriminator2/conv4",net.get_shape())

        #Convolution to 7*7*512
        net = slim.conv2d(net, 512, [3,3], stride=2, padding = 'SAME', activation_fn=None, scope='conv5')
        net = lrelu(net)
        print("Discriminator2/conv5",net.get_shape())

        #Fully-connected to 4096
        net = slim.flatten(net)
        net = slim.fully_connected(net, 4096, activation_fn=None, scope='fc_shared')
        net = lrelu(net)
        print("Discriminator2/fc_shared",net.get_shape())

        endpoints['fc_shared']=net

        net = slim.fully_connected(endpoints['fc_shared'], z2_dim, activation_fn=None, scope='fc_recon')
        print("Discriminator2/fc_recon",net.get_shape())
        net = tf.nn.sigmoid(net)
        endpoints['fc_recon']=net

        net = slim.fully_connected(endpoints['fc_shared'], 1, activation_fn=None, scope='fc_adv')
        print("Discriminator2/fc_adv",net.get_shape())
        endpoints['fc_adv_logits'] = net
        net = tf.nn.sigmoid(net)
        endpoints['fc_adv'] = net

    return endpoints['fc_adv_logits'], endpoints['fc_recon']

def generator0(z0, y, scope='Generator0', reuse=False):
    #The first block of generator
    print("~~~~~~~~~~Generator0~~~~~~~~~~~~~~~~~~")
    fc_bn1 = batch_norm(name='fc_bn1')
    bn1 = batch_norm(name='bn1')
    bn2 = batch_norm(name='bn2')

    with tf.variable_scope(scope,reuse=reuse):
        input_vec = tf.concat(1,[z0,y],name='concat_input')
        print('concat_input',input_vec.get_shape())
        #Convert z to tensorblock
        with tf.variable_scope('fc1',reuse=reuse):
            net = slim.fully_connected(input_vec, 512*7*7 , activation_fn=None, scope='fc1')
            net=lrelu(net)
            print('fc1',net.get_shape())
            net = tf.reshape(net,[-1,7,7,512])
            print('fc1',net.get_shape())

            #batch normalization of fully-connected
            net = fc_bn1(net)
            print('bn1',net.get_shape())

        #Deconvolution to 14*14*512
        with tf.variable_scope('deconv1',reuse=reuse):
            #Deconvolution + batch norm + lrelu
            net = slim.conv2d_transpose(net,512,[3,3],stride=2,activation_fn=None,
                        padding='SAME',scope='deconv1')
            print('deconv1/deconv1',net.get_shape())
            net = bn1(net)
            print('deconv1/bn1',net.get_shape())
            net = lrelu(net,name='relu1')
            print('deconv1/relu1',net.get_shape())

            #The following two 'SAME' convolution
            #[TODO] Add batch normalization to the net
            net = slim.conv2d(net, 512, [3,3], activation_fn=None, scope='conv1')
            net = lrelu(net)
            net = slim.conv2d(net, 512, [3,3], activation_fn=None, scope='conv2')
            net = lrelu(net)
            print('deconv1/conv1,2',net.get_shape())

        #Deconvolution to 28*28*256
        with tf.variable_scope('deconv2',reuse=reuse):
            net = slim.conv2d_transpose(net,512,[3,3],stride=2,activation_fn=None,
                        padding='SAME',scope='deconv2')
            print('deconv2/deconv2',net.get_shape())

            net = bn2(net)
            print('deconv2/bn2',net.get_shape())

            net = lrelu(net)
            print('deconv2/relu2',net.get_shape())

            #The following two conv to shrink channel
            #[TODO] Add batch normalization to the net
            net = slim.conv2d(net,512,[3,3], activation_fn=None, scope='conv1')
            net = lrelu(net)
            net = slim.conv2d(net,256,[3,3], activation_fn=None, scope='conv2')
            net = lrelu(net)
            print('deconv2/conv1,2',net.get_shape())
        
    return net
    

def generator1(z1, input_g1, scope='Generator1', reuse=False):
    #The second block of generator
    fc_bn2 = batch_norm(name='fc_bn2')
    bn3 = batch_norm(name='bn3')
    bn4 = batch_norm(name='bn4')
    print("~~~~~~~~~~~Generator1~~~~~~~~~~~~~~~~~~")
    with tf.variable_scope(scope,reuse=reuse):
        #convert z1 to desire shape
        with tf.variable_scope('fc2',reuse=reuse):
            net = slim.fully_connected(z1, 1*28*28, activation_fn=None, scope= 'fc2')
            net = lrelu(net)
            print('fc2',net.get_shape())
            net = tf.reshape(net,[-1,28,28,1])
            print('fc2',net.get_shape())
            net = fc_bn2(net)
            print('bn2',net.get_shape())

        input_vec = tf.concat(3,[input_g1,net],name='concat_input')
        print('concat_input',input_vec.get_shape())

        #Deconvolution to 56*56*128
        with tf.variable_scope('deconv3',reuse=reuse):
            net = slim.conv2d_transpose(input_vec,257,[3,3],stride=2,activation_fn=None,
                        padding='SAME',scope='deconv3')
            print('deconv3/deconv3',net.get_shape())
            net = bn3(net)
            print('deconv3/bn3',net.get_shape())
            net = lrelu(net)
            print('deconv3/lrelu3',net.get_shape())

            #The following two conv to shrink channel
            #[TODO] Add batch normalization to the net
            net = slim.conv2d(net,257,[3,3],activation_fn=None ,scope='conv1')
            net = lrelu(net)
            net = slim.conv2d(net,128,[3,3],activation_fn=None ,scope='conv2')
            net = lrelu(net)
            print('deconv3/conv1,2',net.get_shape())

    with tf.variable_scope(scope,reuse=reuse):
        #Deconvoltion to 112*112*64
        with tf.variable_scope('deconv4',reuse=reuse):
            net = slim.conv2d_transpose(net,128,[3,3],stride=2,activation_fn=None, 
                        padding='SAME',scope='deconv4')
            print('deconv4/deconv4',net.get_shape()) 
            net = bn4(net)
            print('deconv4/bn4',net.get_shape())
            net = lrelu(net)
            print('deconv4/lrelu4',net.get_shape())

            #The following two conv to shrink channel
            #[TODO] Add batch normalization to the net
            net = slim.conv2d(net,128,[3,3], activation_fn=None, scope='conv1')
            net = lrelu(net)
            net = slim.conv2d(net,64,[3,3], activation_fn=None, scope='conv2')
            net = lrelu(net)
            print('deconv4/conv1,2',net.get_shape()) 

    return net

def generator2(z2, input_g2, scope='Generator2', reuse=False):
    #The third block of generator
    fc_bn3 = batch_norm(name='fc_bn3')
    bn5 = batch_norm(name='bn3')
    bn6 = batch_norm(name='bn4')
 
    print("~~~~~~~~~~~Generator2~~~~~~~~~~~~~~~~~~~~")
    with tf.variable_scope(scope,reuse=reuse):
        #convert z2 to desire shape
        with tf.variable_scope('fc3',reuse=reuse):
            net = slim.fully_connected(z2, 1*112*112,activation_fn=None, scope='fc3')
            net = lrelu(net)
            print('fc3',net.get_shape())
            net = tf.reshape(net,[-1,112,112,1])
            print('fc3',net.get_shape())
            net = fc_bn3(net)
            print('bn3',net.get_shape())
        
        input_vec = tf.concat(3,[input_g2,net],name='concat_input')
        print('concat_input',input_vec.get_shape())
        #Deconvolution to 224*224*3
        with tf.variable_scope('deconv5',reuse=reuse):
            net = slim.conv2d_transpose(input_vec,65,[3,3],stride=2,activation_fn=None,
                        padding='SAME',scope='deconv5')
            print('deconv5/deconv5',net.get_shape())
            net = bn5(net)
            print('deconv5/bn5',net.get_shape())
            net = lrelu(net)
            print('deconv4/relu5',net.get_shape())

            #The following two conv to shrink channel
            #[TODO] Add batch normalization to the net
            net = slim.conv2d(net,65,[3,3], activation_fn=None, scope='conv1')
            net = lrelu(net)
            net = slim.conv2d(net,3,[3,3], activation_fn=None, scope='conv2')
            net = lrelu(net)
            print('deconv5/conv5',net.get_shape())
            net = tf.nn.tanh(net)

    return net

def z0_sampler(batch_size, dim):
    z_mean = 0
    z_std = 1
    z = tf.random_normal([batch_size, dim], 
                          z_mean,
                          z_std,
                          dtype=tf.float32,
                          name='z0_sampler')

    return z

def z1_sampler(batch_size, dim):
    z_mean = 0
    z_std = 1
    z = tf.random_normal([batch_size, dim], 
                          z_mean,
                          z_std,
                          dtype=tf.float32,
                          name='z1_sampler')

    return z

def z2_sampler(batch_size, dim):
    z_mean = 0
    z_std = 1
    z = tf.random_normal([batch_size, dim], 
                          z_mean,
                          z_std,
                          dtype=tf.float32,
                          name='z2_sampler')

    return z

