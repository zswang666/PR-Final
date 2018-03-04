from __future__ import print_function

import tensorflow as tf
import sys

import models.encoder as old_net
import models.SGAN as new_net

old_network_ckpt = './checkpoint/vgg16_encoder20170106-150113.ckpt-32000'
new_network_save_path = './vgg16_pretrained/encoder-32000.ckpt'
new_scope = 'E/'

image_ph = tf.placeholder(tf.float32, [3, 224, 224, 3], 'image_ph')

old_network = old_net.vgg16_finetune(image_ph, n_classes=10) # no outest scope
new_network = new_net.encoder(image_ph, 'E', n_classes=10) # outest scope = 'E/'

new_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'E/')
old_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for i in range(len(new_vars)):
    try:
        old_vars.remove(new_vars[i])
    except:
        print('No such vars in old to be excluded')
# now new_vars contains E/newworkA/, and old_vars contains networkA/

#new_vars = [None]*len(old_vars)
ass = [None]*len(old_vars)
for i in range(len(old_vars)):
    #new_vars[i] = tf.identity(old_vars[i], name=new_scope+old_vars[i].op.name)
    ass[i] = new_vars[i].assign(old_vars[i])
    print(old_vars[i].op.name,' --> ',new_vars[i].op.name)
#sys.exit()

with tf.control_dependencies(ass):
    rename = tf.no_op('rename')

init_new = tf.initialize_variables(new_vars) 

restorer = tf.train.Saver(old_vars)
saver = tf.train.Saver(new_vars)

with tf.Session() as sess:
    sess.run(init_new)
    restorer.restore(sess, old_network_ckpt)
    
    sess.run(rename)

    saver.save(sess, new_network_save_path)
