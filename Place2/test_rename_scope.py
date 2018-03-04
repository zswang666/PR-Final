import tensorflow as tf

import models.SGAN as network
from data_loader import indoor_input as data_input 

load_ckpt_path = './vgg16_pretrained/encoder-32000.ckpt'
DI = data_input('./data/eval_indoor',
                 2, 224, True, True)

image = tf.placeholder(tf.float32, [2,224,224,3], 'image_ph')

E_eps = network.encoder(image, n_classes=10)
pred = E_eps['predictions']

restorer = tf.train.Saver()

with tf.Session() as sess:
    restorer.restore(sess, load_ckpt_path)
    
    for i in range(2):
        image_in, label_in = DI.next_batch()
        feed_dict = {image: image_in}
        pred_ = sess.run(pred, feed_dict=feed_dict)
        print(pred_) 
