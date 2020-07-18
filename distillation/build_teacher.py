import tensorflow as tf
import numpy as np


with tf.variable_scope("teacher"):
    input_image = tf.placeholder(dtype=tf.float32, shape=[10, 224, 224, 3], name="input")
    conv1 = tf.layers.conv2d(inputs=input_image, filters=64, kernel_size=[3, 3], padding='same')
    conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=[3, 3], padding='same')
    conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=[3, 3], padding='same')
    conv4 = tf.layers.conv2d(conv3, filters=64, kernel_size=[3, 3], padding='same')
    conv5 = tf.layers.conv2d(conv4, filters=128, kernel_size=[3, 3], padding='same')
    conv6 = tf.layers.conv2d(conv5, filters=128, kernel_size=[3, 3], padding='valid')
    conv7 = tf.layers.conv2d(conv6, filters=128, kernel_size=[3, 3], padding='valid')
    conv8 = tf.layers.conv2d(conv7, filters=128, kernel_size=[3, 3], padding='valid')
    shape = int(np.prod(conv8.get_shape()[1:]))
    flat = tf.reshape(conv8, [-1, shape])
    fc1 = tf.layers.dense(flat, units=100)
    fc2 = tf.layers.dense(fc1, units=10, name="logit")

image = np.ones(shape=[10, 224, 224, 3])

with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.save(sess,  "./model/teacher")

    print(sess.run(fc2, feed_dict={input_image: image}))
