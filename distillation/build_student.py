import tensorflow as tf
import numpy as np


with tf.variable_scope("student"):
    input_label = tf.placeholder(dtype=tf.float32, shape=[10, 10], name="label")
    input_image = tf.placeholder(dtype=tf.float32, shape=[10, 224, 224, 3], name="input")
    conv1 = tf.layers.conv2d(inputs=input_image, filters=64, kernel_size=[3, 3], padding='same')
    conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=[3, 3], padding='same')
    conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=[3, 3], padding='same')
    shape = int(np.prod(conv3.get_shape()[1:]))
    flat = tf.reshape(conv3, [-1, shape])
    fc1 = tf.layers.dense(flat, units=100)
    fc2 = tf.layers.dense(fc1, units=10, name="logit")
    probability = tf.nn.softmax(fc2)
    loss = tf.losses.softmax_cross_entropy(input_label, fc2)
    print(input_label)

image = np.ones(shape=[10, 224, 224, 3])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.save(sess,  "./student/student")
    print(sess.run(probability, feed_dict={input_image: image}))
