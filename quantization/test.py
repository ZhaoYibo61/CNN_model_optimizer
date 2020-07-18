import numpy as np
import tensorflow as tf

# var = tf.get_variable(name='var', dtype=tf.float32, initializer=tf.constant([1.2, 2.2, 3.2]))
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
#
# saver = tf.train.Saver()
# saver.save(sess, './model')
# print("asdfasdfsadf", sess.run(var))
# sess.close()

ckpt = tf.train.get_checkpoint_state('./')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')

op = tf.get_default_graph().get_operation_by_name('var')
assign = tf.get_default_graph().get_operation_by_name('var/Assign')
const = tf.get_default_graph().get_operation_by_name('Const')
op._set_attr('dtype', tf.AttrValue(type=tf.float16.as_datatype_enum))
assign._set_attr('T', tf.AttrValue(type=tf.float16.as_datatype_enum))
const._set_attr('T', tf.AttrValue(type=tf.float16.as_datatype_enum))

with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("asdadddddddddddddd", sess.run(tf.get_default_graph().get_tensor_by_name('var:0')))



