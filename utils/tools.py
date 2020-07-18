import os
import cv2
import numpy as np
import tensorflow as tf


def build_lables(file, sess, input_name="resnet_v1_50/input:0", output_name="resnet_v1_50/predictions/Softmax:0"):
    """
    :param output_name:
    :param input_name:
    :param file: the dir of picture
    :param sess: session
    :return: the label
    """
    input_image = sess.graph.get_tensor_by_name(input_name)
    output_tensor = sess.graph.get_tensor_by_name(output_name)
    batches = []
    file_name = os.listdir(file)
    filelist = [os.path.join(file, name) for name in file_name]
    i = 0
    while i < len(filelist):
        tag = 0
        batch = []
        while True:
            if tag == 10:
                break
            png = cv2.imread(filelist[i])
            png = cv2.resize(png, (224, 224))
            batch.append(png)
            tag += 1
            i += 1
        batches.append(batch)
    one_hot = []
    for tem in range(80):
        result = np.squeeze(sess.run([output_tensor], feed_dict={input_image: batches[tem]}))
        one = [np.argmax(x) for x in result]
        one_hot.extend(one)
    one_hot = np.asarray(one_hot)
    np.savetxt("label.txt", one_hot.astype(int), fmt='%d')
    return one_hot


def props_to_onehot(props):
    """对于二维矩阵可以转成one_hot编码"""
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b


def load_checkpoint_model(checkpoint_dir='../model/'):
    """
    :param checkpoint_dir:
    :return: session
    """
    graph = tf.Graph()
    with graph.as_default():
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # 通过检查点文件锁定最新的模型
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
    sess = tf.Session(graph=graph)
    saver.restore(sess, ckpt.model_checkpoint_path)
    return sess


def load_pb_model(pd_dir="../model/model.pb"):
    """
    :param pd_dir:
    :return: sess
    """
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(pd_dir, 'rb') as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')  # imports graph_def into the current default Graph
    sess = tf.Session(graph=graph)
    return sess


def save_pb(sess, output_node_list, pb_dir):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    output_graph_def = convert_variables_to_constants(sess, sess.graph_def,
                                                      output_node_names=output_node_list)
    with tf.gfile.FastGFile(pb_dir, mode='wb') as f:
        f.write(output_graph_def.SerializeToString())


def run(sess=None):
    with sess.graph.as_default():
        y_label = tf.constant(value=np.ones(shape=(10, 1000)), dtype=tf.float32)
        x_image = np.ones(shape=(10, 224, 224, 3))

        softmax = tf.get_default_graph().get_tensor_by_name("resnet_v1_50/predictions/Softmax:0")
        input = tf.get_default_graph().get_tensor_by_name("resnet_v1_50/input:0")
        loss = tf.reduce_sum(softmax - y_label)
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        grads_origin = optimizer.compute_gradients(loss, tf.trainable_variables())
        grads_prune = []
        for grad in grads_origin:
            print(grad)
            grads_prune.append((grad[0] * 0, grad[1]))
        train_op = optimizer.apply_gradients(grads_prune)
        #  train_op = optimizer.minimize(loss)
        tf.summary.FileWriter("./log/", sess.graph)
        sess.run(tf.variables_initializer(optimizer.variables()))
        for i in range(10):
            print("train：{}".format(i))
            a, b = sess.run([grads_prune, train_op], feed_dict={input: x_image})
            #  print(a)
