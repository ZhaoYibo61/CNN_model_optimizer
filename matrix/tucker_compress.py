import os
import tensorly
import numpy as np
from utils import VBMF
import tensorflow as tf
from utils import model_wrapper
from utils.tools import load_checkpoint_model, load_pb_model
from tensorly.decomposition import partial_tucker

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.DEBUG)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("cp_tucker_compress_path", "model", "the path for pruning_mode")


class TuckerCompress(object):
    def __init__(self, model, new_model_name, new_model_dir):
        self._model = model
        self.new_model_name = new_model_name
        self.new_model_dir = new_model_dir
        self.convs = self._model.get_operations_by_type()
        self.new_weight = {}
        self.old_weight = {}
        self.old_input_tensor = {}
        self.old_output_tensor = {}
        self.new_input_tensor = {}
        self.new_output_tensor = {}
        self._build()

    def _build(self):
        self.get_old_input_tensor()
        self.get_old_output_tensor()
        self.get_new_weight()

    @staticmethod
    def estimate_ranks(weights):
        """
        :param weights: matrix
        :return: the ranks
        """
        print("原始模型参数：", np.shape(weights))
        unfold_0 = tensorly.base.unfold(weights, 0)
        unfold_1 = tensorly.base.unfold(weights, 1)
        try:
            _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
            _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
        except MemoryError:
            tf.logging.info('this conv layer can\'t be decomposition because of MemoryError')
            return None

        if isinstance(diag_0, int):
            return None
        ranks = [diag_0.shape[0], diag_1.shape[1]]
        if ranks[0] == 0 or ranks[1] == 0:
            return None
        return ranks

    def get_weight(self, weights):
        """
        :param weights:
        :return: tucker decomposition
        """
        weight = np.transpose(np.squeeze(weights), (3, 2, 0, 1))
        ranks = self.estimate_ranks(weight)
        if ranks is None:
            return None
        print("new ranks : ({})".format(ranks))
        core, [last, first] = partial_tucker(weight, modes=[0, 1], ranks=ranks, init='svd')
        weight = []
        weight_1 = first[np.newaxis, np.newaxis, :, :]
        weight.append(weight_1)
        weight_2 = np.transpose(core, (2, 3, 1, 0))
        weight.append(weight_2)
        weight_3 = np.transpose(last, (1, 0))[np.newaxis, np.newaxis, :, :]
        weight.append(weight_3)
        return weight

    def get_new_weight(self):
        for item in self.convs:
            print(item.name)
            weight = self._model.get_var_by_op(item)
            if weight.shape[0] == 1 and weight.shape[1] == 1:
                self.new_weight[item.name] = None
                continue
            print(weight.shape)
            self.old_weight[item.name] = weight
            weights = self.get_weight(weight)
            if weights is None:
                self.new_weight[item.name] = None
                continue
            self.new_weight[item.name] = weights

    def get_old_input_tensor(self):
        for item in self.convs:
            input_tensor = self._model.get_input_by_op(item)
            self.old_input_tensor[item.name] = input_tensor

    def get_old_output_tensor(self):
        for item in self.convs:
            self.old_output_tensor[item.name] = self._model.get_output_by_op(item)

    def initialize_uninitialized(self):
        global_vars = tf.global_variables()
        is_not_initialized = self._model.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        if len(not_initialized_vars):
            self._model.sess.run(tf.variables_initializer(not_initialized_vars))

    def compress(self):
        with self._model.g.as_default():

            for i, item in enumerate(self.convs):
                print("i:", i)
                weights = self.new_weight[item.name]
                if weights is None:
                    continue

                input_tensor = self.old_input_tensor[item.name]
                definition = self._model.get_conv_def(item)
                k_h, k_w, in_c, out_c = definition['h'], definition['w'], definition['c'], definition['n']
                pad = definition['padding']
                stride = definition['strides']

                with tf.variable_scope(item.name + "_1"):
                    print(weights[0].shape)
                    w1 = tf.get_variable("weight1", [1, 1, in_c, weights[0].shape[3]],
                                         initializer=tf.constant_initializer(weights[0]))
                    conv1 = tf.nn.conv2d(input=input_tensor, filter=w1, strides=[1, 1, 1, 1], padding='SAME',
                                         name="conv1")
                    print(weights[1].shape)
                    w2 = tf.get_variable("weight2", [k_h, k_w, weights[1].shape[2], weights[1].shape[3]],
                                         initializer=tf.constant_initializer(weights[1]))
                    conv2 = tf.nn.conv2d(input=conv1, filter=w2, strides=stride, padding=pad, name="conv2")

                    print(weights[2].shape)
                    w3 = tf.get_variable("weight3", [1, 1, weights[2].shape[2], out_c],
                                         initializer=tf.constant_initializer(weights[2]))
                    conv3 = tf.nn.conv2d(input=conv2, filter=w3, strides=[1, 1, 1, 1], padding='SAME',
                                         name="conv3")

                consumers = self.old_output_tensor[item.name].consumers()
                for consumer in consumers:
                    consumer._update_input(0, conv3)

            self.initialize_uninitialized()

        self.save_model()

    def save_model(self):
        """ save the current model to the `FLAGS.channel_pruned_path`"""
        with self._model.g.as_default():
            saver = tf.train.Saver()
            is_exists = os.path.exists(self.new_model_dir)
            if not is_exists:
                os.makedirs(FLAGS.cp_tucker_compress_path)
            saver.save(self._model.sess, self.new_model_dir + "/" + self.new_model_name)
            tf.logging.info('saved pruned model to {}'.format(self.new_model_dir))


if __name__ == "__main__":
    sess = load_pb_model('./model/model_vgg_16.pb')
    model = model_wrapper.Model(sess=sess)
    tucker_model = TuckerCompress(model=model, new_model_name="vgg", new_model_dir="./model")
    tucker_model.compress()
    # tf.summary.FileWriter("./log/", sess.graph)
