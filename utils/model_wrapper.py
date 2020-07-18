"""Model warpper for easier graph manipulation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops


FLAGS = tf.app.flags.FLAGS


class Model(object):
    """The model wraper make it easier to do some operation on a tensorflow model"""

    def __init__(self, sess):
        self.sess = sess
        self.g = self.sess.graph
        self._param_data = {}
        self._variable_prunable = {}
        self.flops = {}
        self.fathers = {}  # the input op of an op
        self.children = {}  # the input op of an op
        self.data_format = self.get_data_format()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

    def get_data_format(self):
        with self.g.as_default():
            for operation in self.g.get_operations():
                if operation.type == 'Conv2D':
                    return operation.get_attr('data_format').decode("utf-8")
        return 'NHWC'

    def get_operations_by_type(self, op_type='Conv2D'):
        with self.g.as_default():
            operations = []
            for i in self.g.get_operations():
                if i.type == op_type:
                    operations.append(i)
        return operations

    def get_outputs_by_ops(self, operations=None):
        with self.g.as_default():
            outputs = []
            for i in operations:
                outputs.append(self.get_output_by_op(i))
            return outputs

    def get_output_by_op(self, op):
        with self.g.as_default():
            # op.outputs will return a tensor(=g.get_tensor_by_name("op.name"+":0"))
            output = op.outputs
            try:
                assert len(output) == 1, 'the length of output should be 1'
            except AssertionError as error:
                tf.logging.error(error)
            return output[0]

    def get_input_by_op(self, op):
        # return input op
        with self.g.as_default():
            inputs = op.inputs
            real_inputs = []
            for inp in inputs:
                if not 'read' in inp.name and 'Const' not in inp.name:
                    real_inputs.append(inp)
            if len(real_inputs) != 1:
                print(real_inputs)
            try:
                assert (len(real_inputs) == 1), 'the number of real inputs of {} should be 1'.format(op.name)
            except AssertionError as error:
                tf.logging.error(error)
            return real_inputs[0]

    def get_var_by_op(self, op):
        """ get the weights of an operation (k_h,k_w,c,n)"""
        with self.g.as_default():
            if op.name in self._param_data:
                return self._param_data[op.name]
            else:
                input_tensor = op.inputs
                try:
                    assert (len(input_tensor) == 2), 'the number of inputs of {} should be 2'.format(op.name)
                except AssertionError as error:
                    tf.logging.error(error)
                w_tensor = None
                for w_tensor in input_tensor:
                    if 'weight' in w_tensor.name or 'bias' in w_tensor.name:
                        break
                try:
                    assert (w_tensor != 2), 'the "weight" or "bias" must be included in the name of var'
                except AssertionError as error:
                    tf.logging.error(error)
                w_name = w_tensor.name.split('/read:0')[0]
                value_var = self.sess.run(w_tensor)
                self._param_data[op.name] = value_var
                return value_var

    def param_shape(self, op):
        w_var = self.get_var_by_op(op)
        return np.shape(w_var)

    def get_conv_def(self, op):
        """ get the definition of an operation which contains the following information:
          `ksizes`: kernel sizes
          `padding`: paddings
          `h`: the kernel of height
          `w`: the kernel of weight
          `c`: channels
          `n`: output channels
        """
        with self.g.as_default():
            definition = dict()
            definition['padding'] = op.get_attr('padding')
            definition['strides'] = op.get_attr('strides')
            s = self.param_shape(op)
            definition['ksizes'] = [1, s[0], s[1], 1]
            definition['h'] = s[0]
            definition['w'] = s[1]
            definition['c'] = s[2]
            definition['n'] = s[3]
            return definition

    def output_width_height(self, name):
        """the height and weight """
        # definition = dict()
        # definition["height"] = self.g.get_tensor_by_name(name).shape[1].value
        # definition["width"] = self.g.get_tensor_by_name(name).shape[2].value
        return self.g.get_tensor_by_name(name).shape[1].value, self.g.get_tensor_by_name(name).shape[2].value

    def output_channels(self, name):
        return self.g.get_tensor_by_name(name).shape[3].value

    def compute_layer_flops(self, op):
        with self.g.as_default():
            opname = op.name
            if opname in self.flops:
                flops = self.flops[opname]
            else:
                flops = tf_ops.get_stats_for_node_def(self.g, op.node_def, 'flops').value
                flops = flops / FLAGS.batch_size
                self.flops[opname] = flops
        return flops

    def get_add_if_op_is_first_after_resblock(self, op):
        """ check whether the input operation is first layer after sum in a resual branch.
        Args: 'op' an operation
        Return: the name of the Add operation is the last of a residual block
        conv   \
        \      \
        add <--
        \
        conv <<---this op
        \
        """
        curr_op = op
        is_first = True
        while True:
            curr_op = self.get_input_by_op(curr_op).op
            if curr_op.type == 'DepthwiseConv2dNative' or curr_op.type == 'Conv2D':
                is_first = False
                break
            if curr_op.type == 'Add':
                break
        if is_first:
            return curr_op.outputs[0]
        return None

    @classmethod
    def get_add_if_op_is_last_in_resblock(cls, op):
        """ check whether the input operation is last layer before sum in a resual branch.
        Args:
          'op': an operation
        Return:
          the name of the Add operation is the last of a residual block

                   relu    \
                    \      \
        this op-->>conv    \
                    \      \
                    add <--
                    \
                    conv
                    \
        """
        curr_op = op
        is_last = True
        while True:
            next_ops = curr_op.outputs[0].consumers()
            go_on = False
            for curr_op in next_ops:
                if curr_op.type in ['Relu', 'FusedBatchNorm', 'DepthwiseConv2dNative', 'MaxPool', 'Relu6']:
                    go_on = True
                    break
            if go_on:
                continue
            if curr_op.type == 'Add':
                break
            is_last = False
            break
        if is_last:
            return curr_op.outputs[0]
        return None

    def is_weight_prunable(self, conv):
        """ if the op's input channels can be pruned"""
        conv_name = conv.name
        weight_prunable = True
        tem = conv
        while True:
            tem = self.get_input_by_op(tem).op
            if tem.type in ['Relu', 'FusedBatchNorm', 'MaxPool', 'BiasAdd', 'Identity', 'Relu6']:
                continue
            if tem.type == 'Conv2D' or tem.type == 'DepthwiseConv2dNative':
                break
            weight_prunable = False
            break
        if weight_prunable:
            self.fathers[conv_name] = tem.name
            self.children[tem.name] = conv_name
        else:
            # if not ,this conv is orphan
            self.fathers[conv_name] = None
            self.children[tem.name] = None
        self._variable_prunable[conv_name] = weight_prunable
        return weight_prunable
