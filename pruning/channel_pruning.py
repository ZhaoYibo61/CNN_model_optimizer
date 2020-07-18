"""Channel Pruner
--Ref.
A. https://arxiv.org/abs/1707.06168
"""
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import model_wrapper
from collections import OrderedDict
from timeit import default_timer as timer
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from utils.tools import load_checkpoint_model, load_pb_model, save_pb

tf.logging.set_verbosity(tf.logging.DEBUG)
slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('cp_lasso', True,
                            'If True use lasso and reconstruction otherwise prune according to weight magnitude')
tf.app.flags.DEFINE_boolean('cp_quadruple', False, 'Restric the channels after pruning is a mutiple of 4')
tf.app.flags.DEFINE_string('cp_reward_policy', 'accuracy',
                           'If reward_policy equals accuracy, it means learning to achieve guaranted accuracy.')
tf.app.flags.DEFINE_integer('cp_nb_points_per_layer', 10, 'Sample how many point for each layer')
tf.app.flags.DEFINE_integer('cp_nb_batches', 80, 'Input how many bathes data into a model')
tf.app.flags.DEFINE_integer('batch_size', 10, 'batch_size')
tf.app.flags.DEFINE_float('cp_preserve_ratio', 0.7, 'the ratio of preserve')
tf.app.flags.DEFINE_boolean('debug', False, "debug")
tf.app.flags.DEFINE_string("cp_prune_option", "None", "some stratege of prune")
tf.app.flags.DEFINE_string("cp_channel_pruned_path", "model", "the path for pruning_mode")


class ChannelPruner(object):
    """ The Channel Prunner """

    def __init__(self, model, images="D:/Downloads/DIV2K_train_HR", labels="./result.txt",
                 mem_images="resnet_v1_50/input",
                 mem_labels="resnet_v1_50/predictions/Softmax", lbound=0):
        # images为string,表示图片的地址
        # labels为string,表示标签的地址
        self._model = model
        self.data_format = self._model.data_format
        self.images = images
        self.labels = labels
        self.mem_images = self._model.g.get_tensor_by_name(mem_images + ":0")
        self.mem_labels = self._model.g.get_tensor_by_name(mem_labels + ":0")
        self.state = 0  # the index of conv
        self.lbound = lbound
        self.best = -math.inf
        self.bestinfo = []
        self.states = []
        self.names = []  # the name of conv and res_conv_add
        self.thisconvs = self._model.get_operations_by_type()  # 所有的conv的operation
        self.drop_trainable_vars = set([])
        self.currentStates = []
        self.desired_reduce = 0
        self.feats_dict = {}  # 每一个卷积层在所有采样点的集合
        self.points_dict = {}  # 主要两个作用，一个是存储输入数据（batch,0/1），另一个是(batch,name,"x_samples/y_samples"),
        self.desired_preserve = 0
        self.drop_conv = set([])
        self.layer_flops = 0
        self.model_flops = 0
        self.max_reduced_flops = 0
        self.config = tf.ConfigProto()
        self.config.gpu_options.visible_device_list = str(0)
        self.max_strategy_dict = {}
        self.fake_pruning_dict = {}
        self.extractors = {}
        self.__build()

    def __build(self):
        self.__extract_output_of_conv_and_sum()
        self.__create_extractor()
        self.initialize_state()
        self.extract_features()

    def __extract_output_of_conv_and_sum(self):
        """Extract output tensor name of convolution layers and sum layers in a residual block"""
        operations = self._model.get_operations_by_type()
        conv_outputs = [x.outputs[0] for x in operations]
        conv_add_outputs = []
        for conv in conv_outputs:
            conv_add_outputs.append(conv)
            add_output = self._model.get_add_if_op_is_last_in_resblock(conv.op)
            if add_output is not None:
                conv_add_outputs.append(add_output)
        tf.logging.debug('extracted outputs {}'.format(conv_add_outputs))
        self.names = [x.name for x in conv_add_outputs]

    def __create_extractor(self):
        """ create extracters which would be used to extract input of a convolution"""
        with self._model.g.as_default():
            ops = self._model.get_operations_by_type()
            self.extractors = {}
            for op in ops:
                input_op = self._model.get_input_by_op(op)
                if self.data_format == 'NCHW':
                    input_op = tf.transpose(input_op, [0, 2, 3, 1])
                defs = self._model.get_conv_def(op)
                strides = defs['strides']
                extractor = tf.extract_image_patches(input_op, ksizes=defs['ksizes'],
                                                     strides=strides, rates=[1, 1, 1, 1],
                                                     padding=defs['padding'])
                self.extractors[op.name] = extractor

    def initialize_state(self):
        """Initialize state"""
        self.best = -math.inf
        self.bestinfo = []
        allstate = []
        self.state = 0
        while self.state < len(self.thisconvs):
            allstate.append(self.get_state(self.thisconvs[self.state]))
            self.state += 1
        feature_names = ['layer', 'n', 'c', 'H', 'W', 'stride', 'maxreduce', 'layercomp']
        states = pd.DataFrame(allstate, columns=feature_names)
        self.state = 0
        self.states = states / states.max()  # don't know why
        self.layer_flops = np.array(self.states['layercomp'].tolist())
        self.model_flops = self.__compute_model_flops()
        tf.logging.info('The original model flops is {}'.format(self.model_flops))

        self.currentStates = self.states.copy()
        self.desired_reduce = (1 - FLAGS.cp_preserve_ratio) * self.model_flops
        self.desired_preserve = FLAGS.cp_preserve_ratio * self.model_flops
        self.max_strategy_dict = {}  # collection of intilial max [inp preserve, out preserve]
        self.fake_pruning_dict = {}  # collection of fake pruning indices
        for i, conv in enumerate(self.thisconvs):
            if self._model.is_weight_prunable(conv):
                tf.logging.info('current conv ' + conv.name)
                father_conv_name = self._model.fathers[conv.name]
                father_conv = self._model.g.get_operation_by_name(father_conv_name)
                if father_conv.type == 'DepthwiseConv2dNative':
                    if self._model.is_weight_prunable(father_conv):
                        self.max_strategy_dict[self._model.fathers[father_conv_name]][1] = self.lbound
                else:
                    self.max_strategy_dict[father_conv_name][1] = self.lbound
            if not (i == 0 or i == len(self.thisconvs) - 1):
                self.max_strategy_dict[conv.name] = [self.lbound, 1.]
            else:
                self.max_strategy_dict[conv.name] = [1., 1.]
            conv_def = self._model.get_conv_def(conv)
            self.fake_pruning_dict[conv.name] = [[True] * conv_def['c'], [True] * conv_def['n']]

        tf.logging.info('current states:\n {}'.format(self.currentStates))
        tf.logging.info('max_strategy_dict\n {}'.format(self.max_strategy_dict))

    def get_state(self, op):
        """Get state"""
        conv_def = self._model.get_conv_def(op)
        n, c, _, _ = conv_def['n'], conv_def['c'], conv_def['h'], conv_def['w']
        conv = op.name + ":0"
        H, W = self._model.output_width_height(conv)
        stride = conv_def['strides'][1]
        return [self.state, n, c, H, W, stride, 1., self._model.compute_layer_flops(op)]

    def __action_constraint(self, action):
        """constraint action during reinfocement learning search"""
        action = float(action)
        if action > 1.:
            action = 1.
        if action < 0.:
            action = 0.
        # final layer is not prunable
        if self.finallayer():
            return 1
        conv_op = self.thisconvs[self.state]
        prunable = self._model.is_weight_prunable(conv_op)
        if prunable:
            father_opname = self._model.fathers[conv_op.name]
        conv_left = self.__conv_left()
        this_flops = 0
        other_flops = 0
        behind_layers_start = False
        for conv in conv_left:
            curr_flops = self._model.compute_layer_flops(conv)
            if prunable and conv.name == father_opname:
                this_flops += curr_flops * self.max_strategy_dict[conv.name][0]
            elif conv.name == conv_op.name:
                this_flops += curr_flops * self.max_strategy_dict[conv.name][1]
                behind_layers_start = True
            elif behind_layers_start and 'pruned' not in conv.name:
                other_flops += curr_flops * self.max_strategy_dict[conv.name][0] * \
                               self.max_strategy_dict[conv.name][1]
            else:
                other_flops += curr_flops * self.max_strategy_dict[conv.name][0] * \
                               self.max_strategy_dict[conv.name][1]

        self.max_reduced_flops = other_flops + this_flops * action
        if FLAGS.cp_reward_policy != 'accuracy' or self.state == 0:
            return action
        recommand_action = (self.desired_preserve - other_flops) / this_flops
        tf.logging.info('max_reduced_flops {}'.format(self.max_reduced_flops))
        tf.logging.info('desired_preserce {}'.format(self.desired_preserve))
        tf.logging.info('this flops {}'.format(this_flops))
        tf.logging.info('recommand action {}'.format(recommand_action))
        return np.minimum(action, recommand_action)

    def __conv_left(self):
        """Get the left convolutions after pruning so far"""
        conv_ops = self._model.get_operations_by_type()
        conv_left = []
        for conv in conv_ops:
            if conv.name not in self.drop_conv:
                conv_left.append(conv)
        tf.logging.debug('drop conv {}'.format(self.drop_conv))
        tf.logging.debug('conv left {}'.format(conv_left))
        return conv_left

    def __compute_model_flops(self, fake=False):
        """Compute the convolution computation flops of the model"""
        conv_left = self.__conv_left()
        flops = 0.
        for op in conv_left:
            if fake:
                flops += self._model.compute_layer_flops(op) * \
                         self.max_strategy_dict[op.name][0] * self.max_strategy_dict[op.name][1]
            else:
                flops += self._model.compute_layer_flops(op)
        tf.logging.info('The current model flops is {}'.format(flops))
        return flops

    @property
    def model(self):
        """Return the model"""
        return self._model

    @staticmethod
    def picture_read(data_file, batch_size=FLAGS.batch_size, height=224, width=224):
        """
        :param data_file:
        :param batch_size:
        :param height:
        :param width:
        :return:
        """
        file_name = os.listdir(data_file)
        filelist = [os.path.join(data_file, name) for name in file_name]
        file_queue = tf.train.string_input_producer(filelist, shuffle=False)
        reader = tf.WholeFileReader()
        key, value = reader.read(file_queue)
        image = tf.image.decode_png(value)
        image_resize = tf.image.resize_images(image, [height, width])
        image_resize.set_shape([height, width, 3])
        image_resize_batch = tf.train.batch([image_resize], batch_size=batch_size, num_threads=10, capacity=32)
        return image_resize_batch

    def extract_features(self, names=None, init_fn=None, images=None, labels=None):
        """
        Extract feature-maps and do sampling for some convolutions with given images
        Args:
          names: convolution operation names
          init_fn: initialization function
          images: input images
          labels: input lables
        """
        if names is None:
            names = self.names

        def set_points_dict(name, data):
            """ set data to point dict"""
            if name in points_dict:
                raise ValueError("{} is in the points_dict".format(name))
            points_dict[name] = data

        # remove duplicates
        names = list(OrderedDict.fromkeys(names))
        points_dict = dict()
        feats_dict = {}  # 每个卷积层的抽样结果的输出值
        shapes = {}
        #  每个卷积层采样点个数*一个batch的个数=每个batch获取样本点的个数
        nb_points_per_batch = FLAGS.cp_nb_points_per_layer * FLAGS.batch_size
        set_points_dict("nb_points_per_batch", nb_points_per_batch)  # {nb_points_per_batch:10*10 }

        nb_points_total = nb_points_per_batch * FLAGS.cp_nb_batches  # 100*30
        for name in names:
            width, height = self._model.output_width_height(name)
            shapes[name] = (height, width)
            feats_dict[name] = np.ndarray(shape=(nb_points_total, self._model.output_channels(name)))

        # extract bathes of input images and labels
        tf.logging.info('Start input data ...')
        image_batch = self.picture_read(self.images)
        batches = []
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord=coord)
            for tem in range(FLAGS.cp_nb_batches):
                np_images_raw = np.squeeze(sess.run([image_batch]))
                np_labels = self._model.sess.run(self.mem_labels, feed_dict={self.mem_images: np_images_raw})
                np_labels = np.argmax(np_labels, axis=1)
                batches.append([np_images_raw, np_labels])
            coord.request_stop()
            coord.join(threads)
            data_and_label = batches
        tf.logging.info('End input data !')
        # get the output of corresponding layer and do sampling
        idx = 0
        for batch in range(FLAGS.cp_nb_batches):
            data = data_and_label[batch][0]
            label = data_and_label[batch][1]
            set_points_dict((batch, 0), data)
            set_points_dict((batch, 1), label)
            with tf.variable_scope('train'):
                feats = self._model.sess.run(names, feed_dict={self.mem_images: data})
            for feat, name in zip(feats, names):
                shape = shapes[name]
                x_samples = np.random.randint(0, shape[0] - 0, FLAGS.cp_nb_points_per_layer)
                y_samples = np.random.randint(0, shape[1] - 0, FLAGS.cp_nb_points_per_layer)
                set_points_dict((batch, name, "x_samples"), x_samples.copy())
                set_points_dict((batch, name, "y_samples"), y_samples.copy())
                if self.data_format == 'NCHW':
                    feats_dict[name][idx:(idx + nb_points_per_batch)] = \
                        np.transpose(feat[:, :, x_samples, y_samples], (0, 2, 1)).reshape((nb_points_per_batch, -1))
                else:
                    feats_dict[name][idx:(idx + nb_points_per_batch)] = \
                        feat[:, x_samples, y_samples, :].reshape((nb_points_per_batch, -1))
            idx += nb_points_per_batch

        self.feats_dict = feats_dict
        self.points_dict = points_dict

    def __extract_new_features(self, names=None):
        """ extract new feature map via re-sampling some points"""
        nb_points_per_batch = self.points_dict["nb_points_per_batch"]
        feats_dict = {}
        shapes = {}
        nb_points_total = nb_points_per_batch * FLAGS.cp_nb_batches
        idx = 0

        for name in names:
            width, height = self._model.output_width_height(name)
            shapes[name] = (height, width)
            feats_dict[name] = np.ndarray(shape=(nb_points_total, self._model.output_channels(name)))

        for batch in range(FLAGS.cp_nb_batches):
            feats = self._model.sess.run(names, feed_dict={self.mem_images: self.points_dict[(batch, 0)]})
            for feat, name in zip(feats, names):
                x_samples = self.points_dict[(batch, name, "x_samples")]
                y_samples = self.points_dict[(batch, name, "y_samples")]
                if self.data_format == 'NCHW':
                    feats_dict[name][idx:(idx + nb_points_per_batch)] = \
                        np.transpose(feat[:, :, x_samples, y_samples], (0, 2, 1)).reshape((nb_points_per_batch, -1))
                else:
                    feats_dict[name][idx:(idx + nb_points_per_batch)] = feat[:, x_samples, y_samples, :].reshape(
                        (nb_points_per_batch, -1))
            idx += nb_points_per_batch
        return feats_dict

    def __extract_input(self, conv):
        # 抽出一个卷积层的输入，个数为N的小长条（k_h, k_w, c)
        """extract the input X (k_h, k_w, c) of a conv layer
        Args:
            conv: a convolution operation
        Returns:
            bathces of X (N, k_h, k_w, c)
        """
        opname = conv.name
        outname = opname + ":0"
        extractor = self.extractors[opname]
        Xs = []
        def_ = self._model.get_conv_def(conv)
        for batch in range(FLAGS.cp_nb_batches):
            feat = self._model.sess.run(extractor, feed_dict={self.mem_images: self.points_dict[(batch, 0)]})
            x_samples = self.points_dict[(batch, outname, "x_samples")]
            y_samples = self.points_dict[(batch, outname, "y_samples")]

            X = feat[:, x_samples, y_samples, :].reshape((-1, feat.shape[-1]))
            X = X.reshape((X.shape[0], def_['h'], def_['w'], def_['c']))  # 小条
            Xs.append(X)
        return np.vstack(Xs)

    def accuracy(self):
        """Calculate the accuracy of pruned model"""
        # 测试剪枝后的精度
        acc_list = []
        for batch in range(FLAGS.cp_nb_batches):
            metrics = self._model.sess.run(self.mem_labels, feed_dict={self.mem_images: self.points_dict[(batch, 0)]})
            acc_list.extend(metrics)
        res = np.argmax(acc_list, axis=1)
        labels = []
        for batch in range(FLAGS.cp_nb_batches):
            label = self.points_dict[(batch, 1)]
            labels.extend(label)
        # labels = self.labels
        num = 0
        for x, y in zip(res, labels):
            if x == y:
                num += 1
        return float(num / len(labels))

    @classmethod
    def rel_error(cls, A, B):
        """calcualte relative error"""
        return np.mean((A - B) ** 2) ** .5 / np.mean(A ** 2) ** .5

    @classmethod
    def featuremap_reconstruction(cls, x, y, copy_x=True, fit_intercept=False):
        """Given changed input X, used linear regression to reconstruct original Y
          Args:
            :param y: The original feature map of the convolution layer
            :param x: The pruned input
            :param copy_x:
            :param fit_intercept:
          Return:
            new weights and bias which can reconstruct the feature map with small loss given X

        """
        _reg = LinearRegression(n_jobs=-1, copy_X=copy_x, fit_intercept=fit_intercept)
        _reg.fit(x, y)
        return _reg.coef_, _reg.intercept_

    def compute_pruned_kernel(self, X, W2, Y, alpha=1e-4, c_new=None, tolerance=0.02):
        """compute which channels to be pruned by lasso"""

        tf.logging.info('computing pruned kernel')
        nb_samples = X.shape[0]
        c_in = X.shape[-1]
        c_out = W2.shape[-1]
        samples = np.random.randint(0, nb_samples, min(400, nb_samples // 20))
        # print("samples:", samples)
        reshape_X = np.rollaxis(np.transpose(X, (0, 3, 1, 2)).reshape((nb_samples, c_in, -1))[samples], 1, 0)
        reshape_W2 = np.transpose(np.transpose(W2, (3, 2, 0, 1)).reshape((c_out, c_in, -1)), [1, 2, 0])
        product = np.matmul(reshape_X, reshape_W2).reshape((c_in, -1)).T
        reshape_Y = Y[samples].reshape(-1)
        # feature
        tmp = np.nonzero(np.sum(np.abs(product), 0))[0].size
        if FLAGS.debug:
            tf.logging.info('feature num: {}, non zero: {}'.format(product.shape[1], tmp))

        solver = LassoLars(alpha=alpha, fit_intercept=False, max_iter=3000)

        def solve(alpha):
            """ Solve the Lasso"""
            solver.alpha = alpha
            solver.fit(product, reshape_Y)
            idxs = solver.coef_ != 0.
            tmp = sum(idxs)
            return idxs, tmp, solver.coef_

        tf.logging.info('pruned channel selecting')
        start = timer()

        if c_new == c_in:
            idxs = np.array([True] * c_new)
        else:
            left = 0
            right = alpha  # 0.0001
            lbound = max(1, c_new - tolerance * c_in / 2)  # 32 - 0.01*64 = 26
            rbound = c_new + tolerance * c_in / 2  # 32 + 0.01*64 = 38

            while True:
                _, tmp, coef = solve(right)
                if tmp < c_new:
                    break
                else:
                    right *= 2  # Changing this parameter
                    if FLAGS.debug:
                        tf.logging.debug("relax right to {}".format(right))
                        tf.logging.debug(
                            "we expect got less than {} channels, but got {} channels".format(c_new, tmp))
            while True:
                idxs, tmp, coef = solve(alpha)
                # print loss
                loss = 1 / (2 * float(product.shape[0])) * \
                       np.sqrt(np.sum((reshape_Y - np.matmul(product, coef)) ** 2, axis=0)) + alpha * np.sum(
                    np.fabs(coef))
                print("alpha:", alpha * np.sum(np.fabs(coef)))
                if FLAGS.debug:
                    tf.logging.debug('loss: {}, alpha: {}, feature nums: {}, left: {}, right: {}, \
                          left_bound: {}, right_bound: {}'.format(loss, alpha, tmp, left, right, lbound, rbound))

                if FLAGS.debug:
                    tf.logging.info('tmp {}, lbound {}, rbound {}, alpha {}, left {}, right {}'.format(
                        tmp, lbound, rbound, alpha, left, right))
                if FLAGS.cp_quadruple:
                    if tmp % 4 == 0 and abs(tmp - lbound) <= 2:
                        break
                if lbound <= tmp and tmp <= rbound:
                    if FLAGS.cp_quadruple:
                        if tmp % 4 == 0:
                            break
                        elif tmp % 4 <= 2:
                            rbound = tmp - 1
                            lbound = lbound - 2
                        else:
                            lbound = tmp + 1
                            rbound = rbound + 2
                    else:
                        break
                elif abs(left - right) <= right * 0.1:
                    if lbound > 1:
                        lbound = lbound - 1
                    if rbound < c_in:
                        rbound = rbound + 1
                    left = left / 1.2
                    right = right * 1.2
                elif tmp > rbound:
                    left = left + (alpha - left) / 2
                else:
                    right = right - (right - alpha) / 2

                if alpha < 1e-10:
                    break

                alpha = (left + right) / 2
            c_new = tmp
        tf.logging.info('Channel selection time cost: {}s'.format(timer() - start))
        start = timer()
        tf.logging.info('Feature map reconstructing')
        newW2, _ = self.featuremap_reconstruction(X[:, :, :, idxs].reshape((nb_samples, -1)), Y, fit_intercept=False)
        tf.logging.info('Feature map reconstruction time cost: {}s'.format(timer() - start))
        return idxs, newW2

    def residual_branch_diff(self, sum_name):
        """ calculate the difference between before and after weight pruning for a certain branch sum"""
        tf.logging.info("approximating residual branch diff")
        feats_dict = self.__extract_new_features([sum_name])
        residual_diff = (self.feats_dict[sum_name]) - (feats_dict[sum_name])
        return residual_diff

    def prune_kernel(self, op, nb_channel_new):
        """prune the input of op by nb_channel_new
        Args:
            op: the convolution operation to be pruned.
            nb_channel_new: preserving ratio (0, 1]
        Return:
            idxs: the indices of channels to be kept
            newW2: new weight after pruning
            nb_channel_new: actual channel after pruned
        """
        tf.logging.info('pruning kernel')
        definition = self._model.get_conv_def(op)
        k_h, k_w, c, n = definition['h'], definition['w'], definition['c'], definition['n']
        try:
            assert nb_channel_new <= 1., \
                'pruning rate should be less than or equal to 1, while it\'s {}'.format(nb_channel_new)
        except AssertionError as error:
            tf.logging.error(error)
        # 需要保留的通道个数
        nb_channel_new = max(int(np.around(c * nb_channel_new)), 1)

        # newX为原始输入的采样，Y代表相应的输出, add没有的话则为None，W2为卷积核参数
        newX = self.__extract_input(op)
        Y = self.feats_dict[op.name + ":0"]
        add = self._model.get_add_if_op_is_last_in_resblock(op)
        if add is not None:
            Y = Y + self.residual_branch_diff(add.name)
            tf.logging.debug('residual_branch_diff: {}'.format(self.residual_branch_diff(add.name)))
        W2 = self._model.get_var_by_op(op)
        tf.logging.debug('original feature map rmse: {}'.format(
            self.rel_error(newX.reshape(newX.shape[0], -1).dot(W2.reshape(-1, W2.shape[-1])), Y)))

        # 默认使用cp_lasso,idxs为剪枝序号，newW2为新的filter
        if FLAGS.cp_lasso:
            idxs, newW2 = self.compute_pruned_kernel(newX, W2, Y, c_new=nb_channel_new)
        else:
            idxs = np.argsort(-np.abs(W2).sum((0, 1, 3)))
            mask = np.zeros(len(idxs), bool)
            idxs = idxs[:nb_channel_new]
            mask[idxs] = True
            idxs = mask
            reg = LinearRegression(fit_intercept=False)
            reg.fit(newX[:, :, :, idxs].reshape(newX.shape[0], -1), Y)
            newW2 = reg.coef_
            # tf.logging.info('idxs2 {}'.format(idxs))

        tf.logging.debug('feature map rmse: {}'.format(
            self.rel_error(newX[:, :, :, idxs].reshape(newX.shape[0], -1).dot(newW2.T), Y)))
        tf.logging.info('Prune {} c_in from {} to {}'.format(op.name, newX.shape[-1], sum(idxs)))
        nb_channel_new = sum(idxs)
        newW2 = newW2.reshape(-1, k_h, k_w, nb_channel_new)
        newW2 = np.transpose(newW2, (1, 2, 3, 0))
        return idxs, newW2, nb_channel_new / len(idxs)

    def finallayer(self, offset=1):
        """ whether final layer reached"""
        return len(self.thisconvs) - offset == self.state

    def __add_drop_train_vars(self, op):
        """ Add the dropped train variable to the `self.drop_trainable_vars`
          and dropped convolution name to the `drop_conv`
          Args:
            op: An drop operation
        """
        with self._model.g.as_default():
            train_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=os.path.split(op.name)[0] + '/')
            train_vars += tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=os.path.split(op.name)[0].replace('MobilenetV1/MobilenetV1', 'MobilenetV1') + '/')
            train_vars_names = list(map(lambda x: x.name, train_vars))
            self.drop_trainable_vars.update(train_vars_names)
            self.drop_conv.update([op.name])

    def prune_W1(self, father_conv, idxs):
        # 剪除filter
        """ Prune the previous layer weight (channel dimension of the input)
          Args:
            father_conv: previous convolution
            idxs: the indices of channels to be kept
            conv_input: the original input of the convolution operation
          return:
            the output of the previous layer after pruning
        """
        #  [1]代表filter的剪枝级别
        self.max_strategy_dict[father_conv.name][1] = sum(idxs) / len(idxs)
        self.fake_pruning_dict[father_conv.name][1] = idxs

        # assign fake pruned weights
        weight = self._model.get_var_by_op(father_conv)
        fake_pruned_weight = weight  # .eval(self._model.sess)
        not_idxs = [not i for i in idxs]
        if father_conv.type == 'DepthwiseConv2dNative':
            fake_pruned_weight[:, :, not_idxs, :] = 0
        else:
            fake_pruned_weight[:, :, :, not_idxs] = 0

        variable_weight = slim.get_variables_by_name(os.path.split(father_conv.name)[0] + "/weights")[0]
        self._model.sess.run(tf.assign(variable_weight, fake_pruned_weight))

        # assign fake pruned bias
        bias_list = slim.get_variables_by_name(os.path.split(father_conv.name)[0] + '/biases')
        if bias_list:
            bias = bias_list[0]
            fake_pruned_bias = bias.eval(self._model.sess)
            fake_pruned_bias[not_idxs] = 0
            self._model.sess.run(tf.assign(bias, fake_pruned_bias))
        output = None
        return output

    def prune_W2(self, conv_op, idxs, W2=None):
        # 剪除kernel
        """ Prune the current layer weight (channel dimension of the output)
          Args:
            conv_op: the current convolution operation
            W2: the new W2
          return:
            the output of the current convolution opeartion
            :param idxs:
        """
        # [0]代表kernel级别
        self.max_strategy_dict[conv_op.name][0] = sum(idxs) / len(idxs)
        self.fake_pruning_dict[conv_op.name][0] = idxs
        # assign fake pruned weights
        weight = self._model.get_var_by_op(conv_op)
        fake_pruned_weight = weight  # .eval(self._model.sess)
        if W2 is not None:
            fake_pruned_weight[:, :, idxs, :] = W2
        not_idxs = [not i for i in idxs]
        fake_pruned_weight[:, :, not_idxs, :] = 0
        variable_weight = slim.get_variables_by_name(os.path.split(conv_op.name)[0] + "/weights")[0]
        self._model.sess.run(tf.assign(variable_weight, np.asarray(fake_pruned_weight)))

        output = None

        return output

    def compress(self, c_ratio):
        """ Compress the model by channel pruning
        Args:
            action: preserving ratio
            :param c_ratio:
        """
        # 第一层卷积不进行剪裁, 而且要打印出模型的原始准确度
        if self.state == 0:
            c_ratio = 1.0
            self.accuracy()
        # 最后一层不进行剪裁
        if self.finallayer():
            c_ratio = 1
        # 进行强化学习时c_ratio需要处理的问题
        if FLAGS.cp_prune_option == 'auto':
            tf.logging.info('preserve ratio before constraint {}'.format(c_ratio))
            c_ratio = self.__action_constraint(c_ratio)
            tf.logging.info('preserve ratio after constraint {}'.format(c_ratio))
        #  thisconvs 是所有卷积的列表集合，conv_op表示列表中的第self.state个卷积operation
        conv_op = self.thisconvs[self.state]
        # 如果压缩比例为1，主要是为了automl后面使用的使用运用
        if c_ratio == 1:
            if FLAGS.cp_prune_option == 'auto':
                self.max_strategy_dict[conv_op.name][0] = c_ratio
                if self._model.is_weight_prunable(conv_op):
                    self.max_strategy_dict[self._model.fathers[conv_op.name]][1] = c_ratio

        # 如果压缩比例是一个具体的值比如0.8
        else:
            idxs, W2, c_ratio = self.prune_kernel(conv_op, c_ratio)  # idxs剪枝列表, W2新的filter, c_ratio实际的剪枝比例
            with self._model.g.as_default():
                if self._model.is_weight_prunable(conv_op):
                    father_conv = self._model.g.get_operation_by_name(self._model.fathers[conv_op.name])
                    while father_conv.type in ['DepthwiseConv2dNative']:
                        if self._model.is_weight_prunable(father_conv):
                            father_conv = self._model.g.get_operation_by_name(self._model.fathers[father_conv.name])
                    tf.logging.info('father conv {}'.format(father_conv.name))
                    tf.logging.info('father conv input {}'.format(father_conv.inputs[0]))
                    # 先剪上层的filter
                    self.prune_W1(father_conv, idxs)
                # 然后剪本层的filter中的kernel， 本质上都是置0
                self.prune_W2(conv_op, idxs, W2)

        tf.logging.info('Channel pruning the {} layer, the pruning rate is {}'.format(conv_op.name, c_ratio))

        if self.finallayer():
            acc = self.accuracy()
            tf.logging.info('Pruning accuracy {}'.format(acc))
            pruned_flops = self.__compute_model_flops(fake=True)
            tf.logging.info('Pruned flops {}'.format(pruned_flops))

            preserve_ratio = pruned_flops / self.model_flops
            reward = [acc, pruned_flops]
            tf.logging.info(
                'The accuracy is {} and the flops after pruning is {}'.format(reward[0], reward[1]))
            tf.logging.info('The speedup ratio is {}'.format(preserve_ratio))
            tf.logging.info('The original model flops is {}'.format(self.model_flops))
            tf.logging.info('The pruned flops is {}'.format(pruned_flops))
            tf.logging.info('The max strategy dict is {}'.format(self.max_strategy_dict))

            state, reward = self.currentStates.loc[self.state].copy(), reward
            # if FLAGS.prune_option != 'auto':
            self.save_model()
            return state, reward, True, c_ratio
        # 强化学习时候用的
        reward = [0, 1]
        self.state += 1
        if FLAGS.cp_prune_option == 'auto':
            self.currentStates['maxreduce'][self.state] = self.max_reduced_flops / self.model_flops
        state = self.currentStates.loc[self.state].copy()
        return self.state, reward, False, c_ratio

    def save_model(self):
        """ save the current model to the `FLAGS.channel_pruned_path`"""
        with self._model.g.as_default():
            saver = tf.train.Saver()
            is_exists = os.path.exists(FLAGS.cp_channel_pruned_path)
            if not is_exists:
                os.makedirs(FLAGS.cp_channel_pruned_path)
            saver.save(self._model.sess, FLAGS.cp_channel_pruned_path + "/model")
            tf.logging.info('saved pruned model to {}'.format(FLAGS.cp_channel_pruned_path))

    def prune_total_model(self, c_ratio=0.6, layer_num=5):
        tag = False
        self.state = 0
        while tag is False:
            state, reward, tag, _ = self.compress(c_ratio)
            self.state = state
            if state == layer_num:
                return

    def finetune_model(self, epoch):

        with self._model.g.as_default():

            maskable_var_names = [(slim.get_variables_by_name(os.path.split(op_name)[0] + "/weights")[0]).name
                                  for op_name, ratio in self.fake_pruning_dict.items()]
            maskable_var_names = [name.split(":")[0] for name in maskable_var_names]

            val = np.random.normal(size=[10, 1000])
            labels = tf.constant(value=val, shape=[10, 1000])
            predicts = self.mem_labels
            loss = tf.losses.softmax_cross_entropy(labels, predicts)
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            grads_origin = optimizer.compute_gradients(loss, tf.trainable_variables())  # 优化所有变量

            grads_pruned, masks = self.get_grads_pruned(grads_origin, maskable_var_names)
            train_op = optimizer.apply_gradients(grads_pruned)

            train_init_op = tf.initialize_variables(optimizer.variables() + masks)
            # init variable
            self._model.sess.run(train_init_op)
            # train model
            image_batch = self.picture_read(self.images)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord=coord)

            for i in range(epoch):
                for tem in range(FLAGS.cp_nb_batches):
                    print("epoch:{}-batch:{}".format(i, tem))
                    np_images_raw = np.squeeze(sess.run([image_batch]))
                    self._model.sess.run(train_op, feed_dict={self.mem_images: np_images_raw})
            coord.request_stop()
            coord.join(threads)

            # save model
            self.save_model()

    def get_grads_pruned(self, grads_origin, maskable_var_names):
        grads_pruned = []
        masks = []

        for grad in grads_origin:
            if grad[1].name not in maskable_var_names:
                print(grad[1].name)
                grads_pruned.append(grad)
            else:
                pruned_idxs = self.fake_pruning_dict[grad[1].name]
                mask_tensor = np.ones(grad[0].shape)
                mask_tensor[:, :, [not i for i in pruned_idxs[0]], :] = 0
                mask_tensor[:, :, :, [not i for i in pruned_idxs[1]]] = 0
                mask_initializer = tf.constant_initializer(mask_tensor)
                mask = tf.get_variable(grad[1].name.split(':')[0] + '_mask', shape=mask_tensor.shape,
                                       initializer=mask_initializer, trainable=False)
                grads_pruned.append((grad[0] * mask, grad[1]))
                masks.append(mask)
        return grads_pruned, masks


if __name__ == "__main__":
    graph = tf.Graph()
    with graph.as_default():
        ckpt = tf.train.get_checkpoint_state('../model/')  # 通过检查点文件锁定最新的模型
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
    sess = tf.Session(graph=graph)
    saver.restore(sess, ckpt.model_checkpoint_path)
    model = model_wrapper.Model(sess=sess)
    prune_model = ChannelPruner(model=model)
    prune_model.prune_total_model(0.5)
    # for op in prune_model.thisconvs:
    #     print(op.name)
    #     print(prune_model.fake_pruning_dict[op.name][0])
    #     print(prune_model.fake_pruning_dict[op.name][1])
    prune_model.finetune_model(2)

