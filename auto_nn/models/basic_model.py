#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: basic_model.py
# @brief: 
# @author: niezhipeng(@baidu.com)
# @Created on 2018/4/30
# *************************************************************************************
from abc import abstractmethod
import numpy as np
import tensorflow as tf
from common.layers import apply_layer
import common.tf_utils as tf_utils


def logit_2_score(logit_list, paths=None, mode="add"):
    """ 针对单个time_step，取多任务联合最大的score
    args:
        logit_list: [task_1_logit, task_2_logit, ...]
        paths: LIST of (task_1_indice, task_2_indice, ...)
        mode: "add" or "product"
    return:
    """
    max_prob = -10000
    score_list = []
    if paths is not None and len(paths) > 0:
        for path in paths:
            prob = np.sum([logit[i] for logit, i in zip(logit_list, path)])
            if prob > max_prob:
                max_prob = prob
                score_list = list(path)
        return score_list, max_prob

    score_list = [np.argmax(logit) for logit in logit_list]
    return score_list


class BasicModel(object):
    """ Basic sequence label model, supporting multi-task
    need a conf dict, including:
        "feature_dim":      INT,
        "target_dim_list":  LIST of INT, for multi-task,
        "hidden_layer":     LIST,
        "output_layer":     LIST for multi-task,
    """
    def __init__(self, feature_dim, target_dim_list, model_cfg):
        self._cfg = model_cfg
        self._feature_dim = feature_dim
        self._target_dim_list = target_dim_list
        self._task_num = len(self._target_dim_list)
        self._logits_list = list()
        self._targets_list = list()
        self._scores_list = list()
        self._train_op = None
        self._loss_list = list()
        self._loss = None

        self._is_training = tf.placeholder(
            dtype=tf.bool, shape=None, name="is_training")

    @property
    def loss_list(self):
        return self._loss_list

    @property
    def label_mask(self):
        return self._label_mask

    @property
    def feature_dim(self):
        """
        return feature dim
        """
        return self._feature_dim

    @property
    def target_dim_list(self):
        """
        return LIST of target dim for multi-task
        """
        return self.target_dim_list

    @property
    def loss(self):
        """
        return weighted loss for train
        """
        return self._loss

    @property
    def train_op(self):
        """
        return train optimizer, default = adam
        """
        return self._train_op

    @property
    def logits_list(self):
        """
        return LIST of Tensor([B, T, D], tf.float32)
        """
        return self._logits_list

    @property
    def scores_list(self):
        """
        return LIST of Tensor([B, T], tf.int32)
        """
        return self._scores_list

    '''
    def initialize(self, inputs, input_lenghts, targets_list=None, hparams=None):
        self._inputs = inputs
        self._seq_len = input_lenghts
        self._targets_list = targets_list
        # self._dropout = tf.cond(self._is_training, lambda :float(hparams["dropout"]), lambda :0.)
        self._dropout = tf.placeholder(dtype=tf.float32, shape=())
        #self._label_mask = _get_dyz_label_mask(self._inputs, self._targets_list[-1])
        self._build_graph()
    '''

    def make_feed_inputs(self, inputs, seq_lens=None, targets_list=None, label_masks=None,
                         dropout=0., is_training=False):
        """ make feed dict , trans inputs, targets and label_masks to model inputs
        Args:
            inputs: NP.ARRAY([B, T, D], float)
            targets_list:   LIST of NP.ARRAY([B, T], int)
            label_masks_list:   LIST of (NP.ARRAY([B, T]))
            dropout: FLOAT
            is_training: True, if train task; False, otherwise
        Returns:
            feed_dict for training
        """
        # input dropout
        # 仅对字向量进行dropout,one-hot特征保留
        # inputs[:,:,:128] = tf.nn.dropout(inputs[:,:,:128], 1. - dropout)
        feed_dict = dict()
        feed_dict[self._inputs] = inputs
        feed_dict[self._seq_len] = seq_lens
        # feed_dict[self._seq_len] = calc_sequence_lengths(inputs)
        feed_dict[self._dropout] = dropout
        # for batch_normalize in conv1d
        feed_dict[self._is_training] = is_training

        if targets_list is None:
            return feed_dict

        for i in range(self._task_num):
            feed_dict[self._targets_list[i]] = targets_list[i]
        # feed_dict[self._label_mask] = get_seq_padding_mask(self._seq_len, len(inputs[0]))
        return feed_dict

    def build_graph(self, scope=None):
        """ build graph, need to be called explicitly
        """
        tf.set_random_seed(1)

        self._init_tensors()
        scope = scope or "model"
        with tf.name_scope(scope):
            _, outputs_list = self._init_layers(self._inputs, self._cfg)
            self._logits_list = outputs_list
            self._init_scoring()

        self.calc_loss()

    def _init_layers(self, layer_in, model_cfg, scope=None):
        """ build tensor graph according to hidden_layer, output_layer
            each layer should be DICT, including:
                "layer_type": STR,
        args:
            layer_in: Tensor([B, T, D], tf.float32)

        :return:
        """
        hidden_layers = list()
        output_layers = list()

        for idx, params in enumerate(model_cfg["hidden_layer"]):
            layer_type = params["layer_type"]
            layer_out = apply_layer(
                layer_in,
                params,
                self._seq_len,
                dropout=self._dropout,
                is_training=self._is_training,
                scope="hidden_layer_%d_%s" % (idx, layer_type)
            )
            print("[DEBUG]: hidden_layer_%d_%s" % (idx, layer_type))
            print("\t\tparams =", params)
            hidden_layers.append(layer_out)
            layer_in = layer_out

        for idx, params in enumerate(model_cfg["output_layer"]):
            params["num_units"] = self._target_dim_list[idx]
            layer_out = apply_layer(
                layer_in,
                params,
                dropout=0.0,
                is_training=False,
                scope="output_layer_%d" % idx,
            )
            print("[DEBUG]: output_layer_%d" % idx)
            print("\t\tparams =", params)
            output_layers.append(layer_out)
            return hidden_layers, output_layers

    # TODO: 打分机制---默认最大概率，后期可扩展crf
    def _init_scoring(self):
        """
        logits to scores, default = indices with maximum softmax value
        """
        self._scores_list = []
        for i in range(self._task_num):
            score = tf.argmax(tf_utils.softmax(self._logits_list[i]), axis=-1)
            self._scores_list.append(tf.cast(score, dtype=tf.int32))

    def calc_loss(self, loss_cfg=None):
        """ calculate weighted loss for train, support multi-task
            self._loss --- Float value for train
            self._loss_list --- task loss and hidden loss
        args:
            loss_weights: LIST or NP.ARRARY
        return:
            loss_list: LIST
        """
        self._loss_list = list()
        loss_cfg = self._cfg["losses"]
        with tf.name_scope("calc_loss"):
            for i in range(self._task_num):
                _loss = self.calc_softmax_focal_loss(
                    self._targets_list[i],
                    self._logits_list[i],
                    loss_cfg[i])
                _loss *= self._label_mask
                aver_loss = tf.reduce_sum(_loss) / tf.reduce_sum(self._label_mask)
                self._loss_list.append(aver_loss)
                tf.summary.scalar("loss_%d" % i, self._loss_list[i])

            # weighted loss for multi-task
            try:
                task_weights = [float(x) for x in self._cfg["task_weights"]]
                # assert(len(task_weights) == len(self._loss_list))
            except (KeyError, IndexError) as e:
                print(e)
                task_weights = [1.0] * len(self._loss_list)
            self._loss = sum([l * w for l, w in zip(self._loss_list, task_weights)])

        return self._loss_list

    def apply_optimizer(self, train_cfg, global_step=None):
        """ apply (adam) optimizer for train, decay learning rate as:
                lr * decay_lr_rate ^ (global_step / decay_lr_steps),
            clip gradient when use_clip is True.
        args:
            global_step: 记录迭代次数，一个batch算一次迭代
            train_cfg: DICT {
                "optim_name": STR,
                "optim_params": DICT, {
                    "learning_rate":  FLOAT,
                    "decay_lr_steps": INT, ,
                    "decay_lr_rate":  ,
                    "beta1":          FLOAT, optional for adam,
                    "beta2":          FLOAT, optional for adam,
                    "use_clip":       BOOL, optional,
                    "max_grad_value": FLOAT, clip gradient by
                                      [-max_grad_value, max_grad_value],
                    "max_grad_norm":  FLOAT, rescale gradient
                                      when larger than max_grad_norm
                }
            }
        return:
            operator for train
        """
        with tf.name_scope("optimizer"):
            optim_name = train_cfg["optim_name"]
            params = train_cfg["optim_params"]
            use_clip = "use_clip" in params and params["use_clip"]

            self._init_lr = tf.convert_to_tensor(params["learning_rate"])
            '''
            self._init_lr = tf.train.exponential_decay(
                float(params["learning_rate"]),
                global_step,
                decay_steps=int(params["decay_lr_steps"]),
                decay_rate=float(params["decay_lr_rate"]),
                staircase=True,
                name="AnnealedLearningRate")
            '''
            tf.summary.scalar("lr", self._init_lr)
            
            optimizer = tf.train.AdamOptimizer(self._init_lr,
                                               params["beta1"],
                                               params["beta2"])

            if use_clip:
                (_grads, _vars) = zip(*optimizer.compute_gradients(self._loss))

                # clip gradients to solve gradients explosion or gradients vanishing
                self._gradients = _grads
                '''
                #print(_grads)
                # cnn with batch norm: (kernel, bias, bn_1, bn_2)
                print(_grads[1])
                tf.summary.scalar("hidden_layer_0_grad_norm", tf.norm(_grads[1]))
                #print(_grads[13])
                #tf.summary.scalar("hidden_layer_1_grad_norm", tf.norm(_grads[13]))

                # print(_grads[25])
                # tf.summary.scalar("hidden_layer_2_grad_norm", tf.norm(_grads[25]))
                # print(_grads[37])
                # tf.summary.scalar("hidden_layer_3_grad_norm", tf.norm(_grads[37]))

                print(_grads[-2])
                tf.summary.scalar("output_layer_1_grad_norm", tf.norm(_grads[-2]))
                tf.summary.scalar("grad_norm", tf.global_norm(self._gradients))
                '''

                clipped_grads = _grads
                # Clip gradients by value magnitude.
                if "max_grad_value" in params:
                    max_value = float(params["max_grad_value"])
                    clipped_grads = [tf.clip_by_value(grad, -max_value, max_value)
                        for grad in clipped_grads]

                # Rescale gradients to a maximum norm.
                if "max_grad_norm" in params:
                    gradnorm = tf.global_norm(_grads)
                    max_grad_norm = float(params["max_grad_norm"])
                    clipped_grads, gradnorm = tf.clip_by_global_norm(
                        clipped_grads, max_grad_norm, use_norm=gradnorm)

                # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
                # https://github.com/tensorflow/tensorflow/issues/1122
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    self._train_op = optimizer.apply_gradients(zip(clipped_grads, _vars),
                                                               global_step=global_step)
            else:
                self._train_op = optimizer.minimize(self._loss)

            return self._train_op

    @staticmethod
    def calc_softmax_ce_loss(targets, logits, hparams):
        """ calc softmax cross entropy, support label_smooth
        args:
            targets: Tensor([B, T], tf.int32)
            logits: Tensor([B, T, num_classes], tf.float32)
            hparams: DICT, {
                "alpha": [1.0, 1.0, ...],
                "label_smooth":
            }
        return:
            Tensor([B, T, num_classes], tf.float32)
        """
        num_classes = logits.get_shape().as_list()[-1]
        try:
            alpha = hparams["alpha"]
        except KeyError:
            alpha = [1.0 for _ in range(num_classes)]
        print("[INFO] label_alpha: %s" % str(alpha))
        weights = tf.nn.embedding_lookup(
            tf.constant(alpha, tf.float32), targets)

        try:
            epsilon = float(hparams["label_smooth"])
            print("[INFO] use label smoothing: %.5f" % epsilon)
            onehot_label = tf.one_hot(targets, num_classes)
            # label smoothing
            smooth_label = (1 - epsilon) * onehot_label + (epsilon / num_classes)
            ce_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=smooth_label,
                logits=logits,
            )
        except KeyError:
            ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets,
                logits=logits
            )
        return ce_loss * weights

    @staticmethod
    def calc_softmax_focal_loss(targets, logits, hparams):
        """ calc softmax focal loss,
        Ref: https://arxiv.org/pdf/1708.02002.pdf
        args:
            targets: Tensor([B, T], tf.int32)
            logits: Tensor([B, T, num_classes], tf.float32)
            hparams: DICT, {
                "alpha": LIST, [1.0, 1.0, 1.0, 0.7, 1.0, ...]
                "gamma": INT, default=2
            }
        return:
            Tensor([B, T, num_classes], tf.float32)
        """
        num_classes = logits.get_shape().as_list()[-1]
        try:
            alpha = hparams["alpha"]
        except KeyError:
            alpha = [1.0 for _ in range(num_classes)]
        print("[INFO] label_alpha: %s" % str(alpha))
        weights = tf.nn.embedding_lookup(
            tf.constant(alpha, tf.float32), targets)

        probs = tf.add(tf.nn.softmax(logits, dim=-1), 1.e-10)
        onehot_labels = tf.one_hot(targets, num_classes)
        # ce_loss
        # _loss = tf.reduce_sum(tf.multiply(onehot_labels, -tf.log(probs)), axis=-1)
        p_t = tf.reduce_sum(tf.multiply(onehot_labels, probs), axis=-1)

        gamma = float(hparams["gamma"]) if "gamma" in hparams else 2.0
        focal_loss = tf.multiply(tf.pow(1.-p_t, gamma), -tf.log(p_t))
        # focal_loss = tf.multiply(tf.pow(1.-p_t, gamma)/p_t, -tf.log(p_t))
        return focal_loss * weights


    def _init_tensors(self):
        """ define inputs, seq_len and targets for graph
        """
        self._inputs = tf.placeholder(
            shape=(None, None, self._feature_dim),
            dtype=tf.float32,
            name='inputs',
        )

        # self._seq_len = tf_utils.calc_sequence_lengths(self._inputs)
        
        self._seq_len = tf.placeholder(
            shape=(None),
            dtype=tf.int32,
            name='seq_len',
        )
        
        self._label_mask = tf_utils.get_seq_padding_mask(self._seq_len)

        self._targets_list = [tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='targets_%d' % i,
        ) for i, dim in enumerate(self._target_dim_list)]

        self._dropout = tf.placeholder(
            shape=None,
            dtype=tf.float32,
            name="dropout",
        )
