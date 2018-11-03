#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: prosody_model.py
# @brief: 
# @author: niezhipeng(@baidu.com)
# @Created on 2018/4/10
# *************************************************************************************

import os
import sys
import copy
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from common.layers import *
import common.tf_utils as tf_utils
from models.basic_model import BasicModel
import common.layers as tf_layers
from common.text_box import TextProcessBox
from models.densely_attention_net import DenselyAttentionNet
from tensorflow import set_random_seed
set_random_seed(1)

class CBHGModel(BasicModel):
    """
    CBHG based sequence label model
    need a conf dict, including:
        "feature_dim",
        "target_dim_list",
        "hidden_layer",
        "output_layer"
    """
    def __init__(self, feature_dim, target_dim_list, model_cfg):
        super(CBHGModel, self).__init__(feature_dim, target_dim_list, model_cfg)

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

    def build_graph(self):
        """ build graph, need to be called explicitly
        """
        
        tf.set_random_seed(1)

        self._init_tensors()
        _, outputs_list = self._init_layers(self._inputs, self._cfg, scope="model")
        self._logits_list = outputs_list
        self._init_scoring()
        self.calc_loss(self._cfg["loss_weights"])

    def _init_layers(self, layer_in, model_cfg, scope=None):
        """ initialize cbhg layers
        args:
            layer_in:
            model_cfg:
            scope:
        :return:
        """
        hidden_layers = list()
        output_layers = list()
        scope = scope or "cbhg_model"
        # layers
        with tf.name_scope(scope):
            layer_in = self._inputs
            for idx, params in enumerate(model_cfg["hidden_layer"]):
                layer_type = params["layer_type"]
                layer_out = apply_layer(
                    layer_in,
                    params,
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

    def calc_loss(self, loss_weights=None):
        """ 计算loss, 支持multi-task
        args:
            loss_weights: LIST or NP.ARRARY
        return:
            loss_list: LIST
        """

        for i in range(self._task_num):
            # self._losses[i] = get_loss(train_args["loss_function"])(
            self._loss_list.append(tf.losses.sparse_softmax_cross_entropy(
                labels=self._targets_list[i],
                logits=self._logits_list[i],
                weights=self._label_mask,
            ))
            # tf.summary.scalar("loss_%d" % i, self._loss_list[i])
        # weighted loss for multi-task
        if loss_weights is not None:
            assert (len(loss_weights) == len(loss_weights))
            self._loss_weights = [float(x) for x in loss_weights]
        else:
            self._loss_weights = [1.] * len(self._loss_list)
        self._loss = sum([l * w for l, w in zip(self._loss_list, self._loss_weights)])
        return self._loss_list


class DilatedCNNModel(BasicModel):
    """
    Dilated CNN based sequence label model
    need a conf dict, including:
        "feature_dim",
        "target_dim_list",
        "hidden_layer",
        "output_layer"
    """

    def __init__(self, feature_dim, target_dim_list, model_cfg):
        super(DilatedCNNModel, self).__init__(feature_dim, target_dim_list, model_cfg)

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

    def build_graph(self, scope=None):
        """ build graph, need to be called explicitly
        """
        tf.set_random_seed(1)

        self._init_tensors()
        '''
        # add position embedding to features
        position_embedding = self.position_embedding(self._inputs, self.feature_dim)
        tf_utils.plot_2d_tensor(position_embedding[0,:,:], "position_embedding_0")
        _inputs = self._inputs + position_embedding
        _mask = tf.expand_dims(self._label_mask, axis=-1)
        _mask = tf.concat([_mask] * self.feature_dim, axis=-1)
        # print(_mask)
        layer_in = tf.multiply(_inputs, _mask)
        '''
        layer_in = self._inputs
        tf.summary.histogram("inputs", self._inputs)
        max_val = tf.reduce_max(self._inputs)
        tf.summary.scalar("input_max_val", max_val)

        min_val = tf.reduce_min(self._inputs)
        tf.summary.scalar("input_min_val", min_val)

        scope = scope or "model"
        with tf.name_scope(scope):
            hidden_layers, outputs_list = self._init_layers(layer_in, self._cfg)
            self._logits_list = outputs_list
            self._init_scoring()
            self._hidden_layers = hidden_layers

        # print the block corrcoef matrix
        block_in = self._hidden_layers[0]
        tf.summary.histogram("block_in", block_in)
        for idx, block_out in enumerate(self._hidden_layers[1:-1], 1):
            corr_0 = tf_utils.calc_cosine_coef(block_out[0, :self._seq_len[0], :],
                                               block_in[0, :self._seq_len[0], :])
            # print(corr_0)
            # corr = tf.clip_by_value(tf.abs(corr), 0, 1.0)
            corr_0 = tf.abs(corr_0)
            corr_0 = tf.where(tf.greater(corr_0, 0.1), corr_0, tf.zeros_like(corr_0))
            # tar = tf.expand_dims(tf.cast(self._targets_list[0][0,:self._seq_len[0]],
            #       dtype=tf.float32), axis=-1)
            pred_0 = tf.one_hot(self._scores_list[0][0, :self._seq_len[0]],
                               6, on_value=0.1, dtype=tf.float32)
            img_t_0 = tf.concat([pred_0, corr_0], axis=-1)
            tf_utils.plot_2d_tensor(
                img_t_0, "hidden_layer_%d/outputs_0_corr" % (idx))

            corr_1 = tf_utils.calc_cosine_coef(block_out[-1, :self._seq_len[-1], :],
                                               block_in[-1, :self._seq_len[-1], :])
            # print(corr_1)
            corr_1 = tf.abs(corr_1)
            corr_1 = tf.where(tf.greater(corr_1, 0.1), corr_1, tf.zeros_like(corr_1))
            pred_1 = tf.one_hot(self._scores_list[0][-1, :self._seq_len[-1]],
                               6, on_value=0.1, dtype=tf.float32)
            img_t_1 = tf.concat([pred_1, corr_1], axis=-1)
            tf_utils.plot_2d_tensor(
                img_t_1, "hidden_layer_%d/outputs_1_corr" % (idx))

        self.calc_loss(self._cfg["loss_weights"])

    def _init_layers(self, layer_in, model_cfg, scope=None):
        """ initialize dilated cnn block layers
        args:
            layer_in:
            model_cfg:
            scope:
        """
        hidden_layers = list()
        output_layers = list()

        # print(self._inputs.get_shape())
        # print(self._targets_list[0].get_shape())
        # sentence = tf
        # sentence = u""
        # print(sentence.encode("gbk"))
        # tf.summary.text("sentence", tf.convert_to_tensor(sentence.encode("gbk")))
        # layer_in = self.prenet(layer_in, model_cfg["prenet"])

        for idx, params in enumerate(model_cfg["hidden_layer"]):
            layer_type = params["layer_type"].lower()

            # if idx == 1:
            #    block_in = layer_in

            if layer_type == "id_cnn_block":
                layer_block = params["layers"]
                residual = ("residual" in params) and params["residual"]
                block_concat = ("block_concat" in params) and params["block_concat"]
                block_residual = ("block_residual" in params) and params["block_residual"]
                try:
                    block_repeats = int(params["block_repeats"])
                except:
                    block_repeats = 1
                layer_out = None
                for k in range(block_repeats):
                    block_reuse = k > 0
                    layer_out = self.id_cnn_block(
                        layer_in,
                        layer_block,
                        residual=residual,
                        block_concat=block_concat,
                        block_residual=block_residual,
                        block_reuse=block_reuse,
                        scope="hidden_layer_%d_%s" % (idx, layer_type)
                    )
                    layer_in = layer_out
            elif layer_type == "position_embedding":
                position_dim = int(params["position_dim"])
                type = "add"
                gain = float(params["gain"])
                layer_out = self.add_position_embedding(layer_in, position_dim,
                                                        gain=gain, type=type)
                layer_in = layer_out
                print("[DEBUG] %s\n\tparams = %s" % (layer_type, str(params)))
            else:
                _scope = "hidden_layer_%d_%s" % (idx, layer_type)
                layer_out = apply_layer(
                    layer_in,
                    params,
                    self._seq_len,
                    dropout=self._dropout,
                    is_training=self._is_training,
                    scope=_scope
                )
                print("[DEBUG] %s\n\tparams = %s" % (_scope, str(params)))

            # tf.summary.histogram("hidden_layer_%d_%s/outputs_time_0_l2" % (idx, layer_type),
            #                      tf.norm(tf.slice(layer_out, [0, 0, 0], [-1, 1, -1]), axis=-1))
            # tf.summary.histogram("hidden_layer_%d_%s/outputs_time_5_l2" % (idx, layer_type),
            #                     tf.norm(tf.slice(layer_out, [0, 5, 0], [-1, 1, -1]), axis=-1))

            hidden_layers.append(layer_out)
            layer_in = layer_out

        for idx, params in enumerate(model_cfg["output_layer"]):
            params["num_units"] = self._target_dim_list[idx]
            layer_out = apply_layer(
                layer_in,
                params,
                dropout=self._dropout,
                is_training=False,
                scope="output_layer_%d" % idx,
            )
            print("[DEBUG] output_layer_%d" % idx)
            print("\tparams = %s" % str(params))
            output_layers.append(layer_out)
        return hidden_layers, output_layers

    def id_cnn_block(self, inputs, layer_block,
                     residual=False, block_concat=False, block_residual=False,
                     block_reuse=False,
                     scope=None):
        """ iterable dilated cnn block
        args:
            inputs: Tensor([B, T, D], tf.float32)
            layer_block: LIST,
            residual: BOOL, whether to add shortcut connection in each layer
            block_concat: BOOL, whether to concat outputs of all layers
            block_residual: BOOL, whether to add shortcut connection for block
            block_reuse: BOOL, whether to reuse params of block layers
            scope: "id_cnn_block"
        :return:
        """
        scope = scope or "id_cnn_block"
        layer_out = None
        outputs_list = list()

        #with tf.name_scope(scope):
        with tf.variable_scope(scope, reuse=block_reuse):
            layer_out = inputs
            for idx, params in enumerate(layer_block):
                layer_in = layer_out
                layer_type = params["layer_type"].lower()
                layer_out = apply_layer(
                    layer_in,
                    params,
                    self._seq_len,
                    dropout=self._dropout,
                    is_training=self._is_training,
                    scope="%s_%d" % (layer_type, idx)
                )
                print("[DEBUG] %s_%s_%d" % (scope, layer_type, idx))
                if not block_reuse:
                    print("\tparams = %s" % str(params))
                else:
                    print("\treuse the previous block")

                if residual:
                    if layer_out.shape[-1] != layer_in.shape[-1]:
                        layer_out = tf_layers.conv1d(
                            layer_out,
                            kernel_size=1,
                            channels=layer_in.shape[-1],
                            activation="relu",
                            is_training=self._is_training,
                            scope="%s_%d_res_layer" % (scope, idx)
                        )
                        print("[DEBUG] %s_%d_res_layer" % (scope, idx))
                    layer_out = layer_out + layer_in

                outputs_list.append(layer_out)

            if block_concat:
                layer_out = tf.concat(outputs_list, axis=-1)
            elif block_residual:
                layer_out = layer_out + inputs
                layer_out = tf.nn.relu(layer_out)
            return layer_out

    def tcn_block(self, inputs, params,
                  block_residual=False, block_reuse=False, scope=None):
        """ TCN block:
            -> [dilated conv -> WeightNorm -> ReLU -> Dropout] * 2  -> + -> ReLU
            |-----------------------------> 1*1 conv -------------------------> |
        args:
            inputs: Tensor([B, T, D], tf.float32)
            layer_params: DICT, include: {
                layer_type: "atrous_conv1d" or "casual_conv1d",
                kernel_size: INT,
                dilation_rate: INT, default:1,
                initializer: kernel_initializer,
                activation: ,
                use_bn:
                use_wn:
            }
            block_residual: BOOL, whether to add shortcut connection for block
            block_reuse: BOOL, whether to reuse params of block layers
            scope: "tcn_block"
        return:
            Tensor([B, T, D_OUT], tf.float32)
        """
        scope = scope or "tcn_block"
        layer_out = None
        outputs_list = list()

        #with tf.name_scope(scope):
        with tf.variable_scope(scope, reuse=block_reuse):
            layer_type = "casual_conv1d"
            channels = params["channels"]
            params_1 = copy.deepcopy(params)
            params_1["layer_type"] = layer_type
            params_1["channels"] = int(channels / 2)
            out1 = apply_conv1d(
                inputs,
                params,
                is_training=self._is_training,
                dropout=self._dropout,
                scope="%s_%d" % (scope, 0))
            print("[DEBUG]: %s_%s_0" % (scope, layer_type))
            if not block_reuse:
                print("\tparams = %s" % params_1)
            else:
                print("\treuse the previous block")
            outputs_list.append(out1)

            params_2 = copy.deepcopy(params)
            params_2["layer_type"] = layer_type
            params_2["dilation_rate"] = params["dilation_rate"] + 1
            out2 = apply_conv1d(
                out1,
                params,
                is_training=self._is_training,
                dropout=self._dropout,
                scope="%s_%d" % (scope, 1))
            print("[DEBUG] %s_%s_1" % (scope, layer_type))
            if not block_reuse:
                print("\tparams = %s" % str(params_2))
            else:
                print("\treuse the previous block")
            outputs_list.append(out2)

            if block_residual:
                if layer_out.shape[-1] != inputs.shape[-1]:
                    res_layer_out = tf_layers.atrous_conv1d(
                        inputs,
                        kernel_size=1,
                        channels=layer_out.shape[-1],
                        activation="linear",
                        use_bn=False,
                        is_training=self._is_training,
                        scope="%s_residual_layer" % (scope)
                    )
                    print("[DEBUG] %s_residual_layer" % (scope))
                    layer_out = tf.nn.relu(layer_out + res_layer_out)
                else:
                    layer_out = tf.nn.relu(layer_out + inputs)

        return layer_out

    def prenet(self, layer_in, params, dropout=0., scope="prenet"):
        """ preproccessing networks, including
            transform feature, position_embedding
            x --> conv --> ReLU(bn) --> conv --> ReLU(bn) --> conv --|
                                              |----------------------|-> concat -->
                        |--------------------------------------------|

        args:
            layer_in: Tensor([B, T, D], tf.float32)
            params: DICT, {"channels", "activation", "initializer", "use_bn"}
            scope:
        return:
            Tensor([B, T, channels * num_layer], tf.float32)
        """
        channels = int(params["channels"])
        if "initializer" not in params:
            _initializer = "xavier_normal"
        else:
            _initializer = params["initializer"].lower()
        use_bn = "use_bn" in params and params["use_bn"]

        #layer_in = tf.nn.dropout(layer_in, keep_prob=1.0-dropout)

        with tf.name_scope(scope):
            conv1 = tf_layers._conv1d(layer_in, kernel_size=1, channels=channels,
                                     add_bias=False,
                                     is_relu=True,
                                     kernel_initializer=_initializer,
                                     scope="conv1")

            conv1_out = tf.nn.relu(conv1)
            #print(conv1_out)
            if use_bn:
                conv1_out = tf.layers.batch_normalization(conv1_out, training=self._is_training)

            conv2 = tf_layers._conv1d(conv1_out, kernel_size=1, channels=channels,
                                     add_bias=False,
                                     is_relu=True,
                                     kernel_initializer=_initializer,
                                     scope="conv2")

            '''
            conv2_out = tf.nn.relu(conv2)
            if use_bn:
                conv2_out = tf.layers.batch_normalization(conv2_out, training=self._is_training)
            conv3 = tf_layers._conv1d(conv2_out, kernel_size=1, channels=channels,
                                      add_bias=False,
                                      is_relu=True,
                                      kernel_initializer=_initializer,
                                      scope="conv3")
            '''
            outputs = tf.concat((conv1, conv2), axis=-1)
            outputs = tf.nn.relu(outputs)
            if use_bn:
                outputs = tf.layers.batch_normalization(outputs, training=self._is_training)

            return outputs

    def add_position_embedding(self, inputs, position_size, gain=1., type="add"):
        """
        :param inputs:
        :param position_size:
        :param gain:
        :param type:
        :return:
        """
        if type == "add":
            position_size = inputs.get_shape().as_list()[-1]
        tf.summary.histogram("position_input", inputs)
        position_embedding = gain * self.position_embedding(inputs, position_size)
        tf.summary.histogram("position_embedding", position_embedding)
        tf_utils.plot_2d_tensor(position_embedding[0,:,:], "position_embedding_0")
        if type == "add":
            _inputs = inputs + position_embedding
        elif type == "concat":
            _inputs = tf.concat([inputs, position_embedding], axis=-1)
        else:
            _inputs = inputs

        return _inputs

    def position_embedding(self, inputs, position_size, scope=None):
        """ generate position embedding tensor, according to
                PE_2i(p) = sin(p / 10000 ^ (2i/dpos))
                PE_2i+1(p) = cos(p / 10000 ^ (2i/dpos))
            where
                p --- position id in sequence
                dpos --- position size
        notice:
            shield padding value when added(concated) to features
        args:
            inputs: Tensor([B, T, D], tf.float32)
            position_size: INT
            scope: STR, default is "position_embedding"
        return:
        """
        scope = scope or "position_embedding"
        with tf.variable_scope(scope):
            batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]

            position_j = tf.range(position_size / 2, dtype=tf.float32)
            position_j = 1. / tf.pow(10000., 2 * position_j / position_size)

            position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)

            position_ij = tf.matmul(tf.expand_dims(position_i, 1),
                                    tf.expand_dims(position_j, 0))
            position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
            #position_embedding = tf.stack([position_ij] * batch_size, axis=0)
            position_embedding = tf.expand_dims(position_ij[:,:position_size], 0) \
                                 + tf.zeros((batch_size, seq_len, position_size))
            return position_embedding

    def calc_loss(self, loss_cfg=None):
        """ 计算loss, 支持multi-task
        args:
            loss_weights: LIST or NP.ARRARY
        return:
            loss_list: LIST
        """
        self._loss_list = list()
        loss_cfg = self._cfg["losses"]
        with tf.name_scope("calc_loss"):
            for i in range(self._task_num):
                # self._losses[i] = get_loss(train_args["loss_function"])(
                # class weight: 增加错误#3的loss权重
                params = loss_cfg[i]
                try:
                    alpha = params["alpha"]
                except KeyError:
                    alpha = [1.0 for _ in range(self._target_dim_list[i])]
                print("[INFO] prosody_label_alpha: %s" % str(alpha))
                weights = tf.nn.embedding_lookup(
                    tf.constant(alpha, tf.float32), self._targets_list[i])
                num_classes = self._logits_list[i].get_shape().as_list()[-1]

                try:
                    epsilon = float(params["label_smooth"])
                    print("[INFO] use label smoothing: %.5f" % epsilon)
                    onehot_label = tf.one_hot(self._targets_list[i], num_classes)
                    # label smoothing
                    smooth_label = (1-epsilon) * onehot_label + (epsilon / num_classes)
                    ce_loss = tf.nn.softmax_cross_entropy_with_logits(
                        labels=smooth_label,
                        logits=self._logits_list[i],
                    )
                except KeyError:
                    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self._targets_list[i],
                        logits=self._logits_list[i]
                    )
                ce_loss = ce_loss * weights * self._label_mask
                _loss = tf.reduce_sum(ce_loss) / tf.reduce_sum(self._label_mask)
                '''
                weighted_label = tf.multiply(onehot_label, alpha)
                _loss = tf.losses.softmax_cross_entropy(
                    onehot_labels=tf.reshape(weighted_label, [-1, num_classes]),
                    logits=tf.reshape(self._logits_list[i], [-1, num_classes]),
                    weights=tf.reshape(self._label_mask, [-1]),
                    label_smoothing=label_smooth
                )
                '''
                self._loss_list.append(_loss)
                tf.summary.scalar("loss_%d" % i, self._loss_list[i])

            # weighted loss for multi-task
            try:
                task_weights = [float(x) for x in self._cfg["task_weights"]]
                # assert(len(task_weights) == len(self._loss_list))
            except (KeyError, IndexError) as e:
                print(e)
                task_weights = [1.0] * len(self._loss_list)
            self._loss = sum([l * w for l, w in zip(self._loss_list, task_weights)])
            '''
            # add hidden_classifier loss for hidden layer
            hidden_targets_1 = self._targets_list[0]
            #hidden_targets_1 = tf.clip_by_value(self._targets_list[0], 0, 1)
            _, hidden_loss_1 = self.hidden_classifier(self._hidden_layers[-3], 6, hidden_targets_1,
                                                      self._seq_len, scope="hidden_classifier_1")
            print("[INFO] add hidden_classifier in hidden_layer[-3]")
            self._loss_list.append(hidden_loss_1)
            self._loss += hidden_loss_1
            '''
        return self._loss_list

    def hidden_classifier(self, inputs, dim, targets, seq_lens, scope=None):
        """ hidden layer classifier, and calc the loss according to targets
        args:
            layer_id: INT
            dim: INT
            targets: Tensor([B,T,D], tf.float32)
            seq_lens: Tensor([B,], tf.int32)
        return:
        """
        scope = scope or "hidden_classifier"
        outputs = None
        params = {"num_units": dim, "activation": "linear"}
        outputs, _ = apply_dense(
            inputs,
            params,
            scope=scope
        )
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=targets,
            logits=outputs,
            weights=tf_utils.get_seq_padding_mask(seq_lens)
        )

        return outputs, loss

    def _build_dilated_block(self, params, scope=None):
        scope = scope or "dilated_cnn"
        outputs = None
        return outputs

# DenselyAttentionNet
class MultiSpeakerModel(DilatedCNNModel):
    """ sequence label model for multi speaker, supported multi-task,
    apply word embedding and speaker embedding in model,
    need a config dict, including:
        "feature_dim_list": LIST of INT, the first is wordvec dim
        "target_dim_list" : LIST of INT
        "hidden_layer"    : LIST,
        "output_layer"    : LIST for multi-task
        "loss_weight"     : LIST of FLOAT, for multi-task
        "wordvec_path"    : STR, for word_embedding as initialized value
        "speaker_num"     : INT, for speaker_embedding
        "domain_sep"      : BOOL, whether to apply speaker embedding
    """
    def __init__(self, feature_dim_list, target_dim_list, model_cfg):
        self._feature_dim_list = feature_dim_list
        # load wordvec to embedding
        print("[DEBUG] MultiSpeakerModel: wordvec_path = \n\t%s" %
              model_cfg["wordvec_path"])
        self._wordvec = TextProcessBox.read_wordvec(model_cfg["wordvec_path"])
        (vec_num, vec_dim) = self._wordvec.shape
        if self._feature_dim_list[0] != vec_dim:
            print("[WARNING] change wordvec dim = %d" % vec_dim)
            self._feature_dim_list[0] = self._wordvec[1]

        print("[DEBUG] MultiSpeakerModel: embedded feature_dim = %d" %
              sum(self._feature_dim_list))
        super(MultiSpeakerModel, self).__init__(
            len(self._feature_dim_list), target_dim_list, model_cfg)

        self._speaker_num = model_cfg["speaker_num"]

    def build_graph(self, scope=None):
        """ build graph, need to be called explicitly
        """
        tf.set_random_seed(1)

        self._init_tensors()
        scope = scope or "model"
        with tf.name_scope(scope):
            self._hidden_layers, outputs_list = self._init_layers(
                self._inputs, self._cfg)
            self._logits_list = outputs_list
            self._init_scoring()
        self.calc_loss()

        #tf.summary.histogram("inputs", self.embedding_inputs)
        # print block_out corrcoef matrix in tensorboard
        # block_in = self._hidden_layers[0]
        # tf.summary.histogram("block_in", block_in)

        '''
        for idx, block_out in enumerate(self._hidden_layers[1:-1], 1):
            
            tf.summary.histogram("block_%d_output" % idx, block_out)
            corr_0 = tf_utils.calc_cosine_coef(block_out[0,:self._seq_len[0],:], 
                                                block_in[0,:self._seq_len[0],:])
            #corr = tf.clip_by_value(tf.abs(corr), 0, 1.0)
            corr_0 = tf.abs(corr_0)
            corr_0 = tf.where(tf.greater(corr_0, 0.1), corr_0, tf.zeros_like(corr_0))
            #tar = tf.expand_dims(tf.cast(self._targets_list[0][0,:self._seq_len[0]], dtype=tf.float32), axis=-1)
            pred_0 = tf.one_hot(self._scores_list[0][0,:self._seq_len[0]], 6, on_value=0.1, dtype=tf.float32)
            img_t_0 = tf.concat([pred_0, corr_0], axis=-1)
            tf_utils.plot_2d_tensor(img_t_0, "hidden_layer_%d/outputs_0_corr" % (idx))
            
            corr_1 = tf_utils.calc_cosine_coef(block_out[-1,:self._seq_len[-1],:],
                                                block_in[-1,:self._seq_len[-1],:])
            corr_1 = tf.abs(corr_1)
            corr_1 = tf.where(tf.greater(corr_1, 0.1), corr_1, tf.zeros_like(corr_1))
            pred_1 = tf.one_hot(self._scores_list[0][-1,:self._seq_len[-1]], 6, on_value=0.1, dtype=tf.float32)
            img_t_1 = tf.concat([pred_1, corr_1], axis=-1)
            tf_utils.plot_2d_tensor(img_t_1, "hidden_layer_%d/outputs_1_corr" % (idx))
        '''
    
    def _init_layers(self, layer_in, model_cfg, scope=None):
        """
        :param layer_in:
        :return:
        """
        with tf.name_scope("feature_embedding"):
            embedding_inputs = self._embedding_feature_lookup(layer_in)
            # print(embedding_inputs)
            #self.embedding_inputs = embedding_inputs
        tf.summary.histogram("embedding_inputs", embedding_inputs)
        print("[DEBUG] embedding_inputs dim = %d" % self._feature_dim)
        hidden_layers, output_layers = super(MultiSpeakerModel, self)._init_layers(
            embedding_inputs, model_cfg)
        return hidden_layers, output_layers

    '''
    def prenet(self, layer_in, params, scope="ms_prenet"):
        embedding_inputs = self._embedding_feature_lookup(layer_in)
        # print(embedding_inputs)
        _mask = tf.expand_dims(self._label_mask, axis=-1)
        _mask = tf.concat([_mask] * self.feature_dim, axis=-1)
        # print(_mask)
        layer_in = tf.multiply(embedding_inputs, _mask)
        layer_in = super(MultiSpeakerModel, self).prenet(layer_in, params, scope)
        return layer_in
	'''

    def _embedding_feature_lookup(self, features):
        """ translate feature ids to embedding vector
            first ids is wordvec, others should be one-hot
        args:
            features: Tensor([B, T, num_features], tf.int32)
        return:
            inputs: embedding features, Tensor([B, T, feature_dim], tf.float32)
        """
        # vec_fn = os.path.abspath("../../dict/word2vec_decompress.feat")
        # word2vec = TextProcessBox.read_wordvec(vec_fn, is_compress=False)
        embedding_inputs = list()
        if self._wordvec.shape[1] != self._feature_dim_list[0]:
            sys.exit("unexpected word2vec shape = %s, should be %d" %
                     (str(self._wordvec.shape), self._feature_dim_list[0]))
        # scaled word embedding
        scale = 1.0
        wordvec = self._wordvec * scale
        #word_table = tf.Variable(wordvec, name="word_embedding", trainable=False)
        word_table = tf.get_variable("word_embedding", initializer=wordvec, trainable=False)
        tf.summary.histogram("word_embedding", word_table)
        if self._cfg["domain_sep"]:
            speaker_table = tf.get_variable("speaker_embedding",
                shape=(self._speaker_num, self._wordvec.shape[1]),
                initializer=tf.random_normal_initializer(0, stddev=0.1),
                trainable=True)
            tf.summary.histogram("speaker_embedding", speaker_table)
            tf_utils.plot_2d_tensor(speaker_table, "speaker_embedding")
            table = tf.concat([speaker_table, word_table], axis=0)
        else:
            table = word_table
        word_embedding = tf.nn.embedding_lookup(table, tf.cast(features[..., 0], tf.int32))

        try:
            word2vec_layers = self._cfg["word2vec_layer"]
        except KeyError:
            word2vec_layers = []

        layer_out = word_embedding
        for i, params in enumerate(word2vec_layers):
            layer_type = params["layer_type"].lower()
            scope = "word2vec_layer_%d_%s" % (i, layer_type)
            layer_out = tf_layers.apply_layer(
                layer_out,
                params,
                dropout=0.,
                is_training=self._is_training,
                scope=scope
            )
            print("[DEBUG] %s\n\tparams:%s" % (scope, str(params)))

        # layer_out = layer_out * 0.1
        embedding_inputs.append(layer_out)

        onehot_inputs = list()
        for i in range(1, len(self._feature_dim_list)):
            if self._feature_dim_list[i] == 1:
                onehot_inputs.append(tf.expand_dims(features[..., i], -1))
            else:
                onehot_inputs.append(tf.one_hot(
                    tf.cast(features[..., i], tf.int32),
                    depth=self._feature_dim_list[i],
                    on_value=1., off_value=0., dtype=tf.float32))
        onehot_inputs = tf.concat(onehot_inputs, axis=2)

        try:
            onehot_layers = self._cfg["onehot_layer"]
        except KeyError:
            onehot_layers = []
        scale = 1.0
        layer_out = onehot_inputs * scale
        for i, params in enumerate(onehot_layers):
            layer_type = params["layer_type"].lower()
            scope = "onehot_layer_%d_%s" % (i, layer_type)
            layer_out = tf_layers.apply_layer(
                layer_out,
                params,
                dropout=0.,
                is_training=self._is_training,
                scope=scope
            )
            print("[DEBUG] %s\n\tparams:%s" % (scope, str(params)))
        # scale onehot embedding
        # layer_out = tf.Print(layer_out, [layer_out[0,2,:]], message="onehot_inputs: ")
        embedding_inputs.append(layer_out)
        inputs = tf.concat(embedding_inputs, axis=2)
        inputs = tf.nn.dropout(inputs, keep_prob=1.0-self._dropout)
        self._feature_dim = inputs.get_shape().as_list()[-1]

        _mask = tf.tile(tf.expand_dims(self._label_mask, axis=-1),
                        [1, 1, self._feature_dim])
        #print(_mask)
        inputs = inputs * _mask
        #inputs = inputs * tf.expand_dims(self._label_mask, axis=-1)
        return inputs

    def calc_loss(self, loss_cfg=None):
        """ 计算loss, 支持multi-task
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
                _loss *=  self._label_mask
                aver_loss = tf.reduce_sum(_loss) / tf.reduce_sum(self._label_mask)
                '''
                weighted_label = tf.multiply(onehot_label, alpha)
                _loss = tf.losses.softmax_cross_entropy(
                    onehot_labels=tf.reshape(weighted_label, [-1, num_classes]),
                    logits=tf.reshape(self._logits_list[i], [-1, num_classes]),
                    weights=tf.reshape(self._label_mask, [-1]),
                    label_smoothing=label_smooth
                )
                '''
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
            '''
            # add hidden_classifier loss for hidden layer
            hidden_targets_1 = self._targets_list[0]
            #hidden_targets_1 = tf.clip_by_value(self._targets_list[0], 0, 1)
            _, hidden_loss_1 = self.hidden_classifier(self._hidden_layers[-3], 6, hidden_targets_1,
                                                      self._seq_len, scope="hidden_classifier_1")
            print("[INFO] add hidden_classifier in hidden_layer[-3]")
            self._loss_list.append(hidden_loss_1)
            self._loss += hidden_loss_1
            '''
        return self._loss_list