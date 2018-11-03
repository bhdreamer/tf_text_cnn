#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: densely_attention_net.py
# @brief: 
# @author: niezhipeng
# @Created on 2018/9/4
# *************************************************************************************

import numpy as np
import tensorflow as tf
from common import tf_utils
from models.basic_model import BasicModel

from common import layers as tf_layers
from common.conv1d_layer import conv1d
from tensorflow import set_random_seed
set_random_seed(1)

class DenselyAttentionNet(BasicModel):
    """ densely cnn + multi-scale attention
    Ref: http://coai.cs.tsinghua.edu.cn/hml/media/files/2018wangshiyao_DenselyCNN.pdf
    """
    def __init__(self, feature_dim, target_dim_list, model_cfg):
        super(DenselyAttentionNet, self).__init__(
            feature_dim, target_dim_list, model_cfg)

    def build_graph(self, scope=None):
        """ build graph
        """
        self._init_tensors()

        tf.summary.histogram("inputs", self._inputs)
        scope = scope or "model"
        with tf.name_scope(scope):
            hidden_layers, outputs_list = self._init_layers(self._inputs, self._cfg)
            self._logits_list = outputs_list
            self._init_scoring()
        self.calc_loss()

    def _init_layers(self, layer_in, model_cfg, scope=None):
        """ initialize prenet, densely_cnn_block, output layers
        args:
            layer_in:
            model_cfg:
            scope:
        """
        hidden_layers = list()
        output_layers = list()

        # prenet
        block_in = self.prenet(layer_in, model_cfg["prenet"],
                               dropout=self._dropout,
                               scope="prenet")
        print("[DEBUG] prenet\n\tparams = %s" % str(model_cfg["prenet"]))
        tf.summary.histogram("prenet_out", block_in)
        hidden_layers.append(block_in)

        # densely_cnn_block
        for idx, params in enumerate(model_cfg["hidden_layer"]):
            layer_type = params["layer_type"].lower()
            _scope = "hidden_layer_%d_%s" % (idx, layer_type)
            if layer_type == "densely_cnn_block":
                block_out = self.densely_cnn_block(
                    block_in,
                    params,
                    dropout=self._dropout,
                    is_training=self._is_training,
                    scope=_scope
                )
            elif layer_type == "attention":
                block_out = self.attention_layer(
                    block_in,
                    params,
                    dropout=0.,
                    is_training=self._is_training,
                    scope=_scope
                )
            elif layer_type == "highway":
                block_out = self.highway_net(
                    block_in,
                    params,
                    dropout=0.,
                    scope=_scope
                )
            else:
                block_out = tf_layers.apply_layer(
                    block_in,
                    params,
                    self._seq_len,
                    dropout=self._dropout,
                    is_training=self._is_training,
                    scope=_scope
                )
            print("[DEBUG] %s\n\tparams = %s" % (_scope, str(params)))
            hidden_layers.append(block_out)
            block_in = block_out

        # attention layers
        #atten_outputs = self.attention_layers(block_out, model_cfg["attention_layers"])

        # output layers
        for idx, params in enumerate(model_cfg["output_layer"]):
            params["num_units"] = self._target_dim_list[idx]
            layer_out = tf_layers.apply_layer(
                block_in,
                params,
                dropout=self._dropout,
                is_training=False,
                scope="output_layer_%d" % idx,
            )
            print("[DEBUG] output_layer_%d" % idx)
            print("\tparams = %s" %  str(params))
            output_layers.append(layer_out)
        return hidden_layers, output_layers

    def prenet(self, layer_in, params, dropout=0., scope="prenet"):
        """ preproccessing networks, including
            transform feature, position_embedding
            x --> conv --> ReLU(bn) --> conv --|-> concat --> ReLU(bn) -->
                        |----------------------|
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
        use_wn = "use_wn" in params and params["use_wn"]
        try:
            activation = tf_utils.get_activation(params["activation"].lower())
            use_relu = params["activation"].lower() == "relu"
        except KeyError:
            activation = lambda x: x
            use_relu = False


        with tf.variable_scope(scope):
            layer_in = tf.nn.dropout(layer_in, keep_prob=1.0-dropout)
            conv1 = conv1d(layer_in, kernel_size=1, channels=channels*2,
                           add_bias=(not use_bn),
                           is_relu=use_relu,
                           use_wn=use_wn,
                           kernel_initializer=_initializer,
                           scope="conv1")
            outputs = conv1
            #outputs = activation(outputs)

            if use_bn:
                outputs = tf.layers.batch_normalization(outputs, training=self._is_training)
            outputs = activation(outputs)

            conv2 = conv1d(outputs, kernel_size=1, channels=channels,
                           add_bias=(not use_bn),
                           is_relu=use_relu,
                           use_wn=use_wn,
                           kernel_initializer=_initializer,
                           scope="conv2")

            '''
            conv2_out = tf.nn.relu(conv2)
            if use_bn:
                conv2_out = tf.layers.batch_normalization(conv2_out, training=self._is_training)
            conv3 = conv1d(conv2_out, kernel_size=1, channels=channels,
                           add_bias=False,
                           is_relu=True,
                           kernel_initializer=_initializer,
                           scope="conv3")
            '''
            #outputs = tf.concat((conv1, conv2), axis=-1)
            outputs = conv2
            # outputs = activation(outputs)
            if use_bn:
                outputs = tf.layers.batch_normalization(outputs, training=self._is_training)
            outputs = activation(outputs)
            return outputs

    def densely_cnn_block(self, inputs, params, dropout=0., is_training=False, scope=None):
        """
            inputs --> conv --|--> concat --> activation --> bn --> (pooling) -->
                    |---------|
        args:
            inputs: Tensor([B, T, D], tf.float32)
            params: DICT, {
                    "kernel_size",
                    "channels",
                    "dilation_rate",
                    "initializer",
                    "activation",
                    "use_bn"
            }
            is_training:
            scope:
        return:
        """
        scope = scope or "densely_conv1d"
        kernel_size = int(params["kernel_size"])
        channels = int(params["channels"])
        try:
            dilation_rate = params["dilation_rate"]
        except KeyError:
            dilation_rate = 1

        try:
            _initializer = params["initializer"].lower()
        except KeyError:
            _initializer = "xavier_normal"

        try:
            activation = tf_utils.get_activation(params["activation"])
        except KeyError:
            activation = lambda x: x

        use_bn = "use_bn" in params and params["use_bn"]
        use_wn = "use_wn" in params and params["use_wn"]

        with tf.variable_scope(scope):
            outputs = inputs
            outputs = conv1d(outputs, kernel_size, channels,
                             dilation_rate=dilation_rate,
                             kernel_initializer=_initializer,
                             is_relu=True,
                             use_wn=use_wn,
                             add_bias=(not use_bn))
            outputs = tf.nn.dropout(outputs, keep_prob=1.0-dropout)
            tf.summary.histogram("%s_conv" % scope, outputs)
            # outputs = activation(outputs)
            if use_bn:
                # outputs = tf_utils.group_norm(outputs, G=32)
                outputs = tf.layers.batch_normalization(outputs, training=is_training)
            outputs = activation(outputs)
            # pooling
            '''
            outputs = tf.nn.pool(outputs,
                                 window_shape=[],
                                 pooling_type="AVG",
                                 padding="SAME")
            '''
            tf.summary.histogram("%s_bn" % scope, outputs)

            outputs = tf.concat((inputs, outputs), axis=-1)
            return outputs

    def highway_net(self, layer_in, params, dropout=0., scope=None):
        scope = scope or "highway"
        num_layers = int(params["num_layers"])
        try:
            activation = params["activation"].lower()
        except Exception as e:
            print(e)
            activation = "relu"
        num_units = layer_in.shape.as_list()[-1]
        layer_out = layer_in
        for i in range(num_layers):
            layer_out = tf.nn.dropout(layer_out, keep_prob=1.0-dropout)
            layer_out = tf_layers.highwaynet(
                layer_out, num_units,
                activation=activation,
                scope="%s_%d" % (scope, i))
        return layer_out

    def group_conv1d(self, layer_in, groups, channels, scope=None):
        values = tf.split(layer_in, groups, axis=-1)
        for i in range(groups):
            values[i] = tconv1d(values[i], 1, channels,
                                kernel_initializer="xavier_normal",
                                is_relu=True,
                                add_bias=False)
        return tf.concat(values, axis=-1)

    def attention_layer(self, layer_in, params, dropout=0., is_training=False, scope=None):
        """ group_conv
            [B, T, D] --> [B, T, G, D//G] --> pooling([B, T, 1, D//G]) -->
            highway --> [B, T, D//G]
        args:
            layer_in: Tensor([B, T, D], tf.float32)
            params: DICT, {"channels", "groups", "activation", "use_wn"}
            scope:
        return:
        """
        G = params["groups"]
        input_dim = layer_in.shape.as_list()[-1]

        with tf.name_scope(scope):
            dims = layer_in.shape.ndims
            C = input_dim // G
            spatial_shape = [tf.shape(layer_in)[i] for i in range(dims-1)]
            # [B, T, G, C//G]
            inputs = tf.reshape(layer_in, spatial_shape + [G, C])

            # max-over-group pooling
            pooling_out = tf.nn.max_pool(inputs, [1, 1, G, 1], [1, 1, 1, 1], "VALID")
            print(pooling_out)
            # highway
            outputs = tf.squeeze(pooling_out, axis=2)
            params = params["highway"]
            params["num_units"] = C
            outputs = self.highway_net(outputs, params, scope="highway")

            print(outputs)
            return outputs

    '''
    def attention_layer(self, layer_in, params, dropout=0., is_training=False, scope=None):
        """ group_conv
            [B, T, D] --> [B, T, G, D//G] -->
            conv2d([1, G, D//G, channels]) --> relu -->
            [B,T,channels]
        args:
            layer_in: Tensor([B, T, D], tf.float32)
            params: DICT, {"channels", "groups", "activation", "use_wn"}
            scope:
        return:
        """
        G = params["groups"]
        input_dim = layer_in.shape.as_list()[-1]
        channels = params["channels"]
        try:
            activation = tf_utils.get_activation(params["activation"])
        except KeyError:
            activation = lambda x: x
        use_relu = params["activation"] == "relu"
        use_wn = "use_wn" in params and params["use_wn"]
        use_bn = "use_bn" in params and params["use_bn"]
        with tf.name_scope(scope):
            dims = layer_in.shape.ndims
            C = input_dim // G
            spatial_shape = [tf.shape(layer_in)[i] for i in range(dims-1)]
            # [B, T, G, C//G]
            inputs = tf.reshape(layer_in, spatial_shape + [G, C])
            _kernel = tf_utils.get_initialize_variable(
                "%s_kernel" % scope,
                shape=(1, G, C, channels),
                initializer="xavier_normal",
                is_relu=use_relu)
            if use_wn:
                _kernel = tf_utils.weight_norm(_kernel, scope)
            # conv2d([1, G, C//G, channels])
            conv2d_out = tf.nn.conv2d(inputs,
                                      _kernel,
                                      strides=(1, 1, 1, 1),
                                      padding="VALID")

            outputs = activation(conv2d_out)
            if use_bn:
                # outputs = tf_utils.group_norm(outputs, G=32)
                outputs = tf.layers.batch_normalization(outputs, training=is_training)
            # [B, T, 1, channels] --> [B, T, channels]
            outputs = tf.reshape(outputs, spatial_shape + [channels])
            print(outputs)
            return outputs
    '''
    '''
    def attention_layer(self, layer_in, params, dropout=0., is_training=False, scope=None):
        """ weighted n-gram layer
        args:
            layer_in: Tensor([B, T, D], tf.float32)
            params: DICT, {"groups", "channels", "use_bn", ""}
            scope:
        return:
        """
        groups = params["groups"]
        channels = params["channels"]
        use_bn = "use_bn" in params and params["use_bn"]
        use_wn = "use_wn" in params and params["use_wn"]
        scope = scope or "attention_layer"

        def _group_conv1d(x, groups, channels, use_wn, scope=None):
            values = tf.split(x, groups, axis=-1)
            for i in range(groups):
                with tf.variable_scope("%s/conv1d_%d" % (scope, i)):
                    values[i] = conv1d(values[i], 1, channels,
                                       kernel_initializer="xavier_normal",
                                       is_relu=True,
                                       add_bias=False,
                                       use_wn=use_wn)
            return tf.concat(values, axis=-1)

        with tf.name_scope(scope):
            # [B, T, channels*groups]
            out1 = _group_conv1d(layer_in, groups, channels, use_wn, "group_conv1")

            out2 = tf.nn.relu(out1)
            if use_bn:
                out2 = tf.layers.batch_normalization(out2, training=self._is_training)

            # [B, T, channels*groups]
            out2 = _group_conv1d(out2, groups, channels, use_wn, "group_conv2")

            out3 = tf.nn.relu(out2)
            if use_bn:
                out3 = tf.layers.batch_normalization(out3, training=self._is_training)

            # [B, T, channels] * groups
            values = tf.split(out3, num_or_size_splits=groups, axis=-1)
            # [B, T, 1] * groups
            s = [tf.reduce_sum(values[i], axis=-1, keep_dims=True)
                      for i in range(groups)]
            # [B, T, 1*groups]
            alphas = tf.nn.softmax(tf.concat(s, axis=-1))
            tf_utils.plot_2d_tensor(tf.transpose(alphas[0,:,:]), "attention_weight_0")
            tf_utils.plot_2d_tensor(tf.transpose(alphas[-1,:,:]), "attention_weight_1")
            # [B, T, 1] * groups
            weights = tf.split(alphas, num_or_size_splits=groups, axis=-1)
            # [B, T, channels] * groups
            atten_values = list()
            for i in range(groups):
                atten_values.append(
                    tf.multiply(values[i], tf.tile(weights[i], [1, 1, channels])))

            return  tf.add_n(atten_values)
    '''
