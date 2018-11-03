#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: conv1d_layer.py
# @brief: 
# @author: niezhipeng
# @Created on 2018/11/2
# *************************************************************************************

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.framework import tensor_shape
import common.tf_utils as tf_utils


class Conv1d(base.Layer):
    """ 1D convolution layer (e.g. temporal convolution)
        support dilated convolution
    """
    def __init__(self, kernel_size, filters,
                 dilation_rate=1,
                 strides=1,
                 padding="same",
                 kernel_initializer=None,
                 add_bias=True,
                 use_wn=False,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Conv1d, self).__init__(trainable=trainable, name=name,
                                      activity_regularizer=None,
                                      **kwargs)
        self.rank = 1
        self.filters = filters
        self.kernel_size = utils.normalize_tuple(
            kernel_size, self.rank, "kernel_size")
        self.dilation_rate = utils.normalize_tuple(
            dilation_rate, self.rank, "dilation_rate")
        self.strides = utils.normalize_tuple(
            strides, self.rank, "strides")
        self.padding = utils.normalize_padding(padding)
        self.kernel_initializer = kernel_initializer
        self.add_bias = add_bias
        self.use_wn = use_wn

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_dim = input_shape[-1].value
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_variable(name="kernel",
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        trainable=True,
                                        dtype=self.dtype)
        if self.use_wn:
            self.kernel = tf_utils.weight_norm(self.kernel, self.name)
        img_kernel = tf.reshape(self.kernel, shape=(-1, self.filters))
        tf_utils.plot_2d_tensor(img_kernel, "%s_kernel" % self.name)
        tf.summary.histogram("%s_kernel" % self.name, self.kernel)

        if self.add_bias:
            self.bias = self.add_variable(name="bias",
                                          shape=(self.filters,),
                                          initializer=tf.zeros_initializer(),
                                          trainable=True,
                                          dtype=self.dtype)
            tf.summary.histogram("%s_bias" % self.name, self.bias)
        else:
            self.bias = None
        self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                         axes={-1: input_dim})

        self.built = True

    def call(self, inputs, **kwargs):
        outputs = tf.nn.convolution(
            inputs,
            self.kernel,
            padding=self.padding.upper(),
            strides=self.strides,
            dilation_rate=self.dilation_rate)

        if self.add_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        return outputs

    def _compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                        [self.filters])


def apply_conv1d(inputs, params, is_training, dropout=0., scope=None):
    """ convolution layer
    args:
        inputs: Tensor([B, T, D], tf.float32)
        params: DICT, {
            "kernel_size": int,
            "channels": int,
            "activation": str
        }
        is_training:
        dropout:
        scope:
    :return:
        Tensor([B, T, channels], tf.float32)
    """
    keep_prob = 1. - dropout

    kernel_size = int(params["kernel_size"])
    channels = int(params["channels"])
    activation = params["activation"].lower()

    try:
        initializer = params["initializer"].lower()
    except KeyError:
        initializer = "xavier_normal"

    try:
        dilation_rate = int(params["dilation_rate"])
    except KeyError:
        dilation_rate = 1

    type = params["layer_type"].lower()
    if "atrous" in type:
        scope = scope or "atrous_conv1d"
        use_bn = "use_bn" in params and params["use_bn"]
        use_wn = "use_wn" in params and params["use_wn"]
        inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
        outputs = atrous_conv1d(inputs, kernel_size, channels, dilation_rate,
                                activation,
                                kernel_initializer=initializer,
                                use_bn=use_bn,
                                use_wn=use_wn,
                                is_training=is_training,
                                scope=scope)
        # outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
        return outputs
    elif "casual" in type:
        scope = scope or "casual_conv1d"
        outputs = causal_conv1d(inputs, kernel_size, channels, dilation_rate,
                                activation,
                                kernel_initializer=initializer,
                                is_training=is_training,
                                scope=scope)
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
        return outputs
    elif "gated" in type:
        scope = scope or "gated_conv1d"
        outputs = gated_conv1d(inputs, kernel_size, channels, dilation_rate,
                                activation,
                                kernel_initializer=initializer,
                                is_training=is_training,
                                scope=scope)
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
        return outputs
    else:
        scope = scope or "conv1d"
        outputs = conv1d_layer(inputs, kernel_size, channels,
                         activation, is_training, scope=scope)
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
        return outputs

def conv1d(inputs, kernel_size, channels,
           strides=1,
           dilation_rate=1,
           padding="same",
           kernel_initializer=None,
           is_relu=True,
           add_bias=True,
           use_wn=False,
           trainable=True,
           scope=None,
           reuse=None):
    _initializer = tf_utils.get_initializer(
        kernel_initializer, is_relu)

    layer = Conv1d(kernel_size, channels,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        kernel_initializer=_initializer,
        add_bias=add_bias,
        use_wn=use_wn,
        trainable=trainable,
        name=scope,
        dtype=inputs.dtype.base_dtype,
        _reuse=reuse,
        _scope=scope)
    return layer.apply(inputs)

def conv1d_layer(inputs, kernel_size, channels, activation, is_training, scope=None):
    scope = scope or "conv1d"
    try:
        use_relu = activation in ["relu", "RELU", "ReLU"]
        activation = tf_utils.get_activation(activation)
    except Exception as e:
        print(e)
        activation = None
        use_relu = False

    kernel_initializer = tf_utils.get_initializer("xavier", is_relu=use_relu)

    with tf.variable_scope(scope):
        output = tf.layers.conv1d(
            inputs,
            filters=channels,
            kernel_size=kernel_size,
            activation=activation,
            kernel_initializer=kernel_initializer,
            padding='same')
        output = tf.layers.batch_normalization(output, training=is_training)
        tf.summary.histogram("outputs", output)
        return output

'''
def atrous_conv1d(inputs, kernel_size, channels, params, is_training, scope=None):
    """
    args:
        inputs: Tensor([B, T, D], tf.float32)
        kernel_size: INT, filter width
        channels: INT, output_channels
        params: DICT, {
            "dilation_rate":INT,
            "initializer": STR,
            "activation": ST}
        is_training: BOOL for batch normalization
            True --- train, False --- prediction
        scope:
    return:
    """
    scope = scope or "atrous_conv1d"
    dilation_rate = int(params["dilation_rate"])
    use_relu = params["activation"].lower() == "relu"
    initializer = get_initializer(params["initializer"], use_relu)
    activation = get_activation(params["activation"])
    use_bn = params["use_bn"]
    with tf.variable_scope(scope):
        output = tf.layers.conv1d(
            inputs,
            filters=channels,
            kernel_size=kernel_size,
            activation=activation,
            kernel_initializer=initializer,
            dilation_rate=dilation_rate,
            padding="same")
        if use_bn:
            output = tf.layers.batch_normalization(output,
                        training=is_training)
        return output
'''


def atrous_conv1d(inputs, kernel_size, channels,
                  dilation_rate=1, activation=None, kernel_initializer=None,
                  use_bn=False, use_wn=False, is_training=True, scope=None):
    """ atrous cnn for text
    args:
        inputs: Tensor([B, T, D], tf.float32)
        kernel_size: INT, filter width
        channels: INT, output_channels
        dilation_rate: INT,
        activation: STR
        kernel_initializer: STR
        use_bn: whether to use batch normalization
        is_training: BOOL for batch normalization
            True --- train, False --- prediction
        scope:
    return:
    """
    scope = scope or "atrous_conv1d"

    use_relu = activation in ["relu", "ReLU", "RELU"]
    outputs = inputs

    try:
        activation = tf_utils.get_activation(activation)
    except Exception as e:
        print(e)
        activation = None

    # if kernel_initializer is None:
    #    kernel_initializer = xavier_initializer(use_relu)

    with tf.variable_scope(scope):
        outputs = conv1d(inputs, kernel_size, channels,
                         dilation_rate=dilation_rate,
                         kernel_initializer=kernel_initializer,
                         is_relu=use_relu,
                         add_bias=(not use_bn),
                         use_wn=use_wn)
        '''
        corr_0 = tf_utils.calc_cosine_coef(output[0, :20, :], output[0, :20, :])
        corr_0 = tf.abs(corr_0)
        corr_0 = tf.where(tf.greater(corr_0, 0.1), corr_0, tf.zeros_like(corr_0))
        corr_0 = tf.Print(corr_0, [corr_0])
        tf_utils.plot_2d_tensor(
            corr_0, "%s/outputs_0_self_corr" % scope)

        corr_1 = tf_utils.calc_cosine_coef(output[-1, :20, :], output[-1, :20, :])
        corr_1 = tf.abs(corr_1)
        corr_1 = tf.where(tf.greater(corr_1, 0.1), corr_1, tf.zeros_like(corr_1))
        corr_1 = tf.Print(corr_1, [corr_1])
        tf_utils.plot_2d_tensor(
            corr_1, "%s/outputs_1_self_corr" % scope)
        '''
        tf.summary.histogram("conv_out", outputs)

        if activation is not None:
            outputs = activation(outputs)

        if use_bn:
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
        tf.summary.histogram("bn_out", outputs)
        return outputs


def causal_conv1d(inputs, kernel_size, channels,
                  dilation_rate=1, activation=None, kernel_initializer=None,
                  use_bn=True, use_wn=False, is_training=True, scope=None):
    """ causal conv1d by
            padding (kernel_size - 2) * dilation_rate elements before inputs
    args:
        inputs: Tensor([B, T, D], tf.float32)
        kernel_size: INT, filter width
        channels: INT, output_channels
        dilation_rate: INT,
        activation: STR
        kernel_initializer: STR
        use_bn: whether to use batch normalization
        is_training: BOOL for batch normalization
            True --- train, False --- prediction
        scope:
    return:
    """
    scope = scope or "causal_conv1d"
    output = None
    with tf.variable_scope(scope):
        padded_num = (kernel_size - 2) * dilation_rate

        padded = tf.pad(inputs, [[0, 0], [padded_num, 0], [0, 0]])

        output = atrous_conv1d(padded, kernel_size, channels,
                               dilation_rate=dilation_rate,
                               activation=activation,
                               kernel_initializer=kernel_initializer,
                               use_bn=use_bn,
                               use_wn=use_wn,
                               is_training=is_training,
                               scope=scope)

        output = tf.slice(output, [0, 0, 0], [-1, tf.shape(inputs)[1], -1])
    return output


def gated_conv1d(inputs, kernel_size, channels,
                 dilation_rate=1, activation=None, kernel_initializer=None,
                 use_bn=False, use_wn=False, is_training=True, scope=None):
    """ gated conv1d:
                      |--> conv1d --> activation -->|
            inputs -->|                             * --> bn
                      |--> conv1d -->   sigmoid  -->|
    args:
        inputs: Tensor([B, T, D], tf.float32)
        kernel_size: INT, filter width
        channels: INT, output_channels
        dilation_rate: INT,
        activation: STR
        kernel_initializer: STR
        use_bn: whether to use batch normalization
        is_training: BOOL for batch normalization
            True --- train, False --- prediction
        scope:
    return:
    """
    scope = scope or "gated_conv1d"

    use_relu = activation in ["relu", "ReLU", "RELU"]

    try:
        activation = tf_utils.get_activation(activation)
    except Exception as e:
        print(e)
        activation = None

    # if kernel_initializer is None:
    #    kernel_initializer = xavier_initializer(use_relu)

    with tf.variable_scope(scope):
        conv_out = conv1d(inputs, kernel_size, channels,
                          dilation_rate=dilation_rate,
                          kernel_initializer=kernel_initializer,
                          is_relu=use_relu,
                          add_bias=(not use_bn),
                          use_wn=use_wn,
                          scope="filter")

        if activation is not None:
            conv_out = activation(conv_out)

        gated = conv1d(inputs, kernel_size, channels,
                        dilation_rate=dilation_rate,
                        kernel_initializer=kernel_initializer,
                        is_relu=False,
                        add_bias=True,
                        use_wn=use_wn,
                        scope="gate")
        gated_out = tf.nn.sigmoid(gated)
        tf.summary.histogram("%s_gated_out" % scope, gated_out)

        output = conv_out * gated_out

    if use_bn:
        output = tf.layers.batch_normalization(output, training=is_training)
    return output


def conv_bank_pooling1d(conv_banks, pool_type="max", scope=None):
    """ merge the last dim of Tensors acoording to the pooling type
    args:
        conv_banks: LIST of Tensor([B, T, D], )
        type: "max" or "avg"
        scope:
    return:
        Tensor([B,T,D])
    """
    scope = scope or "conv_banks_pooling"

    input_shape = conv_banks[0].get_shape()
    # print(input_shape)
    dim = input_shape[-1].value

    pool_type = pool_type.lower()
    assert pool_type in ["max", "avg", "average"], \
        "\"%s\" not be supported in conv_banks_pooling" % type
    outputs = None
    with tf.variable_scope(scope):
        pool_inputs = [tf.reshape(x, (-1, dim)) for x in conv_banks]
        pool_inputs = tf.stack(pool_inputs, axis=1)
        if pool_type == "max":
            max_outputs = tf.layers.max_pooling1d(
                pool_inputs,
                pool_size=len(conv_banks),
                strides=1,
                padding="valid")
            outputs = tf.reshape(max_outputs, [tf.shape(conv_banks[0])[0], -1, dim])
        elif pool_type == "avg" or pool_type == "average":
            avg_outputs = tf.layers.average_pooling1d(
                pool_inputs,
                pool_size=len(conv_banks),
                strides=1,
                padding="valid")
            outputs = tf.reshape(avg_outputs, [tf.shape(conv_banks[0])[0], -1, dim])
        return outputs
