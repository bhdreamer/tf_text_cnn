#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: conv1d_layer_v1.py
# @brief: 
# @author: niezhipeng
# @Created on 2018/11/2
# *************************************************************************************
import tensorflow as tf
import common.tf_utils as tf_utils


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
            dilation_rate=1, kernel_initializer=None,
            is_relu=False, add_bias=True, use_wn=False, scope=None):
    """ atrous conv1d op:
            convolution(kernel, input) + bias
    args:
        inputs: Tensor([B, T, D], tf.float32)
        kernel_size: INT, filter_size
        channels: INT, output_channels
        dilation_rate: INT,
        kernel_initializer: STR
        is_relu: True for ReLU; otherwise, False
        add_bias: True(default), False for using batch norm
        use_wn: True for weight normalization
        scope:
    return:
        Tensor([B, T, channels], tf.float32)
    """
    scope = scope or "conv1d"

    input_shape = inputs.get_shape().as_list()
    input_dim = input_shape[-1]
    kernel_shape = (kernel_size, input_dim, channels)
    # TODO: 用通用的接口替换
    _kernel = tf_utils.get_initialize_variable(
        '%s_kernel' % scope,
        shape=kernel_shape,
        initializer=kernel_initializer,
        is_relu=is_relu)

    if use_wn:
        # v_norm = tf.nn.l2_normalize(_kernel.initialized_value(), [0, 1])
        # m_init, v_init = tf.nn.moments(_kernel.initialized_value(), [0, 1])
        # scale_init = v_norm / v_init
        _kernel = tf_utils.weight_norm(_kernel, scope)
        '''
        _g = tf.get_variable(name="%s_g" % scope,
                             shape=[channels],
                             initializer=tf.ones_initializer(1.),
                             dtype=_kernel.dtype,
                             trainable=True)
        _kernel = tf.reshape(_g, [1, 1, channels]) * tf.nn.l2_normalize(_kernel, dim=[0, 1, 2])
        '''
    img_kernel = tf.reshape(_kernel, shape=(-1, channels))
    tf_utils.plot_2d_tensor(img_kernel, "%s/kernel" % scope)
    # tf_utils.plot_2d_tensor(kernel[0, :, :], "%s/kernel_0" % scope)
    tf.summary.histogram("%s_kernel" % scope, _kernel)

    conv_out = tf.nn.convolution(
        inputs,
        filter=_kernel,
        padding="SAME",
        strides=(1,),
        dilation_rate=(dilation_rate,))

    if not add_bias:
        return conv_out

    _bias = tf.get_variable('%s_bias' % scope,
                            shape=(channels,),
                            initializer=tf.constant_initializer(0.),
                            trainable=True,
                            dtype=tf.float32)
    tf.summary.histogram("%s_bias" % scope, _bias)
    return tf.nn.bias_add(conv_out, _bias)

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
    output = inputs

    try:
        activation = tf_utils.get_activation(activation)
    except Exception as e:
        print(e)
        activation = None

    # if kernel_initializer is None:
    #    kernel_initializer = xavier_initializer(use_relu)

    with tf.variable_scope(scope):
        output = _conv1d(output, kernel_size, channels,
                         dilation_rate=dilation_rate,
                         kernel_initializer=kernel_initializer,
                         is_relu=use_relu,
                         add_bias=(not use_bn),
                         use_wn=use_wn,
                         scope="atrous")
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
        tf.summary.histogram("conv_out", output)

        if activation is not None:
            output = activation(output)

        if use_bn:
            output = tf.layers.batch_normalization(output, training=is_training)
        tf.summary.histogram("bn_out", output)
        return output


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
        conv_out = _conv1d(inputs, kernel_size, channels,
                           dilation_rate=dilation_rate,
                           kernel_initializer=kernel_initializer,
                           is_relu=use_relu,
                           add_bias=(not use_bn),
                           use_wn=use_wn,
                           scope="filter")

        if activation is not None:
            conv_out  = activation(conv_out)

        gated = _conv1d(inputs, kernel_size, channels,
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
