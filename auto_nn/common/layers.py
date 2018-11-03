#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: basic_model.py
# @brief: 
# @author: niezhipeng(@baidu.com)
# @Created on 2018/4/11
# *************************************************************************************
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.contrib import rnn
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from common.conv1d_layer_v1 import apply_conv1d
import common.tf_utils as tf_utils


def apply_layer(inputs, params, seq_len=None, dropout=0., is_training=False, scope=None):
    layer_type = params["layer_type"].lower()
    if layer_type in ["fc", "dnn"]:
        layer_out, _ = apply_dense(
            inputs,
            params,
            is_training=is_training,
            dropout=dropout,
            scope=scope)
    elif "rnn" in layer_type:
        cell_name, params = check_rnn_params(params)
        if cell_name is None:
            print("[Error] invalid params in \"%s\"" % params["layer_type"])
            return None
        if seq_len is None:
            seq_len = tf_utils.calc_sequence_lengths(inputs)
        if params["struct"] == "bi":
            layer_out, _, _ = apply_bi_rnn(
                inputs,
                seq_len,
                cell_name,
                params,
                dropout=dropout,
                scope=scope)
        else:
            layer_out, _, _ = apply_uni_rnn(
                inputs,
                seq_len,
                cell_name,
                params,
                dropout=dropout,
                scope=scope)
    elif layer_type == "cbhg":
        if seq_len is None:
            seq_len = tf_utils.calc_sequence_lengths(inputs)
        layer_out = cbhg_v2(
            inputs,
            seq_len,
            params,
            is_training=is_training,
            scope=scope)
    elif "cnn" in layer_type or "conv" in layer_type:
        layer_out = apply_conv1d(
            inputs,
            params,
            is_training,
            dropout=dropout,
            scope=scope)
    elif "highway" in layer_type:
        layer_out = apply_highway(
            inputs,
            params,
            dropout=dropout,
            scope=scope)
    else:
        print("[Error] \"%s\" not be supported !" % params["layer_type"])
        layer_out = inputs
    return layer_out


def apply_dense(inputs, params, is_training=False, dropout=0.0, scope=None):
    """
    full connection layer
    num_params: w: in_dim * num_units, b: num_units

    Args:
        inputs (): 输入, [batch, len, dim]
        params (): DICT, {"num_units", "activation"}
        dropout (): dropout
        scope (): tensorflow域名

    Returns: dense层输出, 需要初始化的变量

    """
    num_units = int(params["num_units"])

    try:
        activation = params["activation"].lower()
    except KeyError:
        activation = "linear"
    try:
        initializer = params["initializer"].lower()
    except KeyError:
        initializer = "xavier_normal"
    use_bn = "use_bn" in params and params["use_bn"]

    temp = set(tf.global_variables())
    scope = scope or "dense"
    with tf.variable_scope(scope):
        shape = inputs.get_shape()
        input_dim = shape[-1].value
        inputs_2d = tf.reshape(inputs, [-1, input_dim])
        inputs_2d = tf.nn.dropout(inputs_2d, keep_prob=1.0-dropout)
        initializer = tf_utils.get_initializer(initializer, is_relu=(activation == "relu"))
        w = tf.get_variable("w", dtype=tf.float32,
                            shape=[input_dim, num_units],
                            initializer=initializer)
        b = tf.get_variable("b", dtype=tf.float32,
                            shape=[1, num_units],
                            initializer=tf.zeros_initializer())
        outputs = tf.matmul(inputs_2d, w) + b

        outputs = tf_utils.get_activation(activation)(outputs)

        outputs = tf.reshape(outputs, [tf.shape(inputs)[0], -1, num_units])
               
        if use_bn:
            outputs = tf.layers.batch_normalization(outputs, training=is_training)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("outputs", outputs)
        # outputs = tf.nn.dropout(outputs, keep_prob=1.0-dropout)
    init_var_set = set(tf.global_variables()) - temp
    return outputs, init_var_set


def apply_bi_rnn(inputs, seq_len, cell_name, params, dropout=0.0, scope=None):
    """
    申请双向rnn层
    Args:
        inputs (): 输入, (batch, len, dim)
        seq_len (): 序列长度, (batch, )
        cell_name (): cell名称
        params (): cell配置参数
        dropout (): dropout
        scope (): tensorflow域名

    Returns: 输出, 最后一个state, 需要初始化的参数

    """
    temp = set(tf.global_variables())

    cell_params = params["cell_params"]
    keep_prob = 1. - dropout
    outputs = None
    scope = scope or "bidirectional_rnn"

    if "concat_output" not in params:
        concat_output = "num_proj" not in cell_params
    else:
        concat_output = params["concat_output"]

    with tf.variable_scope(scope):
        fw_rnn_cell = getattr(rnn, cell_name)(**cell_params)
        # fw_rnn_cell = rnn.DropoutWrapper(fw_rnn_cell,output_keep_prob=keep_prob)
        bw_rnn_cell = getattr(rnn, cell_name)(**cell_params)
        # bw_rnn_cell = rnn.DropoutWrapper(bw_rnn_cell, output_keep_prob=keep_prob)
        outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
            fw_rnn_cell, bw_rnn_cell, inputs,
            sequence_length=seq_len,
            initial_state_fw=None,
            initial_state_bw=None,
            dtype="float",
            scope="bidirectional_rnn"
        )

        if concat_output:
            outputs = tf.concat(outputs, 2)
        else:
            outputs = outputs[0] + outputs[1]
        # final_state = tf.concat((fw_state, bw_state), 1)

        final_state = None
        if isinstance(fw_state, rnn.LSTMStateTuple):
            # state_c = tf.concat((fw_state.c, bw_state.c), 1)
            # state_h = tf.concat((fw_state.h, bw_state.h), 1)
            # final_state = rnn.LSTMStateTuple(c=state_c, h=state_h)
            final_state = tf.concat((fw_state.c, bw_state.c), 1)
        elif isinstance(fw_state, tf.Tensor):
            final_state = tf.concat((fw_state, bw_state), 1)

        # outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
        # tf.summary.histogram("fw_state", fw_state)
        # tf.summary.histogram("bw_state", bw_state)
        tf.summary.histogram("final_state", final_state)
        tf.summary.histogram("outputs", outputs)
    init_var_set = set(tf.global_variables()) - temp
    return outputs, final_state, init_var_set


def apply_uni_rnn(inputs, seq_len, cell_name, params, dropout=0.0, scope=None):
    """
    申请单向rnn层
    Args:
        inputs (): 输入, (batch, len, dim)
        seq_len (): 序列长度, (batch, )
        cell_name (): cell名称
        params (): cell配置参数
        dropout (): dropout
        scope (): tensorflow域名

    Returns: 输出, 最后一个state, 需要初始化的参数

    """
    cell_params = params["cell_params"]
    temp = set(tf.global_variables())

    keep_prob = 1. - dropout
    outputs = None
    scope = scope or "unidirectional_rnn"
    with tf.variable_scope(scope):
        rnn_cell = getattr(rnn, cell_name)(**cell_params)
        # add for rnn dropout
        rnn_cell = rnn.DropoutWrapper(rnn_cell, output_keep_prob=keep_prob)
        outputs, final_state = tf.nn.dynamic_rnn(
            rnn_cell, inputs,
            initial_state=None,
            dtype="float",
            scope="unidirectional_rnn",
            sequence_length=seq_len)
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
        tf.summary.histogram("final_state", final_state)
        tf.summary.histogram("outputs", outputs)
    init_var_set = set(tf.global_variables()) - temp
    return outputs, final_state, init_var_set


def apply_cu_rnn(inputs, cell_name, cell_params, scope=None):
    """
    申请cudnn实现的rnn层
    Args:
        inputs (): 输入, (batch, len, dim)
        cell_name (): cell名称
        cell_params (): cell配置参数
        scope (): tensorflow域名

    Returns: 输出, 最后一个state, 需要初始化的参数

    """
    temp = set(tf.global_variables())
    outputs = None
    scope = scope or cell_name
    cell_params["input_size"] = inputs.get_shape()[2].value
    cu_rnn = getattr(cudnn_rnn_ops, cell_name)(**cell_params)
    temporary_graph = tf.Graph()
    with tf.Session(graph=temporary_graph) as session:
        num_params = session.run(cu_rnn.params_size())
    with tf.variable_scope(scope):
        variable_initializer = tf_utils.get_initializer("xavier")
        params = tf.get_variable(
            "rnn_params", shape=[num_params], dtype=tf.float32,
            initializer=variable_initializer)
        num_directions = 2 if cell_params["direction"] == "bidirectional" else 1
        batch_size = tf.shape(inputs)[0]
        initial_state_h = tf.zeros((batch_size, num_directions, cell_params["num_units"]))
        initial_state_c = tf.zeros((batch_size, num_directions, cell_params["num_units"]))
        outputs, final_state_h, final_state_c = cu_rnn(tf.transpose(inputs, [1, 0, 2]),
                                                       tf.transpose(initial_state_h, [1, 0, 2]),
                                                       tf.transpose(initial_state_c, [1, 0, 2]),
                                                       params, is_training=True)
        outputs = tf.transpose(outputs, [1, 0, 2])
        final_state_h = tf.transpose(final_state_h, [1, 0, 2])
        final_state_c = tf.transpose(final_state_c, [1, 0, 2])
        final_state = (final_state_h, final_state_c)
        # tf.summary.histogram("final_state", final_state)
        tf.summary.histogram("outputs", outputs)
    init_var_set = set(tf.global_variables()) - temp
    return outputs, final_state, init_var_set


def apply_highway(inputs, params, dropout=0., scope=None):
    """ apply highway layer
    args:
        inputs: Tensor([B, T, D], tf.float32)
        params: DICT, {
            "num_units":INT, "activation": STR, "gate_bias":FLOAT
        }
        is_training:
        dropout:
        scope:
    return:
    """
    num_units = int(params["num_units"])
    try:
        activation = params["activation"].lower()
    except Exception as e:
        print(e)
        activation = "relu"
    try:
        gate_bias = float(params["gate_bias"])
    except Exception as e:
        print(e)
        gate_bias = -0.5
    outputs = highwaynet(inputs, num_units,
                         activation=activation,
                         gate_bias=gate_bias,
                         scope=scope)
    return outputs


def cbhg(inputs, input_lengths, params, is_training, scope="cbhg"):
    """ cbhg model
        conv1d_bank --> concat --> maxpooling --> conv1d:3 * 2 --> skip connection
        --> fc --> hightway * 4 --> BiGRU
    args:
        inputs: Tensor[N, T, D]
        input_lengths: Tensor[N]
        params: DICT {
            "K": INT, number of conv filter
            "proj_nums": LIST or "128&128", projection dim
            "num_units": INT,
        }
        is_training: True, batch_normalize or False
        scope:
    :return:
    """
    if isinstance(params["kernel_size"], (list, tuple)):
        kernel_size_bank = params["kernel_size"]
    else:
        kernel_size_bank = range(1, params["kernel_size"] + 1)

    num_units = int(params["num_units"])

    proj_nums = params["proj_nums"]
    inputs_shape = inputs.get_shape().as_list()
    # inputs: [N, T, D]
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_bank'):
            # Convolution bank: concatenate on the last axis
            # to stack channels from all convolutions
            conv_outputs = tf.concat(
                [conv1d(inputs, k, num_units, tf.nn.relu, is_training, 'conv1d_%d' % k)
                 for k in kernel_size_bank],
                axis=-1
            )
        # conv_outputs: [N, T, k*128]
        # Maxpooling:
        maxpool_output = tf.layers.max_pooling1d(
            conv_outputs,
            pool_size=2,
            strides=1,
            padding='same',
            name="maxpool")
        # maxpool_output: [N, T, k*128]

        # Two projection layers:
        proj1_output = conv1d(maxpool_output, 3, proj_nums[0], "relu", is_training,
                              'proj_conv1d_1')
        proj2_output = conv1d(proj1_output, 3, inputs_shape[-1], None, is_training,
                              'proj_conv1d_2')
        # proj2_output: [N, T, D]

        # Residual connection:
        highway_input = proj2_output + inputs
        # highway_input: [N, T, D]

        # Handle dimensionality mismatch:
        if highway_input.shape[2] != num_units:
            highway_input = tf.layers.dense(highway_input, num_units)
        # highway_input: [N, T, 128]

        # 4-layer HighwayNet:
        for i in range(4):
            highway_input = highwaynet(highway_input, num_units, 'highway_%d' % (i + 1))
        rnn_input = highway_input
        # rnn_input: [N, T, 128]

        # Bidirectional RNN
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            rnn.GRUCell(num_units),
            rnn.GRUCell(num_units),
            rnn_input,
            sequence_length=input_lengths,
            dtype=tf.float32,
            scope="biGRU")
    # outputs: ([N, T, 128], [N, T, 128])
    return tf.concat(outputs, axis=2)  # Concat forward and backward


def cbhg_v2(inputs, input_lengths, params, is_training, scope="cbhg_v2"):
    """ cbhg model
        conv1d_bank --> concat --> maxpooling --> conv1d:3*3 --> skip connection
        --> fc --> highway * 2 --> BiGRU
    args:
        inputs: Tensor[N, T, D]
        input_lengths: Tensor[N]
        params: DICT {
            "K": INT, number of conv filter
            "proj_nums": LIST or "128&128", projection dim
            "num_units": INT,
        }
        is_training: True, batch_normalize or False
        scope:
    :return:
    """
    if isinstance(params["kernel_size"], (list, tuple)):
        kernel_size_bank = params["kernel_size"]
    else:
        kernel_size_bank = range(1, params["kernel_size"] + 1)

    num_units = int(params["num_units"])

    proj_dims = params["proj_dims"]
    num_highway = params["num_highway"]
    inputs_shape = inputs.get_shape().as_list()
    # inputs: [N, T, D]
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_bank'):
            # Convolution bank:  from all convolutions
            conv_bank = [conv1d(inputs, k, num_units, "relu", is_training, 'conv1d_%d' % k)
                         for k in kernel_size_bank]

        # Maxpooling:
        maxpool_inputs = tf.concat(conv_bank, axis=-1)
        # conv_outputs: [N, T, k*128]
        maxpool_output = tf.layers.max_pooling1d(
            maxpool_inputs,
            pool_size=2,
            strides=1,
            padding='same',
            name="maxpool")
        # maxpool_output: [N, T, k*128]

        '''
        maxpool_output = conv_bank_pooling1d(conv_bank, pool_type="max")
        # maxpool_output: [N, T, 128]
        print(maxpool_output)
        '''

        # Two projection layers:
        proj_output = maxpool_output
        num_proj = len(proj_dims) if proj_dims is not None else 1
        for i in range(num_proj - 1):
            proj_output = conv1d(proj_output, 3, proj_dims[i], "relu", is_training,
                                 "proj_conv1d_%d" % i)
        proj2_output = conv1d(proj_output, 1, inputs_shape[-1], None, is_training,
                              'proj_conv1d_%d' % (num_proj - 1))
        # proj2_output: [N, T, D]

        # Residual connection:
        highway_input = proj2_output + inputs
        # highway_input: [N, T, D]

        # Handle dimensionality mismatch:
        if highway_input.shape[2] != num_units:
            highway_input = tf.layers.dense(highway_input, num_units)
        # highway_input: [N, T, 128]

        # 4-layer HighwayNet:
        for i in range(num_highway):
            highway_input = highwaynet(highway_input, num_units, 'highway_%d' % (i + 1))
        rnn_input = highway_input
        # rnn_input: [N, T, 128]

        # Bidirectional RNN
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            rnn.GRUCell(num_units),
            rnn.GRUCell(num_units),
            rnn_input,
            sequence_length=input_lengths,
            dtype=tf.float32,
            scope="biGRU")
    # outputs: ([N, T, 128], [N, T, 128])
    return tf.concat(outputs, axis=2)  # Concat forward and backward


def check_rnn_params(params):
    """ check params for rnn,
        default initializer for LSTM =
    args:
        params:
        DICT for GRUCell {
            "num_units":
            "activation": (tf.tanh)
        }
        DICT for LSTMCell or RNNCell {
            "num_units":
            "activation": (tf.tanh)
            "initializer": (tf.orthogonal_initializer)
            "use_peepholes":
            "num_proj":
            "forget_bias"
        }
    :return:
    """
    cell_name = None

    def check_cell_params(_name, _params):
        _cell_params = dict()
        try:
            _cell_params["num_units"] = int(_params["num_units"])
        except (KeyError, TypeError):
            Exception("[ERROR] check_rnn_params: 'num_units' is needed, should be an integer")

        try:
            activation = tf_utils.get_activation(_params["activation"])
            _cell_params["activation"] = activation
        except Exception as e:
            print(e)
            _cell_params["activation"] = tf.tanh

        use_relu = _params["activation"].lower() == "relu"
        try:
            initializer = tf_utils.get_initializer(_params["initializer"],
                                                   is_relu=use_relu)
        except Exception as e:
            #print(e)
            gain = np.sqrt(2) if use_relu else 1.0
            initializer = tf.orthogonal_initializer(gain=gain)

        if _name == "GRUCell":
            # GRUCell不需要kernel_initializer, 否则报错
            # _cell_params["kernel_initializer"] = initializer
            return _cell_params

        _cell_params["initializer"] = initializer

        try:
            if not isinstance(_params["use_peepholes"], bool):
                raise KeyError
        except KeyError:
            if _params["use_peepholes"].upper() in ['TRUE', 'T', 'YES', 'Y']:
                _cell_params["use_peepholes"] = True
            else:
                _cell_params["use_peepholes"] = False

        if "num_proj" in _params:
            _cell_params["num_proj"] = int(_params["num_proj"])
        if "forget_bias" in _params:
            _cell_params["forget_bias"] = float(_params["forget_bias"])
        return _cell_params

    try:
        cell_name = params["cell_name"].lower()
        if cell_name in ["rnn", "rnncell"]:
            cell_name = "RNNCell"
        elif cell_name in ["lstm", "lstmcell"]:
            cell_name = "LSTMCell"
        elif cell_name in ["gru", "grucell"]:
            cell_name = "GRUCell"
        else:
            print("[Error] invalid cell_name \"%s\" in rnn" % params["cell_name"])
            return None
    except KeyError:
        print("[Error] no cell_name !")
        return None

    try:
        cell_params = check_cell_params(cell_name, params["cell_params"])
    except KeyError:
        cell_params = check_cell_params(cell_name, params)
    params["cell_params"] = cell_params

    if "concat_output" not in params:
        params["concat_output"] = ("num_proj" not in params)

    return cell_name, params


#def highwaynet(inputs, num_units, scope):
def highwaynet(inputs, num_units, activation="relu", gate_bias=-1.0, scope=None):
    """
        activation(x) * T + x * (1 - T)
    args:
        inputs: Tensor([B, T, D], tf.float32)
        num_units: INT, same dimension with inputs
        activation: STR, for transform layer
        gate_bias: FLOAT, bias for transform gate, default=-1.0
        scope:
    return:
    """
    scope = scope or "highwaynet"
    try:
        activation = tf_utils.get_activation(activation)
    except Exception as e:
        print(e)
        activation = None

    with tf.variable_scope(scope):
        H = tf.layers.dense(
            inputs,
            units=num_units,
            activation=activation,
            name='H')
        T = tf.layers.dense(
            inputs,
            units=num_units,
            activation=tf.nn.sigmoid,
            name='T',
            bias_initializer=tf.constant_initializer(gate_bias))
        outputs = tf.add(H * T, inputs * (1.0 - T), "highway_output")
        tf.summary.histogram(scope + "/H", H)
        tf.summary.histogram(scope + "/T", T)
        tf.summary.histogram(scope + "/outputs", outputs)
    return outputs

