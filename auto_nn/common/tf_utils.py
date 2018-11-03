#!/usr/bin/env python
# -*- coding: utf-8 -*-

# *********************************************************************
# @file: tf_utils.py
# @brief: common functions for tensorflow
# @author: niezhipeng(@baidu.com)
# @Created on 2017/9/23
# *********************************************************************

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.init_ops import VarianceScaling

def get_host_ip():
    """获得主机host和ip
    """
    import socket
    # 获取本机电脑名
    myname = socket.getfqdn(socket.gethostname())
    # 获取本机ip
    myaddr = socket.gethostbyname(myname)
    return myname, myaddr


def start_tensorboard(log_dir, port):
    """start tensorboard
    """
    if not port:
        port = 0
    port += 8000
    command = "nohup tensorboard --logdir=%s --port=%s & echo $! > pid.txt" % (log_dir, str(port))
    os.system(command)
    pid = str(os.popen("head pid.txt").read())
    pid = pid.strip()
    os.system("rm -rf pid.txt")
    _, ip = get_host_ip()
    addr = r"http:\\%s:%s" % (str(ip), str(port))
    return addr, pid


def get_activation(identifier):
    """ return activation function according its name
    Args:
        identifier: STR, name of activation function
    Returns:
        Func()
    """
    if identifier is None:
        return None
    identifier = identifier.lower()
    linear = lambda x: x
    if identifier == "selu":
        return selu
    elif identifier == "swish":
        return swish
    elif hasattr(tf.nn, identifier):
        return getattr(tf.nn, identifier)
    elif identifier == "linear":
        return linear
    else:
        Exception("[Error] activation \"%s\" not be supported!" % identifier)
        return None


def get_initializer(identifier, is_relu=False):
    """ return initializer
    args:
        identifier: STR,
        is_relu: True, initialzer for ReLU units; default = False
    return:
        a callable object (obj or function)
    """
    if identifier is None:
        return initializer(is_relu=is_relu)
    identifier = identifier.lower()
    if identifier == "zero":
        return tf.zeros_initializer()
    elif identifier == "random":
        return tf.truncated_normal_initializer(stddev=0.1)
    elif identifier == "varscale":
        return tf.variance_scaling_initializer()
    elif identifier == "orthogonal":
        gain = np.sqrt(2) if is_relu else 1.0
        return tf.orthogonal_initializer(gain=gain)
    elif identifier == "uniform":
        return tf.uniform_unit_scaling_initializer()
    elif identifier == "xavier_normal":
        return initializer(distribution="normal", is_relu=is_relu)
    elif identifier == "xavier_uniform":
        return initializer(distribution="uniform", is_relu=is_relu)
    elif identifier == "he_uniform":
        return initializer(distribution="uniform", is_relu=True)
    elif identifier == "he_normal":
        return initializer(distribution="normal", is_relu=True)
    else:
        Exception("[Error] initializer \"%s\" not be supported" % identifier)
        return None

def get_initialize_variable(name, shape, initializer, is_relu=False, gain=1.0, divisor=1.0):
    if initializer is not None and initializer.lower() == "identity":
        # print(shape)
        middle0 = int(shape[0] / 2)
        #middle1 = int(shape[1] / 2)
        if shape[-2] == shape[-1]:
            array = np.zeros(shape, dtype='float32')
            identity = np.eye(shape[-2], shape[-1])
            array[middle0] = identity
        else:
            m1 = divisor / shape[-2]
            m2 = divisor / shape[-1]
            sigma = 1e-5 * m2
            array = np.random.normal(loc=0, scale=sigma, size=shape).astype('float32')
            for i in range(shape[-2]):
                for j in range(shape[-1]):
                    if int(i*m1) == int(j*m2):
                        array[middle0, i, j] = m2
        return tf.get_variable(name, initializer=array)
    else:
        try:
            initializer = get_initializer(initializer, is_relu)
        except Exception as e:
            print(e)
            initializer = initializer(is_relu=is_relu)
    return tf.get_variable(name, shape, initializer=initializer, trainable=True)


def weight_norm(v, scope="weight_norm"):
    """ weight normalization by chenchangbin,
        initialize by norm of v_init
        Ref: https://arxiv.org/pdf/1602.07868.pdf
    args:
        v:
    return:

    """
    with tf.variable_scope(scope):
        def _norm(x, axis=0, epsilon=1e-12):
            return tf.sqrt(tf.maximum(tf.reduce_sum(x**2, axis), epsilon))
        output_size = v.get_shape()[-1].value
        v_norm = _norm(tf.reshape(v, [-1, output_size]), axis=0)
        flat_init = tf.reshape(v.initialized_value(), [-1, output_size])
        v_norm_init = _norm(flat_init, axis=0)
        g = tf.get_variable("g", dtype=tf.float32, initializer=v_norm_init)
        w = g * (v / v_norm)
        return w


def layer_norm(x, eps=1e-8, scope="layer_norm"):
    """ layer normalization
    """
    with tf.variable_scope(scope):
        dim = x.get_shape().as_list()[-1]
        scale = tf.get_variable("norm_scale", [dim],
                                initializer=tf.ones_initializer())
        bias = tf.get_variable("norm_bias", [dim],
                               initializer=tf.zeros_initializer())
        mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(tf.square(x-mean), axis=[-1], keep_dims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + eps)
        return norm_x * scale + bias


def group_norm(x, G, eps=1e-5, scope="group_norm"):
    """ group normalization
        Ref: https://arxiv.org/pdf/1803.08494.pdf
    args:
        x: Tensor([B, ..., C], tf.float32)
        G: INT, num_groups for channels
        eps: FLOAT,
        scope: "group_norm"
    return:
        Tensor of the same shape as x
    """
    with tf.variable_scope(scope):
        x = tf.convert_to_tensor(x)
        ori_shape = tf.shape(x)
        dims = x.shape.ndims
        C = x.get_shape().as_list()[-1]
        # G should be less than C
        G = min(G, C)
        g_shape = [tf.shape(x)[i] for i in range(dims-1)] + [C//G, G]
        x = tf.reshape(x, g_shape)
        # calc mean and var along spatial dims and C//G
        moments_axes = [i+1 for i in range(dims-1)]
        mean, var = tf.nn.moments(x, moments_axes, keep_dims=True)
        x = (x - mean) * tf.rsqrt(var + eps)
        # gamma and beta for per channel
        broadcast_shape = [1] * (dims-1) + [C]
        gamma = tf.get_variable('gamma', broadcast_shape,
                                initializer=tf.constant_initializer(1.0),
                                trainable=True)
        beta = tf.get_variable('beta', broadcast_shape,
                               initializer=tf.constant_initializer(0.0),
                               trainable=True)
        output = tf.reshape(x, ori_shape) * gamma + beta
        return output


def calc_sequence_lengths(batch_data, padding_value=0.):
    """ calculate sequence length of a batch data
    args:
        batch_data: Tensor([B, T, D],tf.float)
        padding_value: FLOAT, default = 0.
    returns:
        Tensor([B,], tf.int64)
    """
    with tf.name_scope("GetSequenceLength"):
        batch_max = tf.reduce_max(tf.abs(batch_data), reduction_indices=2)
        mask = tf.cast(tf.greater(batch_max, padding_value), tf.int32)
        seq_len = tf.reduce_sum(mask, reduction_indices=1)
        seq_len = tf.cast(seq_len, tf.int32)
    return seq_len


def get_padding_mask(inputs, padding_value=0):
    """ get padding mask
    Args:
        inputs: Tensor(batch_size, seq_len, ...)
        padding_value: default=0
    Returns:
        Tensor(batch_size, ...), with
            0 for padding location,
            1 for non-padding location
    """
    with tf.name_scope("GetPaddingMask"):
        padding_value = tf.convert_to_tensor(padding_value, inputs.dtype)
        if len(inputs.get_shape()) < 3:
            mask = tf.not_equal(tf.abs(inputs), padding_value)
        else:
            mask = tf.not_equal(
                tf.reduce_max(tf.abs(inputs), axis=-1), padding_value)
    return tf.to_float(mask)

def get_seq_padding_mask(seq_lens, max_len=None):
    """ indice the real sample (1.0) or the padding sample (0.0)
        by comparing seq_len and seq_max_len for each sequence
    args:
        seq_lens: Tensor([batch_size], tf.int64)
        seq_max_len: int
    return:
        Tensor([batch_size, max_len], tf.float32) which value is 0. or 1.
    """
    with tf.name_scope("GetSeqPaddingMask"):
        seq_lens = tf.convert_to_tensor(seq_lens, dtype=tf.int32)
        batch_size = tf.shape(seq_lens)[0]
        if max_len is None:
            max_len = tf.reduce_max(seq_lens)

        indices = tf.reshape(tf.tile(tf.range(max_len), [batch_size]), [batch_size, -1])
        mask = tf.less(indices, tf.expand_dims(seq_lens, axis=1))
        return tf.cast(mask, dtype=tf.float32)


def bin_values(values, num_bins, min_value, max_value, scope=None):
    """
    Bin a continuous set of values into indices.
    Args:
        values (): Tensor of values to bin (elementwise).
        num_bins (): Number of bins to create.
        min_value (): Minimum value for the smallest bin.
        max_value (): Maximum value for the largest bin.
        scope (): Scope to put operations in.

    Returns: a tf.int32 tensor of the same shape as `values` containing indices.

    """
    with tf.name_scope(scope or "BinValues"):
        increment = (max_value - min_value) / num_bins
        values = tf.clip_by_value(values, min_value, max_value)
        values -= min_value
        binned = tf.cast(values / increment, tf.int32)
    return binned


def unbin_values(values, num_bins, min_value, max_value, scope=None):
    """
    Undo binning and convert from binned indices to a continuous range.
    Args:
        values (): Tensor of values to unbin (elementwise).
        num_bins (): Number of bins used.
        min_value (): Minimum value for the smallest bin.
        max_value (): Maximum value for the largest bin.
        scope (): Scope to put operations in.

    Returns: a tf.float32 tensor of the same shape as `values` containing values.

    """
    with tf.name_scope(scope or "UnbinValues"):
        # Set the reference values to be in the middle of the buckets, to avoid
        # a bias towards the left edge of the buckets.
        increment = (max_value - min_value) / num_bins
        reference = (tf.linspace(min_value, max_value - increment, num_bins)
                     + increment / 2)
        return tf.cast(tf.nn.embedding_lookup(reference, values), tf.float32)


def leaky_relu(x, leak=0.2, name="leak_relu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >0.0, x, alpha * tf.exp(x) - alpha)


def swish(x):
    return x * tf.nn.relu(x)


def initializer(mode="FAN_IN", distribution="normal", is_relu=False, seed=None):
    """ create xavier initializer, include he_initializer for relu
    args:
        mode: "FAN_IN"(default) or "FAN_OUT" or "FAN_AVG"
        distribution: "normal"(default) or "uniform"
        is_relu: True, norm variance by 2
                    in case we output to a ReLU unit
    Reference: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
               https://arxiv.org/abs/1502.01852 by Kaiming He
    return:
        initializer op

    """
    mode = mode.lower()
    scale = 2. if is_relu else 1.
    return VarianceScaling(
        scale=scale, mode=mode, distribution=distribution, seed=seed)


def softmax(logits, axis=-1):
    """
    calculate the softmax value of a darray
    for each i:
        softmax[i] = exp(logits[i] - max(logits)) / sum(exp(logits) - max(logits))
    args:
        logits: NP.ARRAY()
        axis: The dimension softmax would be performed on
    :return:
    """
    if not isinstance(logits, np.ndarray):
        return logits

    exp_x = np.exp(logits - np.max(logits))
    return exp_x / np.expand_dims(np.sum(exp_x, axis=axis), axis=axis)


def rewrite_padded_tensor(x, padding_mask):
    """ reduct padded tensor for sequence batch,
        zero-clear the last dimension value according to padding_mask
    args:
        x:    Tensor([d1, d2, ..., dim])
        mask: Tensor([d1, d2, ...]), value is [0, 1] or [True, False]
    return:
    """
    shape = tf.shape(x)
    mask = tf.cast(padding_mask, x.dtype)
    # [B, T, D] --> [B*T, D] --> [D, B*T]
    temp = tf.transpose(tf.reshape(x, [-1, shape[-1]]))
    # broadcast multiply([D, B*T], [B*T]) with broadcast
    temp = tf.multiply(temp, tf.reshape(mask, [-1]))
    return tf.reshape(tf.transpose(temp), shape)


def calc_cosine_coef(t1, t2):
    """ calculate the cosine coef between rows of t1 and t2
    :param t1: Tensor()
    :param t2: Tensor(), which last dim is same with t1
    :return:
    """
    normed_t1 = tf.nn.l2_normalize(t1, dim=-1)
    normed_t2 = tf.nn.l2_normalize(t2, dim=-1)
    #print(normed_t1, normed_t2)
    sim = tf.matmul(normed_t1, normed_t2, transpose_b=True)
    return sim


def calc_corrcoef(t1, t2):
    """ calculate the pearson coef between rows of t1 and t2
    :param t1: Tensor()
    :param t2: Tensor(), which last dim is same with t1
    :return:
    """
    shape = t1.get_shape().as_list()
    if len(shape) > 2:
        t1 = tf.reshape(t1, (-1, shape[-1]))
        t2 = tf.reshape(t2, (-1, shape[-1]))
    dim = t1.get_shape().as_list()[-1]
    t1_mean, t1_var = tf.nn.moments(t1, axes=[1], keep_dims=True)
    t2_mean, t2_var = tf.nn.moments(t2, axes=[1], keep_dims=True)
    cov = tf.matmul(t1-t1_mean, t2-t2_mean, transpose_b=True) / (dim)
    t1_cov = tf.diag(1/(tf.sqrt(tf.squeeze(t1_var) + 1e-10)))
    t2_cov = tf.diag(1/(tf.sqrt(tf.squeeze(t2_var) + 1e-10)))
    coef = tf.matmul(tf.matmul(t1_cov, cov), t2_cov)
    # coef = t1_cov @ cov @ t2_cov
    #return coef
    shape[-1] = -1
    return tf.reshape(coef, shape)


def plot_2d_tensor(tensor, name):
    # x_min = tf.reduce_min(tensor)
    # x_max = tf.reduce_max(tensor)
    # img_tensor = tf.constant(255, dtype=tf.float32) * (tensor - x_min) / (x_max - x_min + 1e-10)

    img_tensor = tf.expand_dims(tf.expand_dims(tensor, axis=-1), axis=0)
    tf.summary.image(name, img_tensor, max_outputs=1)

#def plot_tensor_correlation(tensor1, tensor2, name=None):
#    pearson_r = tf.contrib.metrics.streaming_pearson_correlation(tensor1, tensor2)

def print_seq_tag(t, target):
    np_t = tf.concat([t, tf.expand_dims(target, axis=-1)], axis=-1)
    str_t = tf.as_string(np_t)
    shape = np_t.get_shape().as_list()
    return tf.string_join(tf.unstack(str_t, axis=0), " ")

    #return " ".join(["%d_%d_%d" %(np_t[i][0], np_t[i][1], np_t[i][-1]) for i in range(shape[0])])