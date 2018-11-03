# -*- coding: utf-8 -*-
################################################################################
# file: auto_train_dyz_cbhg.py
# brief: fullPYLSTM训练脚本
# @author: niezhipeng(@baidu.com)
# Created on Tue May 02 10:18:57 2017
# Update: 1、加入初始模型的训练; 2、加入多任务损失权重; 
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import random
import re
import struct
import numpy as np

import sys
import os
import csv
import ConfigParser
import json
import codecs
from functools import wraps
import inspect

import tensorflow as tf
from cbhg_model import cbhg
import data_reader_cbhg as reader
#import keras_evaluate as ke


def create_path(output_path):
    
    pred_path = output_path + 'prediction/'
    report_path = output_path + 'report/'
    model_path = output_path + 'model/'

    if not os.path.exists(output_path):
        print ("Creating path: '%s'" % output_path)
        os.makedirs(output_path)

    try:
        os.mkdir(pred_path)
    except OSError:
        print ("Path: '%s' already exist, the original prediction files (if exist) could be overwrited." % (pred_path))
    else:
        print ("Creating folder: '%s'" % (pred_path))

    try:
        os.mkdir(report_path)
    except OSError:
        print ("Path: '%s' already exist, the original report files (if exist) could be overwrited." % (report_path))
    else:
        print ("Creating folder: '%s'" % (report_path))

    try:
        os.mkdir(model_path)
    except OSError:
        print ("Path: '%s' already exist, the original models (if exist) could be overwrited." % (model_path))
    else:
        print ("Creating folder: '%s'" % (model_path))

    #print ("Working in '%s'" % output_path)

    return pred_path, report_path, model_path

def read_section_params(conf, section):
    options = conf.items(section)
    params = dict(([key,value] for (key, value) in options))
    params["name"] = section
    return params

def check_fc_params(func):
    @wraps(func)
    def wrapper(*a, **k):
        new_args = inspect.getcallargs(func, *a, **k)
        #check params
        try:
            params = new_args['params']
        except KeyError:
            sys.exit("[ERROR] args 'params' does not exist, should be needed as a dict")
        try:
            layer_name = params['name']
        except KeyError:
            sys.exit("[ERROR] not find 'layer_name' in args 'params'")
        # check node_num, needed
        try:
            params["node_num"] = int(params["node_num"])
        except (KeyError, ValueError):
            sys.exit("[ERROR] invalid 'node_num' in %s, should be an interger" % layer_name)
        # check activation, needed
        try:
            activation = params["activation"].lower()
        except KeyError:
            activation = None
            #exit_with_info("invalid \"activation\" in %s, should be a string: linear, sigmoid, relu"% args['layer_name'])
        finally:
            if activation in ['linear', 'sigmoid', 'relu']:
                params["activation"] = activation
            else:
                sys.exit("[ERROR] invalid 'activation' in %s, should be a string: linear, sigmoid, relu"% layer_name)  
        # read params[initializer], default is sigmoid for activation == sigmoid, adjust_normal for others
        try:
            initializer = params["initializer"].lower()
        except KeyError:
            initializer = ""
        finally:
            if params["activation"] == "sigmoid":
                params['initializer'] = "sigmoid"
            elif initializer in ['fixed', 'fixed_normal']:            
                params["initializer"] = "fixed_normal"
            elif initializer in ['adjust', '', 'adjust_normal']:
                params["initializer"] = "adjust_normal"
        # read params[regularizer], default is None
        try:        
            regularizer = params["regularizer"]
        except KeyError:
            regularizer = ""
        finally:
            if regularizer.lower() in ['l1','l1_loss']:
                params["regularizer"] = "l1_loss"
            elif regularizer.lower() in ['l2', 'l2_loss']:
                params["regularizer"] = "l2_loss"
            elif regularizer.lower() == "":
                params["regularizer"] = None
            else:
                sys.exit("[ERROR] regularizer '%s' in %s not be supported" % (regularizer, layer_name))
        new_args['params'] = params
        return func(**new_args)
    return wrapper

    
def check_lstm_params(func):
    @wraps(func)
    def wrapper(*a, **k):
        new_args = inspect.getcallargs(func, *a, **k)
        ### check params
        try:
            params = new_args['params']
        except KeyError:
            sys.exit("[ERROR] args 'params' does not exist, should be needed as a dict")
        try:
            layer_name = params['name']
        except KeyError:
            sys.exit("[ERROR] not find 'layer_name' in args 'params'")
        # check node_num, needed
        try:
            params["num_units"] = int(params["num_units"])
        except (KeyError, ValueError):
            sys.exit("[ERROR] invalid 'num_units' in %s, should be an interger" % layer_name)
        
        try:
            use_peephole = params["use_peephole"].upper()
            params["use_peephole"] = True if use_peephole in ['TRUE', 'T', 'YES', 'Y', 'ON', '1'] else False
        except KeyError:
            params["use_peephole"] = False
        
        try:
            params["cell_clip"] = float(params["cell_clip"])
        except (ValueError, KeyError):
            params["cell_clip"] = None
            
        #TODO: support initializer for weights and projection matrices
        try:
            params["initializer"] = None    #params["initializer"].lower()
            #initializer = None
        except KeyError:
            params["initializer"] = None
    
        try:
            params["num_proj"] = int(params["num_proj"])
        except (KeyError, ValueError):
            params["num_proj"] = None
    
        try:
            params["proj_clip"] = float(params["proj_clip"])
        except (KeyError, ValueError):
            params["proj_clip"] = None
        
        try:
            params["forget_bias"] = float(params["forget_bias"])
        except (KeyError, ValueError):
            params["forget_bias"] = 1.0
        
        try:
            inner_activation = params["inner_activation"]
        except KeyError:
            inner_activation = ""
        finally:
            inner_activation = inner_activation.lower()
            if inner_activation in ["tanh", ""]:
                params["inner_activation"] = tf.tanh
            elif inner_activation == "sigmoid":
                params["inner_activation"] = tf.sigmoid
            else:
                sys.exit("[ERROR] inner_activation '%s' in %s not be supported" % (inner_activation, layer_name))
        new_args['params'] = params
        return func(**new_args)
    return wrapper



@check_fc_params
def full_layer(inputs, params, keep_prob=1.0, scope = "fc"):   
    shape = inputs.get_shape().as_list()
    #print(shape)
    shape = [-1 if x is None else x for x in shape]
    in_size = shape[-1]
    out_size = params["node_num"]
    
    if  params["initializer"] == 'fixed_normal':    
        # 1. original weights initialization
        weights = tf.Variable(tf.random_normal([in_size, out_size],
                                           mean=0.0, stddev=0.02), name='W')
    elif params["initializer"] == "adjust_normal":   
        # 2. weights initialization with adjustment
        weights = tf.Variable(tf.random_normal([in_size, out_size],
                                         stddev=tf.sqrt(1.0 / in_size)), name='W')
    elif params["activation"] == "sigmoid":
        # 3. Glorot & Bengio (2010) for sigmoid
        weights = tf.Variable(tf.random_uniform([in_size, out_size],
                          minval=-tf.sqrt(6. / (in_size + out_size)) * 4.,
                          maxval=tf.sqrt(6. / (in_size + out_size)) * 4.), name='W')

    biases = tf.Variable(tf.zeros([out_size]), name='b')

    wx_plus_b = tf.add(tf.matmul(tf.reshape(inputs,[-1, in_size]), weights), biases)

    # 激活函数
    if   params["activation"] == "linear":
        outputs = wx_plus_b
    elif params["activation"] == "sigmoid":
            outputs = tf.nn.sigmoid(wx_plus_b)
    elif params["activation"] == "relu":
            outputs = tf.nn.relu(wx_plus_b)
    else:
        sys.exit("Activation function '%s' is not supported in this version." % params['activation'])

    # 正则惩罚项
    if params['regularizer'] == "l1_loss":
        regular_loss = tf.nn.l1_loss(weights)
    elif params['regularizer'] == "l2_loss":
        regular_loss = tf.nn.l2_loss(weights)
    else:
        regular_loss = 0.

    # dropout
    outputs = tf.nn.dropout(outputs, keep_prob)
    
    #恢复数据流的shape
    shape[-1] = out_size
    return tf.reshape(outputs, shape), regular_loss


# TODO: complete different initializers
@check_lstm_params
def dynamic_birnn_layer(inputs, actual_seq_len, params, keep_prob=1.0, scope="BiRNN"):
    # set initializer
    #
    #lstm_cell_fw = tf.contrib.rnn.LSTMCell()
    #print("inner_activation=%s,num_proj=%d" % (params["inner_activation"],params["num_proj"]))
    '''
    lstm_cell_fw = tf.contrib.rnn.LSTMCell(
        params["num_units"],                   #隐层状态维数(输出维数)
        use_peepholes=params["use_peephole"],  #diagonal/peephole conections(cell状态连接到各个门的输入, 默认False)
        cell_clip=params["cell_clip"],         #
        initializer=params["initializer"],     #权值初始化方法
        num_proj=params["num_proj"],           #映射矩阵输出维数(在LSTM层后面加入输出层) None---无输出映射
        proj_clip=params["proj_clip"],         #float 需要num_proj > 0 配合
        forget_bias=params["forget_bias"],     #忘记门偏置---设为0，默认为1 
        state_is_tuple=True,                   #True --- 返回值为[输出, 胞元状态]
        activation=params["inner_activation"]) #内部状态的激活函数, 默认为tanh 但softsign更有效
    '''
    #lstm_cell_fw = tf.contrib.rnn.LSTMCell(64)
    lstm_cell_fw = tf.nn.rnn_cell.GRUCell(params["num_units"])
    lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw, output_keep_prob=keep_prob)                                            
    '''
    lstm_cell_bw = tf.contrib.rnn.LSTMCell(
        params["num_units"],
        use_peepholes=params["use_peephole"],
        cell_clip=params["cell_clip"],
        initializer=params["initializer"],
        num_proj=params["num_proj"],
        proj_clip=params["proj_clip"],
        forget_bias=params["forget_bias"],
        state_is_tuple=True,
        activation=params["inner_activation"])
    '''
    lstm_cell_bw = tf.nn.rnn_cell.GRUCell(params["num_units"])
    lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw, output_keep_prob=keep_prob)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=lstm_cell_fw,   # "left-to-right" RNNCell
        cell_bw=lstm_cell_bw,   # "right-to-left" RNNCell
        inputs=inputs,          # [batch_size, T, feature_dim]
        dtype=tf.float32,       
        sequence_length=actual_seq_len, # batch中序列的实际长度 [batch_size]
        scope=scope)

    if params["num_proj"] is None:
        outputs = tf.concat(values=outputs, axis=2)
    else:
        outputs = outputs[0] + outputs[1]
    
    #if keep_prob < 1.0:
    outputs = tf.nn.dropout(outputs, keep_prob)
    return outputs

""""""
#brief: 根据配置文件中的模型结构以及各层参数，自动建立模型
#params[in1]: X --- tf.darray[batch * T, feat_dim] or [batch_size, T, feat_dim]
#params[in2]: actual_seq_len --- 实际序列长度tf.darray[batch_size]
#params[in3]: keep_prob ---
#params[in4]: initial_dim ---
#params[in5]: final_dim ---
#params[in6]: tone_dim ---
#params[in7]: model_conf ---
#
#return[1]: initial_outputs ---
#return[2]: final_outputs ---
#return[3]: tone_outputs ---
#retuan[4]: total_regular_loss ---
""""""
def auto_model_struct(X, actual_seq_len, keep_prob, outputs_dim_list, model_conf):
    ######################## read model config #################################
    try:
        model_struct = model_conf.get("model", "struct")
    except ConfigParser.NoOptionError:
        sys.exit("[ERROR] 'struct' does not exist in section [model]")
    ############################################################################   
    # dropout for inputs    
    X = tf.nn.dropout(X, keep_prob)
    
    #reshape X ==> [batch_size, T, feat_dim]
    #X = tf.reshape(X, [batch_size, -1, feat_dim])

    total_regular_loss = 0.0  #the sum of regulared weights loss     
    layer_id = 1                
    sublayer_id = 0           #layer id for branch
    branch_id = 0             #branch: 0,master; >0,branch
    outputs = [X]             #[list]current layer outputs for branch: 0, master
    for char in model_struct:
        if 'F' == char:
            scope = "fc%d" % layer_id
            layer_name = "layer_%d" % layer_id
            # layer in branch            
            if branch_id > 0:
                sublayer_id += 1
                layer_name += "@%d_%d" % (branch_id, sublayer_id)
                scope += "@%d_%d" % (branch_id, sublayer_id)
            else:
                layer_id += 1

            with tf.name_scope(scope):
                params = read_section_params(model_conf, layer_name)
                outputs[branch_id], regular_loss = full_layer(outputs[branch_id], params, keep_prob=keep_prob, scope=scope)
                total_regular_loss += regular_loss        
        elif 'B' == char:
            scope = "bi%d" % layer_id
            layer_name = "layer_%d" % layer_id
            # layer in branch            
            if branch_id > 0:
                sublayer_id += 1
                layer_name += "@%d_%d" % (branch_id, sublayer_id)
                scope += "@%d_%d" % (branch_id, sublayer_id)
            else:
                layer_id += 1

            with tf.name_scope(scope):
                params = read_section_params(model_conf, layer_name)
                outputs[branch_id] = dynamic_birnn_layer(outputs[branch_id], actual_seq_len, params, keep_prob=keep_prob, scope=scope)
        #inter branch
        elif '[' == char or '|' == char:
            branch_id += 1
            sublayer_id = 0
            try:
                outputs[branch_id] = outputs[0]
            except IndexError:
                outputs.append(outputs[0])
        #return master
        elif ']' == char:
            branch_id = 0
            sublayer_id = 0
            #分支输出合并
        #merge branch outputs
        #elif char.isdigit():
            
    task_outputs = []
    for i, dim in enumerate(outputs_dim_list, 1):
        scope = "fc_out%d" % i
        with tf.name_scope(scope):
            params = {}
            params['name'] = "layer_out%d" % i
            params['node_num'] = dim
            params['activation'] = "linear"
            params['regularizer'] = "l2_loss"
            tmp, regular_loss = full_layer(outputs[0], params, keep_prob=1.0, scope=scope)
            task_outputs.append(tmp)    
            total_regular_loss += regular_loss   
  
    return task_outputs, total_regular_loss


def model_struct(X, actual_seq_len, keep_prob, outputs_dim_list, conf):
    
    total_regular_loss = 0.0
    #
    #对input以较小的drop_rate进行dropout，以防onehot特征丢失
    #X = tf.nn.dropout(X, keep_prob)
    # 必须对输入进行dropout, 否则会过拟合
    #
    # 仅对字向量进行dropout,one-hot特征保留
    #X[:,:,:128] = tf.nn.dropout(X[:,:,:128], keep_prob)
    #X = tf.concat(values=[tf.nn.dropout(X[:,:,:128], keep_prob), X[:,:,128:]], axis=2)
    # 保留dyzid的one-hot特征，不进行dropout
    #x_keep_prob = 0.1 if keep_prob < 1.0 else 1.0
    #x_keep_prob = tf.cond(keep_prob < tf.constant(1.0), lambda:tf.constant(0.9),
    #        lambda:tf.constant(1.0))
    #X = tf.concat(values=[tf.nn.dropout(X[:,:,:173], x_keep_prob), X[:,:,173:]], axis=2)
    #X = tf.cond(keep_prob < tf.constant(1.0), 
    #        lambda: tf.concat(values=[tf.nn.dropout(X[:,:,:173],1.0),
    #        X[:,:,173:]], axis=2), lambda: X)
    #[batch_size, T, feat_dim] ==> [batch_size * T, feat_dim]
    #X = tf.reshape(X, [batch_size, -1, feat_dim])

    is_training = keep_prob < 1.0
    #layer_1: birnn_layer
    with tf.name_scope("cbhg"):
        # 6,10
        cbhg_out = cbhg(X, actual_seq_len, is_training, "pre_cbhg", 6, [128, 214])
    
    with tf.name_scope("bi1"):
        lstm_params = read_section_params(conf, "layer_1")
        l1_out = dynamic_birnn_layer(cbhg_out, actual_seq_len, lstm_params, keep_prob=keep_prob, scope="bi1")

    with tf.name_scope("bi2"):
        lstm_params = read_section_params(conf, "layer_2")
        l2_out = dynamic_birnn_layer(l1_out, actual_seq_len, lstm_params, keep_prob=keep_prob, scope="bi2")
 
    with tf.name_scope("bi3"):
        lstm_params = read_section_params(conf, "layer_3")
        l3_out = dynamic_birnn_layer(l2_out, actual_seq_len, lstm_params, keep_prob=keep_prob, scope="bi3")
    
    #layer_4: full_connected (sigmoid)
    #with tf.name_scope("fc4"):
    #    params = read_section_params(conf, "layer_4")   
    #    l4_out, regular_loss = full_layer(l3_out, keep_prob, params)
    #    total_regular_loss += regular_loss        
        #outputs = full_layer(l4_out, l4_size, n_classes, 1. ,activation_function = tf.nn.softmax)
    
    #layer_5: full_connected (sigmoid)
    #with tf.name_scope("fc5"):
    #    params = read_section_params(conf, "layer_5")
    #    l5_out, regular_loss = full_layer(l4_out, keep_prob, params)
    #    total_regular_loss += regular_loss
    task_outputs = []
    with tf.name_scope("fc_out1"):
        params = {}
        params['name'] = "layer_out1"
        params['node_num'] = outputs_dim_list[0]
        params['activation'] = "linear"
        params['regularizer'] = "l2_loss"
        tmp, regular_loss = full_layer(l3_out, params, 1.0, scope="fc_out1")
        task_outputs.append(tmp)    
        total_regular_loss += regular_loss

    with tf.name_scope("fc_out2"):
        params = {}
        params['name'] = "layer_out2"
        params['node_num'] = outputs_dim_list[1]
        params['activation'] = "linear"
        params['regularizer'] = "l2_loss"
        tmp, regular_loss = full_layer(l3_out, params, 1.0, scope="fc_out2")
        task_outputs.append(tmp)    
        total_regular_loss += regular_loss

    with tf.name_scope("fc_out3"):
        params = {}
        params['name'] = "layer_out3"
        #params['node_num'] = 128
        #params['activation'] = "sigmoid"
        params['regularizer'] = "l2_loss"
        #tmp1, regular_loss = full_layer(l3_out, params, 1.0, scope="fc_out3@1")
        #total_regular_loss += regular_loss
        params['node_num'] = outputs_dim_list[2]
        params['activation'] = "linear"
        tmp, regular_loss1 = full_layer(l3_out, params, 1.0, scope="fc_out3")
        task_outputs.append(tmp)    
        total_regular_loss += regular_loss
        
    return task_outputs, total_regular_loss

def get_dyz_label_mask(inputs, y_tone):
    padding_mask = tf.sign(tf.reduce_max(inputs, reduction_indices=2))
    

def build_model(inputs_dim, targets_dim_list, batch_size, model_conf):
    print ('Building & initializing model...')    

    with tf.name_scope("inputs"):
        x_in = tf.placeholder(tf.float32, (batch_size, None, inputs_dim))
        targets_y = [tf.placeholder(tf.float32, (batch_size, None, dim)) for dim in targets_dim_list]
        # add to indice the valid label(dyz && no_tone --> invalid)
        label_mask = tf.placeholder(tf.float32, (batch_size, None))

    with tf.name_scope("keep_prob"):
        keep_prob = tf.placeholder(tf.float32)
    
    #with tf.name_scope("seq_len"):
    #    seq_len = tf.placeholder(tf.int64, [batch_size])
    
    # [batch_size, T] 标记实际样本(1)和填充样本(0)
    padding_mask = tf.sign(tf.reduce_max(x_in, reduction_indices=2)) #[batch_size, T]
    #print(padding_mask.dtype)
    #print(padding_mask.get_shape().as_list())
    # [batch_size] 实际序列长度
    act_seq_len = tf.reduce_sum(tf.cast(padding_mask, tf.int64), reduction_indices=1)    
    
    # net_model
    #preds_y, regular_loss = auto_model_struct(x_in, act_seq_len, keep_prob, targets_dim_list, model_conf)
    preds_y, regular_loss = model_struct(x_in, act_seq_len, keep_prob, targets_dim_list, model_conf)
    #print("model_struct over")
    weights = tf.multiply(padding_mask, label_mask)
    #weights = padding_mask
    print("loss is cross_entropy")
    loss_list = [tf.contrib.losses.softmax_cross_entropy(
                logits=tf.reshape(preds_y[i], [-1, dim]),
                onehot_labels=tf.reshape(targets_y[i], [-1, dim]),
                weights=tf.reshape(weights, [-1]))
                for i, dim in enumerate(targets_dim_list)]
   
    #loss = sum(loss_list)
    #loss += alpha * regular_loss
    
    # train operation
    # 1. use clip_by_global_norm
    #tvars = tf.trainable_variables()
    #grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 15)
    #train_op = tf.train.MomentumOptimizer(learning_rate = lr, momentum = 0.9).apply_gradients(zip(grads, tvars))

    # 2. normal train
    #train_op = tf.train.MomentumOptimizer(learning_rate = lr, momentum = 0.9).minimize(cost)

    # 3. adadelta
    #train_op = tf.train.AdadeltaOptimizer().minimize(loss + alpha * regular_loss)

    # 4. adam
    #train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss + alpha * regular_loss)
    
    # change loss -> loss_list by niezhipeng for weighted loss in multitask
    return (x_in, targets_y, label_mask, keep_prob), (padding_mask, preds_y, loss_list, regular_loss)


"""查看各类别的TP"""    
def result_watcher_test(targets, results, padding_mask, mark_dict_reverse):

    y_real_indices = np.argmax(targets, 2)
    #print(y_real_indices.dtype)
    y_pred_indices = np.argmax(results, 2) 
    #print(y_pred_indices.dtype)
    
    TP = np.zeros(mark_dict_reverse.dim, dtype=int)  
    CNT = np.copy(TP)
    
    padding_mask = (padding_mask.astype(int)==1)
    TP_mask = np.logical_and(np.equal(y_real_indices, y_pred_indices), padding_mask)
    #TP_mask = np.multiply(TP_mask, padding_mask)    
    
    #print(mask.dtype)
    for ind,mark in mark_dict_reverse.dict.items():
        #print(ind, mark)
        TP[ind] = np.sum(np.logical_and(y_pred_indices==ind, TP_mask))
        CNT[ind] = np.sum(np.logical_and(y_real_indices==ind, padding_mask))
    
    return TP, CNT



#查看单一任务的TP和CNT
#def result_watcher(targets, results, padding_mask):
#    mask = tf.equal(padding_mask, tf.constant(1., dtype=tf.float32)) #[batch_size, T]
    
#    y_real_indices = tf.argmax(targets, 2)
#    y_pred_indices = tf.argmax(results, 2)

#    correct_pred = tf.equal(y_real_indices, y_pred_indices)
#    TP = tf.reduce_sum(tf.cast(tf.logical_and(correct_pred, mask), dtype=tf.float32))
#    CNT = tf.reduce_sum(padding_mask)
#    return TP, CNT


def result_watcher(targets, results, padding_mask):
    padding_mask = (padding_mask.astype(int)==1)
    y_real_indices = np.argmax(targets, 2)
    y_pred_indices = np.argmax(results, 2)
    correct_pred = np.equal(y_real_indices, y_pred_indices)
    TP = np.sum(np.logical_and(correct_pred, padding_mask))
    CNT = np.sum(padding_mask)
    return TP, CNT

def softmax(input, axis=-1):
    exp_x = np.exp(input - np.max(input))
    return exp_x / np.expand_dims(np.sum(exp_x, axis=axis), axis=axis)

"""计算多音字最可能的读音"""
def calc_dyz_py_prob(model_outputs, mark_trans_dict, split_py_list):
    #max_prob = 0.
    #out_py = []
    result = {}
    keys = ["initial", "final", "tone"]
    for py in split_py_list:
        py_list = py.strip().split()
        prob = 1.
        for i, key in enumerate(keys):
            ind = mark_trans_dict[key].value(py_list[i])
            prob *= model_outputs[i][ind]
        result[py] = prob
        #if prob > max_prob:
        #    max_prob = prob
        #    out_py = py
    return result

"""查看多任务的合并TP和CNT"""
def dyz_result_watcher(targets_y, preds_y, padding_mask, dyz_mask):
    #mask = tf.equal(padding_mask, tf.constant(1., dtype=tf.float32)) #[batch_size, T]
        
    padding_mask = (padding_mask.astype(int)==1)
    dyz_mask = np.logical_and(padding_mask, dyz_mask)
    dyz_CNT = np.sum(dyz_mask)
    ndyz_mask = np.logical_and(padding_mask, np.logical_not(dyz_mask))
    ndyz_CNT = np.sum(ndyz_mask)    
    for i, y in enumerate(targets_y):
        dyz_mask = np.logical_and(np.equal(np.argmax(y, 2),np.argmax(preds_y[i], 2)), dyz_mask)
        ndyz_mask = np.logical_and(np.equal(np.argmax(y, 2),np.argmax(preds_y[i], 2)), ndyz_mask)
    return np.sum(dyz_mask), dyz_CNT, np.sum(ndyz_mask), ndyz_CNT


def print_result(Plist, mark_dict_reverse):
    Dict = {}    
    for indice, mark in mark_dict_reverse.dict.items():
        Dict[mark] = Plist[indice]
    print (Dict)

"""模型预测结果写入文件: 多音字,计算最大概率读音; 非多音字，按照argmax拼接initial, final, tone"""
def write_result_file_new(out_file, textlist, outputs_list, mark_dict, mark_dict_reverse, dyz_py_dict):
    keys = ["initial", "final", "tone"]
    out_file = codecs.open(out_file, 'a', 'gb18030')
    for i, item in enumerate(textlist):
        #写入原始文本
        out_file.write("%s\n" % item[0])
        for j, label in enumerate(item[1]):
            outline = label
            dyz = outline[0]
            # find dyz_id in label
            #dyz_id = filter(str.isdigit, label)
            #cur_outputs = [np.exp(x[i,j]) for x in outputs_list]
            cur_outputs = [softmax(x[i,j]) for x in outputs_list]
            #cur_outputs = [x[i,j] for x in outputs_list]
            #print(cur_outputs)
            try:
                py_list = dyz_py_dict[dyz]
                #print(item[0].encode("gb18030"))
                #print(cur_outputs)
                #out_py, _  = calc_dyz_py_prob(cur_outputs, mark_dict, py_list)
                #print(dyz)
                prob_dict = calc_dyz_py_prob(cur_outputs, mark_dict, py_list)
                #for py,prob in prob_dict.items():
                #    print("%s: %.20f" % (py, prob))
                out_py = max(prob_dict.items(), key=lambda x:x[1])[0]
                #print("out_py = %s" % out_py)
                outline += "|%s" % out_py
            except KeyError:
                indices = [np.argmax(x,0) for x in cur_outputs]
                #print(indices)
                out_py = [mark_dict_reverse[key].value(ind) for (key,ind) in zip(keys, indices)]
                outline += "|%s" % " ".join(out_py)
            out_file.write("%s\n" % outline)
        out_file.write("\n")
    out_file.close()

"""将测试集结果写入文件"""
def write_result_file(out_file, textlist, outputs_list, mark_dict_reverse):
    #orifile = codecs.open(filename, 'r', 'gbk')
    # [batch_size, X] * task_num
    indices_batch_list = [np.argmax(x, 2) for x in outputs_list]
    out_file = codecs.open(out_file, 'a', 'gb18030')
    for i, item in enumerate(textlist):
        out_file.write("%s\n" % item[0])
        for j, label in enumerate(item[1]):
            outline = label
            outline += "|%s" % mark_dict_reverse['initial'].dict[indices_batch_list[0][i,j]]
            outline += " %s" % mark_dict_reverse['final'].dict[indices_batch_list[1][i,j]]
            outline += " %s" % mark_dict_reverse['tone'].dict[indices_batch_list[2][i,j]]
            out_file.write("%s\n" % outline)
        out_file.write("\n")
    out_file.close()


def write_csv_report(output_dir, epoch_id, out_list):
    csvfile = open(output_dir + 'all_report.csv', 'ab')
    writer = csv.writer(csvfile)
    if epoch_id == 0:
        #writer.writerow(["epoch_id", 
        #                 "train_loss", "dev_loss",
        #                 "train_precision_initial", "dev_precision_intial",
        #                 "train_precision_final", "dev_precision_final",
        #                 "train_precision_tone", "dev_precision_tone"])
        writer.writerow(["epoch_id", "train_loss", "dev_loss","dev_precision_dyz", "dev_precision_ndyz"])
    writer.writerow([epoch_id] + out_list)


def test_operator(conf_fn, mailto):
    print("[INFO] JOB_TYPE = TEST")
    device = conf.get("device", "device_id")
    device = ""
    if device != "hdfs" and device.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    print("[INFO] CUDA_VISIBLE_DEVICES: %s" % device)
    ############################# read test params ##################################
    batch_size = int(conf.get("model", "batch_size"))
    test_data_path = conf.get("test", "test_text_database")    
    
    task_list = ['initial', 'final', 'tone']
    task_num = 3
    # model path
    model_name = conf.get("test", "model_name")
    #dyzlist_path = conf.get("dict", "dyz_list_path")
    dyz_py_dict_path = conf.get("dict", "dyz_py_list_path")
    #################################################################################
    #dyzlist = reader.read_dyzlist(dyzlist_path)
    dyz_py_dict,_ = reader.read_dyz_py_dict(dyz_py_dict_path)
    dyzlist = sorted(dyz_py_dict.keys())
    print("[INFO] read dyz_py_dict success, total number = %d" % len(dyzlist))
    #base_dir = conf.get("dir", "base_dir")
    #output_dir = "%s%sdriver_%s_test/" % (base_dir, conf.get("dir", "output_dir"), device)
    test_output_dir = conf.get("test", "output_dir")
    pred_path, report_path, model_path = create_path(test_output_dir)
    model_path += model_name

    ### Prepare dict ###
    feature_dict, mark_dict, mark_dict_reverse = reader.prepare_dict(conf)

    #read text_data for train
    test_dataset, max_len, textlist,is_dyz_list = reader.read_rawdata_dyz(test_data_path, feature_dict, mark_dict, dyzlist)
    feature_dim = len(test_dataset[0]["feature"][0])
    #for feats,text in zip(test_dataset, textlist):
    #    print(text[0].encode("gb18030"))
    #    for feat in feats["feature"]:
    #        print(", ".join(["[%d]%.7f" % (i, x) for i,x in enumerate(feat)]))

    #print(test_dataset[0]["feature"])
    print("[INFO] feature_dim = %d" % feature_dim)
    ### read mean_var ###
    mean_var_path = conf.get("test", "mean_var_path")    
    (mean_vec, var_vec) = reader.get_mean_var(test_dataset, feature_dim, mean_var_path, updating=False)
    mean_var = (mean_vec, var_vec)
    
    # feature normalize
    test_dataset = reader.dataset_normalized(test_dataset, feature_dim, mean_var)
    
    test_sample_list = range(len(test_dataset))
    print("the size of test_dataset = %d(sentence)" % len(test_dataset))
    
    ### Build model ###
    targets_dim_list = [mark_dict[x].dim for x in task_list]
    #x_in, targets_y, keep_prob, padding_mask, preds_y, loss, regular_loss
    ((x_in, targets_y, label_mask, keep_prob),
    (padding_mask, preds_y, loss_list, regular_loss)) = build_model(feature_dim, targets_dim_list, batch_size, conf)
    
    loss = sum(loss_list)

    pred_filename = "%s%s_pred_result.txt" % (pred_path, model_name)
    if os.path.exists(pred_filename):
        os.remove(pred_filename)
    pred_file = codecs.open(pred_filename, "w", "gb18030")
    pred_file.write("test_file is %s\n" % test_data_path)
    pred_file.close()
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # TODO: begin training on previous model
        # load  model data        
        saver.restore(sess, model_path)

        costs = 0.
        dyz_TP_sum = 0
        dyz_CNT_sum = 0
        ndyz_TP_sum = 0
        ndyz_CNT_sum = 0        
        #TPs_sum = np.zeros(task_num, dtype=np.float32)
        #CNTs_sum = TPs_sum.copy()
        #TPs_sum  = [np.zeros([dim], dtype=np.float32) for dim in targets_dim_list]
        #CNTs_sum = [np.zeros([dim], dtype=np.float32) for dim in targets_dim_list]
        TPs_sum = [0.] * task_num
        CNTs_sum = [0.] * task_num
        start_time = time.time()
        for i, (x_test, y_test, sample_inds, _) in enumerate(reader.batch_generator(test_dataset, test_sample_list, mean_var, batch_size, is_training=False)):
            #fetches = [loss, TPs, CNTs]
            batch_shape = list(y_test.shape)[0:-1]
            fetches = [loss, preds_y, padding_mask]
            feed_dict = {}
            feed_dict[x_in] = x_test
            targets_y_test = []
            st = 0
            #print(x_test[0,:,:])
            #不能在使用i,会与外循环的冲突
            for k,dim in enumerate(targets_dim_list):
                targets_y_test.append(y_test[:,:,st:st+dim])
                feed_dict[targets_y[k]] = targets_y_test[k]
                st += dim
            
            dyz_mask = np.zeros(batch_shape, dtype = np.float32)
            for k,ind in enumerate(sample_inds):
                dyz_mask[k,0:len(is_dyz_list[ind])] = np.array(is_dyz_list[ind], dtype=np.float32)
            dyz_mask = np.sign(dyz_mask)

            label_mask_test = np.sign(np.amax(targets_y_test[-1][:,:,1:], axis=2))
            #label_mask_test = np.logical_or(np.logical_not(dyz_mask),
            #    np.logical_and(label_mask_test, dyz_mask))
            label_mask_test = label_mask_test.astype(np.float32)
            
            feed_dict[label_mask] = label_mask_test
            feed_dict[keep_prob] = 1.0
            cost_test, preds_y_test, padding_mask_test = sess.run(fetches, feed_dict=feed_dict)
            
            write_result_file_new(pred_filename, textlist[batch_size * i: min(batch_size* (i+1),len(textlist))],
                    preds_y_test, mark_dict, mark_dict_reverse, dyz_py_dict)
            #write_result_file(pred_filename, textlist[batch_size * i: min(batch_size*(i+1), len(textlist))], preds_y_test, mark_dict_reverse)
            
            costs += cost_test

            #print(targets_y_test[2][0])
            #print(preds_y_test[2][0])
            dyz_label_mask = np.logical_and(np.sign(np.amax(targets_y_test[-1][:,:,1:], axis=2)),
                dyz_mask)
            dyz_TP, dyz_CNT, ndyz_TP, ndyz_CNT = dyz_result_watcher(targets_y_test, preds_y_test,
                    padding_mask_test, dyz_label_mask)
            dyz_TP_sum += dyz_TP
            dyz_CNT_sum += dyz_CNT
            ndyz_TP_sum += ndyz_TP
            ndyz_CNT_sum += ndyz_CNT
            
            for k,task in enumerate(task_list):
                TP, CNT = result_watcher(targets_y_test[k], preds_y_test[k], padding_mask_test)
                TPs_sum[k] += TP
                CNTs_sum[k] += CNT

        test_loss = costs
        dev_precision_dyz = 1.0 * dyz_TP_sum / dyz_CNT_sum if dyz_CNT_sum > 0 else 0.
        dev_precision_ndyz = 1.0 * ndyz_TP_sum / ndyz_CNT_sum if ndyz_CNT_sum > 0 else 0.
        dev_precision_initial = 1.0 * TPs_sum[0] / CNTs_sum[0] if CNTs_sum[0] > 0 else 0.
        dev_precision_final = 1.0 * TPs_sum[1] / CNTs_sum[1] if CNTs_sum[1] > 0 else 0.
        dev_precision_tone = 1.0 * TPs_sum[2] / CNTs_sum[2] if CNTs_sum[2] > 0 else 0.
        #print("batch size: %d, batch num: %d" % (batch_size , it))
        print("test_loss: %.3f, Speed: %.3f sec/epoch" % (test_loss, (time.time() - start_time)))
        print(dyz_TP_sum)
        print(dyz_CNT_sum)
        print("dev_precision_dyz  = %0.6f" % dev_precision_dyz)
        print("dev_precision_ndyz = %0.6f" % dev_precision_ndyz)                
        print("dev_precision_%s = %0.6f" % (task_list[0], dev_precision_initial))
        print("dev_precision_%s = %0.6f" % (task_list[1], dev_precision_final))
        print("dev_precision_%s = %0.6f" % (task_list[2], dev_precision_tone))
    
def multi_test_main(conf_fn, mailto, init, end):
    print("[INFO] JOB_TYPE = MULTI-TEST")
    device = conf.get("device", "device_id")
    if device != "hdfs" and device.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    print("[INFO] CUDA_VISIBLE_DEVICES: %s" % device)
    ############################# read test params ##################################
    batch_size = int(conf.get("model", "batch_size"))
    test_data_path = conf.get("test", "test_text_database")    
    
    task_list = ['initial', 'final', 'tone']
    task_num = 3

    #dyzlist_path = conf.get("dict", "dyz_list_path")
    dyz_py_dict_path = conf.get("dict", "dyz_py_list_path")
    #################################################################################
    #dyzlist = reader.read_dyzlist(dyzlist_path)
    dyz_py_dict,_ = reader.read_dyz_py_dict(dyz_py_dict_path)
    dyzlist = sorted(dyz_py_dict.keys())
    print("[INFO] read dyz_py_dict success, total number = %d" % len(dyzlist))
    #base_dir = conf.get("dir", "base_dir")
    #output_dir = "%s%sdriver_%s_test/" % (base_dir, conf.get("dir", "output_dir"), device)
    test_output_dir = conf.get("test", "output_dir")
    pred_path, report_path, model_path = create_path(test_output_dir)
    #model_path += model_name

    ### Prepare dict ###
    feature_dict, mark_dict, mark_dict_reverse = reader.prepare_dict(conf)

    #read text_data for train
    test_dataset, max_len, textlist,is_dyz_list = reader.read_rawdata_dyz(test_data_path, feature_dict, mark_dict, dyzlist)
    feature_dim = len(test_dataset[0]["feature"][0])
    print("[INFO] feature_dim = %d" % feature_dim)
    ### read mean_var ###
    mean_var_path = conf.get("test", "mean_var_path")    
    (mean_vec, var_vec) = reader.get_mean_var(test_dataset, feature_dim, mean_var_path, updating=False)
    mean_var = (mean_vec, var_vec)
    
    # feature normalize
    test_dataset = reader.dataset_normalized(test_dataset, feature_dim, mean_var)
    
    test_sample_list = range(len(test_dataset))
    print("the size of test_dataset = %d(sentence)" % len(test_dataset))
    
    ### Build model ###
    targets_dim_list = [mark_dict[x].dim for x in task_list]
    #x_in, targets_y, keep_prob, padding_mask, preds_y, loss, regular_loss
    ((x_in, targets_y, label_mask, keep_prob),
    (padding_mask, preds_y, loss_list, regular_loss)) = build_model(feature_dim, targets_dim_list, batch_size, conf)
    
    loss = sum(loss_list)

    #pred_filename = "%s%s_pred_result.txt" % (pred_path, model_name)
    #if os.path.exists(pred_filename):
    #    os.remove(pred_filename)
    #pred_file = codecs.open(pred_filename, "w", "gb18030")
    #pred_file.write("test_file is %s\n" % test_data_path)
    #pred_file.close()
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        report_file = codecs.open(os.path.join(report_path, "model_test.txt"), "w", "gb18030")
        # TODO: begin training on previous model
        for model_id in range(int(init), int(end)): 
            model_fn = os.path.join(model_path, "model-%d" % model_id)
            # load  model data        
            saver.restore(sess, model_fn)

            costs = 0.
            dyz_TP_sum = dyz_CNT_sum = 0
            ndyz_TP_sum = ndyz_CNT_sum = 0        
            #TPs_sum = np.zeros(task_num, dtype=np.float32)
            #CNTs_sum = TPs_sum.copy()
            #TPs_sum  = [np.zeros([dim], dtype=np.float32) for dim in targets_dim_list]
            #CNTs_sum = [np.zeros([dim], dtype=np.float32) for dim in targets_dim_list]
            TPs_sum = [0.] * task_num
            CNTs_sum = [0.] * task_num
            start_time = time.time()
            for i, (x_test, y_test, sample_inds, _) in enumerate(reader.batch_generator(test_dataset, test_sample_list, mean_var, batch_size, is_training=False)):
                #fetches = [loss, TPs, CNTs]
                batch_shape = list(y_test.shape)[0:-1]
                fetches = [loss, preds_y, padding_mask]
                feed_dict = {}
                feed_dict[x_in] = x_test
                targets_y_test = []
                st = 0
                #不能在使用i,会与外循环的冲突
                for k,dim in enumerate(targets_dim_list):
                    targets_y_test.append(y_test[:,:,st:st+dim])
                    feed_dict[targets_y[k]] = targets_y_test[k]
                    st += dim
                    
                    
                dyz_mask = np.zeros(batch_shape, dtype = np.float32)
                for k,ind in enumerate(sample_inds):
                    dyz_mask[k,0:len(is_dyz_list[ind])] = np.array(is_dyz_list[ind], dtype=np.float32)
                dyz_mask = np.sign(dyz_mask)

                label_mask_test = np.sign(np.amax(targets_y_test[-1][:,:,1:], axis=2))
                #label_mask_test = np.logical_or(np.logical_not(dyz_mask),
                #    np.logical_and(label_mask_test, dyz_mask))
                label_mask_test = label_mask_test.astype(np.float32)
                
                feed_dict[label_mask] = label_mask_test
                feed_dict[keep_prob] = 1.0
                cost_test, preds_y_test, padding_mask_test = sess.run(fetches, feed_dict=feed_dict)
                
                #write_result_file_new(pred_filename, textlist[batch_size * i: min(batch_size* (i+1),len(textlist))],
                #        preds_y_test, mark_dict, mark_dict_reverse, dyz_py_dict)
                #write_result_file(pred_filename, textlist[batch_size * i: min(batch_size*(i+1), len(textlist))], preds_y_test, mark_dict_reverse)
                
                costs += cost_test

                #print(targets_y_test[2][0])
                #print(preds_y_test[2][0])
                dyz_label_mask = np.logical_and(np.sign(np.amax(targets_y_test[-1][:,:,1:], axis=2)),
                    np.sign(dyz_mask))
                dyz_TP, dyz_CNT, ndyz_TP, ndyz_CNT = dyz_result_watcher(targets_y_test,
                        preds_y_test, padding_mask_test, dyz_label_mask)
                dyz_TP_sum += dyz_TP
                dyz_CNT_sum += dyz_CNT
                ndyz_TP_sum += ndyz_TP
                ndyz_CNT_sum += ndyz_CNT
                
                for k,task in enumerate(task_list):
                    TP, CNT = result_watcher(targets_y_test[k], preds_y_test[k], padding_mask_test)
                    TPs_sum[k] += TP
                    CNTs_sum[k] += CNT

            test_loss = costs
            dev_precision_dyz = dyz_TP_sum / dyz_CNT_sum
            dev_precision_ndyz = ndyz_TP_sum / ndyz_CNT_sum
            dev_precision_initial = TPs_sum[0] / CNTs_sum[0]
            dev_precision_final = TPs_sum[1] / CNTs_sum[1]
            dev_precision_tone = TPs_sum[2] / CNTs_sum[2]
            #print("batch size: %d, batch num: %d" % (batch_size , it))
            print("========================model-%d========================" % model_id)
            
            print("test_loss: %.3f, Speed: %.3f sec/epoch" % (test_loss, (time.time() - start_time)))
            print(dyz_TP_sum)
            print(dyz_CNT_sum)
            #report_file.write("dyz_right_num = %d, dyz_num = %d\n" % (dyz_TP_sum, dyz_CNT_sum))
            print("dev_precision_dyz  = %0.6f" % dev_precision_dyz)
            report_file.write("%d %0.6f\n" % (model_id, dev_precision_dyz))
            print("dev_precision_ndyz = %0.6f" % dev_precision_ndyz)                
            print("dev_precision_%s = %0.6f" % (task_list[0], dev_precision_initial))
            print("dev_precision_%s = %0.6f" % (task_list[1], dev_precision_final))
            print("dev_precision_%s = %0.6f" % (task_list[2], dev_precision_tone))
        
        report_file.close()
            

            
def train_operator(conf, mailto):
    print("[INFO] JOB_TYPE = TRAIN")
    device = conf.get("device", "device_id")
    if device != "hdfs" and device.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    print("[INFO] CUDA_VISIBLE_DEVICES: %s" % device)
    
    task_list = ['initial', 'final', 'tone']
    task_num = 3
    ################################ read train config #########################################
    n_epoches = int(conf.get("train", "rounds")) #201
    print_freq = int(conf.get("train", "print_freq")) #1
    batch_size = int(conf.get("model", "batch_size"))
    train_data_path = conf.get("train", "train_text_database")
    dev_data_path = conf.get("train", "dev_text_database")
    train_keep_prob = 1.0 - float(conf.get('train', 'dropout_rate'))
    lr = float(conf.get("train", "learning_rate"))
    alpha = 0.0
    dyz_py_list_path = conf.get("dict", "dyz_py_list_path")
    
    #加载初始模型
    try:
        init_model_path = conf.get("model", "initial_model")
    except ConfigParser.NoOptionError:
        init_model_path = ""
    try:
        start_epoch = conf.getint("train", "start")
    except (ConfigParser.NoOptionError, ValueError):
        start_epoch = 0
    
    #init_model_name = "model-%d" % start_epoch
    
    ############################## read train config end #######################################
    
    dyz_py_dict, _ = reader.read_dyz_py_dict(dyz_py_list_path)
    dyzlist = dyz_py_dict.keys()
    base_dir = conf.get("dir", "base_dir")
    output_dir = "%s%sdriver_%s/" % (base_dir, conf.get("dir", "output_dir"), device)
    #output_dir += "driver_%s/" % device
    pred_path, report_path, model_path = create_path(output_dir)
    
    #model_name = conf.get("train", "model_name")
    
    #print(pred_path)
    ### Prepare dict ###
    #feature_dim, feature_dict, mark_dict, mark_dict_reverse = reader.prepare_dict(conf)
    feature_dict, mark_dict, _ = reader.prepare_dict(conf)
    
    #read text_data for train
    dataset, max_len, _, is_dyz_list_train = reader.read_rawdata_dyz(train_data_path, feature_dict, mark_dict, dyzlist)
    feature_dim = len(dataset[0]['feature'][0])
    print("[INFO] feature_dim = %d" % feature_dim)
    ### read mean_var ###
    mean_var_path = conf.get("train", "mean_var_path")    
    (mean_vec, var_vec) = reader.get_mean_var(dataset, feature_dim, mean_var_path, updating = False)
    mean_var = (mean_vec, var_vec)    

    dev_dataset, max_len, _, is_dyz_list_dev = reader.read_rawdata_dyz(dev_data_path, feature_dict, mark_dict, dyzlist)
    
    #print("total dyz number in dev = %d" % sum(np.sum(np.array(is_dyz_list))))
    
    # feature normalize
    dataset = reader.dataset_normalized(dataset, feature_dim, mean_var)
    dev_dataset = reader.dataset_normalized(dev_dataset, feature_dim, mean_var)
    
    #extract dataset for dev
    #rnds = range(len(dataset))
    #random.shuffle(rnds)
    #dev_size = int(math.floor(len(rnds) * 0.1))
    #print(dev_size)
    #dev_sample_list = rnds[:dev_size]
    train_sample_list = range(len(dataset))
    print("the size of train_dataset = %d(sentence)" % len(train_sample_list))
    dev_sample_list = range(len(dev_dataset))
    print("the size of dev_dataset   = %d(sentence)" % len(dev_sample_list))    
    
    
    ### Build model ###
    targets_dim_list = [mark_dict[x].dim for x in task_list]
    ((x_in, targets_y, label_mask, keep_prob),
    (padding_mask, preds_y, loss_list, regular_loss)) = build_model(feature_dim, targets_dim_list, batch_size, conf)
    
    ### calculate loss ###
    try:
        loss_weight = conf.get('train', 'multi_loss_weight')
        loss = sum([loss_list[i] * float(w) for i,w in enumerate(loss_weight.split('|'))])
        print("[INFO] weighted loss = %s" % loss_weight)
    except Exception:
        loss = sum(loss_list)
    except (IndexError,ValueError):
        sys.exit("[ERROR] loss_weight does not match loss_list for multi-task")
    
    #loss = sum(loss_list)
    loss += alpha * regular_loss
    #loss = sum(loss_list)
    ### calculate TP and CNT ###
    #TPs = [0.] * task_num
    #CNTs = [0.] * task_num
    #for i,task in enumerate(task_list):
    #    TPs[i], CNTs[i] = result_watcher(targets_y[i], preds_y[i], padding_mask)

    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    
    #init = tf.global_variables_initializer()
    model_saver = tf.train.Saver(sharded=False, max_to_keep=None)    

    with tf.Session() as sess:
        
        # begin training on previous model; otherwise, begin on 0      
        if init_model_path == "":
            print("[INFO] a new model will be built")
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            print("[INFO] initial model is '%s'" % (init_model_path))
            model_saver.restore(sess, init_model_path)
        
        current_epoch = start_epoch
        start_time = time.time()
        aver_loss = 0.
        
        print("[INFO] Training ...")
         
        while current_epoch < n_epoches:
         
            #if current_epoch % print_freq == 0: 
            #    start_time = time.time()
            #    costs = 0.
            #    TPs_sum = [0.0] * task_num
            #    CNTs_sum = [0.0] * task_num
            
            batch_num = 0
            # Generate mini-batch data and train the model
            for (x_train, y_train, sample_inds, _) in reader.batch_generator(dataset, train_sample_list, mean_var, batch_size):

                y_train_list = []
                feed_dict = {}
                feed_dict[x_in] = x_train
                st = 0
                for i,dim in enumerate(targets_dim_list):
                    y_train_list.append(y_train[:,:,st:st+dim])
                    feed_dict[targets_y[i]] = y_train_list[i]
                    st += dim
                
                # 计算label_mask
                batch_shape = list(y_train.shape)[0:-1]
                #print(batch_shape)
                dyz_mask_train = np.zeros(batch_shape, dtype=np.float32)
                for k,ind in enumerate(sample_inds):
                    dyz_mask_train[k,0:len(is_dyz_list_train[ind])] = np.array(is_dyz_list_train[ind], dtype=np.float32)
                dyz_mask_train = np.sign(dyz_mask_train)
                #print("dyz_mask:")
                #print(dyz_mask_train)
                label_mask_train = np.sign(np.amax(y_train_list[-1][:,:,1:], axis=2))
                #print("tone_valid:")
                #print(label_mask_train)
                
                #label_mask_train = np.logical_or(np.logical_not(dyz_mask_train),
                #    np.logical_and(label_mask_train, dyz_mask_train))
                label_mask_train = label_mask_train.astype(np.float32)
                #print("label_mask_train:")
                #print(label_mask_train)
                feed_dict[label_mask] = label_mask_train
                
                feed_dict[keep_prob] = train_keep_prob
                ### feed data for train ###
                train_step.run(feed_dict=feed_dict)
                #fetches = loss
                #train_coss = train_step.run(fetches, feed_dict=feed_dict)
                
                #TODO:不太合理，计算训练集准确率时，每个batch模型都会更新
                if current_epoch % print_freq == 0:
                    #feed traindata for test
                    #fetches = [loss, TPs, CNTs]
                    fetches = loss
                    
                    feed_dict[keep_prob] = 1.0
                    # 不要与fetches中参数重名,否则 第二次运行时，参数类型不再是tensor
                    train_cost = sess.run(fetches, feed_dict=feed_dict)
                    aver_loss += train_cost
                    batch_num += 1
                    
                    #for i in range(task_num):
                    #    TPs_sum[i] += TP_train[i]
                    #    CNTs_sum[i] += CNT_train[i]
            #
            if current_epoch % print_freq == 0:
                
                model_name = "model-%d" % current_epoch
                
                #save current model(weights): write_meta_graph=True, save graph and data; False, save data only
                save_dir = model_saver.save(sess, os.path.join(model_path, model_name), write_meta_graph=False)
                
                aver_loss /= 0.01 * print_freq * batch_num
                #aver_precision_initial = TPs_sum[0] / CNTs_sum[0]
                #aver_precision_final = TPs_sum[1] / CNTs_sum[1]
                #aver_precision_tone = TPs_sum[2] / CNTs_sum[2]
                ### print evalution results ###
                print ("==================Epoch: %d==================" %(current_epoch))

                print("train_loss: %.3f, Speed: %.3f sec/epoch" % (aver_loss, (time.time()-start_time) / print_freq))
                #print("train_precision_%s = %0.6f" % (task_list[0], aver_precision_initial))
                #print("train_precision_%s = %0.6f" % (task_list[1], aver_precision_final))
                #print("train_precision_%s = %0.6f" % (task_list[2], aver_precision_tone))
                
                #for i, task in enumerate(task_list)):
                #    print("train_precision_%s = %0.6f" % (task, TPs_sum[i] / CNTs_sum[i])))
                
                ########################### dev test ###########################
                dev_loss = 0.
                #TPs_sum = [0.0] * task_num
                #CNTs_sum = [0.0] * task_num
                dyz_TP_sum = dyz_CNT_sum = 0
                ndyz_TP_sum = ndyz_CNT_sum = 0

                start_time = time.time()
                batch_num = 0
                for (x_test, y_test, sample_inds, _) in reader.batch_generator(dev_dataset, dev_sample_list, mean_var, batch_size, is_training=False):
                    #fetches = [loss, TPs, CNTs]
                    batch_shape_dev = list(y_test.shape)[0:-1]
                    #print(batch_shape)
                    fetches = [loss, preds_y, padding_mask] 
                    feed_dict = {}
                    feed_dict[x_in] = x_test
                    y_test_list = []
                    st = 0
                    for i,dim in enumerate(targets_dim_list):
                        y_test_list.append(y_test[:,:,st:st+dim])
                        feed_dict[targets_y[i]] = y_test_list[i]
                        st += dim

                    dyz_mask = np.zeros(batch_shape_dev, dtype=np.float32)
                    for k,ind in enumerate(sample_inds):
                        dyz_mask[k,0:len(is_dyz_list_dev[ind])] = np.array(is_dyz_list_dev[ind], dtype=np.float32)
                    dyz_mask = np.sign(dyz_mask)
                    
                    label_mask_dev = np.sign(np.amax(y_test_list[-1][:,:,1:], axis=2))
                    #label_mask_dev = np.logical_or(np.logical_not(dyz_mask),
                    #    np.logical_and(label_mask_dev, dyz_mask))
                    label_mask_dev = label_mask_dev.astype(np.float32)
                    
                    
                    feed_dict[label_mask] = label_mask_dev
                    feed_dict[keep_prob] = 1.0
                    #cost_dev, TP_dev, CNT_dev = sess.run(fetches, feed_dict=feed_dict)
                    cost_dev, preds_y_dev, padding_mask_dev = sess.run(fetches, feed_dict=feed_dict)                   
                    dev_loss += cost_dev
                    batch_num += 1                    
                    
                    #batch_shape = list(y_test.shape)[0:-1]
                    
                    dyz_label_mask = np.logical_and(np.sign(np.amax(y_test_list[-1][:,:,1:], axis=2)),
                        np.sign(dyz_mask))
                    dyz_TP, dyz_CNT, ndyz_TP, ndyz_CNT = dyz_result_watcher(y_test_list,
                            preds_y_dev, padding_mask_dev, dyz_label_mask)
                    dyz_TP_sum += dyz_TP
                    dyz_CNT_sum += dyz_CNT
                    ndyz_TP_sum += ndyz_TP
                    ndyz_CNT_sum += ndyz_CNT
                    
                    #for i in range(task_num):
                    #    TPs_sum[i] += TP_dev[i]
                    #    CNTs_sum[i] += CNT_dev[i]
                
                dev_loss /= 0.01 * batch_num
                dev_precision_dyz = dyz_TP_sum / dyz_CNT_sum
                dev_precision_ndyz = ndyz_TP_sum / ndyz_CNT_sum
                #dev_precision_initial = TPs_sum[0] / CNTs_sum[0]
                #dev_precision_final = TPs_sum[1] / CNTs_sum[1]
                #dev_precision_tone = TPs_sum[2] / CNTs_sum[2]
                #print("batch size: %d, batch num: %d" % (batch_size , it))
                print("dyz:total_number = %d, right_number = %d" % (dyz_CNT_sum, dyz_TP_sum))
                print("dev_loss: %.3f, Speed: %.3f sec/epoch" % (dev_loss, (time.time() - start_time) / print_freq))
                print("dev_precision_dyz  = %0.6f" % dev_precision_dyz)
                print("dev_precision_ndyz = %0.6f" % dev_precision_ndyz)                
                #print("dev_precision_%s = %0.6f" % (task_list[0], dev_precision_initial))
                #print("dev_precision_%s = %0.6f" % (task_list[1], dev_precision_final))
                #print("dev_precision_%s = %0.6f" % (task_list[2], dev_precision_tone))
                
                #for i, task in enumerate(task_list)):
                #   print("dev_precision_%s = %0.6f" % (task, TPs_sum[i] / CNTs_sum[i])))
                
                #将结果打印到文本
                #output_list = [aver_loss, dev_loss,
                #               aver_precision_initial, dev_precision_initial,
                #               aver_precision_final, dev_precision_final,
                #               aver_precision_tone, dev_precision_tone]            
                #output_list = [aver_loss, dev_loss, dev_precision_initial, dev_precision_final, dev_precision_tone]                
                output_list = [aver_loss, dev_loss, dev_precision_dyz, dev_precision_ndyz]              
                #print(output_list)
                write_csv_report(report_path, current_epoch, output_list)
                
                start_time = time.time()

            current_epoch += 1
        
        

mailto = ''
if __name__ == '__main__':
        
    if len(sys.argv) < 3:
        exit("need at least 2 args: argv[1]: config filename\n" + \
             "                      argv[2]: job type (train/test)\n" + \
             "                      argv[2]: e-mail url(N/R)")
    elif len(sys.argv) > 3:
        mailto = sys.argv[3]

    # get_report(logfile)
    print(tf.__version__)
    conf =ConfigParser.ConfigParser()
    conf.read(sys.argv[1])
    
    #json conf
    #conf = reader.read_config_file(sys.argv[1])

    if sys.argv[2] == "train":        
        train_operator(conf, mailto)
    elif sys.argv[2] == "test":
        test_operator(conf, mailto)
    elif sys.argv[2] == "multi-test":
        multi_test_main(conf, mailto, int(sys.argv[3]), int(sys.argv[4]))
    else:
        exit("[ERROR] invalid args! argv[1]: config file path\n" + \
             "                      argv[2]: job type (train/test)\n")        
    

        #TODO: visualization with tensorboard

        #writer = tf.train.SummaryWriter("./logs/", sess.graph)


#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

