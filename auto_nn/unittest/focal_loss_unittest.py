#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: focal_loss_unittest.py
# @brief: 
# @author: niezhipeng(@baidu.com)
# @Created on 2018/8/6
# *************************************************************************************
import os
import sys
import numpy as np
import tensorflow as tf

''' 
def calc_focal_loss(labels, logits, mask, alpha, gamma=2):
    """
        reference: https://github.com/zhezh/focalloss/blob/master/focalloss.py
    """
    epsilon = 1.e-9
    labels = tf.convert_to_tensor(labels, tf.int64)
    logits = tf.convert_to_tensor(logits, tf.float32)
    alpha = tf.convert_to_tensor(alpha, tf.float32)
    label_mask = tf.cast(mask, dtype=tf.float32)
    num_cls = logits.get_shape().as_list()[-1]

    model_out = tf.add(tf.nn.softmax(logits, dim=-1), epsilon)
    onehot_labels = tf.one_hot(labels, num_cls)
    ce = tf.multiply(onehot_labels, -tf.log(model_out))
    weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))

    _loss = tf.multiply(alpha, tf.multiply(weight, ce))
    return tf.reduce_sum(_loss) / tf.reduce_sum(label_mask)
'''


def calc_focal_loss(labels, logits, padding_mask, alpha, gamma=2):
    """ focal loss for multi-classes,
            FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        where
            p_t: the softmax value with true labels as index
    args:
        labels: Tensor([B, T], tf.int64), ground true label
        logits: Tensor([B, T, num_classes], tf.float32), model output
        mask:   Tensor([B, T], tf.bool), padding_mask
        alpha:  LIST, [1.0, 1.0, 1.0, 0.7, 1.0]
        gamma:  INT, default=2
    return:
    """
    labels = tf.convert_to_tensor(labels, tf.int64)
    logits = tf.convert_to_tensor(logits, tf.float32)
    alpha = tf.convert_to_tensor(alpha, tf.float32)
    label_mask = tf.cast(padding_mask, dtype=tf.float32)

    num_cls = logits.get_shape().as_list()[-1]
    probs = tf.add(tf.nn.softmax(logits, dim=-1), 1.e-10)
    onehot_labels = tf.one_hot(labels, num_cls)
    weight = label_mask * tf.nn.embedding_lookup(alpha, labels)
    # ce_loss
    # _loss = tf.reduce_sum(tf.multiply(onehot_labels, -tf.log(probs)), axis=-1)
    p_t = tf.reduce_sum(tf.multiply(onehot_labels, probs), axis=-1)
    _loss = tf.multiply(tf.pow(tf.subtract(1., p_t), gamma), -tf.log(p_t))
    aver_loss = tf.reduce_sum(_loss * weight) / tf.reduce_sum(label_mask)
    return aver_loss


def calc_ce_loss(labels, logits, mask, alpha):
    labels = tf.convert_to_tensor(labels, tf.int64)
    logits = tf.convert_to_tensor(logits, tf.float32)
    alpha = tf.convert_to_tensor(alpha, tf.float32)
    label_mask = tf.cast(mask, dtype=tf.float32)

    num_cls = logits.get_shape().as_list()[-1]
    weighted_label = tf.one_hot(labels, num_cls)
    weighted_label *= alpha
    smooth_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=tf.reshape(weighted_label, [-1, num_cls]),
        logits=tf.reshape(logits, [-1, num_cls]),
        weights=tf.reshape(label_mask, [-1]),
        label_smoothing=0
    )
    '''
    # old version
    weights = tf.nn.embedding_lookup(alpha, labels)
    ce_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits,
        weights=label_mask * weights,
    )
    '''
    return smooth_loss

def calc_ce_loss_v2(labels, logits, mask, alpha):
    def label_smoothing(onehot_label, epsilon):
        num_cls = onehot_label.get_shape().as_list()[-1]
        return ((1-epsilon) * onehot_label) + (epsilon / num_cls)

    labels = tf.convert_to_tensor(labels, tf.int64)
    logits = tf.convert_to_tensor(logits, tf.float32)
    alpha = tf.convert_to_tensor(alpha, tf.float32)
    label_mask = tf.cast(mask, dtype=tf.float32)
    num_cls = logits.get_shape().as_list()[-1]
    onehot_label = tf.one_hot(labels, num_cls)

    weights = tf.nn.embedding_lookup(alpha, labels)
    epsilon = 0.
    if 0. < epsilon < 1.:
        onehot_label = (1-epsilon) * onehot_label + epsilon / num_cls

    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=onehot_label,
        logits=logits) * weights * label_mask

    return (tf.reduce_sum(loss) / tf.reduce_sum(label_mask))

def test_main():
    num_cls = 5
    alpha = tf.constant([1, 1, 0.2, 1, 1], tf.float32)
    labels = tf.constant([[0, 1, 2, 3, 0],[0, 1, 0, 2, 3]], tf.int64)
    pre_score = tf.constant([[0, 1, 2, 4, 4],[0,2,0,3, 4]], tf.int64)
    pre_logits = tf.one_hot(pre_score, depth=num_cls, on_value=5.)
    logits = tf.random_normal([2,5,5], mean=0, stddev=1.0) + pre_logits
    mask = tf.constant([[True, True, True, True, False], [True, True, True, True, True]])

    #fl = calc_focal_loss(labels, logits, mask, alpha)

    ce1 = calc_ce_loss(labels, logits, mask, alpha)
    ce2 = calc_ce_loss_v2(labels, logits, mask, alpha)
    with tf.Session() as sess:
        [ce1_out, ce2_out] = sess.run([ce1, ce2])
        print(ce1_out)
        print(ce2_out)

if __name__ == "__main__":
    test_main()