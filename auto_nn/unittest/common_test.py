#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: common_test.py
# @brief: 
# @author: niezhipeng
# @Created on 2018/10/17
# *************************************************************************************
import numpy as np
import tensorflow as tf

def padding_mask_test():
    inputs = tf.random_normal([2, 5, 5], mean=0., stddev=1.0)
    padding_mask = tf.to_float(tf.constant([[True, True, True, False, False],
                                            [True, True, True, True, True]]))
    dim = 5

    _mask1 = tf.expand_dims(padding_mask, axis=-1)
    #_mask1 = tf.concat([_mask1] * dim, axis=-1)
    # print(_mask)
    out1 = tf.multiply(inputs, _mask1)

    #_mask2 = tf.tile(tf.expand_dims(padding_mask, axis=-1),
    #                [1, 1, dim])
    _mask2 = tf.expand_dims(padding_mask, axis=-1)
    out2 = inputs * _mask2

    with tf.Session() as sess:
        m1, o1, m2, o2 = sess.run([_mask1, out1, _mask2, out2])
        print(m1)
        print(o1)
        print(m2)
        print(o2)

if __name__ == "__main__":
    padding_mask_test()

