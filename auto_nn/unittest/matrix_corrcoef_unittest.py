#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: .py
# @brief: 
# @author: niezhipeng(@baidu.com)
# @Created on 2018/6/29
# *************************************************************************************
import scipy.stats as stats
import tensorflow as tf
import numpy as np
#m1 = np.array([[1,2,2,1],[1,2,3,2], [1,1,1,1]])
m1 = np.array([[0,0,0,1],[1,0,0,0], [0,1,0,0]])
m2 = np.array([[2,4,5,3],[3,5,6,4], [0,0,0,1]])
#m1 = np.random.random_integers(0, 10, (2,4))
#m2 = np.random.random_integers(0, 10, (3,4))
print(m1)
print(m2)
print(np.corrcoef(m2, m1, rowvar=True))
#print(stats.pearsonr(m1[0], m2[1]))

def calc_2d_cosine_coef(t1, t2):
    normed_t1 = tf.nn.l2_normalize(t1, dim=-1)
    normed_t2 = tf.nn.l2_normalize(t2, dim=-1)
    #print(normed_t1, normed_t2)
    sim = tf.matmul(normed_t1, normed_t2, transpose_b=True)
    return sim


def calc_2d_corrcoef(t1, t2):
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
    coef = t1_cov @ cov @ t2_cov
    #return coef
    shape[-1] = -1
    return tf.reshape(coef, shape)

if __name__ == "__main__":
    with tf.Session() as sess:
        t1 = tf.constant(m1, dtype=tf.float32)
        t2 = tf.constant(m2, dtype=tf.float32)
        cor = calc_2dtensor_corrcoef(t2, t1)
        print(cor.eval())
        cosine_coef = calc_2dtensor_cosinecoef(t2, t1)
        print(cosine_coef.eval())

