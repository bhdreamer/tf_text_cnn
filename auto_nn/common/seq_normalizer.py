#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: seq_normalizer.py
# @brief: 
# @author: niezhipeng(@baidu.com)
# @Created on 2017/7/15
# @Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
# *************************************************************************************
import os
import sys
import codecs
import numpy as np
from collections import Iterator


class GaussNormalizer(object):
    def __init__(self, mean_var_path="", updating=False):
        self._mean_var_path = mean_var_path
        if not updating and os.path.exists(self._mean_var_path):
            self.read_mean_var(self._mean_var_path)
            self._updating = False
        else:
            self._mean_vec = None  # 均值
            self._var_vec = None  # 方差
            self._recip_var_vec = None  # 方差倒数，方便归一化运算
            self._updating = True

    def is_valid(self, feature_dim):
        return (isinstance(self._mean_vec, np.ndarray) and
                isinstance(self._var_vec, np.ndarray) and
                self.dim == feature_dim)

    @property
    def mean_var_vec(self):
        return (self._mean_vec, self._var_vec)

    @property
    def dim(self):
        return len(self._mean_vec)

    def normalize(self, features):
        """ 归一化
        args:
            features: n-dimensions list, entry in each dimension has the same size
        :return:
            np.array, the same shape with input, np.float32
        """
        features = np.asarray(features, dtype=np.float32)
        if features.shape[-1] != self.dim:
            return features
        return (features - self._mean_vec) * self._recip_var_vec

    def batch_normalize(self, batch_features):
        """ 序列归一化
        args:
            seq_list: LIST of list of feature,
            each feature has the same dimension
        :return
            LIST of np.array,
        """
        #print(seq_list)
        if not self.is_valid(len(batch_features[0][0])):
            return batch_features

        norm_features = []
        for seq in batch_features:
            norm_features.append(self.normalize(seq))
        return norm_features

    def anti_normalize(self, norm_features):
        """ 反归一化
        :param norm_features --- 多维列表形式
        :return --- np.array, the same shape with input
        """
        features = np.array(norm_features)
        if features.shape[-1] != self.dim:
            return features
        return features * self._var_vec + self._mean_vec

    def read_mean_var(self, mean_var_path):
        """ read mean, std_var from file
        :param mean_var_path:
        :return:
        """
        with codecs.open(mean_var_path, 'r', encoding='gbk') as file:
            self._mean_var_path = mean_var_path
            means = []
            vars = []
            for line in file:
                mean, var = line.strip().split()
                means.append(float(mean))
                vars.append(float(var))
            assert len(means) == len(vars)
            means[128:] = [0.] * (len(means)-128)
            vars[128:] = [1.] * (len(means)-128)
            self._mean_vec = np.array(means, dtype=np.float32)
            self._var_vec = np.array(vars, dtype=np.float32)
            self._recip_var_vec = 1.0 / self._var_vec
            self._dim = len(self._mean_vec)
            # print("mean", self._mean_vec)
            # print("recip_var", self._recip_var_vec)
            return True

    def gen_mean_var(self, feature_dataset, mean_var_path):
        """ 计算均值和方差, 并写入文件 format: mean[i]<\t>std_var[i]
        args:
            feature_dataset --- Iterable, have the same length and dim
        """
        if isinstance(feature_dataset, Iterator):
            pass

        self._mean_var_path = mean_var_path

        sum_x = None
        sum_x2 = None

        num = 0
        for feature in feature_dataset:
            feature_vec = np.asarray(feature, dtype=np.float32)
            dim = feature_vec.shape[-1]
            feature_vec.reshape([-1, dim])
            #print(feature_vec)
            try:
                sum_x += np.sum(feature_vec, axis=0)
                sum_x2 += np.sum(feature_vec * feature_vec, axis=0)
            except TypeError:
                sum_x = np.sum(feature_vec, axis=0)
                sum_x2 = np.sum(feature_vec * feature_vec, axis=0)

            num += len(feature)

        self._mean_vec = sum_x / float(num)
        self._var_vec = np.sqrt(np.abs(sum_x2 / float(num) - self._mean_vec ** 2))
        self._var_vec = np.maximum(self._var_vec, np.ones_like(self._var_vec) * 1e-8)
        self._recip_var_vec = 1.0 / self._var_vec
        self._dim = len(self._mean_vec)

        with codecs.open(mean_var_path, 'w', 'gb18030') as out_file:
            for mean, var in zip(self._mean_vec, self._var_vec):
                out_file.write("%.8f\t%.8f\n" % (mean, var))
        return (self._mean_vec, self._var_vec)


