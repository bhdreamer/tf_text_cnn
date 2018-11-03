#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: transformer_unittest.py
# @brief: 
# @author: niezhipeng
# @Created on 2018/10/17
# *************************************************************************************

import os
import sys
import unittest
import numpy as np
import tensorflow as tf
from models.transformer import Transformer

PAD = "<PAD>"
PAD_ID = 0
EOS = "<EOS>"
EOS_ID = 1

class TransformerUnitTest(unittest.TestCase):
    def setUp(self):
        self._feature_dims = [128, 4, 39]
        self._targets_dim_list = [5]
        model_cfg = {"vocab_size": 128}
        self.model = Transformer(self._feature_dims, self._targets_dim_list, model_cfg)
        self.sess = tf.Session()

    def _test_position_encoding(self):
        print("running test | position_encoding...")
        batch_size = 1
        length = 5
        dim = 7
        pos_encoding1 = self.model.position_encoding_v1(batch_size, length, dim)
        pos_encoding2 = self.model.position_encoding(batch_size, length, dim)
        pos_encoding3 = self.model.position_encoding_v3(length, dim)
        print(pos_encoding1)
        print(pos_encoding2)
        print(pos_encoding3)
        pos_enc1, pos_enc2 = self.sess.run([pos_encoding1, pos_encoding2])
        print(pos_enc1)
        print(pos_enc2)
        print(pos_encoding3)

    def _test_multihead_attention(self):
        print("running test | multihead attention ... ")
        query = tf.random_normal([3, 6, 5], mean=0., stddev=1.)
        key = tf.random_normal([3, 5, 5], mean=0., stddev=1.)
        value = key
        key_padding =  tf.to_float(tf.constant([
            [True, True, True, False, False],
            [True, True, True, True, True],
            [True, False, False, False, False]]))
        hidden_size = 8
        num_heads = 8
        causal = False
        scope = "self_attention"
        outputs, atten_matrix = self.model.multihead_attention(
            query, key, value, key_padding,
            hidden_size=hidden_size,
            num_heads=num_heads,
            causal=causal,
            scope=scope)
        print(atten_matrix)
        init_args = tf.global_variables_initializer()
        self.sess.run(init_args)
        out, atten_mtx = self.sess.run([outputs, atten_matrix])
        # print(out)
        print(atten_mtx)

    def _test_encoder_block(self):
        print("running test | encoder block...")
        inputs = tf.random_normal([3, 5, 8], mean=0., stddev=1.)
        paddings =  tf.to_float(tf.constant([
            [True, True, True, False, False],
            [True, True, True, True, True],
            [True, False, False, False, False]]))
        hparams = {
            "hidden_size":8,
            "num_heads":8,
            "filter_size": 32,
            "dropout":0.2
        }

        outputs, _ = self.model.encoder_block(
            inputs, hparams, paddings,
            is_training=True,
            scope="encode")
        print(outputs)
        init_args = tf.global_variables_initializer()
        self.sess.run(init_args)
        out= self.sess.run(outputs)
        print(out)

    def test_init_layer(self):
        print("running test | build encoder-decoder ...")
        encoder_out = tf.random_normal([3, 5, 8], mean=0., stddev=1.)
        enc_paddings =  tf.to_float(tf.constant([
            [True, True, True, False, False],
            [True, True, True, True, True],
            [True, False, False, False, False]]))
        inputs = tf.constant([[[1],[2],[3],[1],[4]],
                              [[2],[1],[4],[3],[0]],
                              [[3],[2],[1],[0],[0]]])

        targets = tf.constant([[1, 2, 3, 1, 4, 0],
                               [2, 1, 3, 2, 1, 4],
                               [1, 4, 0, 0, 0, 0]], tf.int32)
        self.model._targets_list = list()
        self.model._targets_list.append(targets)
        #targets = tf.expand_dims(targets, axis=-1)
        # targets = tf.random_normal([3, 6, 8], mean=0., stddev=1.0)
        dec_paddings = tf.to_float(tf.constant([
            [True, True, True, True, False, False],
            [True, True, True, True, True, True],
            [True, False, False, False, False, False]]))

        model_cfg = {
            "encoder": {
                "hidden_size": 8,
                "num_heads": 8,
                "filter_size": 32,
                "num_blocks": 6,
                "dropout": 0.2
            },
            "output_layer": [
                {
                "num_classes": 5,
                "max_seq_len": 256,
                "hidden_size": 8,
                "num_heads": 8,
                "filter_size": 32,
                "num_blocks": 6,
                "dropout": 0.2
                }
            ]
        }
        _wordvec = np.random.normal(size=6*8)
        self.model._wordvec = np.reshape(_wordvec, [6, 8])
        _, outputs = self.model._init_layers(encoder_out, model_cfg)

        print(outputs)
        init_args = tf.global_variables_initializer()
        self.sess.run(init_args)
        out = self.sess.run(outputs[0])
        print(out)

if __name__ == "__main__":
    unittest.main()
