#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: deq2seq_model.py
# @brief: 
# @author: niezhipeng(@baidu.com)
# @Created on 2017/7/2
# @Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
# *************************************************************************************

import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.seq2seq as seq2seq

class DynamicSeq2seq(object):
    """ dynamic_rnn_seq2seq with tensorflow-1.0.0
        encoder_inputs:  [batch_size, encoder_max_sen_len, 1 or feature_dim]
        decoder_targets: [batch_size, decoder_max_seq_len, 1]
        support attention, either "bahdanau" or "luong"
    args:
        encoder_cell --- rnn cell
        encoder_num_layers --- number of rnn layers for encoder
        decoder_cell --- rnn cell
        decoder_num_layers --- number of rnn layer for decoder
        encoder_vocab_size --- when embedding encoder inputs, the size of vocabulary
        decoder_vocab_size --- when embedding decoder inputs, the size of vocabulary
        embedding_size ---
    """
    PAD = 0
    EOS = 2
    UNK = 3
    def __init__(self, encoder_cell,
                 encoder_num_layers,
                 decoder_cell,
                 decoder_num_layers,
                 encoder_input_dim=None,
                 encoder_vocab_size=None,
                 decoder_vocab_size=None,
                 embedding_size=None,
                 bidirectional=True,
                 attention_type=None,
                 time_major=False):
        self.encoder_cell = encoder_cell
        self.encoder_num_layers = encoder_num_layers
        self.decoder_cell = decoder_cell
        self.decoder_num_layers = decoder_num_layers

        self.encoder_input_dim = encoder_input_dim
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.embedding_size = embedding_size
        initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
        if self.encoder_vocab_size and self.encoder_vocab_size > 0:
            self.encoder_embedding_matrix = tf.get_variable(
                name="encoder_embedding_matrix",
                shape = [self.encoder_vocab_size, self.embedding_size],
                initializer = initializer,
                dtype=tf.float32
            )
        if self.decoder_vocab_size and self.decoder_vocab_size > 0:
            self.decoder_embedding_matrix = tf.get_variable(
            name="decoder_embedding_matrix",
            shape=[self.decoder_vocab_size, self.embedding_size],
            initializer=initializer,
            dtype=tf.float32
        )
        """
        if attention_type and attention_type.lower() in []
        else:
            self.attention_type = None
        """
        self.attention_type = attention_type
        self.bidirectional = bidirectional
        self.time_major = time_major
        self.global_step = tf.Variable(-1, trainable=False)

    def build_graph(self):
        # 创建占位符
        self._init_placeholders()

        self._init_decoder_train_feeds()

        self._init_embeddings()

        # 创建encoder
        if self.bidirectional:
            self._bidirectional_rnn_encoder()
        else:
            self._simple_rnn_encoder()

        # 创建decoder
        self._rnn_decoder()

        self._apply_optimizer()

    def train_step(self, encoder_inputs, encoder_seq_lens, decoder_targets, decoder_seq_lens, sess):
        with sess:
            feed_dict = {}
            feed_dict[self.encoder_inputs] = encoder_inputs
            feed_dict[self.encoder_seq_lens] = encoder_seq_lens
            feed_dict[self.decoder_targets] = decoder_targets
            feed_dict[self.decoder_seq_lens] = decoder_seq_lens
            #feea_dict[self.train_keep_prob] = keep_prob
            fetches = [self.loss]
            train_loss = self.train_op.run(fetches=fetches, feed_dict=feed_dict)
        return train_loss

    def predict_step(self, encoder_inputs, encoder_seq_lens, sess):
        feed_dict = {}
        feed_dict[self.encoder_inputs] = encoder_inputs
        feed_dict[self.encoder_seq_lens] = encoder_seq_lens
        fetches = [self.decoder_logits_inference, self.decoder_padding_mask]
        preds_logits_y, padding_mask = sess.run(fetches=fetches, feed_dict=feed_dict)
        return preds_logits_y, padding_mask

    @property
    def decoder_hidden_units(self):
        return self.decoder_cell.output_size

    @staticmethod
    def get_seq_padding_mask(seq_lens, seq_max_len):
        """ 根据序列长度得到真实样本的标记: 1.0---真实样本, 0.0---填充
        :param seq_lens: [batch_size], int
        :return: [batch_size, max_len]
        """
        batch_size = tf.shape(seq_lens)[0]
        #max_len = tf.reduce_max(seq_lens)
        indices = tf.reshape(tf.tile(tf.range(seq_max_len), [batch_size]), [batch_size, -1])
        mask = tf.less(indices, tf.cast(tf.expand_dims(seq_lens, axis=1), dtype=tf.int32))
        return tf.cast(mask, dtype=tf.float32)

    def get_sequence_loss(self, logits, targets, weights):
        return seq2seq.sequence_loss(logits=logits, targets=targets, weights=weights)


    def _init_placeholders(self):
        if self.encoder_vocab_size and self.encoder_vocab_size > 0:
            self.encoder_inputs = tf.placeholder(
                shape=(None, None),
                dtype=tf.int32,
                name='encoder_inputs',
            )
        else:
            self.encoder_inputs = tf.placeholder(
                shape=(None, None, self.encoder_input_dim),
                dtype=tf.float32,
                name='encoder_inputs',
            )

        self.encoder_seq_lens = tf.placeholder(
            shape=(None),
            dtype=tf.int32,
            name='encoder_seq_length',
        )

        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.float32,
            name='decoder_targets'
        )

        self.decoder_seq_lens = tf.placeholder(
            shape=(None),
            dtype=tf.int32,
            name='decoder_seq_length',
        )

    def _init_decoder_train_feeds(self):
        with tf.name_scope("decoder_train_feeds"):
            #sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets)
            #print(tf.unstack(tf.shape(self.decoder_targets)))
            #batch_size, decoder_max_seq_len = tf.unstack(value=tf.shape(self.decoder_targets))
            batch_size = tf.unstack(value=tf.shape(self.decoder_targets))[0]
            #print(batch_size)
            EOS_SLICE = tf.ones([batch_size, 1], dtype=tf.float32) * self.EOS
            PAD_SLICE = tf.ones([batch_size, 1], dtype=tf.float32) * self.PAD

            self.decoder_inputs_train = tf.concat([EOS_SLICE, self.decoder_targets], axis=-1)

            self.decoder_seq_lens_train = self.decoder_seq_lens + 1

            # [batch_size, max_seq_len]
            decoder_targets_train = tf.concat([self.decoder_targets, PAD_SLICE], axis=-1)
            #batch_size, max_seq_len = tf.unstack(tf.shape(decoder_targets_train))
            max_seq_len = tf.unstack(tf.shape(decoder_targets_train))[1]
            # 标记每个序列结束 EOS
            decoder_targets_eos_mask = tf.one_hot(self.decoder_seq_lens_train - 1,
                                                        max_seq_len,
                                                        on_value=float(self.EOS), off_value=0.,
                                                        dtype=tf.float32)
            # TODO: why???
            #decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])
            self.decoder_targets_train = tf.add(decoder_targets_train,
                                           decoder_targets_eos_mask)

            self.decoder_padding_mask = self.get_seq_padding_mask(self.decoder_seq_lens_train, max_seq_len)

    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope:
            initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))

            if self.encoder_vocab_size and self.encoder_vocab_size > 0:
                self.encoder_embedding_matrix = tf.get_variable(
                    name="encoder_embedding_matrix",
                    shape=[self.encoder_vocab_size, self.embedding_size],
                    initializer=initializer,
                    dtype=tf.float32
                )
                self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                    self.encoder_embedding_matrix, tf.to_int32(self.encoder_inputs)
                )
            else:
                self.encoder_embedding_matrix = None
                self.encoder_inputs_embedded = None

            self.decoder_embedding_matrix = tf.get_variable(
                name="decoder_embedding_matrix",
                shape=[self.decoder_vocab_size, self.embedding_size],
                initializer=initializer,
                dtype=tf.float32
            )
            self.decoder_inputs_train_embedded = tf.nn.embedding_lookup(
                self.decoder_embedding_matrix, tf.to_int32(self.decoder_inputs_train)
            )


    def _apply_optimizer(self):
        # 整理输出并计算loss
        #self.logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        #self.targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.loss = self.get_sequence_loss(logits=self.decoder_logits_train,
                                           targets=self.decoder_targets_train,
                                           weights=self.decoder_padding_mask)
        opt = tf.train.AdamOptimizer()
        train_op = opt.minimize(self.loss)
        self.train_op = train_op
        # add
        params = tf.trainable_variables()
        self.gradient_norms = []
        self.updates = []
        self.max_gradient_norm = 5

        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         self.max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

        saver = tf.train.Saver(tf.global_variables())
        self.saver = saver
        return train_op, saver


    def _simple_rnn_encoder(self):
        """ simple multi-layers rnn encoder
        """
        with tf.variable_scope("rnn_encoder") as scope:
            if self.encoder_vocab_size and self.encoder_vocab_size > 0:
                _inputs = self.encoder_inputs_embedded
            else:
                _inputs = self.encoder_inputs
            #print(len(tf.unstack(tf.shape(_inputs))))
            encoder_cell = rnn.MultiRNNCell([self.encoder_cell] * self.encoder_num_layers)

            if self.time_major:
                _inputs = tf.transpose(_inputs, [1, 0, 2])

            (self.encoder_outputs, self.encoder_state) = tf.nn.dynamic_rnn(
                cell=encoder_cell,
                inputs=_inputs,
                sequence_length=self.encoder_seq_lens,
                time_major=self.time_major,
                dtype=tf.float32
            )
            #print(self.encoder_state)

    def _bidirectional_rnn_encoder(self):
        """multi-layers bidirectional rnn encoder
        """
        with tf.variable_scope(None, default_name="bi_rnn_encoder") as scope:
            if self.encoder_vocab_size and self.encoder_vocab_size > 0:
                _inputs = self.encoder_inputs_embedded

            else:
                _inputs = self.encoder_inputs

            if self.time_major:
                _inputs = tf.transpose(_inputs, [1, 0, 2])
            print(isinstance(self.encoder_cell, rnn.LSTMCell))
            #print(tf.unstack(tf.shape(_inputs)))
            fw_outputs = None
            bw_outputs = None
            fw_state = None
            bw_state = None
            for i in range(self.encoder_num_layers):
                ((fw_outputs, bw_outputs),
                (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.encoder_cell,
                    cell_bw=self.encoder_cell,
                    inputs=_inputs,
                    sequence_length=self.encoder_seq_lens,
                    time_major=self.time_major,
                    dtype=tf.float32,
                    scope="bi_rnn_%d" % i   #需要指定scope，否则参数重复定义
                )
                _inputs = tf.concat((fw_outputs, bw_outputs), 2)


            self.encoder_outputs = tf.concat((fw_outputs, bw_outputs), 2,
                                         name="bi_encoder_output")
            if isinstance(fw_state, rnn.LSTMStateTuple):
                encoder_state_c = tf.concat((fw_state.c, bw_state.c), 1, name="bi_concat_c")
                encoder_state_h = tf.concat((fw_state.h, bw_state.h), 1, name="bi_concat_h")
                self.encoder_state = rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
            elif isinstance(fw_state, tf.Tensor):
                self.encoder_state = tf.concat((fw_state, bw_state), 1, name="bi_concat_state")


    def _rnn_decoder(self):
        """ rnn decoder with attention, which is either "bahdanau" or "luong"
            thought tf.contrib.seq2seq.dynamic_rnn_decoder
        :return:
        """
        with tf.variable_scope("rnn_decoder") as scope:
            output_fn = lambda x: tf.contrib.layers.linear(x, self.decoder_vocab_size,
                                                           scope="inference")
            # attention_states: size [batch_size, max_time, num_units]
            #attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
            # TODO: 需要确认是否可行?
            # 扩展encoder_state, 以匹配decoder的多层RNN的初始状态
            if len(self.encoder_state) < self.decoder_num_layers:
                _encoder_state = tuple([self.encoder_state] * self.decoder_num_layers)
            else:
                _encoder_state = self.encoder_state

            if self.attention_type is None:
                decoder_fn_train = seq2seq.simple_decoder_fn_train(
                    encoder_state=_encoder_state)
                decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=_encoder_state,
                    embeddings=self.decoder_embedding_matrix,
                    start_of_sequence_id=int(self.EOS),
                    end_of_sequence_id=int(self.EOS),
                    maximum_length=tf.reduce_max(self.encoder_seq_lens) + 100, # TODO: why???
                    num_decoder_symbols=self.decoder_vocab_size
                )
            else:
                print(self.attention_type)
                print(self.encoder_state)
                attention_states = self.encoder_outputs
                #attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
                (attention_keys,
                attention_values,
                attention_score_fn,
                attention_construct_fn) = seq2seq.prepare_attention(
                    attention_states=attention_states,
                    attention_option=self.attention_type,
                    num_units=self.decoder_hidden_units,
                )
                print(attention_keys)
                print(attention_values)

                decoder_fn_train = seq2seq.attention_decoder_fn_train(
                    encoder_state=_encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    name='attention_decoder',
                )
                decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=_encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    embeddings=self.decoder_embedding_matrix,
                    start_of_sequence_id=int(self.EOS),
                    end_of_sequence_id=int(self.EOS),
                    maximum_length=tf.reduce_max(self.encoder_seq_lens) + 100,
                    num_decoder_symbols=self.decoder_vocab_size,
                )
            # TODO: MultiRNNCell导致encoder_state与dynamic_rnn_decoder中的状态对不上
            # decoder 需要和encoder具有相同的层数
            _cell = rnn.MultiRNNCell([self.decoder_cell] * self.decoder_num_layers)
            print(isinstance(_cell, rnn.RNNCell))
            #_cell = self.decoder_cell
            # for train, embedding inputs
            _inputs = self.decoder_inputs_train_embedded
            (self.decoder_outputs_train,
             self.decoder_state_train,
             self.decoder_context_state_train) = seq2seq.dynamic_rnn_decoder(
                cell=_cell,
                decoder_fn=decoder_fn_train,
                inputs=_inputs,
                sequence_length=self.decoder_seq_lens_train,
                time_major=self.time_major,
                scope=scope
            )
            print(self.decoder_outputs_train)
            print(self.decoder_state_train)

            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            if self.time_major:
                self.decoder_logits_train = tf.transpose(self.decoder_logits_train, [1, 0, 2])
            #self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1,
            #                                          name='decoder_prediction_train')

            scope.reuse_variables()
            # for test, no decoder input
            (self.decoder_logits_inference,
             self.decoder_state_inference,
             self.decoder_context_state_inference) = seq2seq.dynamic_rnn_decoder(
                cell=_cell,
                decoder_fn=decoder_fn_inference,
                time_major=self.time_major,
                scope=scope
            )
            if self.time_major:
                self.decoder_logits_inference = tf.transpose(self.decoder_logits_inference, [1, 0, 2])
            #self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1,
            #                                              name='decoder_prediction_inference')
            print(self.decoder_outputs_train)
            print(self.decoder_logits_inference)