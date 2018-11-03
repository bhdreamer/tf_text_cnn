#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: data_feeder.py
# @brief: load and translate examples from file,
#         and feeds batch into a queue on a background thread
# @author: niezhipeng(@baidu.com)
# @Created on 2017/9/23
# *************************************************************************************
import numpy as np
import os
import sys
import codecs
import random
import tensorflow as tf
import threading
from datetime import datetime
import traceback
from common.text_box import TextProcessBox

_batches_per_group = 4
_pad = 0

class BasicFeeder(object):
    def __init__(self, dataset,
                 text_processor,
                 batch_size,
                 embedding=True,
                 b_shuffle=True):

        self._dataset = dataset
        self._processor = text_processor

        self._feature_dim = text_processor.feature_dim
        self._fea_dim_list = text_processor.feature_dim_list
        self._tar_dim_list = text_processor.target_dim_list
        self._tar_name_list = text_processor.target_name_list

        self.embedding = embedding
        self.is_training = len(self._dataset[0]) > 2
        self.b_shuffle = b_shuffle

        self._batch_size = int(batch_size)
        # split samples into batch with index
        sample_inds = list(range(len(self._dataset)))
        if self.is_training and self.b_shuffle:
            # sorted according to the sequence length of inputs
            sample_inds = sorted(sample_inds, key=lambda x: self._dataset[x][1])

        batch_num = int(np.ceil(1.0 * len(sample_inds) / batch_size))
        self._batch_num = batch_num
        self._batch_stack = [sample_inds[i*batch_size:(i+1)*batch_size]
                             for i in range(batch_num)]
        self._batch_idx = 0

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batch_num(self):
        return self._batch_num

    @property
    def sample_num(self):
        return len(self._dataset)

    def start_in_session(self, session):
        self._sess = session

        pass

    def stop(self):
        pass

    def next_batch(self):
        """ generate padding batch from self._batch_stack
            according self._batch_idx
        :return:
            NP.ARRAY()
        """
        self._batch_idx = self._batch_idx % self._batch_num
        if self._batch_idx <= 0 and self.b_shuffle:
            np.random.shuffle(self._batch_stack)
        sample_inds = self._batch_stack[self._batch_idx]
        if self.b_shuffle:
            np.random.shuffle(sample_inds)
        self._batch_idx += 1

        padding_batch = self.gen_padding_batch([self._dataset[i] for i in sample_inds])

        # calculate mask to indices valid targets
        label_mask = None
        return padding_batch

    # TODO:根据需要修改padding方式
    def gen_padding_batch(self, batch_data, padding_value=0.):
        """ 按照最长的序列，在序列末尾填充padding_value
        args:
            batch_data: LIST of example: [seq_features, seq_targets_1, seq_targets_2]
            padding_value:
        return:
            (np.array([B, T, dim], np.int32),
             np.array([B], np.int32),
             np.array([B, T, 1], np.int32))
        """
        def _seq_padding(x, max_len):
            padding_shape = [(0, max_len - len(x))]
            if len(x.shape) > 1:
                padding_shape += [(0, 0) for _ in range(len(x.shape) - 1)]
            return np.pad(x, tuple(padding_shape),
                          mode='constant', constant_values=padding_value)

        batch_size = len(batch_data)
        sample_inds = list(range(batch_size))
        if self.b_shuffle:
            np.random.shuffle(sample_inds)

        padding_data = list()

        # TODO: 根据是否需要embedding features来设置embedding
        inputs, seq_lens, targets_list = self.process(
            [batch_data[ind] for ind in sample_inds],
            embedding=self.embedding
        )

        max_seq_len = max(seq_lens)
        padding_data.append(np.stack(
            [_seq_padding(np.asarray(x), max_seq_len) for x in inputs]))

        padding_data.append(np.asarray(seq_lens))

        for targets in targets_list:
            #max_seq_len = max([len(x) for x in inputs])
            padding_data.append(np.stack(
                [_seq_padding(np.asarray(x), max_seq_len) for x in targets]))

        return tuple(padding_data)

    def process(self, raw_dataset, embedding=True):
        """ padding and normalize raw_dataset
        args:
            raw_dataset: LIST, including:
            [LIST of features, LIST of seq_tar1, LIST of seq_tar2, ...]
            embedding: whether to generate embedding feature
        return:
            inputs: LIST
            seq_lens: LIST
            targets_list: LIST of LIST
        """
        raw_data = [x for x in zip(*raw_dataset)]
        if len(raw_data) > 2:
            targets_list = raw_data[2:]
            is_training = True
        else:
            targets_list = []
            is_training = False

        seq_lens = raw_data[1]

        if embedding:
            inputs = [self._processor.embedding_seq_feature(x)
                      for x in raw_data[0]]
            inputs = self._processor.feature_normalize(inputs)
        else:
            inputs = raw_data[0]

        return (inputs, seq_lens, targets_list)

    def _init_tensors(self):
        # Create placeholders for inputs and targets.
        self._placeholders = [
            tf.placeholder(tf.int32, [None, None, len(self._fea_dim_list)], 'inputs'),
            tf.placeholder(tf.int32, [None], "sequence_lenghts")
        ]

        if self.is_training:
            self._placeholders += [
                tf.placeholder(tf.int32, [None, None], "targets_%s" % name)
                for dim, name in zip(self._tar_dim_list, self._tar_name_list)
            ]


class QueueFeeder(BasicFeeder):

    '''Feeds batches of data into a queue on background threads.'''
    def __init__(self, dataset, text_processor, batch_size, b_shuffle=True, n_threads=1):
        super(QueueFeeder, self).__init__(dataset,
                                          text_processor,
                                          batch_size,
                                          b_shuffle)
        #self._coord = coordinator

        #self._params = hparams
        #self._batch_size = int(hparams["batch_size"])
        self._global_epoch = 0 #训练集遍历轮数
        self._queue_capacity = 8 #队列容量
        self._b_shuffle = b_shuffle
        self._n_threads = int(n_threads)

        self._coord = tf.train.Coordinator()

        #self._dataset = self._text_processor.process(raw_data)



        # Create queue for buffering data, max_size = 8:
        self._queue = tf.FIFOQueue(self._queue_capacity,
                                   [x.dtype for x in self._placeholders],
                                   name='data_queue')
        self._queue_closed = False
        self._enqueue_op = self._queue.enqueue(self._placeholders)

    @property
    def global_epoch(self):
        return self._global_epoch

    def batch_iterator(self):
        self._offset = 0
        try:
            for _ in range(self.batch_num):
                yield self.next_batch()
        except KeyboardInterrupt:
            print("tensorflow queue reader collapse.")
        finally:
            self._coord.request_stop()
            self._coord.join(self._enqueue_threads)

    def next_batch(self):
        """ get tensor batch from queue
        :return:
            (
             Tensor([b, t, feature_dim]),
             Tensor([b, t, tar1_dim]),
             Tensor([b, t, tar2_dim]),
             ...
            )
        """
        #batch_data = self._session.run(self._queue.dequeue())
        batch_data = self._queue.dequeue()
        #batch_data[0] = tf.Print(batch_data[0], data=[self._queue.size()], message="queue size = ")

        for i in range(len(batch_data)):
            batch_data[i].set_shape(self._placeholders[i].shape)

        return tuple(batch_data)

    def start_in_session(self, session=None):
        self._session = session
        # multi threads to enqueue
        self._enqueue_threads = list()
        for _ in range(self._n_threads):
            t = threading.Thread(target=self.enqueue_thread_main,
                                 args=(self._session,))
            # t.daemon = True #Thread will close when parent quits
            self._enqueue_threads.append(t)

        self._coord.join(self._enqueue_threads)

        for t in self._enqueue_threads: t.start()

    def stop(self):
        self._queue_closed = True
        #print(self._queue_closed)
        self._coord.request_stop()
        # close queue and cancel pending enqueue ops
        # cancel_pending_enqueues=True, all pending enqueues will be canceled
        #self._session.run(self._queue.close(cancel_pending_enqueues=True))
        # wait for stop enqueue threading
        try:
            self._coord.join(self._enqueue_threads, stop_grace_period_secs=10)
        except RuntimeError as e:
            while not self._coord.wait_for_stop(10):
                pass

    def enqueue_thread_main(self, session):
        try:
            while not self._coord.should_stop():
                if self._queue_closed:
                    raise Exception
                    #break
                self._enqueue_next_group(session)
                #print(self._queue_closed)
        except Exception as e:
            self.stop()

    def enqueue_from_file(self, meta_file):
        pass

    def _enqueue_next_group(self, session):
        """ feed a group batches of data to queue
        args:
            session:
        return:
        """
        # Read a group of examples:
        n = self._batch_size
        #
        examples = [self._get_next_example() for i in range(n * _batches_per_group)]
        #
        #examples = self._text_processor.process(raw_examples)
        inds = list(range(len(examples)))
        batch_inds = list(range(_batches_per_group))
        if self._b_shuffle:
            # sort by sequence length of feature
            inds.sort(key=lambda x: len(examples[x][0]))
            random.shuffle(batch_inds)

        for ith in batch_inds:
            batch = [examples[i] for i in inds[ith*n : (ith+1)*n]]
            #
            # print(batch[0][0])
            if self._queue_closed:
                break
            #start = datetime.now()
            feed_dict = dict(zip(self._placeholders, self.gen_padding_batch(batch)))
            session.run(self._enqueue_op, feed_dict=feed_dict)
            #print("enqueue runtime: %.6f" % (datetime.now() - start).seconds)

    def _get_next_example(self):
        """ Loads a single sequence example (input, target_list)
            from disk recursively and translate to numerical data
        :return:
            (seq_feature, [seq_tar1, seq_tar2, ...])
        """
        if self._offset >= len(self._dataset):
            self._global_epoch += 1
            if self._b_shuffle:
                random.shuffle(self._dataset)
            self._offset = 0
        # raw_data
        raw_data = self._dataset[self._offset]
        self._offset += 1
        return (raw_data[0], raw_data[1])
        # feature text --> feature sequence
        #norm_seq_feature = self._text_processor.encode_seq_feature(raw_data[0],
        #                                                           use_normalize=True)
        # encode labels
        #seq_target_list = [self._text_processor.encode_seq_target(name, x)
        #    for name, x in zip(self._tar_name_list, raw_data[1])
        #]
        #return (norm_seq_feature, seq_target_list)


class DatasetFeeder(BasicFeeder):
    def __init__(self, dataset, text_processor, batch_size, b_shuffle=True, n_threads=1):
        super(DatasetFeeder, self).__init__(dataset,
                                          text_processor,
                                          batch_size,
                                          b_shuffle)
        self._dataset = tf.contrib.data.Dataset.from_tensor_slices()
        #self._dataset = self._dataset.map(self.process_func)
        self._sess = None
        pass

    def batch_iterator(self):
        #self._sess.run(self._iterator.initializer)
        for _ in range(self.batch_num):
            yield self.next_batch()

    def next_batch(self):

        pass

    def start_in_session(self, session):
        self._sess = session
        pass

    def stop(self):
        pass

def _round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder
