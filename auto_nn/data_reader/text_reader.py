#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *********************************************************************
# @file: text_reader_prosody.py
# @brief: 读取数据、文本数据的转化、batch生成, batch迭代器
# @author: niezhipeng(@baidu.com)
# @Created on 2017/9/23
# *********************************************************************
import codecs
import os
import h5py
import numpy as np
import concurrent
import time
from concurrent.futures import ProcessPoolExecutor
from common.text_box import TextProcessBox
from common.seq_normalizer import GaussNormalizer
from common.config_box import normalize_list
from data_reader.data_feeder import *

class TextReader(object):
    """
    Text Reader
        1. read data file: raw text or hdf5, support multi file
        2. process raw text_data
        3. return _dataset:
            LIST of (seq_features(np.array([seq_len, feature_num])),
                     seq_tar_1(np.array([seq_len, 1])),
                      ...
                    )
    """
    def __init__(self,
                 text_processor,
                 logger,
                 batch_size,
                 data_info,
                 dataset=None,
                 mean_var_path=None,
                 embedding=True,
                 is_training=True,
                 b_shuffle=False):

        assert(isinstance(text_processor, TextProcessBox))
        self._processor = text_processor
        self._logger = logger

        self._feature_dim = self._processor.feature_dim
        self._tar_name_list = self._processor.target_name_list
        self._tar_dim_list = self._processor.target_dim_list
        self._tar_num = len(self._tar_name_list)

        self._batch_size = batch_size
        self.embedding = embedding
        # True --- read target; False --- no target
        self.is_training = is_training
        self.b_shuffle = b_shuffle

        self._mean_var_path = self._processor.mean_var_path

        if isinstance(data_info, (list, tuple)):
            self.data_info_list = data_info
        else:
            self.data_info_list = [data_info]

        if dataset is not None:
            self._dataset = dataset
            self._data_size = len(dataset)
        else:
            self._dataset = []
            self._data_size = 0
            # read data into cache
            self._dataset = self.read_into_cache(self.data_info_list, is_training)
        print(id(self._dataset))
        print(self._data_size)

        # calc normalization params
        if self.is_training and self._processor.normalizer_updating:
            # print("generator mean_var, file is %s" % self._mean_var_path)
            # self._processor.calc_normalize_params(self._dataset)
            pass

        self.feeder = BasicFeeder(self._dataset,
                                  self._processor,
                                  self._batch_size,
                                  self.embedding,
                                  b_shuffle=b_shuffle,)
                                  #n_threads=2)

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def target_dim_list(self):
        return self._tar_dim_list

    @property
    def target_name_list(self):
        return self._tar_name_list

    @property
    def batch_num(self):
        return self.feeder.batch_num

    @property
    def sample_num(self):
        return self._data_size

    @property
    def batch_size(self):
        return self._batch_size

    def batch_iterator(self):
        """ get iterable padding batch data, according self.batch_num
        :return:
            LIST: iterable padding_batch
        """
        self._batch_idx = 0
        for _ in range(self.batch_num):
            yield self.next_batch()

    def next_batch(self):
        return self.feeder.next_batch()

    def start_in_session(self, session):
        self.feeder.start_in_session(session)

    def stop(self):
        self.feeder.stop()

    def read_into_cache(self, data_info_list, is_training=True):
        """
        read dataset from files, support "hdf5"
        args:
            data_info_list: LIST of dict "data_info"
        :return:
        """
        inputs = []
        targets_list = [[] for _ in range(self._tar_num)]
        self._dataset = []
        self._data_size = 0

        for data_info in data_info_list:
            if data_info is None:
                print("data info is None")
                continue
            if data_info["encoding"] == "hdf5":
                hdf5_path = os.path.join(data_info["base_dir"], data_info["hdf5_name"])
                _inputs, _targets_list = self.read_data_hdf5(hdf5_path, is_training)
            else:
                _inputs, _targets_list = self.read(data_info, is_training)

            inputs.extend(_inputs)
            for i, targets in enumerate(_targets_list):
                targets_list[i].extend(targets)
            self._data_size += len(_inputs)

        # inputs = self._feature_normalize(inputs)
        seq_lens = [len(x) for x in inputs]
        if is_training:
            self._dataset = list(zip(*([inputs, seq_lens] + targets_list)))
        else:
            self._dataset = list(zip(*[inputs, seq_lens]))
            #print(inputs)
        return self._dataset

    def read(self, data_info, is_training):
        """
        read text data from multi files
            if is_training, norm_data could be updated
        args:
            file_list: LIST of file path
            is_training:
        return:
            LIST of np.array([seq_len, fea_dim], np.float32)
            [LIST of np.array([seq_len, dim], np.float32)] * target_num
        """
        encoding = data_info["encoding"]
        base_dir = os.path.abspath(data_info["base_dir"])
        #raw_inputs, raw_targets_list = self.read_text_rawdata(text_data_fn, tar_num)
        fea_path = os.path.join(base_dir, data_info["fea_data_name"])
        _inputs = self.read_feature_file(fea_path, encoding)
        _targets_list = []
        if is_training:
            tar_fn_list = normalize_list(data_info["tar_data_names"])
            for fn in tar_fn_list:
                tar_path = os.path.join(base_dir, fn)
                _targets_list.append(self.read_target_file(tar_path, encoding))

        _inputs, _targets_list = self._processor.process_sparse(
            _inputs, _targets_list, is_training
        )
        if is_training:
            # print("_inputs: ", len(_inputs))
            header = {
                "data_size": len(_inputs),
                # "feature_dim": self._feature_dim,
                "target_name_list": self._tar_name_list,
                # "target_dim_list": self._tar_dim_list
            }
            hdf5_path = os.path.join(data_info["base_dir"], data_info["hdf5_name"])
            self.write_data_hdf5(hdf5_path, header, _inputs, _targets_list)

        return _inputs, _targets_list

    # TODO: 根据文本数据格式修改读取代码
    @staticmethod
    def read_feature_file(feature_fn, encoding="gb18030"):
        """ 解析空行分隔的文本特征数据,每个序列元素特征一行
        :param feature_fn:
        :param encoding:
        :return:
        """
        raw_inputs = []
        with codecs.open(feature_fn, 'r', encoding=encoding) as fp:
            x_seq = []
            #orgline = fp.readline()
            line = fp.readline()
            while line != "":
                line = line.strip()
                if line == "" or line.startswith("END"):
                    if len(x_seq) > 0:
                        raw_inputs.append(x_seq)
                    x_seq = []
                    #orgline = fp.readline()
                    line = fp.readline()
                    continue
                #tmplist = line.split()
                tmplist = [x.split("@")[0].strip() for x in line.split()]
                x_seq.append(tmplist)
                line = fp.readline()
        return raw_inputs

    # TODO: 根据文本数据格式修改读取代码
    @staticmethod
    def read_target_file(target_fn, encoding="gb18030"):
        """ 解析空行隔开的文本label数据，每个序列一行，元素label 空格隔开
        :param target_fn:
        :param encoding:
        :return:
        """
        raw_targets = []
        with codecs.open(target_fn, "r", encoding) as fp:
            for line in fp:
                line = line.strip()
                if line != "":
                    labels = [x.split("@")[0].strip() for x in line.split()]
                    raw_targets.append(labels)
        return raw_targets

    # TODO: 根据label的存储格式修改
    @staticmethod
    def write_label_file(pred_labels, pred_fn, encoding="gb18030", is_merge=False):
        with codecs.open(pred_fn, "w", encoding) as fp:
            for seq_label in pred_labels:
                outline = " ".join(["%s@pred" % x for x in seq_label])
                fp.write(outline + "\n")
        return pred_fn

    @staticmethod
    def merge_pred_file(feature_fn, pred_labels, out_fn, encoding="gb18030"):
        out_fp = codecs.open(out_fn, "w", encoding)
        i = 0
        with codecs.open(feature_fn, "r", encoding) as fea_fp:
            features = []
            line = fea_fp.readline()
            while line != "":
                line = line.strip()
                if line == "" or line.startswith("END"):
                    seq_label = pred_labels[i]
                    if len(seq_label) != len(features):
                        pass
                    outlines = ["%s %s@pred\n" % (fea, label)
                                for fea,label in zip(features, seq_label)]
                    out_fp.writelines(outlines)
                    out_fp.write("\n")
                    i += 1
                    features = []
                else:
                    features.append(line)
                line = fea_fp.readline()
        out_fp.close()

    @staticmethod
    def get_batch_seq_lens(batch_data, padding_value=0.):
        """
        calculate sequence lenghts:
            [batch_size, T, dim] --> [batch_size]
            count the value greater than padding_value in sequence
        args:
            inputs: np.array
            padding_value: default(0.)
        return:
            np.array([batch_size], tf.int64)
        """
        if len(batch_data.shape) < 3:
            batch_max = batch_data
        else:
            batch_max = np.max(np.abs(batch_data), axis=-1)

        mask = np.greater(batch_max, padding_value).astype(int)
        return np.sum(mask, axis=-1)

    # TODO: 根据文本数据格式修改读取代码
    @staticmethod
    def read_text_rawdata(text_file, target_num, encoding='gbk'):
        """ 解析空行分隔的训练文本数据,每个序列元素一行，label位于行尾
        args:
            file_list: LIST of file path
            target_num: INT,
            encoding: file encoding, default("gbk")
        :return:
            input_rawdata: LIST of sequence of features
            target_rawdata_list: [[[tar1, tar2, ...], ...], ...]
                if target_num <= 0, return  [[[], ...], ...]
        """
        raw_inputs = []
        # 三个指向同一对象的引用
        # raw_targets_list = [[]] * target_num
        raw_targets_list = [[] for _ in range(target_num)]
        x_seq = []
        y_seq = []

        with codecs.open(text_file, 'r', encoding=encoding) as fp:
            orgline = fp.readline()
            line = fp.readline()
            while line != "":
                line = line.strip()
                if line == "" or line.startswith("END"):
                    raw_inputs.append(x_seq)
                    for i, seq in enumerate(list(zip(*y_seq))):
                        raw_targets_list[i].append(seq)
                    x_seq = []
                    y_seq = []
                    orgline = fp.readline()
                    line = fp.readline()
                    continue
                #tmplist = line.split()
                tmplist = [x.split("@")[0].strip() for x in line.split()]
                x_seq.append(tmplist[: len(tmplist) - target_num])
                y_seq.append(tmplist[-target_num:] if target_num > 0 else [])
                line = fp.readline()

        return raw_inputs, raw_targets_list

    # TODO：根据事先定义的hdf5的header修改
    @staticmethod
    def read_hdf5_header(hdf5_file):
        header = dict()
        with h5py.File(hdf5_file, 'r') as data:
            header["data_size"] = data["data_size"].value
            print("hdf5: data_size =", header["data_size"])
            # header["feature_dim"] = data["feature_dim"].value
            # print("hdf5: feature_dim =" + str(header["feature_dim"]))
            header["target_name_list"] = list(data["target_name_list"].value)
            print("hdf5: target_name_list = ", header["target_name_list"])
            # header["target_dim_list"] = list(data["target_dim_list"].value)
            # print("hdf5: target_dim_list = ", header["target_dim_list"])
        return header

    # TODO：根据事先定义的数据标签修改
    @staticmethod
    def read_data_hdf5(hdf5_file, is_training):
        """
        read data from hdf5 file
        args:
            hdf5_file: should contain at least keys:
                "data_size"
                "feature_dim"
                "target_name_list"
                "target_dim_list"
                "features_<sample_idx>",
                "targets_<sample_idx>.<task_name>"
            is_training: BOOL
                True --- read targets according tar_name_list
        :return:
            inputs: LIST of np.array([seq_len, feature_dim], np.float32)
            targets_list:
                [LIST of np.array([seq_len, tar_dim], np.float32)] * target_num;
                if is_training = False, return []
        """
        header = TextReader.read_hdf5_header(hdf5_file)
        tar_name_list = header["target_name_list"] if is_training else []
        inputs = []
        targets_list = [[] for _ in range(len(tar_name_list))]

        with h5py.File(hdf5_file, 'r') as seq_data:
            for idx in range(header["data_size"]):
                inputs.append(np.asarray(seq_data["features_%d" % idx].value, dtype=np.int32))
                for i, name in enumerate(tar_name_list):
                    targets_list[i].append(np.asarray(
                        seq_data["targets_%d.%s" % (idx, name)].value, dtype=np.int32))
        return inputs, targets_list

    # TODO:根据需要设定hdf5文本格式修改标签
    @staticmethod
    def write_data_hdf5(hdf5_file, header, inputs, targets_list):
        """ write inputs and targets_list to hdf5 file
        :param hdf5_file:
        :param header:
        :param inputs:
        :param targets_list:
        :return:
        """
        with h5py.File(hdf5_file, 'w') as fp:
            fp.create_dataset("data_size", data=header["data_size"])
            # file.create_dataset("feature_dim", data=header["feature_dim"])
            header["target_name_list"] = [np.string_(x) for x in header["target_name_list"]]
            fp.create_dataset("target_name_list", data=header["target_name_list"])
            # file.create_dataset("target_dim_list", data=header["target_dim_list"])
            for i in range(header["data_size"]):
                fp.create_dataset("features_%d" % i, data=inputs[i])
                for j,name in enumerate(header["target_name_list"]):
                    fp.create_dataset("targets_%d.%s" % (i, name),
                                        data=targets_list[j][i])
        return hdf5_file

    def get_label_mask(self, batch_data):

        label_mask = []
        return label_mask

    def _feature_normalize(self, seq_features):
        """
        normalize the sequence features.
        if mean or variance is invalid, generate the new one
        args:
            seq_features: LIST of sequence, each one is list of np.array
        :return:
            LIST of np.array, each one means the feature tensor of a sequence
        """
        if self.is_training and self._processor.normalizer_updating:
            self._logger.info("generate mean_var, file is %s" % self._mean_var_path)
            start_time = time.time()
            self._processor.calc_normalize_params(self._dataset)
            print("\tused time = %.5f" % (time.time() - start_time))
            #self._normalizer.gen_mean_var(seq_features, self._mean_var_path)
        return self._processor.feature_normalize(seq_features)

    #TODO:
    def _load_into_cache(self, read_hander, file_list, n_threads):
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            fs = list()

            for ith_batch in range(self.batch_num):
                fs.append(executor.submit(
                    read_hander, file_list))
            try:
                for future in fs:
                    #self._padding_batch_list.extend(future.result())
                    pass
            finally:
                for future in fs:
                    future.cancel()
        return True
