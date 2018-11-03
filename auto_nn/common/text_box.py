#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *********************************************************************
# @file: text_box.py
# @brief: 转换字典，文本转特征向量，归一化，读取文件列表，
# @author: niezhipeng(@baidu.com)
# @Created on 2017/9/23
# *********************************************************************

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import struct
import numpy as np
import os
import math
from common.seq_normalizer import GaussNormalizer

########################### TransDict ###########################
class TransDict(object):
    """
    TransDict: 特征和label转化字典
    """
    def __init__(self, keys=None, ids=None, embedding_values=None, default_key=None):
        """
        args:
            keys: LIST
            ids: LIST
            embedding_values: LIST
            default_key: 默认项
        """
        self._keys = keys or list()

        self._ids = ids or list()
        if len(self._keys) > 0 and len(self._ids) < 1:
            self._ids = list(range(len(self._keys)))

        self._embedding_values = embedding_values
        self._num = len(self._keys)

        self._default_key = default_key

        if embedding_values is not None:
            self._dim = len(embedding_values[0])
        else:
            self._dim = len(self._keys)

    @property
    def dim(self):
        """
        :return: count of entries
        """
        if self._dim > 0:
            return self._dim

        if self._embedding_values is not None:
            self._dim = len(self._embedding_values[0])
        else:
            self._dim = len(self._keys)
        return self._dim

    @property
    def default_key(self):
        """
        :return: default key
        """
        return self._default_key


    def has_key(self, key):
        """ """
        return key in self._keys

    def keys(self):
        """
        :return: LIST of key of translate dict
        """
        return self._keys

    def append(self, key, id=None, embedding_value=None):
        """ append key, (id, embeding_value) into LIST
        args:
            key:
            id: if None, key_id + 1
            embedding_value: if None, embedding_value = None
        return:
        """
        self._keys.append(key)
        if id is None:
            self._ids.append(self._num)
        else:
            self._ids.append(int(id))
        self._num += 1

        if embedding_value:
            try:
                self._embedding_values.append(embedding_value)
            except Exception:
                self._embedding_values = [embedding_value]

    def get_id(self, key):
        """
        return id with key, if key not exist, return id with default_key
        """
        for _key, _id in zip(self._keys, self._ids):
            if _key == key:
                return int(_id)
        return self.get_id(self._default_key)

    def embedding_value(self, id):
        """ return embedding_value with id,
            if _embedding_values is None, return one_hot
        args:
            id: INT
        return:
        """
        if self._embedding_values:
            return self._embedding_values[id]
        else:
            return self.one_hot(id)

    def value(self, key, embedding=True):
        """ find the value with given key,
            if default not None, return default
        args:
            key:
            embedding: whether to return embedding value
        return:
            value or None
        """
        id = self.get_id(key)
        if embedding:
            return self.embedding_value(id)
        else:
            return id

    def get_key(self, id):
        """ return key with id, if not exist, return default_key
        args:
            id: INT
        :return:
        """
        for _key, _id in zip(self._keys, self._ids):
            if _id == id:
                return _key
        return self._default_key

    def reverse(self):
        """ {key:value} --> {value:key}
            value should be one-to-one with key
            value of default_key --> default_key
        :return:
            reversed TransDict
        """
        default_key = self.value(self._default_key)
        return TransDict(self._ids, self._keys,
                         self._embedding_values, default_key)

    def one_hot(self, idx, on_value=1.0, off_value=0.):
        """ generate a one-hot vector according to ID of key
        :param key: STR
        :param on_value: default(1.0)
        :param off_value: default(0.0)
        :return:
        """
        vec = np.ones([self._dim], dtype=np.float32) * off_value
        try:
            vec[idx] = 1.0 * on_value
        except (TypeError, IndexError) as e:
            print(e)
            pass
        return vec


class ZhuyinInfo(object):
    "zhuyin info for each word"
    def __init__(self, char, id, zylist):
        self._char = char
        self._id = id
        # LIST of [split_zhuyin, zhuyin]
        self._zylist = zylist

    @property
    def py_num(self):
        return len(self._zylist)

    @property
    def id(self):
        return self._id

    @property
    def dyz(self):
        return self._char

    @property
    def zhuyin_list(self):
        return self._zylist

    def get_zhuyin(self, split_zy):
        """ get zhuyin according to the splited py
            eg. "xing2" --> "x ing 2"
            if not found, return None
        args:
            split_py: STR,
        :return:
            STR
        """
        for x, y in zip(*self._zylist):
            if x == split_zy:
                return y
        return None

    def get_split_zhuyin(self, zhuyin):
        """ get splited zhuyin according to the merged py
            eg. "xing2" --> ("x ing 2")
            if not found, return None
        args:
            merge_py: STR
        :return:
            STR
        """
        for x, y in zip(*self._zylist):
            if y == zhuyin:
                return x
        return None


class DyzPyDict(object):
    def __init__(self, dyz_list=None):
        self._items = dyz_list if isinstance(dyz_list, list) else []

    @property
    def dyz_num(self):
        return len(self._items)

    def is_dyz(self, char):
        return char in [x.char for x in self._items]

    def append(self, dyz_info):
        assert isinstance(dyz_info, ZhuyinInfo)
        self._items.append(dyz_info)

    def get_zhuyin_list(self, char):
        """
        :param char: dyz char
        :return:
            LIST of [split_py, zhuyin]
        """
        for x in self._items:
            if x.char == char:
                return x.zhuyin_list
        return None

    def dyzid_trans_dict(self):
        dyzid_dict = TransDict()
        for x in self._items:
            dyzid_dict.append(x.dyz, x.id)
        return dyzid_dict


class TextProcessBox(object):
    """
    TextProcessBox
    """
    def __init__(self,
                 trans_dict_paths,
                 feature_name_list,
                 target_name_list,
                 mean_var_path):

        self._fea_trans_dict = {}
        self._fea_name_list = []
        self._tar_trans_dict = {}
        self._tar_name_list = []
        self._dyz_zy_dict = None
        self._feature_dim = 0

        self._init_trans_dicts(trans_dict_paths,
                               feature_name_list,
                               target_name_list)

        self.mean_var_path = mean_var_path
        self.normalizer = GaussNormalizer(mean_var_path)
        self._norm_updating = self.normalizer._updating

        self._global_max_len = 0

    @property
    def feature_dim_list(self):
        return self._fea_dim_list

    @property
    def feature_dim(self):
        return self._feature_dim

    @feature_dim.setter
    def feature_dim(self, feature_dim):
        assert(feature_dim >= self._feature_dim)
        self._feature_dim = feature_dim

    @property
    def target_num(self):
        return len(self._tar_name_list)

    @property
    def target_dim_list(self):
        """
        Returns:
            list of target_dim to initialize model
        """
        return [self._tar_trans_dict[key].dim for key in self._tar_name_list]

    @property
    def target_name_list(self):
        return self._tar_name_list

    @property
    def target_trans_dict(self):
        return self._tar_trans_dict

    @property
    def reverse_target_trans_dict(self):
        """
        Returns:
            DICT of rev_target_dict to make prediction result
        """
        # return self._tar_trans_dict[name].reverse()
        rev_tar_dicts = {}
        for name, tdict in self._tar_trans_dict.items():
            rev_tar_dicts[name] = tdict.reverse()
        return rev_tar_dicts

    @property
    def normalizer_updating(self):
        return self._norm_updating

    @staticmethod
    def read_dyz_zy_dict(dyz_fn):
        """
        read dyz zhuyin dict
        args:
            dyz_fn:
        :return:
            DyzPyDict:
        """
        with codecs.open(dyz_fn, "r", encoding="gb18030") as dyzfile:
            # print("[DEBUG] openning file is ")
            dyz_cnt = int(dyzfile.readline().strip())
            dyz_py_dict = DyzPyDict()
            for line in dyzfile:
                tmplist = line.strip().split()
                if len(tmplist) < 4 or tmplist[0].startswith("#"):
                    continue
                zy_list = []
                for x in tmplist[3:]:
                    pys = x.strip().split("|")
                    zy_list.append((" ".join(pys[:-1]), pys[-1]))
                dyz_py_dict.append(ZhuyinInfo(tmplist[1], int(tmplist[0]), zy_list))
        return dyz_py_dict

    @staticmethod
    def read_dict_file(file_path, default_key=None, b_reverse=False):
        """
        #brief: 读取文本特征词典
        #file_path: 字典文件
                    format: line 1 : num_entry
                            line 2 : default_key
                            line >2: id entry
        #default: 默认项
        #b_reverse: 反向读取, dafault(False)
        #return:
            tdict: TransDict对象
        """
        tdict = TransDict()
        # print("%s" % filename)
        with codecs.open(file_path, 'r', encoding='gb18030') as dfile:
            # print("[DEBUG] openning file is %s" % dfile.name)
            # tmplist = dfile.readline().strip().split()
            num = int(dfile.readline().strip())
            default = dfile.readline().strip()
            # 读取默认值
            default_key = default_key or default
            for i, line in enumerate(dfile.readlines()):
                dlist = line.strip().split()
                # print(dlist)
                if i >= num:
                    break
                if b_reverse:
                    # print(dlist)
                    tdict.append(int(dlist[0]), dlist[1])
                else:
                    tdict.append(dlist[1], int(dlist[0]))
            tdict._default_key = tdict.value(default_key) if b_reverse else default_key
            #tdict.dim
        return tdict

    @staticmethod
    def read_wordvec(vec_fn, is_compress=False):
        """ read word embedding vector into np.array
        args:
            vec_fn:
            is_compress: True, need decompress wordvec
        return:
            np.array([vec_num, vec_dim], np.float32)
        """
        vec_data = None
        # load dict for char vec
        with codecs.open(vec_fn, 'rb') as vec_fp:
            # 按unsigned int读取两个整数
            if is_compress:
                vec_num, vec_dim, iscompress, factor = struct.unpack('IIII', vec_fp.read(16))
                vec_data = np.fromfile(vec_fp, dtype=np.int16).reshape([-1, vec_dim])
                vec_data = vec_data / factor
            else:
                vec_num, vec_dim = struct.unpack('II', vec_fp.read(8))
                vec_data = np.fromfile(vec_fp, dtype=np.float32).reshape([-1, vec_dim])
            (actual_vec_num, actual_vec_dim) = vec_data.shape
            if vec_num != actual_vec_num or vec_dim != actual_vec_dim:
                print("[ERROR] wordvec: unexpected shape =", vec_data.shape)
        return vec_data

    @staticmethod
    def read_word_dict(word_fn, b_reverse=False):
        tdict = TransDict()
        # print("%s" % filename)
        with codecs.open(word_fn, 'r', encoding='gb18030') as dfile:
            # print("[DEBUG] openning file is %s" % dfile.name)
            # tmplist = dfile.readline().strip().split()
            num = int(dfile.readline().strip())
            default = dfile.readline().strip()
            # 读取默认值
            default_key = default
            for i, line in enumerate(dfile.readlines()):
                dlist = line.strip().split()
                # print(dlist)
                if i >= num:
                    break
                if b_reverse:
                    # print(dlist)
                    tdict.append(int(dlist[0]), dlist[1])
                else:
                    tdict.append(dlist[1], int(dlist[0]) - 1)
            tdict._default_key = tdict.value(default_key) if b_reverse else default_key
            #tdict.dim
        return tdict

    @staticmethod
    def read_wordvec_dict(dict_fn, vec_fn, is_compress=False):
        """
        #brief: 读取文本特征字典与编码文件，生成以文本特征作为关键字，编码为项的字典
        #param[in1]: dict_fn --- 文本特征字典文件(第一行为项个数)
        #param[in2]: vec_fn  --- 编码向量文件(前两个数据为项个数和项维数)
        #return: dic --- class_dict对象
        """
        tdict = {}
        iscompress = 0
        factor = 1.
        # load dict for char vec
        vec_data = TextProcessBox.read_wordvec(vec_fn, is_compress=is_compress)
        (vec_num, vec_dim) = vec_data.shape
        char_list = list()
        vec_list = list()
        # load char vec model
        with codecs.open(dict_fn, 'r', 'gb18030') as fp_dict:
            char_num = int(fp_dict.readline().strip())
            default_value = fp_dict.readline().strip()
            assert char_num == vec_num
            for line in fp_dict:
                index, char = line.strip().split()
                char_list.append(char)
                vec_list.append(vec_data[int(index) - 1, ...])
        # 增加两个默认字符
        char_list.append(u"</s>")
        vec_list.append(vec_data[-1, ...])
        char_list.append(u"<NULL>")
        vec_list.append(np.zeros(vec_dim, dtype=np.float32))
        return char_list, vec_list, vec_dim

    @staticmethod
    def one_hot(indice, depth, on_value=1., off_value=0.):
        """ id --> one-hot vector
        :param indices:
        :param depth:
        :param on_value:
        :param off_value:
        :return:
        """
        onehot_vec = [off_value] * depth
        try:
            onehot_vec[indice] = on_value
        except IndexError:
            pass
        return onehot_vec
    '''
    def process(self, raw_dataset):
        """ translate text data to numeric feature or target data,
            and normalize the feature data
        args:
            raw_dataset --- LIST of [seq_feature, LIST of seq_target]
                            or LIST of [seq_feature]
        :return:
            shape as raw_dataset, but the entry is np.array
            eg. LIST of [np.array([seq_len, feature_dim]), [np.array([seq_len, tar_dim])]]
        """
        is_training = len(raw_dataset[0]) > 1
        inputs = list()
        targets = list()
        for example in raw_dataset:
            seq_feature = np.asarray(self.encode_seq_feature(example[0]),
                                     dtype=np.float32)
            # seq_feature = self._feature_normalize(seq_feature)
            inputs.append(seq_feature)
            # encode multi-target
            if is_training:
                seq_target_list = [np.asarray(self.encode_seq_target(name, x), dtype=np.int32)
                    for name, x in zip(self._tar_name_list, example[1])]
                targets.append(seq_target_list)
        # normalize features
        norm_inputs = self._feature_normalize(inputs)
        return list(zip(*[norm_inputs, targets])) if is_training else norm_inputs
    '''

    def process(self, raw_inputs, raw_targets_list, is_training):
        """
        translate text data to feature_vec or target ids,
        and normalize the feature data
        args:
            raw_inputs: LIST of feature sequence
            raw_targets_list: [LIST of target sequence] * target_num
            is_training: BOOL
        return:
            inputs: LIST of np.array([seq_len, fea_dim], np.float32)
            targets_list: [LIST of np.array([seq_len], np.int32)] * target_num
        """
        if is_training:
            # target_num
            tar_name_list = self._tar_name_list
            assert len(raw_targets_list) == len(tar_name_list)
            # sample_num
            print("%d, %d" % (len(raw_inputs), len(raw_targets_list[0])))
            assert len(raw_inputs) == len(raw_targets_list[0])
        else:
            raw_targets_list = []
            tar_name_list = []

        inputs = []
        for seq_id in range(len(raw_inputs)):
            seq_fea_vec = np.asarray(self.encode_seq_feature(raw_inputs[seq_id]),
                                   dtype=np.float32)
            #seq_fea_vec = self._feature_normalize(seq_fea_vec)
            inputs.append(seq_fea_vec)

        targets_list = list()
        for i, (name, targets) in enumerate(zip(tar_name_list, raw_targets_list)):
            encoded_tars = [np.asarray(self.encode_seq_target(name, x),
                dtype=np.int32) for x in targets]
            targets_list.append(encoded_tars)
        #norm_inputs = self._feature_normalize(inputs)
        return inputs, targets_list

    def process_sparse(self, raw_inputs, raw_targets_list, is_training):
        """
        translate text data to feature ids or target ids,
        args:
            raw_inputs: LIST of feature sequence
            raw_targets_list: [LIST of target sequence] * target_num
            is_training: BOOL
        return:
            inputs: LIST of np.array([seq_len], np.int32)
            targets_list: [LIST of np.array([seq_len], np.int32)] * target_num
        """
        if is_training:
            # target_num
            tar_name_list = self._tar_name_list
            assert len(raw_targets_list) == len(tar_name_list)
            # sample_num
            print("%d, %d" % (len(raw_inputs), len(raw_targets_list[0])))
            assert len(raw_inputs) == len(raw_targets_list[0])
        else:
            raw_targets_list = []
            tar_name_list = []

        inputs = []
        for seq_id in range(len(raw_inputs)):
            seq_fea_ids = np.asarray(
                self.encode_seq_feature(raw_inputs[seq_id], embedding=False),
                dtype=np.int32)
            #seq_fea_vec = self._feature_normalize(seq_fea_vec)
            inputs.append(seq_fea_ids)

        targets_list = list()
        for i, (name, targets) in enumerate(zip(tar_name_list, raw_targets_list)):
            encoded_tars = [np.asarray(self.encode_seq_target(name, x, embedding=False),
                dtype=np.int32) for x in targets]
            targets_list.append(encoded_tars)
        #norm_inputs = self._feature_normalize(inputs)
        return inputs, targets_list

    def feature_normalize(self, seq_features):
        if not os.path.exists(self.mean_var_path):
            return seq_features
        return self.normalizer.batch_normalize(seq_features)

    def calc_normalize_params(self, raw_dataset):
        """ calculate normalize params by iterator
        args:
            raw_dataset:  LIST of sequence data
        :return:
        """
        def feature_iterator():
            for sample in raw_dataset:
                #print(sample[0])
                yield self.embedding_seq_feature(sample[0])

        self.normalizer.gen_mean_var(feature_iterator(), self.mean_var_path)
        self._norm_updating = False

    def embedding_seq_feature(self, seq_feature):
        """ ids feature --> embedding feature for sequence
        args:
            seq_feature: LIST or np.array(), sequence feature
        return:
            np.array([seq_len, feature_dim], np.float32)
        """
        if isinstance(seq_feature, np.ndarray):
            seq_feature = seq_feature.tolist()
        features = [self.gen_feature_vec(x) for x in seq_feature]
        return np.asarray(features, dtype=np.float32)

    def encode_seq_feature(self, raw_seq_feature, embedding=True):
        """ seq_text_feature --> seq_feature (ids/vec)
        args:
            raw_seq_feature: LIST of [STR]
            embedding: whether to generate embedding feature
        :return:
            LIST
        """
        return [self.encode_feature(x, embedding) for x in raw_seq_feature]

    def encode_seq_target(self, target_name, raw_seq_target, embedding=False):
        return [self._tar_trans_dict[target_name].value(x, embedding=embedding)
                for x in raw_seq_target]
        #return [self._gen_target_vec(target_name, x) for x in raw_seq_target]

    def decode_seq_target(self, target_name, seq_values):
        return [self._tar_trans_dict[target_name].get_key(seq_values[i])
                for i in range(len(seq_values))]

    def encode_feature(self, raw_feature, embedding=False):
        """ encode text feature to ids or vector
        args:
            raw_feature: LIST of text feature
            embedding: BOOL
        return:
            if embedding=False, return LIST of feature_id
            else, return np.array([feature_dim], np.float32)
        """
        values = [self._fea_trans_dict[name].value(x, embedding=embedding)
                for name, x in zip(self._fea_name_list, raw_feature)]
        if embedding:
            return np.concatenate(values, axis=-1)
        else:
            return values

    def encode_label(self, name, label):
        return self._tar_trans_dict[name].value(label, embedding=False)

    def decode_label(self, name, value):
        return self._tar_trans_dict[name].get_key(value)

    def decode_merge_label(self, values):
        """ decode merged label for multi task
        args:
            values: LIST, [task_1_value, task_2_value, ...]
        :return:
            LIST, [task_1_label, task_2_label, ...]
        """
        return [self._tar_trans_dict[name].get_key(x)
                for name, x in zip(self._tar_name_list, values)]

    def gen_feature_vec(self, feature_ids, on_value=1., off_value=0.):
        """ list of feature symbols --> vector
            feature should be matched with _fea_name_list
        args:
            features: LIST of feature symbols
        return:
            np.array(feature_dim, np.float32)
        """
        #assert len(feature) == len(self._fea_name_list)
        # fea_vec = np.empty(shape=self.feature_dim)
        values = list()
        for key, id in zip(self._fea_name_list, feature_ids):
            values.append(self._fea_trans_dict[key].embedding_value(int(id)))
        return np.concatenate(values, axis=-1)

    def _gen_dyz_feature_vec(self, feature,
                             dyzid_trans_dict=None,
                             on_value=1.,
                             off_value=0.):
        """
        add one-hot dyzid info in the end of feature vector
        supported to
            1. feature[-1] is dyzid;
            2. feature[-1] not is dyzid, but dyzid_trans_dict be given.
        feture_dim should be the prior as a attribute of TextProcessBox
        args:
            feature: LIST of feature symbols
            dyzid_trans_dict: TransDict
            on_value: default(1.0) for one-hot
            off_value: default(0.0) for padding
        :return:
            np.array(feature_dim, np.float32)
        """
        if len(feature) > 0 and feature[-1].isdigit():
            dyz_id = int(feature[-1])
            feature = feature[0:-1]
        else:
            if len(feature) != len(self._fea_name_list):
                print(feature)
            assert len(feature) == len(self._fea_name_list)
            dyz_id = None
        #print(feature)
        fea_vec = np.ones(shape=self.feature_dim, dtype=np.float32) * off_value
        idx = 0
        for key, item in zip(self._fea_name_list, feature):
            if key == "wordvec":
                vec = self._fea_trans_dict["wordvec"].value(item)
                if isinstance(dyzid_trans_dict, TransDict):
                    dyz_id = dyzid_trans_dict.value(item, 0)

            else:
                vec = self._fea_trans_dict[key].one_hot(item, on_value, off_value)
            #print(key)
            #print(vec)
            fea_vec[idx : idx + len(vec)] = vec
            idx += len(vec)
        # set dyzid
        if isinstance(dyz_id, int):
            fea_vec[idx + dyz_id] = on_value
        return fea_vec

    def _gen_target_vec(self, tar_name, key):
        """ label --> one-hot vector
        args:
            tar_name: STR, should be in tar_name_list
            key: STR, if not found, use default key
        return:
            np.array(dim, np.float32)
        """
        assert tar_name in self._tar_name_list
        return self._tar_trans_dict[tar_name].one_hot(key)

    def _init_trans_dicts(self, trans_dict_path, feature_name_list, target_name_list):
        """ initialize translate dicts for features and targets
            self._fea_trans_dict[key]: TransDict()
            self._tar_trans_dict[key]: TransDict()
        args:
            trans_dict_path:  DICT, {feature name: dict_abs_path}
                supported wordvec {"wordvec":[word_path, vec_path]}
            feature_name_list: LIST
            target_name_list: LIST
        :return:
        """
        self._feature_dim = 0
        self._fea_dim_list = []
        self._fea_trans_dict = {}
        self._fea_name_list = []
        # 读取特征词典
        for name in feature_name_list:
            if name == "wordvec":
                word_path = trans_dict_path["word_dict"]
                vec_path = trans_dict_path["wordvec_dict"]
                char_list, vec_list, xdim = self.read_wordvec_dict(
                    word_path, vec_path, is_compress=False)
                tdict = TransDict(char_list,
                                  embedding_values=vec_list,
                                  default_key=u"<UNK>")
                self._fea_dim_list.append(xdim)
                self._feature_dim += xdim
                self._use_wordvec = True
            elif name in ["word", "word_mix11"]:
                path = trans_dict_path["%s_dict" % name]
                tdict = self.read_word_dict(path)
                # TODO: 目前只支持 128 dim
                self._fea_dim_list.append(128)
                self._feature_dim += 128
                self._use_wordvec = True
            elif "dyz_id" in name:
                path = trans_dict_path["dyz_zy_dict"]
                self._dyz_zy_dict = self.read_dyz_zy_dict(path)
                self._feature_dim += self._dyz_zy_dict.dyz_num + 1
                self._fea_dim_list.append(self._dyz_zy_dict.dyz_num + 1)
                self._fea_name_list.append("dyz_id")
                continue
            else:
                try:
                    path = trans_dict_path["%s_dict" % name]
                    tdict = self.read_dict_file(path)
                    self._fea_dim_list.append(tdict.dim)
                    self._feature_dim += tdict.dim

                except KeyError:
                    exit("not find \"%s_dict\" in trans_dict_path" % name)

            self._fea_trans_dict[name] = tdict
            self._fea_name_list.append(name)
        #print("[DEBUG] _feature_dim = %d" % self._feature_dim)
        self._tar_trans_dict = {}
        self._tar_name_list = []
        # 读取label词典
        for name in target_name_list:
            try:
                path = trans_dict_path["%s_dict" % name]
            except:
                path = ""
                exit("not find \"%s_dict\" in trans_dict_path" % name)
            tdict = self.read_dict_file(path)
            self._tar_trans_dict[name] = tdict
            self._tar_name_list.append(name)
        return True

