#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: split_crf_dict.py
# @brief: 
# @author: niezhipeng(@baidu.com)
# @Created on 2018/4/12
# @Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
# *************************************************************************************
import sys
import os
import codecs

def read_crf_dict(crf_dict_fn):
    items_dict = {}
    with codecs.open(crf_dict_fn, "r", "gb18030") as in_fp:
        item_num = int(in_fp.readline().strip())
        total_freq = int(in_fp.readline().strip())
        for line in in_fp:
            tmplist = line.strip().split()
            item = tmplist[1].strip().split("@")
            if item[1] not in items_dict:
                items_dict[item[-1]] = list()
            items_dict[item[-1]].append((item[0], tmplist[-1]))
    return items_dict

def write_dict_file(items, out_fn, default_item="<UNK>"):
    """ 从0开始编号，第一行：总数, 第二行：默认项, 第三行开始：id    item
    :param items:
    :param out_fn:
    :return:
    """
    with codecs.open(out_fn, "w", "gb18030") as out_fp:
        out_fp.write("%d\n" % len(items))
        out_fp.write("%s\n" % default_item)
        for id, item in enumerate(items, 0):
            out_fp.write("%d\t%s\n" % (id, item[0]))

def split_crf_feat_dict(crf_feat_fn, out_dir):
    feats_dict = read_crf_dict(crf_feat_fn)
    for key, items in feats_dict.items():
        if key == "w":
            feat_fn = os.path.join(out_dir, "char.dict")
            write_dict_file(items, feat_fn)
        elif key == "b":
            feat_fn = os.path.join(out_dir, "seg_token.dict")
            write_dict_file(items, feat_fn)
        elif key == "p":
            feat_fn = os.path.join(out_dir, "seg_prop.dict")
            write_dict_file(items, feat_fn)
        elif key == "tn":
            feat_fn = os.path.join(out_dir, "prosody_tn.dict")
            write_dict_file(items, feat_fn)
        else:
            pass

def trans_crf_label_dict(crf_label_fn, out_fn):
    items = list()
    default_item = ""
    with codecs.open(crf_label_fn, "r", "gb18030") as in_fp:
        item_num = int(in_fp.readline().strip())
        default_val = in_fp.readline().strip()
        for line in in_fp:
            tmplist = line.strip().split()
            item = tmplist[1].strip().split("@")
            if tmplist[0] == default_val:
                default_item = item[0]
            items.append((item[0], 0))
    write_dict_file(items, out_fn, default_item)

if __name__ == "__main__":
    work_dir = r"./data/dict"
    crf_feat_fn = "%s/crf_form/feat.dict" % work_dir
    out_dir = work_dir
    split_crf_feat_dict(crf_feat_fn, out_dir)
    crf_label_fn = "%s/crf_form/label.dict" % work_dir
    out_fn = "%s/label.dict" % work_dir
    trans_crf_label_dict(crf_label_fn, out_fn)


