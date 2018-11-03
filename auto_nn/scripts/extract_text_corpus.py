#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: extract_text_corpus.py
# @brief: 按比例抽取文本语料
# @author: niezhipeng(@baidu.com)
# @Created on 2018/4/24
# @Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
# *************************************************************************************
import os
import sys
import codecs
import numpy as np
from data_reader.text_reader import TextReader

def main(text_fn, out_dir, percent, encoding="gb18030"):
    """ 按比例抽取文本语料
    :param text_fn:
    :param out_dir:
    :param percent:
    :param encoding:
    :return:
    """
    raw_dataset = []
    seq = []
    cnt = 0
    with codecs.open(text_fn, 'r', encoding=encoding) as fp:
        #orgline = fp.readline()
        line = fp.readline()
        while line != "":
            if line.strip() == "" or line.startswith("END"):
                raw_dataset.append(seq)
                cnt += 1
                seq = []
                #orgline = fp.readline()
                line = fp.readline()
                continue
            seq.append(line)
            # tmplist = line.split()
            line = fp.readline()
    train_fp = codecs.open(os.path.join(out_dir, "train.txt"), "w", encoding)
    dev_fp = codecs.open(os.path.join(out_dir, "dev.txt"), "w", encoding)
    dev_inds = np.random.permutation(cnt)[:int(percent * cnt)+1]
    print("train text = %d (sentences), dev text = %d (sentences)" %
          (cnt - len(dev_inds), len(dev_inds)))
    for ind in range(cnt):
        if ind in dev_inds:
            dev_fp.writelines(raw_dataset[ind])
            dev_fp.write("\n")
        else:
            train_fp.writelines(raw_dataset[ind])
            train_fp.write("\n")
    train_fp.close()
    dev_fp.close()


if __name__ == "__main__":
    work_dir = "../../data/20180319_f11/dev_file"
    text_fn = "%s/tag_shuffle_validation_common" % work_dir
    out_dir = "%s/new" % work_dir
    percent = 0.1
    main(text_fn, out_dir, percent)
