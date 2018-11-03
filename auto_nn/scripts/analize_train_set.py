#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: analize_train_set.py
# @brief: 
# @author: niezhipeng(@baidu.com)
# @Created on 2018/7/13
# *************************************************************************************

import os
import sys
import codecs
import numpy as np


g_label = [u"I", u"B1", u"B2", u"B3", u"O", u"T"]

#def calc_label_count(sent_label):


def calc_label_interval(sent_label, label_type):
    cnt = len(sent_label)
    inds = [0]
    icode = g_label.index(label_type)
    for i, x in enumerate(sent_label):
        if x in g_label[icode:]:
            inds.append(i)
    inds.append(cnt-1)
    intervals = []
    for i in range(len(inds)-1):
        if i <= 0 or i >= len(inds)-1:
            continue
        if sent_label[inds[i]] == label_type:
            intervals.append((inds[i]-inds[i-1], inds[i+1]-inds[i]))
    max_dis = [max(x) for x in intervals]
    #print(intervals)
    return max_dis


def print_feature(in_fn, out_fn):
    """ 打印超出间隔限制的特征文本
    :param in_fn:
    :param out_fn:
    :return:
    """
    out_fp = codecs.open(out_fn, "w", "gb18030")
    with codecs.open(in_fn, "r", "gb18030") as in_fp:
        features = []
        line = in_fp.readline()
        while line != "":
            line = line.strip()
            if line == "":
                sent_label = [x.split()[-1].split("@")[0].strip() for x in features]
                B3_ans = calc_label_interval(sent_label, "B3")
                if any([x >= 16 for x in B3_ans]):
                    out_fp.write("\n".join(features))
                    out_fp.write("\n\n")
                features = []
            else:
                features.append(line)
            line = in_fp.readline()
    out_fp.close()


def main(in_fn, out_dir):
    """ 统计B1/B2/B3对应的最大间隔数量
    :param in_fn:
    :param out_dir:
    :return:
    """
    B1_dict = {}
    B2_dict = {}
    B3_dict = {}
    label_dict = dict.fromkeys(g_label, 0)
    len_dict = {}
    out_list = []
    with codecs.open(in_fn, "r", "gb18030") as in_fp:
        for line in in_fp:
            sent_label = [x.split("@")[0].strip() for x in line.strip().split()]
            B1_ans = calc_label_interval(sent_label, "B1")
            B2_ans = calc_label_interval(sent_label, "B2")
            B3_ans = calc_label_interval(sent_label, "B3")
            out_list.append([B1_ans, B2_ans, B3_ans, len(sent_label)])
            for x in B1_ans:
                try:
                    B1_dict[x] += 1
                except KeyError:
                    B1_dict[x] = 1
            for x in B2_ans:
                try:
                    B2_dict[x] += 1
                except KeyError:
                    B2_dict[x] = 1
            for x in B3_ans:
                try:
                    B3_dict[x] += 1
                except KeyError:
                    B3_dict[x] = 1
            sent_len = len(sent_label)

            try:
                len_dict[sent_len] += 1
            except KeyError:
                len_dict[sent_len] = 1

            for key in g_label:
                label_dict[key] += sent_label.count(key)

            '''
            if len(B1_ans) > 0 and any([x > B1_max for x in B1_ans]):
                B1_max = max(B1_ans)
            if len(B1_ans) > 0 and any([x < B1_min for x in B1_ans]):
                B1_min = min(B1_ans)
            B1_aver += sum(B1_ans)
            
            if len(B2_ans) > 0 and any([x > B2_max for x in B2_ans]):
                B2_max = max(B2_ans)
            if len(B2_ans) > 0 and any([x < B2_min for x in B2_ans]):
                B2_min = min(B2_ans)
            B2_aver += sum(B2_ans)
            
            if len(B3_ans) > 0 and any([x > B3_max for x in B3_ans]):
                B3_max = max(B3_ans)
            if len(B3_ans) > 0 and any([x < B3_min for x in B3_ans]):
                B3_min = min(B3_ans)
            B3_aver += sum(B3_ans)
            '''
    with codecs.open(os.path.join(out_dir, "B1.txt"), "w", "utf-8") as out_fp:
        for x1,x2 in sorted(B1_dict.items(), key=lambda x:x[0]):
            out_fp.write("%d %d\n" % (x1, x2))

    with codecs.open(os.path.join(out_dir, "B2.txt"), "w", "utf-8") as out_fp:
        for x1,x2 in sorted(B2_dict.items(), key=lambda x:x[0]):
            out_fp.write("%d %d\n" % (x1, x2))

    with codecs.open(os.path.join(out_dir, "B3.txt"), "w", "utf-8") as out_fp:
        for x1,x2 in sorted(B3_dict.items(), key=lambda x:x[0]):
            out_fp.write("%d %d\n" % (x1, x2))

    with codecs.open(os.path.join(out_dir, "LEN.txt"), "w", "utf-8") as out_fp:
        for x1,x2 in sorted(len_dict.items(), key=lambda x:x[0]):
            out_fp.write("%d %d\n" % (x1, x2))

    for key, cnt in sorted(label_dict.items(), key=lambda x:x[0]):
        print("%s: %d" % (key, cnt))


if __name__ == "__main__":
    #a = [u"B1 I I  B1 I I  B3 I B1 I B2 B1 I I B1 I B1 I I B3 O"]
    #sent_label = [x.strip() for x in a[0].split()]
    #print(calc_label_interval(sent_label, "B3"))

    work_dir = r"/Users/niezhipeng/MyProgram/Python Scripts/baidu/personal-code-nzp/tf_training_prosody_cbhg/data/20180319_f11"
    in_fn = r"%s/tag_shuffle_merge_train.target" % work_dir
    main(in_fn, work_dir)
    # out_fn = r"%s/features_B3_more_than_16.txt" % work_dir
    # print_feature(in_fn, out_fn)