#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: trans_prosody_pred_2_label.py
# @brief: 
# @author: niezhipeng(@baidu.com)
# @Created on 2018/5/31
# *************************************************************************************
import os
import sys
import re
import codecs

g_prosody_dict = {"I":"0", "B1":"1", "B2":"2", "B3":"3", "O":"4", "T":"5"}
g_speaker = ["f28",  "com",  "f20", "f7", "m15",
                 "yyjw", "gezi", "f11", "novel", "news",
                 "miduo"]
g_speaker_tag = ["<%s>" % x for x in g_speaker]
small_punc = ["．", "…", "—", "“", "”", "‘", "’", "：", "《", "》", "（", "）", "、", "·", "『", "』"]
big_punc = ["，", "。", "；", "！", "？"]
g_punc_list = small_punc + big_punc

def trans_sentence(features):
    """将句子的文本特征转化为韵律标注文本"""
    outline = ""
    for fea in features:
        tmplist = fea.strip().split()
        char = tmplist[0].strip()
        pred = tmplist[-1].strip()
        if char in g_speaker_tag:
            if pred != "T":
                print("ERROR: speaker tag is %s" % pred)
            continue
        elif char in g_punc_list:
            if pred != "O":
                print("ERROR: punc(%s) tag is %s" % (char, pred))
            outline += char
        else:
            outline += char
            # 与label文本保持一致
            #token = g_prosody_dict[pred] if pred in ["B1", "B2", "B3"] else ""
            # (#0-#4) 标注格式
            token = ("(#%s)" % g_prosody_dict[pred]) if pred in g_prosody_dict else ""
            outline += token
    return outline

def trans_main(in_fn, out_fn):
    out_fp = codecs.open(out_fn, "w", "gb18030")
    with codecs.open(in_fn, "r", "gb18030") as in_fp:
        features = []
        line = in_fp.readline()
        while line != "":
            line = line.strip()
            if line == "":
                outline = trans_sentence(features)
                out_fp.write(outline + "\n")
                features = []
            else:
                features.append(line)
            line = in_fp.readline()
    out_fp.close()

def filter_label(label_fn, pred_fn, out_label_fn):
    """根据pred中出现的文本过滤label文本"""
    pred_fp = codecs.open(pred_fn, "r", "gb18030")
    out_fp = codecs.open(out_label_fn, "w", "gb18030")
    regex = re.compile(r"[1-3]")
    with codecs.open(label_fn, "r", "gb18030") as label_fp:
        pred_line = pred_fp.readline().strip()
        for line in label_fp:
            line = line.strip()
            if line[0] == pred_line[0]:
                if regex.sub("", line) == regex.sub("", pred_line):
                    out_fp.write(line + "\n")
                    pred_line = pred_fp.readline().strip()
    pred_fp.close()
    out_fp.close()

if __name__ == "__main__":
    work_dir = r"D:/Desktop/Backup/prosody_cbhg/id_cnn_block/test/id_cnn_3_block_75"
    pred_fn = r"%s/prosody_F11_ceshi_prosody_pred_results.txt" % work_dir
    token_fn = r"%s/prosody_F11_ceshi_id_cnn_3_block_75_#0.txt" % work_dir
    trans_main(pred_fn, token_fn)

    label_fn = r"%s/f11_old.txt" % work_dir
    out_fn = r"%s/prosody_test_f11.txt" % work_dir
    #filter_label(label_fn, token_fn, out_fn)




