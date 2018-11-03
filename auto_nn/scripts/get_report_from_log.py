#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: get_report_from_log.py
# @brief: 
# @author: niezhipeng(@baidu.com)
# @Created on 2018/4/23
# @Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
# *************************************************************************************
import os
import sys
import codecs
from common.csv_box import CSVWriter

def parse_report_from_log(log_fn):
    """ 解析tf_*.log中的训练和性能指标，转化为list
    :param log_fn:
    :return:
    """
    outlist = []
    linelist = []
    with codecs.open(log_fn, "r", "gb18030") as log_fp:
        line = log_fp.readline()
        while line != "":
            if "|" in line and "train info" in line:
                if len(linelist) > 0:
                    outlist.append(linelist)

                linelist = []
                epoch = int(line.split(":")[-1].split()[0].strip())
                linelist.append(("epoch", epoch))
            elif "| train speed" in line:
                speed = float(line.split(":")[-1].split()[0].strip())
            elif "| train loss" in line:
                train_loss = float(line.split(":")[-1].strip())
                linelist.append(("train_loss", train_loss))
            elif "|" in line and "dev_loss" in line:
                tmplist = line.split(":")[-1].split(",")
                dev_name = tmplist[0].split("=")[-1].strip()
                loss = float(tmplist[1].split("=")[-1].strip())
                linelist.append(("dev_%s_loss" % dev_name, loss))
                tmplist = log_fp.readline().split(":")[-1].split(",")
                linelist.append(("dev_%s_#1" % dev_name, "%.5f %.5f %.5f %.5f" %
                                tuple([float(x.split("=")[-1].strip()) for x in tmplist])))
                tmplist = log_fp.readline().split(":")[-1].split(",")
                linelist.append(("dev_%s_#2" % dev_name, "%.5f %.5f %.5f %.5f" %
                                tuple([float(x.split("=")[-1].strip()) for x in tmplist])))
                tmplist = log_fp.readline().split(":")[-1].split(",")
                linelist.append(("dev_%s_#3" % dev_name, "%.5f %.5f %.5f %.5f" %
                                tuple([float(x.split("=")[-1].strip()) for x in tmplist])))
            else:
                pass

            line = log_fp.readline()
    return outlist


def main(log_fn, out_fn):
    """ main
    :param log_fn:
    :param out_fn:
    :return:
    """
    report_writer = CSVWriter(out_fn)
    outlist = parse_report_from_log(log_fn)
    for linelist in outlist:
        report_writer.write_row(linelist)
    report_writer.close()


if __name__ == "__main__":
    #log_fn = "../tf_prosody.log"
    #out_fn = "../train_report.csv"
    log_fn = sys.argv[1]
    out_fn = sys.argv[2]
    main(log_fn, out_fn)