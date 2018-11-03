#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#
################################################################################

"""
File   : os_box.py
Author : zhanghuangbin(zhanghuangbin@baidu.com)
Date   : 2017/2/6 20:25
Desc   :
Todo   :
"""

from __future__ import print_function
import os
import shutil
import sys


def get_files_name(scp_file):
    """
    get files from scp
    Args:
        scp_file ():

    Returns:

    """
    files_name = []
    with open(scp_file, "r") as fp:
        lines = [line.strip() for line in fp if line.strip()]
        for file in lines:
            file = file.split()[0]
            file_name = os.path.splitext(os.path.split(file)[1])[0]
            files_name.append(file_name)
    return files_name


def get_files(scp_file):
    """
    get files from scp
    Args:
        scp_file ():

    Returns:

    """
    files = []
    files_name = []
    with open(scp_file, "r") as fp:
        lines = [line.strip() for line in fp if line.strip()]
        for file in lines:
            file = file.split()[0]
            files.append(file)
            file_name = os.path.splitext(os.path.split(file)[1])[0]
            files_name.append(file_name)
    return files, files_name


def mkdir(dir, b_cover=False):
    """
    make dir
    Args:
        dir ():
        b_cover ():

    Returns:

    """
    if os.path.exists(dir) and b_cover is True:
        shutil.rmtree(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return True


def copy_file(files, dir, b_cover=False, tar_names=None):
    """
    copy files
    """
    if type(files) != list:
        files = [files]
    if tar_names is None:
        tar_names = list()
    if type(tar_names) != list:
        tar_names = [tar_names]
    for i, file in enumerate(files):
        if not os.path.exists(file):
            print("file: %s does not exist" % file)
            return False
        if len(tar_names) == 0:
            tar_name = os.path.basename(file)
        else:
            tar_name = tar_names[i]
        out_file = os.path.join(dir, tar_name)
        if os.path.exists(out_file) and b_cover is True:
            os.remove(out_file)
        if not os.path.exists(out_file):
            shutil.copyfile(file, out_file)
    return True


def iter_bar(iterator, num=None):
    """
    迭代器进度条
    Args:
        iterator (): 迭代器
        num (): 迭代器数目

    Returns:

    """
    if num is None:
        num = len(iterator)
    fmt = "\r{:3d}% [{:<100}]".format
    for i, data in enumerate(iterator):
        percentage = int((i + 1) * 100. / num)
        sys.stdout.write(fmt(percentage, "=" * percentage))
        sys.stdout.flush()
        yield data
    print("")


def add_suffix_and_prefix(file_name, prefix=None, suffix=None, splitor="_"):
    """
    给文件添加后缀和前缀
    Args:
        file_name ():
        prefix ():
        suffix ():
        splitor ():

    Returns:

    """
    prefix = "" if prefix is None else prefix
    suffix = "" if suffix is None else suffix
    name, file_type = os.path.splitext(file_name)
    return "{}{}{}{}{}{}".format(prefix, splitor, name, splitor, suffix, file_type)


def read_list_file(file_path, encoding="gbk"):
    """ 读取列表文件, 过滤掉空行和'#'注释行
    args:
        file_path:
        encoding: default("gbk")
    :return:
    """
    outlist = []
    with open(file_path, 'r', encoding=encoding) as fp:
        for line in [x for x in fp if x != "" and not x.startswith("#")]:
            outlist.append(r"%s" % line.strip())
    return outlist

def parse_file_path(path):
    """
    parse file path
    args:
        path: absolute path or relative path
    :return:
        file_dir:
        name:
        ext:
    """
    path = os.path.abspath(path)
    (file_dir, file_name) = os.path.split(path)
    (name, ext) = os.path.splitext(file_name)
    return (file_dir, name, ext)

def join_abs_path(base_dir, paths):
    """
    文件添加目录并转化为绝对路径
    args:
        base_dir: dir path
        paths: LIST or STR
    :return:
        abs paths
    """
    base_dir = os.path.abspath(base_dir)
    if isinstance(paths, (list, tuple)):
        return [os.path.join(base_dir, x) for x in paths]
    else:
        return os.path.join(base_dir, paths)