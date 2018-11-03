#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @file: config_box.py
# @brief: 
# @author: niezhipeng(@baidu.com)
# @Created on 2018/4/19
"""

import sys
import json
import codecs


################ parse ini config file ##################
def parse_conf_file(conf_file):
    """ parse ini config file
    :param conf_file:
    :return:
    """
    if sys.version > '3':
        config_parser = __import__("configparser")
    else:
        config_parser = __import__("ConfigParser")
    conf_parser = config_parser.ConfigParser()
    # 用config对象读取配置文件
    conf_parser.read(conf_file)
    conf = dict()
    for section in conf_parser.sections():
        conf[section] = read_section_params(conf_parser, section)
    return conf


def read_section_params(conf, section):
    """ read ini config section according to section name
    :param conf:
    :param section:
    :return:
    """
    options = dict()
    for (key, value) in conf.items(section):
        options[key] = convert_str(value)
    return options


def parse_model_config(conf):
    """ parse nn layer params
    args:
        conf: DICT
    return:
        DICT: {"hidden_layer": [], "output_layer": []}
    """
    model_cfg = conf["model"]
    if "hidden_layer" in model_cfg:
        return model_cfg

    model_cfg["hidden_layer"] = list()
    model_cfg["output_layer"] = list()
    # parse model config from conf object
    for key, layer in conf.items():
        if key.startswith("hidden_layer"):
            model_cfg["hidden_layer"].append(layer)
        elif key.startswith("output_layer"):
            model_cfg["output_layer"].append(layer)
    return model_cfg


################## parse *.json config conf ##################
def parse_json_conf(conf_file):
    """ parse *.json file, remove "//" or "#" comment
    args:
        conf_file:
    return:
        JSON Object
    """
    text = ""
    with codecs.open(conf_file, "r", "utf-8") as fp:
        text = ""
        for line in fp:
            text += remove_comment(line)
    return json.loads(text)


################# common functions ####################
def remove_comment(str):
    """ remove "//" or "#" comment
    args:
        str:
    return:
        STR
    """
    str = str.rstrip()
    if r"//" in str:
        return str.split(r"//")[0] + "\n"
    elif r"#" in str:
        return str.split(r"#")[0] + "\n"
    else:
        return str + "\n"


def normalize_list(value):
    """ str --> list
    args:
        str: STR or LIST
    return:
        LIST
    """
    if isinstance(value, (list, tuple)):
        return value
    elif isinstance(value, (int, float)):
        return [value]
    else:
        for chr in [",", "&", "|"]:
            if chr in value:
                str = value.strip("[]()")
                return [x.strip() for x in value.split(chr)]
        return [value]


def convert_str(str):
    """ convert str to int, float, list with '&'
    :param str:
    :return:
    """
    str = str.strip()
    if "&" in str:
        return [convert_str(x) for x in str.split("&")]
    elif "|" in str:
        return [convert_str(x) for x in str.split("|")]
    try:
        num = float(str)
        return int(num) if int(num) == num else num
    except ValueError:
        if str.upper() in ["TRUE", "T"]:
            return True
        elif str.upper() in ["FALSE", "F"]:
            return False
        else:
            return str


def normalize_boolean(value):
    """ convert value to boolean
    :param value:
    :return:
    """
    if isinstance(value, bool):
        return value
    try:
        value = value.upper()
        return value in ["T", "TRUE", "Y", "YES"]
    except AttributeError as e:
        raise Exception("can not normalize boolean for ", value)

