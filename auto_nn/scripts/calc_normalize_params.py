#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: calc_normalize_params.py
# @brief: 
# @author: niezhipeng(@baidu.com)
# @Created on 2018/5/24
# *************************************************************************************
import os
import sys
import codecs
import numpy as np
from common.config_box import *
from data_reader.text_reader import *

def calc_normalize_params(data_infos, dict_info, mean_var_path):
    # 读取trans_dict
    dict_base_dir = os.path.abspath(dict_info["base_dir"])
    trans_dict_paths = dict_info
    if isinstance(dict_info["feature_names"], (list, tuple)):
        feature_name_list = dict_info["feature_names"]
    else:
        feature_name_list = [dict_info["feature_names"]]
    if isinstance(dict_info["mark_names"], (list, tuple)):
        target_name_list = dict_info["mark_names"]
    else:
        target_name_list = [dict_info["mark_names"]]

    for key, path in trans_dict_paths.items():
        if key.endswith("_dict"):
            trans_dict_paths[key] = os.path.join(dict_base_dir, path)

    text_processor = TextProcessBox(trans_dict_paths,
                                    feature_name_list,
                                    target_name_list,
                                    mean_var_path)
    print("feature dims = " + str(text_processor.feature_dim_list))
    batch_size = 1

    reader = TextReader(text_processor,
                        None,
                        batch_size,
                        data_infos,
                        mean_var_path=mean_var_path,
                        is_training=True,
                        b_shuffle=True)
    print("dataset: size = %d (sentences)" % reader.sample_num)

    text_processor.calc_normalize_params(reader._dataset)


def main(cfg_file):
    print(cfg_file)
    cfg = parse_json_conf(cfg_file)
    cfg["conf_file"] = os.path.abspath(cfg_file)
    dict_info = cfg["dict"]

    train_cfg = cfg["train"]
    dataset = cfg["data_info"]

    train_dataset = normalize_list(train_cfg["train_dataset"])
    train_data_infos = [dataset["%s_data_info" % x] for x in train_dataset]

    mean_var_path = cfg["model"]["mean_var_path"]
    calc_normalize_params(train_data_infos, dict_info, mean_var_path)


if __name__ == "__main__":
    cfg_file = "../conf/prosody_id_cnn_block.json"
    main(os.path.abspath(cfg_file))


