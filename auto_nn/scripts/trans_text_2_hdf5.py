#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: trans_text_2_hdf5.py
# @brief: 
# @author: niezhipeng
# @Created on 2018/9/29
# *************************************************************************************
import os
import sys
import codecs
import numpy as np
from common.config_box import *
from data_reader.text_reader import *

def trans_text_2_hdf5(data_info, dict_info, mean_var_path):
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
                        [data_info],
                        mean_var_path=mean_var_path,
                        is_training=True,
                        b_shuffle=True)
    print("dataset: size = %d (sentences)" % reader.sample_num)



def main(cfg_file, data_info):
    print(cfg_file)
    cfg = parse_json_conf(cfg_file)
    cfg["conf_file"] = os.path.abspath(cfg_file)
    dict_info = cfg["dict"]

    mean_var_path = cfg["model"]["mean_var_path"]
    trans_text_2_hdf5(data_info, dict_info, mean_var_path)


if __name__ == "__main__":
    root_dir = "/Users/niezhipeng/MyProgram/Python Scripts/baidu/personal-code-nzp/tf_training"
    cfg_file = os.path.join(root_dir, "conf/prosody_id_cnn_block.json")
    data_dir = os.path.join(root_dir, "data/byy_data/data")

    data_info = {
        "base_dir": data_dir,
        "fea_data_name": "prosody_dev_f11.fea",
        "tar_data_names": ["prosody_dev_f11.lab"],
        "encoding": "gbk",
        "hdf5_name": "dev_f11.hdf5"
    }

    main(os.path.abspath(cfg_file), data_info)


