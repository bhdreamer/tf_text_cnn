#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: auto_train_prosody
# @brief: 
# @author: niezhipeng
# @Created on 2018/4/9
# *************************************************************************************

import os
import sys
import codecs
import platform
import multiprocessing
import json
import time
import subprocess
from optparse import OptionParser
import numpy as np
import tensorflow as tf
from common.os_box import mkdir, copy_file
from common.csv_box import CSVWriter
from common.config_box import *
from common.logger_wrapper import LoggerWrapper
import models.prosody_model as prosody_model
from common.text_box import TextProcessBox
import text_eval
from data_reader.text_reader import *
from common.tf_utils import start_tensorboard

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# reload(sys)
# sys.setdefaultencoding("utf-8")

###################### read and parse conf file #########################

small_punc = ["．", "…", "—", "“", "”", "‘", "’", "：", "《", "》",
              "（", "）", "、", "·", "『", "』"]
big_punc = ["，", "。", "；", "！", "？"]


class ProsodyTagger(object):
    def __init__(self, _cfg):
        self._cfg = _cfg
        save_tag = _cfg["save_tag"]
        self.logger = LoggerWrapper.loggerFactory(
			'./tf_%s.log' % save_tag, 'debug')
        # self.logger = None
        self._cur_model_name = ""

        self.report_writer = None

        self._make_model_env()
        self.sess = None
        self.saver = None

    def close(self):
        # data_reader.stop()
        if self.sess:
            self.sess.close()
            self.sess = None
        if self.report_writer:
            self.report_writer.close()
        if self.tb_train_writer:
            self.tb_train_writer.close()
        if self.tb_test_writer:
            self.tb_test_writer.close()

    def train(self):

        self._init_model_train()

        start_epoch = int(self._cfg["train"]["start_epoch"])
        n_epochs = int(self._cfg["train"]["epochs"])
        self._global_train_epoch = start_epoch

        while self._global_train_epoch <= n_epochs:
            output_list = list()
            output_list.append(("epoch", self._global_train_epoch))

            self.logger.info("|==================== train info: epoch: %d ======================" %
                             self._global_train_epoch)

            if self._global_train_epoch > start_epoch:
                # 一轮训练
                train_losses, train_time = self.train_once()
                self.logger.info('| train speed: %.3f sec/epoch' % train_time)

                if len(train_losses) > self.task_num:
                    train_loss = float(np.sum(train_losses[:self.task_num]))
                    self.logger.info("| train loss: %.5f" % train_loss)
                    output_list.append(("train_loss", "%.5f" % train_loss))
                    self.logger.info(
                        "| train hidden_loss: " +
                        ", ".join(["%.5f" % x for x in train_losses[self.task_num:]])
                    )
                    output_list.append(
                        ("train_hidden_loss",
                         ", ".join(["%.5f" % x for x in train_losses[self.task_num:]]))
                    )
                else:
                    train_loss = float(np.sum(train_losses))
                    self.logger.info('| train loss: %.5f' % train_loss)
                    output_list.append(("train_loss", "%.5f" % train_loss))
            # print test summary in tensorboard
            test_summary = self.run_test_summary()
            self.tb_test_writer.add_summary(test_summary, self._global_train_epoch)

            self.logger.info("|---------------------------------------------------------------")

            for dev_name, dev_reader in zip(self.dev_watchers, self.dev_readers):
                dev_losses, eval_results = self.evaluate(dev_reader)
                if len(dev_losses) > self.task_num:
                    dev_loss = float(np.sum(dev_losses[:self.task_num]))
                    self.logger.info("| dev_loss: name = %s, loss = %.5f" %
                                     (dev_name, dev_loss))
                    # self.logger.info("\t hidden_loss: " +
                    #                 ", ".join(["%.5f" % x for x in dev_losses[self.task_num:]]))
                else:
                    dev_loss = float(np.sum(dev_losses))
                    self.logger.info("| dev_loss: name = %s, loss = %.5f" %
                                     (dev_name, dev_loss))
                # dev loss
                output_list.append(("dev_%s_loss" % dev_name, "%.5f" % dev_loss))
                # dev
                self.logger.info("|\t \"#1\": p = %.5f, r = %.5f, f1 = %.5f, f0_5 = %.5f" %
                                 tuple(eval_results[0]["#1"]))
                self.logger.info("|\t \"#2\": p = %.5f, r = %.5f, f1 = %.5f, f0_5 = %.5f" %
                                 tuple(eval_results[0]["#2"]))
                self.logger.info("|\t \"#3\": p = %.5f, r = %.5f, f1 = %.5f, f0_5 = %.5f" %
                                 tuple(eval_results[0]["#3"]))
                output_list.append(("dev_%s_#1_eval" % dev_name,
                                    "%.5f %.5f %.5f %.5f" % tuple(eval_results[0]["#1"])))
                output_list.append(("dev_%s_#2_eval" % dev_name,
                                    "%.5f %.5f %.5f %.5f" % tuple(eval_results[0]["#2"])))
                output_list.append(("dev_%s_#3_eval" % dev_name,
                                    "%.5f %.5f %.5f %.5f" % tuple(eval_results[0]["#3"])))

            self.report_writer.write_row(output_list)

            self._global_train_epoch += 1
        # TODO: stop()
        self.close()

    def test(self, init_model_path):
        test_cfg = self._cfg["test"]
        self._init_model_interface(init_model_path, test_cfg["output_dir"])
        batch_size = int(test_cfg["batch_size"])
        dataset = self._cfg["data_info"]
        for test_name in test_cfg["test_dataset"]:
            test_data_info = dataset["%s_data_info" % test_name]
            test_reader = TextReader(self.text_processor,
                                     self.logger,
                                     batch_size,
                                     test_data_info,
                                     mean_var_path=self.mean_var_path,
                                     is_training=True,
                                     b_shuffle=False)
            self.logger.info("|========================= test info: ============================")
            self.logger.info("| model = %s" % self._cur_model_name)
            self.logger.info("| test dataset: name = %s, size = %d (sentences)" %
                             (test_name, test_reader.sample_num))
            self.logger.info("| \t batch_size = %d, batch_num = %d" %
                             (test_reader.batch_size, test_reader.batch_num))

            test_losses, test_eval_results = tagger.evaluate(test_reader)

            if len(test_losses) > len(self.target_names):
                extra_num = len(test_losses) - len(self.target_names)
                test_loss = float(np.sum(test_losses[:-extra_num]))
                self.logger.info("| test_loss: name = %s, loss = %.5f" %
                                 (test_name, test_loss))
                # self.logger.info("\t hidden_loss: " +
                #                 ", ".join(["%.5f" % x for x in test_losses[-extra_num:]]))
            else:
                test_loss = float(np.sum(test_losses))
                self.logger.info("| test_loss: name = %s, loss = %.5f" %
                                 (test_name, test_loss))

            self.logger.info("|\t \"#1\": p = %.5f, r = %.5f, f1 = %.5f, f0_5 = %.5f" %
                             tuple(test_eval_results[0]["#1"]))
            self.logger.info("|\t \"#2\": p = %.5f, r = %.5f, f1 = %.5f, f0_5 = %.5f" %
                             tuple(test_eval_results[0]["#2"]))
            self.logger.info("|\t \"#3\": p = %.5f, r = %.5f, f1 = %.5f, f0_5 = %.5f" %
                             tuple(test_eval_results[0]["#3"]))

    def predict(self, init_model, input_data_infos, output_dir):
        self._init_model_interface(init_model, output_dir)
        batch_size = 120
        for data_info in input_data_infos:
            fn = data_info["fea_data_name"].split(".")[0].strip()
            pred_reader = TextReader(self.text_processor,
                                     self.logger,
                                     batch_size,
                                     data_info,
                                     mean_var_path=self.mean_var_path,
                                     is_training=False,
                                     b_shuffle=False)
            self.logger.info("|=============== prediction info ===============")
            self.logger.info("| model = %s, input_data = %s" % (self._cur_model_name, fn))
            self.logger.info("| size = %d, batch_size = %d" % (pred_reader.sample_num, batch_size))

            pred_labels_list = [[] for _ in range(len(self.target_names))]
            for i, batch_data in enumerate(pred_reader.batch_iterator()):
                feed_dict = self.model.make_feed_inputs(batch_data[0],
                                                        batch_data[1],
                                                        None,
                                                        dropout=0.,
                                                        is_training=False)
                fetches = (self.model.scores_list)
                pred_scores_list = self.sess.run(fetches, feed_dict=feed_dict)
                for k, task in enumerate(self.target_names):
                    act_seq_len = batch_data[1]
                    #act_seq_len = TextReader.get_batch_seq_lens(batch_data[0])
                    # trans logits to labels, and write to file
                    pred_labels = self.trans_scores_2_label(task, pred_scores_list[k], act_seq_len)
                    pred_labels_list[k].extend(pred_labels)

            data_name = fn.strip().split('.')[0]
            for i, task in enumerate(self.target_names):
                fea_fn = os.path.join(data_info["base_dir"], data_info["fea_data_name"])
                pred_fn = os.path.join(
                    self._cfg["prediction_dir"],
                    "%s_%s_%s_pred_results.txt" % (data_name, self._cur_model_name, task)
                )
                self.logger.info("| task = %s, output_file = %s" % (task, pred_fn))
                TextReader.merge_pred_file(fea_fn, pred_labels_list[i], pred_fn)

    def train_once(self):
        """ train whole dataset for one epoch
        args:
            train_feeder:
        :return:
            aver_train_loss --- 一轮训练的平均损失
        """
        aver_train_losses = None
        start_time = time.time()
        # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        dropout = float(self._cfg["train"]["dropout_rate"])
        for i, batch_data in enumerate(self.train_reader.batch_iterator()):
            feed_dict = self.model.make_feed_inputs(batch_data[0],
                                                    batch_data[1],
                                                    batch_data[2:],
                                                    dropout=dropout,
                                                    is_training=True)
            start = time.time()
            fetches = [self.model.train_op, self.loss_summary, self.model.loss_list]
            _, loss_summary, train_loss = self.sess.run(fetches, feed_dict=feed_dict)
            self.tb_train_writer.add_summary(loss_summary,
                (self._global_train_epoch-1)*self.train_reader.batch_num+i)

            if (i + 1) % self.train_reader.batch_num == 0:
                feed_dict = self.model.make_feed_inputs(batch_data[0],
                                                        batch_data[1],
                                                        batch_data[2:],
                                                        dropout=0,
                                                        is_training=False)
                train_summary = self.sess.run(self.merged_summary, feed_dict=feed_dict)
                self.tb_train_writer.add_summary(train_summary, self._global_train_epoch)

            # print("step %d train_runtime: %.3f" % (i, time.time() - start))
            try:
                aver_train_losses += train_loss
            except TypeError:
                aver_train_losses = np.asarray(train_loss)
                # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                # supported by cuda libcupti
                # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                # timeline_fn = os.path.join(self._cfg["tb_logs_dir"],
                #     "timeline_%d_step_%d.json" % (self._global_train_epoch, i))
                # with open(timeline_fn, "w") as fp:
                #    fp.write(chrome_trace)

                # updating train summary of each bp_step
                # self.tb_writer.add_summary(train_summary, self.global_step)

        train_time = time.time() - start_time
        aver_train_losses /= self.train_reader.batch_num

        # save train model
        self._cur_model_name = "model-%d" % self._global_train_epoch
        model_path = os.path.join(self._cfg["model_dir"], self._cur_model_name)
        self.saver.save(self.sess, model_path, write_meta_graph=False)
        return aver_train_losses, train_time

    # TODO: 根据任务方式以及具体评估指标来修改代码
    def evaluate(self, data_reader):
        aver_loss = None
        pred_labels_list = [[] for _ in range(len(self.target_names))]
        for i, batch_data in enumerate(data_reader.batch_iterator()):
            feed_dict = self.model.make_feed_inputs(batch_data[0],
                                                    batch_data[1],
                                                    batch_data[2:],
                                                    dropout=0.,
                                                    is_training=False)
            fetches = (self.model.scores_list, self.model.loss_list)
            test_scores_list, test_loss = self.sess.run(fetches, feed_dict=feed_dict)
            # print(batch_data[1][0])
            # print(test_scores_list[0][0])
            # print(test_logits_list[0][0])
            try:
                aver_loss += test_loss
            except TypeError:
                aver_loss = np.asarray(test_loss)
            for k, task in enumerate(self.target_names):
                act_seq_len = batch_data[1]
                # act_seq_len = TextReader.get_batch_seq_lens(batch_data[0])
                # trans logits to labels, and write to file
                pred_labels = self.trans_scores_2_label(task, test_scores_list[k], act_seq_len)
                pred_labels_list[k].extend(pred_labels)
        data_info = data_reader.data_info_list[0]
        data_name = data_info["fea_data_name"].strip().split('.')[0]
        pred_files = []
        for i, task in enumerate(self.target_names):
            pred_fn = os.path.join(self._cfg["prediction_dir"],
                                   "%s_%s_pred_results.txt" % (data_name, task))
            TextReader.write_label_file(pred_labels_list[i], pred_fn)
            pred_files.append(pred_fn)

        eval_result = self._calc_evaluate_result(data_info, pred_files)
        aver_loss /= data_reader.batch_num
        # 忽略hidden loss
        return aver_loss[:self.task_num], eval_result

    # TODO: 根据序列的填充策略来修改此处
    def trans_scores_2_label(self, task, scores, actual_seq_len):
        pred_labels = []
        rev_mark_dict = self.text_processor.target_trans_dict[task]
        for ith_sample in range(len(actual_seq_len)):
            # label_id with max score
            # seq_logit = np.argmax(scores[ith_sample], axis=-1)
            seq_len = actual_seq_len[ith_sample]
            seq_score = scores[ith_sample][:seq_len]
            # print(seq_score)
            pred_labels.append([rev_mark_dict.get_key(seq_score[i])
                                for i in range(len(seq_score))])
        return pred_labels

    def _calc_evaluate_result(self, data_info, pred_file_list):
        """ 计算韵律的性能评价指标
        :param data_info:
        :param pred_file_list:
        :return:
            DICT: {'#1':(p, r, f1, f0_5), ...}
        """
        base_dir = os.path.abspath(data_info["base_dir"])
        data_name = data_info["fea_data_name"].split(".feature")[0]
        tar_fn_list = normalize_list(data_info["tar_data_names"])
        ori_fea_file = os.path.join(base_dir, data_info["fea_data_name"])
        eval_result = list()
        for ith_task in range(len(self.target_names)):
            ori_tar_file \
                = os.path.join(base_dir, tar_fn_list[ith_task])
            report_file = os.path.join(self._cfg["report_dir"], "report_%s_%s_task_%d.txt" %
                                       (data_name, self._cur_model_name, ith_task))
            eval_name = ["EvalProsody"]
            try:
                eval_metric = getattr(text_eval, eval_name[ith_task])(report_file)
            except AttributeError:
                eval_result.append(dict())
            else:
                if len(self.target_names) != 1 and eval_name[ith_task] == "EvalProperty":
                    seg_file = os.path.join(base_dir, "%s_segment.target" % data_name)
                    eval_result.append(eval_metric.eval(
                        ori_fea_file, ori_tar_file, pred_file_list[ith_task], seg_file))
                else:
                    eval_result.append(eval_metric.eval(
                        ori_fea_file, ori_tar_file, pred_file_list[ith_task]))
        return eval_result

    def _make_model_env(self):
        # 读取trans_dict
        dict_info = self._cfg["dict"]
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

        self.mean_var_path = self._cfg["model"]["mean_var_path"]
        self.text_processor = TextProcessBox(trans_dict_paths,
                                             feature_name_list,
                                             target_name_list,
                                             self.mean_var_path)

        self.feature_dim = self.text_processor.feature_dim
        #self.feature_dims = self.text_processor.feature_dim_list
        self.target_names = self.text_processor.target_name_list
        self.target_dims = self.text_processor.target_dim_list
        self.task_num = len(self.target_names)
        self.logger.info("[INFO] feature_dim =", str(self.feature_dim))
        self.logger.info("[INFO] target_dims = " + " ".join(["%s:%d" % (name, dim)
                         for name, dim in zip(self.target_names, self.target_dims)]))

        model_cfg = parse_model_config(self._cfg)
        model_name = model_cfg["name"]
        self.model = getattr(prosody_model, model_name)(
            self.feature_dim, self.target_dims, model_cfg)
        self.model.build_graph()

    def _init_model_train(self):
        """ make train dir and build tensor board
        :return:
        """
        train_cfg = self._cfg["train"]

        # mkdir train dir
        self._make_train_dir(b_cover=True)

        copy_file(self._cfg["conf_file"], self._cfg["save_dir"])

        self.report_writer = CSVWriter(os.path.join(self._cfg["report_dir"],
            "train_report_%s.csv" % self._cfg["save_tag"]))

        tf.logging.set_verbosity(tf.logging.WARN)

        self.global_train_step = 0

        # train step, auto added by one
        self.global_bp_step = tf.Variable(0, name='global_step', trainable=False)
        # train_op
        # optim_name = train_cfg["optim_name"]
        # optim_params = train_cfg["optim_params"]
        # train_cfg[optim_name] = self._cfg[optim_name]
        self.train_op = self.model.apply_optimizer(train_cfg, global_step=self.global_bp_step)

        self.saver = tf.train.Saver(max_to_keep=None)

        # start sess
        self.sess_config = tf.ConfigProto(log_device_placement=False,
                                          allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)
        # self.sess.graph.finalize()

        # init model
        init_model = os.path.abspath(train_cfg["init_model"])
        if os.path.exists(init_model + ".index"):
            self.logger.info("init model: %s" % init_model)
            self.saver.restore(self.sess, init_model)
            self._global_train_epoch = int(train_cfg["start_epoch"])
        else:
            init_args = tf.global_variables_initializer()
            self.sess.run(init_args)
            self.logger.info("a new model will be built")
            self._global_train_epoch = 0
        self._cur_model_name = "init_%d" % self._global_train_epoch
        self.saver.save(self.sess, os.path.join(self._cfg["model_dir"], "tf_model.init"))

        # self.merged_summary = tf.summary.merge(
        #     tf.get_collection(tf.GraphKeys.SUMMARIES, scope=None))
        # self.merged_summary = tf.summary.merge_all()
        self.loss_summary = tf.summary.merge(
            tf.get_collection(tf.GraphKeys.SUMMARIES, scope="calc_loss")
        )

        self.merged_summary = tf.summary.merge(
            tf.get_collection(tf.GraphKeys.SUMMARIES, scope="model")
        )
        # config tensorboard
        self.tb_train_writer = tf.summary.FileWriter(os.path.join(self._cfg["tb_logs_dir"], "train"), self.sess.graph)
        self.tb_test_writer = tf.summary.FileWriter(os.path.join(self._cfg["tb_logs_dir"], "test"))
        try:
            self._device_id = int(self._cfg["device_id"])
        except:
            self._device_id = None
        addr, td_pid = start_tensorboard(self._cfg["tb_logs_dir"], self._device_id)
        self.logger.info("visit tensorboard: %s, kill: kill -9 %s" % (addr, td_pid))

        self.mean_var_path = self._cfg["model"]["mean_var_path"]

        train_batch_size = train_cfg["batch_size"]
        # train reader & dev reader
        # TODO: 根据配置文件修改
        dataset = self._cfg["data_info"]
        train_dataset = normalize_list(train_cfg["train_dataset"])
        train_data_infos = [dataset["%s_data_info" % x] for x in train_dataset]
        self.train_reader = TextReader(self.text_processor,
                                       self.logger,
                                       train_batch_size,
                                       train_data_infos,
                                       mean_var_path=self.mean_var_path,
                                       is_training=True,
                                       b_shuffle=True)

        self.logger.info("train dataset: size = %d(sentences), batch_size = %d, batch_num = %d" %
                         (self.train_reader.sample_num,
                          self.train_reader.batch_size,
                          self.train_reader.batch_num))
        # print(self.train_reader.next_batch())
        # dev_reader
        dev_batch_size = train_cfg["dev_batch_size"]
        self.dev_watchers = normalize_list(train_cfg["dev_dataset"])
        self.dev_readers = []
        for i, dev_name in enumerate(self.dev_watchers):
            if dev_name == "train":
                dev_data_info = self.train_reader.data_info_list
                dev_dataset = self.train_reader._dataset
            else:
                try:
                    dev_data_info = [dataset["%s_data_info" % dev_name]]
                except KeyError:
                    # invalid dataset will be rejexted
                    dev_data_info = []
                    print("[ERROR] dev_name = %s not in data_info" % dev_name)
                    continue
                    #sys.exit()
                dev_dataset = None

            dev_reader = TextReader(self.text_processor,
                                    self.logger,
                                    dev_batch_size,
                                    data_info=dev_data_info,
                                    dataset=dev_dataset,
                                    mean_var_path=self.mean_var_path,
                                    is_training=True,
                                    b_shuffle=False)

            self.logger.info("dev dataset: name = %s, size = %d(sentences), " %
                             (dev_name, dev_reader.sample_num) +
                             "batch_size = %d, batch_num = %d" %
                             (dev_reader.batch_size, dev_reader.batch_num))
            # print(dev_reader.next_batch())
            self.dev_readers.append(dev_reader)


    def _make_train_dir(self, b_cover=True):
        """建立训练目录和训练环境
        args:
            _cfg["train"]["train_dir"]
            _cfg["train"]["save_tag"]
        """
        try:
            output_dir = self._cfg["output_dir"]
            if output_dir == "":
                raise KeyError
        except KeyError:
            output_dir = os.path.join(self._cfg["work_space"], "output")

        save_dir = os.path.join(os.path.abspath(output_dir), self._cfg["save_tag"])
        mkdir(save_dir, b_cover=b_cover)
        self._cfg["save_dir"] = save_dir
        self.logger.info("save_dir = %s" % save_dir)
        self._cfg["model_dir"] = os.path.join(save_dir, "model")
        mkdir(self._cfg["model_dir"], b_cover=b_cover)
        self._cfg["prediction_dir"] = os.path.join(save_dir, "prediction")
        mkdir(self._cfg["prediction_dir"], b_cover=b_cover)
        self._cfg["report_dir"] = os.path.join(save_dir, "report")
        mkdir(self._cfg["report_dir"], b_cover=b_cover)
        self._cfg["tb_logs_dir"] = os.path.join(save_dir, "tb_logs")
        mkdir(self._cfg["tb_logs_dir"], b_cover=b_cover)

    def _init_model_interface(self, init_model, output_dir):
        #test_cfg = self._cfg["test"]
        # make test save dir
        save_dir = os.path.abspath(output_dir)
        self._cfg["save_dir"] = save_dir
        mkdir(save_dir, b_cover=False)
        self._cfg["prediction_dir"] = os.path.join(save_dir, "prediction")
        mkdir(self._cfg["prediction_dir"], b_cover=True)
        self._cfg["report_dir"] = os.path.join(save_dir, "report")
        mkdir(self._cfg["report_dir"], b_cover=True)

        copy_file(self._cfg["conf_file"], self._cfg["save_dir"])

        self.mean_var_path = self._cfg["model"]["mean_var_path"]

        # tf_model_saver
        self.saver = tf.train.Saver(max_to_keep=None)

        # start sess
        self.sess_config = tf.ConfigProto(log_device_placement=False,
                                          allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)


        #init_model = os.path.abspath(test_cfg["init_model"])
        self._cur_model_name = os.path.split(init_model)[-1]
        if os.path.exists(init_model + ".index") and \
                os.path.exists(init_model + ".data-00000-of-00001"):
            self.logger.info("init model: %s" % init_model)
            self.saver.restore(self.sess, init_model)
        else:
            self.logger.warning("init_model_path = %s not exist!" % init_model)
            sys.exit()

    def run_test_summary(self):

        sentences = [
            [u"转 L v I", u"过 R v I", u"脸 S n I", u"问 S v I", u"珠 L n I", u"宝 R n I", u"商 L n I", u"人 R n B3", u"： S w O",
             u"这 L r I", u"些 R r I", u"珠 L n I", u"子 R n I", u"你 S r I", u"从 S p I", u"哪 L r I", u"儿 R r I", u"弄 S v I", u"来 S m I", u"的 S u B3", u"？ S w O"],
            [u"你 S r I", u"从 S p I", u"不 L v I", u"会 R v I", u"听 L n I", u"说 M n I", u"过 R n I", u"我 S r I", u"在 S p I",
             u"组 S n I", u"里 S f I", u"有 S v I",u"不 L v I", u"尊 M v I", u"重 R v I", u"任 L r I", u"何 R r I", u"一 L n I", u"个 M n I", u"人 R n B3", u"。 S w O"]
        ]

        labels = [
            [u"I  I B2 B1 I B1 I B3 O  I B1 I B3 B1 I I B1 I I B3 O"],
            [u"B1 I I  B1 I I  B3 I B1 I B2 B1 I I B1 I B1 I I B3 O"]
        ]
        seq_len = [len(x) for x in sentences]
        max_len = max(seq_len)
        inputs = []
        targets = []
        for seq_x, seq_y in zip(sentences, labels):
            inputs.append(np.asarray(self.text_processor.encode_seq_feature(
                [x.strip().split() for x in seq_x], embedding=True),
                dtype=np.float32))
            targets.append(np.asarray(self.text_processor.encode_seq_target(
                "prosody", seq_y[0].strip().split(), embedding=False),
                dtype=np.int32))
        # print(inputs)
        # print(targets)
        norm_inputs = self.text_processor.feature_normalize(inputs)
        feed_dict = self.model.make_feed_inputs(np.stack(norm_inputs),
                                                np.asarray(seq_len),
                                                [np.stack(targets)],
                                                dropout=0.,
                                                is_training=False)

        fetches = (self.merged_summary, self.model.scores_list, self.model.loss_list)
        test_summary, test_scores_list, test_loss = self.sess.run(fetches, feed_dict=feed_dict)
        return test_summary

def simple_usage():
    """
    simple usage
    """
    mess = "usage: %prog [-options] config_file\n"
    mess += "use simple:\n"
    mess += "\t\n1. to train model : %prog -t [config_file]\n"
    mess += "\t\n2. to predict: %prog -p [config_file]\n"
    return mess


if __name__ == "__main__":

    # conf_file = sys.argv[1]
    cfg_file = "../conf/prosody_densely_cnn.json"

    # 读取并解析ini配置文件
    # cfg = parse_conf_file(cfg_file)
    cfg = parse_json_conf(cfg_file)
    cfg["conf_file"] = os.path.abspath(cfg_file)

    job_type = "train"
    tagger = ProsodyTagger(cfg)

    device = cfg["device_id"]
    # device = ""
    if device != "hdfs" and device.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    # print("[INFO] CUDA_VISIBLE_DEVICES: %s" % device)
    tagger.logger.info("[INFO] CUDA_VISIBLE_DEVICES: %s" % device)
    if job_type == "train":
        # tagger.train()
        train_proc = multiprocessing.Process(target=tagger.train)
        train_proc.start()
        train_proc.join()
    elif job_type == "test":
        init_model_path = cfg["test"]["init_model"]
        tagger.test(init_model_path)
        # test_proc = multiprocessing.Process(target=tagger.test,
        #                                     args=(init_model_path,))
        # test_proc.start()
        # test_proc.join()
    elif job_type == "prediction":
        test_cfg = cfg["test"]
        init_model_path = test_cfg["init_model"]
        output_dir = test_cfg["output_dir"]
        input_data_infos = [cfg["data_info"]["%s_data_info" % x]
                            for x in test_cfg["test_dataset"]]
        tagger.predict(init_model_path, input_data_infos, output_dir)
    elif job_type == "multi-test":
        pass
    else:
        pass
