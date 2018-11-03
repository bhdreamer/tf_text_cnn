#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
################################################################################

"""
File   : text_eval.py
Author : bianyanyao(bianyanyao@baidu.com)
Date   : 2017/7/20 10:50
Desc   : basic text model
Todo   :
"""
from __future__ import print_function

import os
import abc
import codecs
#import segword_accuracy as sa
#import postag_accuracy as pa

small_punc = [u"．", u"…", u"—", u"“", u"”", u"‘", u"’",
              u"：", u"《", u"》", u"（", u"）", u"、", u"·", u"『", u"』"]
big_punc = [u"，", u"。", u"；", u"！", u"？"]


class BasicEval(object):
    """
    basic class of evaluation metric
    for different task, specific subclass of BasicEval should be implement
    """

    def __init__(self, report_file):
        self._report_file = report_file

    @abc.abstractmethod
    def eval(self, ori_fea_file, ori_tar_file, pred_tar_file):
        """

        Args:
            ori_fea_file:
            ori_tar_file:
            pred_tar_file:

        Returns:
            a dict contain evaluation result, to be used in tf_text_common._write_train_report
            e.g. {'precision' : 0.9,
                  'recall' : 0.9,
                  'f1' : 0.9}
        """
        eval_result = dict()
        return eval_result

    @staticmethod
    def _merge_result(ori_fea_file, ori_tar_file, pred_tar_file, prefix):
        target_id = 0
        ori_fea_fp = codecs.open(ori_fea_file, "r", "gb18030", "ignoring")
        tar_fp = codecs.open(ori_tar_file, "r", "gb18030")
        pred_fp = codecs.open(pred_tar_file, "r", "gb18030")
        out_file_name = os.path.join(os.path.dirname(pred_tar_file),
                                     prefix + '_' + os.path.basename(pred_tar_file))
        out_file = codecs.open(out_file_name, 'w', "gb18030")
        targets = [x.split("@")[0].strip() for x in next(tar_fp).strip().split()]
        results = [x.split("@")[0].strip() for x in next(pred_fp).strip().split()]
        for i, line in enumerate(ori_fea_fp):
            if line.strip() == '':
                if target_id == 0:
                    continue
                out_file.write('\n')
                try:
                    targets = [x.split("@")[0].strip() for x in next(tar_fp).strip().split()]
                    results = [x.split("@")[0].strip() for x in next(pred_fp).strip().split()]
                    if len(targets) != len(results):
                        print("diff len! ori_fea_file: line num = %d" % (i + 1))
                        break
                except StopIteration:
                    break
                target_id = 0
            else:
                try:
                    fea_line = " ".join([x.split("@")[0].strip() for x in line.strip().split()])
                    out_file.write(
                        fea_line + ' ' + targets[target_id] + ' ' + results[target_id] + '\n')
                    target_id += 1
                except Exception as e:
                    print(e)
                    print("ori_fea_file: line num = %d" % (i + 1))
                    target_id += 1

        tar_fp.close()
        pred_fp.close()
        out_file.close()
        return out_file_name

    @staticmethod
    def merge_result(ori_fea_file, pred_tar_file, prefix):
        """
        use to merge prediction result
        Args:
            ori_fea_file:
            pred_tar_file:
            prefix:

        Returns:

        """
        target_id = 0
        result_fp = open(pred_tar_file)
        out_file_name = os.path.join(os.path.dirname(pred_tar_file),
                                     prefix + '_' + os.path.basename(pred_tar_file))
        out_file = open(out_file_name, 'w')
        results = next(result_fp).strip().split()
        for line in open(ori_fea_file).readlines():
            if line.strip() == '':
                if target_id == 0:
                    continue
                out_file.write('\n')
                try:
                    results = next(result_fp).strip().split()
                except StopIteration:
                    break
                target_id = 0
            else:
                out_file.write(
                    line.strip() + ' ' + results[target_id] + '\n')
                target_id += 1
        out_file.close()
        return out_file_name


class EvalProsody(BasicEval):
    """
    evaluate prosody accuracy
    """

    def __init__(self, report_file):
        super(EvalProsody, self).__init__(report_file)

    def eval(self, ori_fea_file, ori_tar_file, pred_tar_file):
        """

        Args:
            ori_fea_file:
            ori_tar_file:
            pred_tar_file:

        Returns:

        """
        eval_out = open(self._report_file, 'w')
        result_file = self._merge_result(ori_fea_file, ori_tar_file, pred_tar_file, 'prosody')

        res_labs = []
        ans_labs = []
        seg_labs = []
        with codecs.open(result_file, "r", "gb18030", "ignoring") as label_fp:
            lens = len(label_fp.readlines())
            label_fp.seek(0, 0)
            for i, line in enumerate(label_fp):
                if line.strip() == "":
                    continue
                labels = line.strip().split()
                if len(labels) < 2 or i == lens - 1:
                    ans_labs[-1] = 'B4'
                else:
                    try:
                        word = labels[0]
                    except UnicodeDecodeError:
                        try:
                            word = labels[0].decode("utf8")
                        except UnicodeDecodeError:
                            print("cannot decode word: %s", labels[0])
                            word = "<UNK>"
                    except AttributeError:
                        word = labels[0]
                    if word in small_punc:
                        continue
                    if word in big_punc:
                        ans_labs[-1] = 'B4'
                        continue
                    res_labs.append(labels[-1])
                    ans_labs.append(labels[-2])
                    seg_labs.append(labels[1])

        self._calculate(res_labs, ans_labs, seg_labs, eval_out)

        evals_pl = {}
        eval_out.write('#1:' + '\n')
        # rst_ans_1 = rst_ans_11 + rst_ans_12 + rst_ans_13 + rst_ans_22 + rst_ans_23 + rst_ans_33
        eval_out.write(
            ' '.join(['ans:', str(self._ans_1), 'rst:', str(self._rst_1), 'ans_rst:',
                      str(self._rst_ans_1)]) + '\n')
        p = float(self._rst_ans_1) / self._rst_1 if self._rst_1 > 0 else self._rst_1
        r = float(self._rst_ans_1) / self._ans_1 if self._ans_1 > 0 else self._ans_1
        f1 = (2 * p * r) / (p + r + 0.000001)
        f0_5 = ((1 + 0.5 * 0.5) * p * r) / (0.5 * 0.5 * p + r + 1e-6)
        eval_out.write(' '.join(['p:', str(p), 'r:', str(r), 'f1:', str(f1)]) + '\n')

        evals_pl['#1'] = (p, r, f1, f0_5)

        eval_out.write('#2:' + '\n')
        # rst_ans_2 = rst_ans_22 + rst_ans_23 + rst_ans_32 + rst_ans_33
        eval_out.write(
            ' '.join(['ans:', str(self._ans_2), 'rst:', str(self._rst_2), 'ans_rst:',
                      str(self._rst_ans_2)]) + '\n')
        p = float(self._rst_ans_2) / self._rst_2 if self._rst_2 > 0 else self._rst_2
        r = float(self._rst_ans_2) / self._ans_2 if self._ans_2 > 0 else self._ans_2
        f1 = (2 * p * r) / (p + r + 0.000001)
        f0_5 = ((1 + 0.5 * 0.5) * p * r) / (0.5 * 0.5 * p + r + 1e-6)
        eval_out.write(' '.join(['p:', str(p), 'r:', str(r), 'f1:', str(f1)]) + '\n')
        evals_pl['#2'] = (p, r, f1, f0_5)

        eval_out.write('#3:' + '\n')
        eval_out.write(
            ' '.join(['ans:', str(self._ans_3), 'rst:', str(self._rst_3), 'ans_rst:',
                      str(self._rst_ans_3)]) + '\n')
        p = float(self._rst_ans_3) / self._rst_3 if self._rst_3 > 0 else self._rst_3
        r = float(self._rst_ans_3) / self._ans_3 if self._ans_3 > 0 else self._ans_3
        f1 = (2 * p * r) / (p + r + 0.000001)
        f0_5 = ((1 + 0.5 * 0.5) * p * r) / (0.5 * 0.5 * p + r + 1e-6)
        eval_out.write(' '.join(['p:', str(p), 'r:', str(r), 'f1:', str(f1)]) + '\n')
        evals_pl['#3'] = (p, r, f1, f0_5)

        p = (self._rst_ans_1 + self._rst_ans_2 + self._rst_ans_3) / \
                  (float(self._rst_1 + self._rst_2 + self._rst_3) + 1e-6)
        r = (self._rst_ans_1 + self._rst_ans_2 + self._rst_ans_3) / \
                  (float(self._ans_1 + self._ans_2 + self._ans_3) + 1e-6)
        f1 = (2 * p * r) / (p + r + 1e-6)
        f0_5 = ((1 + 0.5 * 0.5) * p * r) / (0.5 * 0.5 * p + r + 1e-6)
        evals_pl['total'] = (p, r, f1, f0_5)
        eval_out.close()
        return evals_pl

    def _calculate(self, res_labs, ans_labs, seg_labs, eval_out):
        rst_ans_00 = 0
        rst_ans_01 = 0
        rst_ans_02 = 0
        rst_ans_03 = 0
        rst_ans_04 = 0
        rst_ans_10 = 0
        rst_ans_11 = 0
        rst_ans_12 = 0
        rst_ans_13 = 0
        rst_ans_14 = 0
        rst_ans_20 = 0
        rst_ans_21 = 0
        rst_ans_22 = 0
        rst_ans_23 = 0
        rst_ans_24 = 0
        rst_ans_30 = 0
        rst_ans_31 = 0
        rst_ans_32 = 0
        rst_ans_33 = 0
        rst_ans_34 = 0
        for i in range(len(res_labs)):
            res_lab = res_labs[i]
            ans_lab = ans_labs[i]
            seg_lab = seg_labs[i]
            if res_lab == 'I' or res_lab == 'O' or seg_lab == 'L' or seg_lab == 'M':
                if ans_lab == 'I' or ans_lab == 'O':
                    rst_ans_00 += 1
                elif ans_lab == 'B1':
                    rst_ans_01 += 1
                elif ans_lab == 'B2':
                    rst_ans_02 += 1
                elif ans_lab == 'B3':
                    rst_ans_03 += 1
                elif ans_lab == 'B4':
                    rst_ans_04 += 1
                elif ans_lab == 'T':
                    continue
                else:
                    raise Exception('rst 0 error' + ans_lab)
            elif res_lab == 'B1':
                if ans_lab == 'I' or ans_lab == 'O':
                    rst_ans_10 += 1
                elif ans_lab == 'B1':
                    rst_ans_11 += 1
                elif ans_lab == 'B2':
                    rst_ans_12 += 1
                elif ans_lab == 'B3':
                    rst_ans_13 += 1
                elif ans_lab == 'B4':
                    rst_ans_14 += 1
                elif ans_lab == 'T':
                    continue
                else:
                    raise Exception('rst 1 error' + ans_lab)
            elif res_lab == 'B2':
                if ans_lab == 'I' or ans_lab == 'O':
                    rst_ans_20 += 1
                elif ans_lab == 'B1':
                    rst_ans_21 += 1
                elif ans_lab == 'B2':
                    rst_ans_22 += 1
                elif ans_lab == 'B3':
                    rst_ans_23 += 1
                elif ans_lab == 'B4':
                    rst_ans_24 += 1
                elif ans_lab == 'T':
                    continue
                else:
                    raise Exception('rst 2 error' + ans_lab)
            elif res_lab == 'B3':
                if ans_lab == 'I' or ans_lab == 'O':
                    rst_ans_30 += 1
                elif ans_lab == 'B1':
                    rst_ans_31 += 1
                elif ans_lab == 'B2':
                    rst_ans_32 += 1
                elif ans_lab == 'B3':
                    rst_ans_33 += 1
                elif ans_lab == 'B4':
                    rst_ans_34 += 1
                elif ans_lab == 'T':
                    continue
                else:
                    raise Exception('rst 3 error' + ans_lab)
        eval_out.write('rst_ans:\n')
        eval_out.write(' '.join(
            ['00', str(rst_ans_00), '01', str(rst_ans_01),
             '02', str(rst_ans_02), '03', str(rst_ans_03),
             '04', str(rst_ans_04)]) + '\n')
        eval_out.write(' '.join(
            ['10', str(rst_ans_10), '11', str(rst_ans_11),
             '12', str(rst_ans_12), '13', str(rst_ans_13),
             '14', str(rst_ans_14)]) + '\n')
        eval_out.write(' '.join(
            ['20', str(rst_ans_20), '21', str(rst_ans_21),
             '22', str(rst_ans_22), '23', str(rst_ans_23),
             '24', str(rst_ans_24)]) + '\n')
        eval_out.write(' '.join(
            ['30', str(rst_ans_30), '31', str(rst_ans_31),
             '32', str(rst_ans_32), '33', str(rst_ans_33),
             '34', str(rst_ans_34)]) + '\n')
        eval_out.write('\n')
        self._ans_3 = rst_ans_03 + rst_ans_13 + rst_ans_23 + rst_ans_33
        self._ans_2 = rst_ans_02 + rst_ans_12 + rst_ans_22 + rst_ans_32 #+ self._ans_3
        self._ans_1 = rst_ans_01 + rst_ans_11 + rst_ans_21 + rst_ans_31 #+ self._ans_2
        self._rst_3 = rst_ans_30 + rst_ans_31 + rst_ans_32 + rst_ans_33
        self._rst_2 = rst_ans_20 + rst_ans_21 + rst_ans_22 + rst_ans_23 #+ self._rst_3
        self._rst_1 = rst_ans_10 + rst_ans_11 + rst_ans_12 + rst_ans_13 #+ self._rst_2
        self._rst_ans_3 = rst_ans_33
        self._rst_ans_2 = rst_ans_22 #+ rst_ans_23 + rst_ans_32 + self._rst_ans_3
        self._rst_ans_1 \
            = rst_ans_11 #+ rst_ans_12 + rst_ans_13 + rst_ans_31 + rst_ans_21 + self._rst_ans_2


class EvalSegment(BasicEval):
    """
    evaluate word segment accuracy
    """

    def __init__(self, report_file):
        super(EvalSegment, self).__init__(report_file)

    def eval(self, ori_fea_file, ori_tar_file, pred_tar_file):
        """

        Args:
            ori_fea_file:
            ori_tar_file:
            pred_tar_file:

        Returns:

        """
        eval_result = {}
        ans_sample_list, pred_sample_list = self._merge_seg_result(ori_fea_file, ori_tar_file, pred_tar_file, 'segment')
        evals_acc = sa.seg_accuracy_sample_list(pred_sample_list, ans_sample_list)
        eval_result["jingpei"] = evals_acc[0]
        eval_result["zhaohui"] = evals_acc[1]
        eval_result["qiyi"] = evals_acc[2]
        eval_result["zhengque"] = evals_acc[3]
        return eval_result

    @staticmethod
    def _merge_seg_result(ori_fea_file, ori_tar_file, pred_tar_file, prefix):
        target_id = 0
        all_targets = iter(open(ori_tar_file).readlines())
        all_results = iter(open(pred_tar_file).readlines())
        targets = all_targets.next().strip().split()
        results = all_results.next().strip().split()
        ans_sample_list = []
        pred_sample_list = []
        ans_sample = []
        pred_sample = []
        out_file_name = os.path.join(os.path.dirname(pred_tar_file),
                                     prefix + '_' + os.path.basename(pred_tar_file))
        out_file = open(out_file_name, 'w')
        for line in open(ori_fea_file).readlines():
            if line.strip() == '':
                if target_id == 0:
                    continue
                ans_sample_list.append(ans_sample)
                pred_sample_list.append(pred_sample)
                out_file.write('\n')
                try:
                    targets = all_targets.next().strip().split()
                    results = all_results.next().strip().split()
                except StopIteration:
                    break
                ans_sample = []
                pred_sample = []
                target_id = 0
            else:
                ans_sample.append((line.strip(), (), targets[target_id]))
                pred_sample.append((line.strip(), (), targets[target_id], results[target_id]))
                out_file.write(
                    line.strip() + ' ' + targets[target_id] + ' ' + results[target_id] + '\n')
                target_id += 1
        out_file.close()
        return ans_sample_list, pred_sample_list

    def _eval_one_vs_one(self, result_file):
        rst_ans_right = 0
        rst_ans_wrong = 0

        report = open(self._report_file, 'w')
        res_labs = []
        ans_labs = []
        with open(result_file) as lab_file:
            label_lines = lab_file.readlines()
            for i in range(len(label_lines)):
                labels = label_lines[i].strip().split()
                if len(labels) > 1:
                    res_labs.append(labels[-1])
                    ans_labs.append(labels[-2])
        for i in range(len(res_labs)):
            res_lab = res_labs[i]
            ans_lab = ans_labs[i]
            if res_lab == ans_lab:
                rst_ans_right += 1
            else:
                rst_ans_wrong += 1

        report.write('rst_ans:\n')
        report.write('rst_ans_right:' + str(rst_ans_right))
        report.write(' ')
        report.write('rst_ans_wrong:' + str(rst_ans_wrong))
        report.write('\n')

        eval_out = {}
        p = float(rst_ans_right) / (rst_ans_right + rst_ans_wrong)
        report.write(' '.join(['p:', str(p)]) + '\n')

        eval_out['p'] = p

        report.close()
        return eval_out

    @staticmethod
    def _lab2sent(result_file):
        label_in = open(result_file)
        sent_out = []
        label_lines = label_in.readlines()

        sent_str = ''
        for i in range(len(label_lines)):
            label_line = label_lines[i]
            try:
                line = label_line.decode('gbk').strip()
            except UnicodeDecodeError:
                try:
                    line = label_line.decode('utf8').strip()
                except UnicodeDecodeError:
                    print("cannot decode line: %s", label_line)
                    exit(1)
            except AttributeError:
                line = label_line.strip()
            labels = line.split(' ')
            if len(labels) > 1:
                word = labels[0]
                lab = labels[-1]
                sent_str += word
                if lab == 'R' or lab == 'S' or lab == 'E':
                    sent_str += ' '
            else:
                sent_out.append(sent_str)
                sent_str = ''
        sent_out.append(sent_str)
        label_in.close()
        return sent_out

    @staticmethod
    def _read_term(seg_sent):
        term = []
        words = seg_sent.strip().split()
        for word in words:
            if len(word) == 1:
                term.append('S')
            else:
                for i in range(len(word)):
                    if i == 0:
                        term.append('L')
                    elif i == len(word) - 1:
                        term.append('R')
                    else:
                        term.append('M')
        return term

    @staticmethod
    def _check_terms(pred_term, n_ans_start_idx, n_ans_end_idx):
        n_pred_term_start_cnt = 0
        n_pred_term_end_cnt = 0
        for i in range(n_ans_start_idx, n_ans_end_idx + 1):
            if pred_term[i] == 'L':
                n_pred_term_start_cnt += 1
            elif pred_term[i] == 'R':
                n_pred_term_end_cnt += 1
            elif pred_term[i] == 'S':
                n_pred_term_start_cnt += 1
                n_pred_term_end_cnt += 1

        if n_pred_term_start_cnt == 1 and n_pred_term_end_cnt == 1:
            if pred_term[n_ans_start_idx] == 'L' and pred_term[n_ans_end_idx] == 'R':
                return 'exact'
            elif pred_term[n_ans_start_idx] == 'S' and pred_term[n_ans_end_idx] == 'S':
                return 'exact'
            else:
                return 'confuse'
        elif n_pred_term_start_cnt <= 1:
            return 'good'

        if (pred_term[n_ans_start_idx] == 'L' or pred_term[n_ans_start_idx] == 'S') and \
                (pred_term[n_ans_end_idx] == 'R' or pred_term[n_ans_end_idx] == 'S'):
            return 'bad'

        return 'confuse'

    def _compare_terms(self, ans_term, pred_term):
        exact_recall = 0
        good_recall = 0
        bad_recall = 0
        confuse = 0

        n_ans_start_idx = -1
        n_ans_end_idx = -1

        for i in range(len(ans_term)):
            if ans_term[i] == 'L':
                if n_ans_start_idx != -1 or n_ans_end_idx != -1:
                    exit('[ERROR] Compare_terms failed!')
                n_ans_start_idx = i

            elif ans_term[i] == 'M':
                if n_ans_start_idx == -1 or n_ans_end_idx != -1:
                    exit('[ERROR] Compare_terms failed!')

            elif ans_term[i] == 'R':
                if n_ans_start_idx == -1 or n_ans_end_idx != -1:
                    exit('[ERROR] Compare_terms failed!')
                n_ans_end_idx = i
                pred_type = self._check_terms(pred_term, n_ans_start_idx, n_ans_end_idx)
                if pred_type == 'exact':
                    exact_recall += 1
                elif pred_type == 'good':
                    good_recall += 1
                elif pred_type == 'bad':
                    bad_recall += 1
                else:
                    confuse += 1
                n_ans_start_idx = -1
                n_ans_end_idx = -1

            elif ans_term[i] == 'S':
                if n_ans_start_idx != -1 or n_ans_end_idx != -1:
                    exit('[ERROR] Compare_terms failed!')
                n_ans_start_idx = i
                n_ans_end_idx = i
                pred_type = self._check_terms(pred_term, n_ans_start_idx, n_ans_end_idx)
                if pred_type == 'exact':
                    exact_recall += 1
                elif pred_type == 'good':
                    good_recall += 1
                else:
                    exit('[ERROR] Compare_terms failed!')
                n_ans_start_idx = -1
                n_ans_end_idx = -1

        cnts = [exact_recall, good_recall, bad_recall, confuse]
        return cnts

    def _work_on_file(self, ans_sent_list, pred_sent_list):
        eval_out = {}

        all_exact_recall = 0
        all_good_recall = 0
        all_bad_recall = 0
        all_confuse = 0

        n_real_seg_ci_chunk_cnt = 0
        n_real_seg_ci_exact_right = 0
        n_real_seg_ci_lidu_right = 0

        if len(ans_sent_list) != len(pred_sent_list):
            exit("[ERROR] Different sent_file length")
        for i in range(len(ans_sent_list)):
            if ans_sent_list[i].strip() != '' and pred_sent_list[i].strip() != '':
                ans_term = self._read_term(ans_sent_list[i])
                pred_term = self._read_term(pred_sent_list[i])
                if len(ans_term) != len(pred_term):
                    exit("[ERROR] Different length of sentence %d" % i)
                cnts = self._compare_terms(ans_term, pred_term)
                exact_recall, good_recall, bad_recall, confuse = cnts
                all_exact_recall += exact_recall
                all_good_recall += good_recall
                all_bad_recall += bad_recall
                all_confuse += confuse

                cnts = self._compare_terms(pred_term, ans_term)
                exact_recall, good_recall, bad_recall, confuse = cnts
                n_real_seg_ci_chunk_cnt += (exact_recall + good_recall + bad_recall + confuse)
                n_real_seg_ci_exact_right += exact_recall
                n_real_seg_ci_lidu_right += bad_recall

        all_ci_cnt = all_exact_recall + all_good_recall + all_bad_recall + all_confuse
        eval_out['jingpei'] = all_exact_recall / all_ci_cnt
        eval_out['zhaohui'] = (all_exact_recall + all_good_recall) / all_ci_cnt
        eval_out['qiyi'] = all_confuse / all_ci_cnt
        eval_out['zhengque'] = \
            (n_real_seg_ci_exact_right + n_real_seg_ci_lidu_right) / n_real_seg_ci_chunk_cnt
        return eval_out


class EvalProperty(BasicEval):
    """
    evaluate word property accuracy
    """

    def __init__(self, report_file):
        super(EvalProperty, self).__init__(report_file)

    def eval(self, ori_fea_file, ori_tar_file, pred_tar_file, seg_tar_file=""):
        """

        Args:
            ori_fea_file:
            ori_tar_file:
            pred_tar_file:
            seg_tar_file:

        Returns:

        """
        eval_result = {}
        #result_file = self._merge_result(ori_fea_file, ori_tar_file, pred_tar_file, 'property')
        ans_sample_list, pred_sample_list = self._merge_pos_result(
            ori_fea_file, ori_tar_file, pred_tar_file, seg_tar_file, 'pos')
        term_right, term_error, sent_right, sent_error = pa.postag_accuracy(pred_sample_list, ans_sample_list)
        term_p = float(term_right) / float(term_right + term_error)
        sent_p = float(sent_right) / float(sent_right + sent_error)
        eval_result["word_precision"] = term_p
        eval_result["sent_precision"] = sent_p
        return eval_result

    @staticmethod
    def _merge_pos_result(ori_fea_file, ori_tar_file, pred_tar_file, seg_tar_file, prefix):
        print (seg_tar_file)
        target_id = 0
        all_targets = iter(open(ori_tar_file).readlines())
        all_results = iter(open(pred_tar_file).readlines())
        targets = all_targets.next().strip().split()
        results = all_results.next().strip().split()
        if seg_tar_file != "":
            all_seg = iter(open(seg_tar_file).readlines())
            seg = all_seg.next().strip().split()
        ans_sample_list = []
        pred_sample_list = []
        ans_sample = []
        pred_sample = []
        out_file_name = os.path.join(os.path.dirname(pred_tar_file),
                                     prefix + '_' + os.path.basename(pred_tar_file))
        out_file = open(out_file_name, 'w')
        for line in open(ori_fea_file).readlines():
            if line.strip() == '':
                if target_id == 0:
                    continue
                ans_sample_list.append(ans_sample)
                pred_sample_list.append(pred_sample)
                out_file.write('\n')
                try:
                    targets = all_targets.next().strip().split()
                    results = all_results.next().strip().split()
                except StopIteration:
                    break
                if seg_tar_file != "":
                    try:
                        seg = all_seg.next().strip().split()
                    except StopIteration:
                        break
                ans_sample = []
                pred_sample = []
                target_id = 0
            else:
                if seg_tar_file == "":
                    w, s = line.strip().split()
                    ans_sample.append((w, s, targets[target_id]))
                    pred_sample.append((w, s, results[target_id]))
                else:
                    ans_sample.append((line.strip(), seg[target_id], targets[target_id]))
                    pred_sample.append((line.strip(), seg[target_id], results[target_id]))
                out_file.write(
                    line.strip() + ' ' + targets[target_id] + ' ' + results[target_id] + '\n')
                target_id += 1
        out_file.close()
        return ans_sample_list, pred_sample_list

    def _eval_one_vs_one(self, result_file):
        rst_ans_right = 0
        rst_ans_wrong = 0

        report = open(self._report_file, 'w')
        res_labs = []
        ans_labs = []
        with open(result_file) as lab_file:
            label_lines = lab_file.readlines()
            for i in range(len(label_lines)):
                labels = label_lines[i].strip().split()
                if len(labels) > 1:
                    res_labs.append(labels[-1])
                    ans_labs.append(labels[-2])
        for i in range(len(res_labs)):
            res_lab = res_labs[i]
            ans_lab = ans_labs[i]
            if res_lab == ans_lab:
                rst_ans_right += 1
            else:
                rst_ans_wrong += 1

        report.write('rst_ans:\n')
        report.write('rst_ans_right:' + str(rst_ans_right))
        report.write(' ')
        report.write('rst_ans_wrong:' + str(rst_ans_wrong))
        report.write('\n')

        eval_out = {}
        p = float(rst_ans_right) / (rst_ans_right + rst_ans_wrong)
        report.write(' '.join(['p:', str(p)]) + '\n')

        eval_out['p'] = p

        report.close()
        return eval_out


class EvalG2P(BasicEval):
    """
    evaluate g2p accuracy
    """

    def __init__(self, report_file):
        super(EvalG2P, self).__init__(report_file)

    def eval(self, ori_fea_file, ori_tar_file, pred_tar_file):
        """

        Args:
            ori_fea_file:
            ori_tar_file:
            pred_tar_file:

        Returns:

        """
        eval_result = {}
        ground_truth = open(ori_tar_file).readlines()
        prediction = open(pred_tar_file).readlines()
        if len(ground_truth) != len(prediction):
            exit("[ERROR] Length not equal between ori_tar_file and pred_tar_file")
        ground_truth_iter = iter(ground_truth)
        prediction_iter = iter(prediction)
        sample_num = len(ground_truth)
        correct = 0.0
        soft_correct = 0.0
        for i in range(sample_num):
            flag = True
            ans = ground_truth_iter.next().strip().split()
            res = prediction_iter.next().strip().split()
            if ans == res:
                correct += 1.0
            if len(ans) == len(res):
                for idx in range(len(ans)):
                    if ans[idx][-1] in ['0', '1', '2']:
                        ans[idx] = ans[idx][:-1]
                    if res[idx][-1] in ['0', '1', '2']:
                        res[idx] = res[idx][:-1]
                    if ans[idx] != res[idx]:
                        flag = False
                        break
                if flag:
                    soft_correct += 1.0
        accuracy = correct / sample_num
        accuracy_without_num = soft_correct / sample_num
        eval_result["word_acc"] = accuracy
        eval_result["word_acc_no_num"] = accuracy_without_num
        self._merge_result(ori_fea_file, ori_tar_file, pred_tar_file, prefix='g2p')
        return eval_result

    @staticmethod
    def _merge_result(ori_fea_file, ori_tar_file, pred_tar_file, prefix):
        all_targets = iter(open(ori_tar_file).readlines())
        all_results = iter(open(pred_tar_file).readlines())
        out_file_name = os.path.join(os.path.dirname(pred_tar_file),
                                     prefix + '_' + os.path.basename(pred_tar_file))
        out_file = open(out_file_name, 'w')
        word_line = ''
        for line in open(ori_fea_file).readlines():
            if line.strip() == '':
                out_file.write(word_line + '\n')
                try:
                    targets = all_targets.next().strip().split()
                    results = all_results.next().strip().split()
                except StopIteration:
                    exit('fea and tars not match')
                else:
                    out_file.write(' '.join(targets[:-1]) + '\n')
                    out_file.write(' '.join(results[:-1]) + '\n\n')
                word_line = ''
            elif line.strip() in ['<eow>', '<eps>']:
                continue
            else:
                word_line += line.strip()
        out_file.close()
        return out_file_name
