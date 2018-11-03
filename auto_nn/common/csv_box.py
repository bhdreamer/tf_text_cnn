#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: csv_box.py
# @brief: 
# @author: niezhipeng(@baidu.com)
# @Created on 2017/10/22
# *************************************************************************************

import os
import csv


class CSVWriter(object):
    """
    csv writer, the first row is header,
    """

    def __init__(self, file_path, fieldnames=None, b_cover=True, restval="", dialect="excel"):
        self._csv_file = None
        mode = "w" if b_cover else "a"
        # for windows, if newline != "", the written line will be followed with a blank.
        # self._csv_file = open(file_path, "a", newline="")
        self._csv_file = open(file_path, mode)
        self._fieldnames = fieldnames
        self._restval = restval
        self._writer = csv.writer(self._csv_file, dialect=dialect)
        if self._fieldnames is not None:
            self.write_header()

    def __del__(self):
        self.close()

    def write_header(self):
        """
        write header
        :return:
        """
        self._writer.writerow(self._fieldnames)

    def write_row(self, items):
        """
        write the next row with list of value.
        if self._header is None, write header firstly
        args:
            in_list: LIST of (name, value)
        return:
        """
        if self._fieldnames is None or len(self._fieldnames) != len(items):
            self._fieldnames = [item[0] for item in items]
            self.write_header()
        rowdict = dict(items)
        self._writer.writerow([rowdict.get(x, self._restval) for x in self._fieldnames])

    def write_item(self, key, value):
        """
        write the pointed key, other values is ""
        args:
            key:
            value:
        :return:
        """
        values = [value if x == key else self._restval for x in self._fieldnames]
        self._writer.writerow(values)

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None