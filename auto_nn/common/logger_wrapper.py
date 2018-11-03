#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#
################################################################################

"""
File   : logger_wrapper.py
Author : zhanghuangbin(zhanghuangbin@baidu.com)
Date   : 2017/2/13 19:47
Desc   :
Todo   : 
"""

import logging


class LoggerWrapper(object):
    """
    logger wrapper
    """
    __logger = None
    __logger_wrapper = None
    def __init__(self):
        """
        init
        """
        return

    @staticmethod
    def basicConfig(logger_path="./default.log",
                    log_level_s="info",
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w'):
        """
        basic config
        :param log_level_s:
        :type log_level_s:
        :param format:
        :type format:
        :param datefmt:
        :type datefmt:
        :param logger_path:
        :type logger_path:
        :param filemode:
        :type filemode:
        :return:
        :rtype:
        """
        if log_level_s == "debug":
            log_level = logging.DEBUG
        elif log_level_s == "info":
            log_level = logging.INFO
        elif log_level_s == "warning":
            log_level = logging.WARNING
        elif log_level_s == "error":
            log_level = logging.ERROR
        elif log_level_s == "critical":
            log_level = logging.CRITICAL
        else:
            log_level = logging.DEBUG
        logging.basicConfig(level=log_level,
                            format=format,
                            datefmt=datefmt,
                            filename=logger_path,
                            filemode=filemode)
        LoggerWrapper.__logger = logging.getLogger()

    @staticmethod
    def getLogger():
        """
        getLogger
        :return:
        :rtype:
        """
        if LoggerWrapper.__logger_wrapper is None:
            return LoggerWrapper()
        else:
            return LoggerWrapper.__logger_wrapper

    @staticmethod
    def loggerFactory(logger_path="./default.log", log_level_s="info"):
        """
        getDefaultLogger
        :param log_level_s:
        :type log_level_s:
        :param logger_path:
        :type logger_path:
        :return:
        :rtype:
        """
        LoggerWrapper.basicConfig(log_level_s=log_level_s, logger_path=logger_path)
        if LoggerWrapper.__logger_wrapper is None:
            return LoggerWrapper()
        else:
            return LoggerWrapper.__logger_wrapper

    def debug(self, mess="", b_print=True):
        """
        debug
        :param mess:
        :param b_print:
        """
        if b_print is True:
            print(mess)
        if self.__logger is not None:
            self.__logger.debug(mess)
            return True
        else:
            return False

    def info(self, mess="", b_print=True):
        """
        info
        :param mess:
        :param b_print:
        """
        if b_print is True:
            print(mess)
        if self.__logger is not None:
            self.__logger.info(mess)
            return True
        else:
            return False

    def warning(self, mess="", b_print=True):
        """
        warning
        :param mess:
        :param b_print:
        """
        if b_print is True:
            print(mess)
        if self.__logger is not None:
            self.__logger.warning(mess)
            return True
        else:
            return False

    def error(self, mess="", b_print=True):
        """
        error
        :param mess:
        :param b_print:
        """
        if b_print is True:
            print(mess)
        if self.__logger is not None:
            self.__logger.error(mess)
            return True
        else:
            return False

    def critical(self, mess="", b_print=True):
        """
        critical
        :param mess:
        :param b_print:
        """
        if b_print is True:
            print(mess)
        if self.__logger is not None:
            self.__logger.critical(mess)
            return True
        else:
            return False
