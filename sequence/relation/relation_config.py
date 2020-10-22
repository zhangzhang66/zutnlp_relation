#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : relation_config
# @Author   : ZhiYi Zhang
# @Time     : 2020/10/22 20:14
from common.config.common_config import CommonConfig
import dynamic_yaml


class SequenceConfig(CommonConfig):

    def __init__(self, config_file):
        super(SequenceConfig, self).__init__()
        self._config_file = config_file
        self.load_config()
        pass

    def load_config(self):
        with open(self._config_file, mode='r', encoding='UTF-8') as f:
            config = dynamic_yaml.load(f)
        return config
        pass
