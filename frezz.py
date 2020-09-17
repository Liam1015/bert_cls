#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-10 09:37:56

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import time

import tensorflow as tf
from tensorflow.python.tools import freeze_graph


def create_pbfile():
    saved_graph_name = './output/graph.pbtxt'
    saved_ckpt_name = './output/model.ckpt-90000'
    output_frozen_graph_name = './model_pb/cls_scene.pb'

    # freeze_graph.freeze_graph(input_graph=saved_graph_name, input_saver='',
    #                           input_binary=False,
    #                           input_checkpoint=saved_ckpt_name,
    #                           output_node_names='loss/Softmax',
    #                           restore_op_name='',
    #                           filename_tensor_name='',
    #                           output_graph=output_frozen_graph_name,
    #                           clear_devices=True,
    #                           initializer_nodes='')

    freeze_graph.freeze_graph(input_graph=saved_graph_name,
                              input_binary=False,
                              input_checkpoint=saved_ckpt_name,
                              output_node_names='loss/Softmax',
                              output_graph=output_frozen_graph_name,
                              clear_devices=True)


if __name__ == '__main__':
    create_pbfile()
    print('save to pb file ok..................')

