# -*-coding: utf-8 -*-
"""
    @Project: tensorflow_models_nets
    @File   : convert_pb.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-08-29 17:46:50
    @info   :
    -通过传入 CKPT 模型的路径得到模型的图和变量数据
    -通过 import_meta_graph 导入模型中的图
    -通过 saver.restore 从模型中恢复图中各个变量的数据
    -通过 graph_util.convert_variables_to_constants 将模型持久化
"""

import tensorflow as tf
# from create_tf_record import *
from tensorflow.python.framework import graph_util
from classifier import *
import time
import tensorflow as tf
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# tf.enable_eager_execution()

def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)


    """The actual input function."""
    batch_size = FLAGS.predict_batch_size

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

def freeze_graph_test(pb_path, text_list):
    processor = MyProcessor()
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    sentences = [["search", str(sentence)] for sentence in text_list]
    #
    # input_ids = sess.graph.get_tensor_by_name("input_id:0")
    # input_mask = sess.graph.get_tensor_by_name("input_mask:0")
    # segment_ids = sess.graph.get_tensor_by_name("label_ids:0")
    # label_ids = sess.graph.get_tensor_by_name("segment_ids:0")
    predict_examples = processor.create_examples(sentences, set_type='pred', file_base=False)
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file='/home/zhaoxj/pycharmProjects/bert_cls/bert/chinese_L-12_H-768_A-12/vocab.txt',
        do_lower_case=FLAGS.do_lower_case)
    features = convert_examples_to_features(predict_examples, label_list, FLAGS.max_seq_length, tokenizer)
    predict_dataset = input_fn_builder(
        features=features,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)
    iterator = predict_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    print(next_element)
    inputs = ["label_ids", "input_ids", "input_mask", "segment_ids"]
    for input in inputs:
        next_element[input] = next_element[input].numpy().tolist()
        print type(next_element[input])
    # print(next_element)

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数


            # json_data = {"model_name": "default", "data": next_element}
            # 定义输出的张量名称
            input_node = sess.graph.get_tensor_by_name("IteratorV2:0")
            # input_id = sess.graph.get_tensor_by_name("IteratorGetNext:0")
            # input_mask = sess.graph.get_tensor_by_name("IteratorGetNext:1")
            # lable_ids = sess.graph.get_tensor_by_name("IteratorGetNext:2")
            # segment_ids = sess.graph.get_tensor_by_name("IteratorGetNext:3")
            keep_prob = sess.graph.get_tensor_by_name("loss/dropout/keep_prob:0")
            # TensorSliceDataset = sess.graph.get_tensor_by_name("TensorSliceDataset:0")
            output_tensor_name = sess.graph.get_tensor_by_name("loss/Softmax:0")
            out = sess.run(output_tensor_name, feed_dict={input_node: next_element, keep_prob: 1.0})
            # out = sess.run(output_tensor_name, feed_dict={
            #     input_id: next_element['input_ids'],
            #     input_mask: next_element['input_mask'],
            #     lable_ids: next_element['label_ids'],
            #     segment_ids: next_element['segment_ids'],
            #     keep_prob: 1.0})
            print("out:{}".format(out))
            # score = tf.nn.softmax(out, name='pre')
            class_id = tf.argmax(out, 1)
            print "pre class_id:{}".format(sess.run(class_id))


def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "loss/Softmax"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

        for op in sess.graph.get_operations():
            print(op.name, op.values())


def freeze_graph2(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "InceptionV3/Logits/SpatialSqueeze"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

        # for op in graph.get_operations():
        #     print(op.name, op.values())


if __name__ == '__main__':
    # 输入ckpt模型路径
    # input_checkpoint = './output/model.ckpt-90000'
    # # 输出pb模型的路径
    out_pb_path = './model_pb/cls_scene.pb'
    # out_pb_path = './model_pb/1560908617/saved_model.pb'
    # # 调用freeze_graph将ckpt转为pb
    # freeze_graph(input_checkpoint, out_pb_path)

    # 测试pb模型
    text_list = ["呵呵", "大陆片老男人","打开洗衣机"]
    freeze_graph_test(pb_path=out_pb_path, text_list=text_list)
# -*- coding:utf-8 -*-