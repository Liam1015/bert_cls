# -*- coding:utf-8 -*-
import tensorflow as tf
import os
from load_embedding import load_bert_embedding_table
import codecs
import numpy as np
# get vocab

def get_look_table():
    # 获取徐训练字典
    pre_vocab_file = '/home/zxj/workspace/bert_cls/bert/chinese_L-12_H-768_A-12/vocab.txt'
    my_vocab_file = './vocab'
    bert_ckpt = '/home/zxj/workspace/bert_cls/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'

    pre_word_dic = {}
    index = 0
    with codecs.open(pre_vocab_file, 'r', 'utf-8') as read_f:
        list_vocab = read_f.readlines()
        for word in list_vocab:
            word = word.strip()
            pre_word_dic[word] = index
            index += 1
    #
    print("pre_vocab_size length is %d" % len(pre_word_dic))
    # print(pre_word_dic['[CLS]'])
    pre_embedding_table = load_bert_embedding_table(bert_ckpt)
    # print np.size(pre_embedding_table[101])
    # word_vector = pre_embedding_table[pre_word_dic['[CLS]']]
    # word_vector1 = pre_embedding_table[101]
    #
    # print word_vector == word_vector1

    word_vectors = []
    with codecs.open(my_vocab_file, 'r', 'utf-8') as read_f:
        for word in read_f.readlines():
            word = word.strip()
            if pre_word_dic.has_key(word):
                word_vector = pre_embedding_table[pre_word_dic[word]]
                word_vectors.append(word_vector)
            else:
                word_vector = pre_embedding_table[pre_word_dic['[UNK]']]
                word_vectors.append(word_vector)

    word_vectors = np.array(word_vectors)
    # print np.shape(word_vectors)
    # print word_vectors[1]
    # print pre_embedding_table[101]
    # print word_vectors[1] == pre_embedding_table[101]
    return word_vectors

if __name__ == '__main__':
    data = get_look_table()
    print(type(data))
    print(np.shape(data))











