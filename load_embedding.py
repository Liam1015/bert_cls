# -*- coding:utf-8 -*-
import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow
import numpy as np
from sklearn.decomposition import PCA

# checkpoint_path = os.path.join(model_dir, model_name)
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# print len(var_to_shape_map)
# for key in var_to_shape_map:
#     print("tensor_name: ", key)
# print(reader.get_tensor('bert/embeddings/token_type_embeddings'))
# print type(reader.get_tensor('bert/embeddings/token_type_embeddings'))

# embedding_table = reader.get_tensor('bert/embeddings/token_type_embeddings')

# embedding_table = reader.get_tensor('bert/embeddings/word_embeddings')
# print np.size(embedding_table)
# print np.shape(embedding_table)
# print embedding_table[101]
# print np.size(embedding_table[101])

def load_bert_embedding_table(bert_ckpt):
    checkpoint_path = os.path.join(bert_ckpt)
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print("tensor_name: ", key)
    #     print(reader.get_tensor(key))
    embedding_table = reader.get_tensor('bert/embeddings/word_embeddings')
    print('embedding_bable shape is {}'.format(np.shape(embedding_table)))
    return embedding_table

if __name__=='__main__':
    # model_dir = '/home/zhaoxj/pycharmProjects/bert_cls/bert/chinese_L-12_H-768_A-12'
    # model_name = 'bert_model.ckpt'
    bert_ckpt = '/home/zhaoxj/pycharmProjects/bert_cls/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
    embedding_table = load_bert_embedding_table(bert_ckpt)
    # print embedding_table
    estimator = PCA(n_components=128)
    pca_embedding_table = estimator.fit_transform(embedding_table)
    pca_dim = np.shape(pca_embedding_table)
    print('pca_embedding_table shape is {}'.format(pca_dim))
    # print pca_embedding_table





