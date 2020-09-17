#!/usr/bin/env bash
#export BERT_BASE_DIR=/home/new/Toxicity/bert_model/models/chinese_L-12_H-768_A-12
#export GLUE_DIR=/home/new/Bert/data
#export MODEL_DIR=/home/new/Bert/output
export BERT_BASE_DIR=/home/zhaoxj/pycharmProjects/bert_cls/bert/chinese_L-12_H-768_A-12
export GLUE_DIR=/home/zhaoxj/pycharmProjects/bert_cls/data
export MODEL_DIR=/home/zhaoxj/pycharmProjects/bert_cls/output

python client.py \
  --data_dir=$GLUE_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$MODEL_DIR/model.ckpt-90000 \
  --max_seq_length=128 \
  --predict_batch_size=1 \
  --output_dir= MODEL_DIR

