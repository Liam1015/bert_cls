#!/usr/bin/env bash
#export BERT_BASE_DIR=/home/new/Toxicity/bert_model/models/chinese_L-12_H-768_A-12
#export GLUE_DIR=/home/new/Bert/data
#export MODEL_DIR=/home/new/Bert/output
#export MODEL_PB_DIR=/home/new/Bert/api/

export BERT_BASE_DIR=/home/zhaoxj/pycharmProjects/bert_cls/bert/chinese_L-12_H-768_A-12
export GLUE_DIR=/home/zhaoxj/pycharmProjects/bert_cls/data
export MODEL_DIR=/home/zhaoxj/pycharmProjects/bert_cls/output
export MODEL_PB_DIR=/home/zhaoxj/pycharmProjects/bert_cls/model_pb/

python export.py \
  --task_name=scene_cls \
  --do_predict=true \
  --data_dir=$GLUE_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --model_dir=$MODEL_DIR/ \
  --serving_model_save_path=$MODEL_PB_DIR



