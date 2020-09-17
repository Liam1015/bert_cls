#!/usr/bin/env bash
# export BERT_BASE_DIR=/home/new/Toxicity/bert_model/models/chinese_L-12_H-768_A-12
# export GLUE_DIR=/home/new/Bert/data
# export OUTPUT_DIR=/home/new/Bert/output
export BERT_BASE_DIR=./bert/chinese_L-12_H-768_A-12
export GLUE_DIR=./data
export OUTPUT_DIR=./output
export TRAINED_CLASSIFIER=./output
python classifier.py \
  --task_name=scene_cls \
  --do_test=true \
  --data_dir=$GLUE_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER/model.ckpt-10000 \
  --max_seq_length=25 \
  --output_dir=$OUTPUT_DIR/ \
  --test_batch_size=128
