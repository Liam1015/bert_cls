#!/usr/bin/env bash
export BERT_BASE_DIR=./bert/chinese_L-12_H-768_A-12
export GLUE_DIR=./data
export OUTPUT_DIR=./output

python classifier.py \
  --task_name=scene_cls \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=25 \
  --train_batch_size=256 \
  --learning_rate=2e-5 \
  --num_train_epochs=20 \
  --output_dir=$OUTPUT_DIR/
