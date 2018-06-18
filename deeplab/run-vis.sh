#!/bin/bash


DATASET="./datasets/tfrecord"

TRAIN_LOGDIR="./datasets/exp/train"
EVAL_LOGDIR="./datasets/exp/val"
VIS_LOGDIR="./datasets/exp/vis"

python ./vis.py \
  --logtostderr \
  --vis_split="val" \
  --dataset="soybean" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size=256 \
  --vis_crop_size=256 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${DATASET}" \
  --max_number_of_iterations=1
