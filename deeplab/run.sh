#!/bin/bash


DATASET="./datasets/tfrecord"

TRAIN_LOGDIR="./datasets/exp/train"

CKPT="./xception_65/model.ckpt"
CKPT="./deeplabv3_pascal_train_aug/model.ckpt"

# Train 10 iterations.
NUM_ITERATIONS=1000

python train.py \
  --logtostderr \
  --initialize_last_layer=False \
  --num_clones=1 \
  --last_layers_contain_logits_only=True \
  --dataset='soybean' \
  --train_split="train" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=256 \
  --train_crop_size=256 \
  --train_batch_size=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="./deeplabv3_pascal_train_aug/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET}"



