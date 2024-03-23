#!/bin/bash
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export CUDA_VISIBLE_DEVICES=1

# SCENE=bikes
# EXPERIMENT=raw
# DATA_DIR=/media/cilab/data/RawNerf/rawnerf/scenes
# CHECKPOINT_DIR=/home/cilab/shreyas/codes/multinerf/ckpts2/"$EXPERIMENT"/"$SCENE"

# rm "$CHECKPOINT_DIR"/*
# python -m train \
#   --gin_configs=configs/llff_raw.gin \
#   --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
#   --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
#   --logtostderr
  
# SCENE=candle
# EXPERIMENT=raw
# DATA_DIR=/media/cilab/data/RawNerf/rawnerf/scenes
# CHECKPOINT_DIR=/home/cilab/shreyas/codes/multinerf/ckpts2/"$EXPERIMENT"/"$SCENE"

# rm "$CHECKPOINT_DIR"/*
# python -m train \
#   --gin_configs=configs/llff_raw.gin \
#   --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
#   --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
#   --logtostderr
  
# SCENE=candlefiat
# EXPERIMENT=raw
# DATA_DIR=/media/cilab/data/RawNerf/rawnerf/scenes
# CHECKPOINT_DIR=/home/cilab/shreyas/codes/multinerf/ckpts2/"$EXPERIMENT"/"$SCENE"

# rm "$CHECKPOINT_DIR"/*
# python -m train \
#   --gin_configs=configs/llff_raw.gin \
#   --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
#   --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
#   --logtostderr
  
SCENE=gardenlights
EXPERIMENT=raw
DATA_DIR=/media/cilab/data/RawNerf/rawnerf/scenes
CHECKPOINT_DIR=/home/cilab/shreyas/codes/multinerf/ckpts2/"$EXPERIMENT"/"$SCENE"

rm "$CHECKPOINT_DIR"/*
python -m train \
  --gin_configs=configs/llff_raw.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr
  
# SCENE=livingroom
# EXPERIMENT=raw
# DATA_DIR=/media/cilab/data/RawNerf/rawnerf/scenes
# CHECKPOINT_DIR=/home/cilab/shreyas/codes/multinerf/ckpts2/"$EXPERIMENT"/"$SCENE"

# rm "$CHECKPOINT_DIR"/*
# python -m train \
#   --gin_configs=configs/llff_raw.gin \
#   --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
#   --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
#   --logtostderr
  
# SCENE=morningkitchen
# EXPERIMENT=raw
# DATA_DIR=/media/cilab/data/RawNerf/rawnerf/scenes
# CHECKPOINT_DIR=/home/cilab/shreyas/codes/multinerf/ckpts2/"$EXPERIMENT"/"$SCENE"

# rm "$CHECKPOINT_DIR"/*
# python -m train \
#   --gin_configs=configs/llff_raw.gin \
#   --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
#   --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
#   --logtostderr
  
# SCENE=nightstreet
# EXPERIMENT=raw
# DATA_DIR=/media/cilab/data/RawNerf/rawnerf/scenes
# CHECKPOINT_DIR=/home/cilab/shreyas/codes/multinerf/ckpts2/"$EXPERIMENT"/"$SCENE"

# rm "$CHECKPOINT_DIR"/*
# python -m train \
#   --gin_configs=configs/llff_raw.gin \
#   --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
#   --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
#   --logtostderr
  
# SCENE=notchbush
# EXPERIMENT=raw
# DATA_DIR=/media/cilab/data/RawNerf/rawnerf/scenes
# CHECKPOINT_DIR=/home/cilab/shreyas/codes/multinerf/ckpts2/"$EXPERIMENT"/"$SCENE"

# rm "$CHECKPOINT_DIR"/*
# python -m train \
#   --gin_configs=configs/llff_raw.gin \
#   --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
#   --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
#   --logtostderr
  
# SCENE=parkstatue
# EXPERIMENT=raw
# DATA_DIR=/media/cilab/data/RawNerf/rawnerf/scenes
# CHECKPOINT_DIR=/home/cilab/shreyas/codes/multinerf/ckpts2/"$EXPERIMENT"/"$SCENE"

# rm "$CHECKPOINT_DIR"/*
# python -m train \
#   --gin_configs=configs/llff_raw.gin \
#   --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
#   --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
#   --logtostderr
  
# SCENE=scooter
# EXPERIMENT=raw
# DATA_DIR=/media/cilab/data/RawNerf/rawnerf/scenes
# CHECKPOINT_DIR=/home/cilab/shreyas/codes/multinerf/ckpts2/"$EXPERIMENT"/"$SCENE"

# rm "$CHECKPOINT_DIR"/*
# python -m train \
#   --gin_configs=configs/llff_raw.gin \
#   --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
#   --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
#   --logtostderr
  
# SCENE=sharpshadow
# EXPERIMENT=raw
# DATA_DIR=/media/cilab/data/RawNerf/rawnerf/scenes
# CHECKPOINT_DIR=/home/cilab/shreyas/codes/multinerf/ckpts2/"$EXPERIMENT"/"$SCENE"

# rm "$CHECKPOINT_DIR"/*
# python -m train \
#   --gin_configs=configs/llff_raw.gin \
#   --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
#   --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
#   --logtostderr
  
# SCENE=stove
# EXPERIMENT=raw
# DATA_DIR=/media/cilab/data/RawNerf/rawnerf/scenes
# CHECKPOINT_DIR=/home/cilab/shreyas/codes/multinerf/ckpts2/"$EXPERIMENT"/"$SCENE"

# rm "$CHECKPOINT_DIR"/*
# python -m train \
#   --gin_configs=configs/llff_raw.gin \
#   --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
#   --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
#   --logtostderr
  
# SCENE=streetcorner
# EXPERIMENT=raw
# DATA_DIR=/media/cilab/data/RawNerf/rawnerf/scenes
# CHECKPOINT_DIR=/home/cilab/shreyas/codes/multinerf/ckpts2/"$EXPERIMENT"/"$SCENE"

# rm "$CHECKPOINT_DIR"/*
# python -m train \
#   --gin_configs=configs/llff_raw.gin \
#   --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
#   --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
#   --logtostderr
  
# SCENE=windowlegovary
# EXPERIMENT=raw
# DATA_DIR=/media/cilab/data/RawNerf/rawnerf/scenes
# CHECKPOINT_DIR=/home/cilab/shreyas/codes/multinerf/ckpts2/"$EXPERIMENT"/"$SCENE"

# rm "$CHECKPOINT_DIR"/*
# python -m train \
#   --gin_configs=configs/llff_raw.gin \
#   --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
#   --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
#   --logtostderr
