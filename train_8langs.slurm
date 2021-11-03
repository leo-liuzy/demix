#!/bin/sh

export NUM_GPUS=8
export DISTRIBUTED_PORT=12345
export MODEL=transformer_lm_gpt3_small
export EXPERIMENT=demix
export DATA_DIR=$(pwd)/data-bin
export DATA_BIN=${DATA_DIR}/cc100/shard0
export EXPERIMENT_SUFFIX=8langs_shard0
export SERIALIZATION_DIR=$(pwd)/8langs

bash demix/train_8langs.sh $NUM_GPUS \
                    $DISTRIBUTED_PORT \
                    $MODEL \
                    $EXPERIMENT \
                    $DATA_BIN \
                    $SERIALIZATION_DIR \
                    $EXPERIMENT_SUFFIX
