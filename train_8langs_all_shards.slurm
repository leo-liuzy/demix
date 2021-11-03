#!/bin/sh

export NUM_GPUS=8
export DISTRIBUTED_PORT=12345
export MODEL=transformer_lm_gpt3_small
export EXPERIMENT=domain_token
export DATA_DIR=$(pwd)/data-bin/cc100
export DATA_BIN=${DATA_DIR}/shard0
NUM_DATA_SHARDS=40
for i in $(seq 1 $(($NUM_DATA_SHARDS-1))); do
    DATA_BIN="${DATA_BIN}:${DATA_DIR}/shard${i}";
done
echo $DATA_BIN

export EXPERIMENT_SUFFIX=8langs_all_shard
export SERIALIZATION_DIR=$(pwd)/8langs

bash demix/train_8langs.sh $NUM_GPUS \
                    $DISTRIBUTED_PORT \
                    $MODEL \
                    $EXPERIMENT \
                    $DATA_BIN \
                    $SERIALIZATION_DIR \
                    $EXPERIMENT_SUFFIX
