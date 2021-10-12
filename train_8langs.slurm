#!/bin/sh
#SBATCH --job-name=demix_8_GPUs_transformer_lm_8langs_shard0
#SBATCH --output=/checkpoint/zeyuliu/2021-10-12/demix_8_GPUs_transformer_lm_8langs_shard0/train.log 
#SBATCH --error=/checkpoint/zeyuliu/2021-09-27/demix_8_GPUs_transformer_lm_8langs_shard0/train.stderr.%j
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10

#SBATCH --time=4320
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180 
#SBATCH --partition=learnaccel
#SBATCH --comment='Code Location: ./slurm_snapshot_code/2021-10-12T01_00_00.000000'
#SBATCH --mem=0

# trap_handler () {
#    echo "Caught signal: " $1
#    # SIGTERM must be bypassed
#    if [ "$1" = "TERM" ]; then
#        echo "bypass sigterm"
#    else
#      # Submit a new job to the queue
#      echo "Requeuing " $SLURM_JOB_ID
#      scontrol requeue $SLURM_JOB_ID
#    fi
# }


# # Install signal handler
# trap '"'"'trap_handler USR1'"'"' USR1
# trap '"'"'trap_handler TERM'"'"' TERM
source /etc/profile.d/modules.sh
source /private/home/zeyuliu/.set_up_fairseq_env.sh
## demix_8_GPUs_transformer_lm_8langs

export NUM_GPUS=8
export DISTRIBUTED_PORT=12345
export MODEL=transformer_lm
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
