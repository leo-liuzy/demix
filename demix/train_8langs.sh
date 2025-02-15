
# Number of GPUs you'd like to train on
NUM_GPUS=$1
# Number of nodes you'd like to train on (assuming 8 GPUs per node)
NUM_NODES=$((${NUM_GPUS}/8))
# Distributed port
PORT=$2
# Fairseq model name (e.g. transformer_lm; see https://github.com/kernelmachine/demix/blob/main/fairseq/models/transformer_lm.py for other options)
ARCH=$3
# Baseline type: choice between demix, dense, unbalanced_dense, and domain_token
EXPERIMENT=$4
# Path to data-bins
DATA_PATH=$5
# path to directory to where you'd like to output the model
SERIALIZATION_DIR=$6
# suffix to append to model output (e.g. "test", "final")
FILE_SUFFIX=$7


# list of domains you'd like to train on, that can be found in $DATA_PATH
domains=en_XX,fr_XX,zh_CN,ru_RU,ja_XX,id_ID,ro_RO,de_DE;
# validation datasets for each domain
valid_subset=valid_en_XX,valid_fr_XX,valid_zh_CN,valid_ru_RU,valid_ja_XX,valid_id_ID,valid_ro_RO,valid_de_DE;
# name of wandb project to track model output (at wandb.ai)
WANDB_PROJECT=domain_token_gpt3_small_experiments;


TOKENS_PER_SAMPLE=1024;
BATCH_SIZE=2;
LOG_INTERVAL=50;
KEEP_INTERVAL_UPDATES=1;


if [[ $ARCH == *"gpt3_small"* ]]; then
     LR=5e-4;
     CLIP_NORM=0.1;
     UPDATE_FREQ=8;
     NUM_STEPS=300000;
     SAVE_INTERVAL_UPDATES=6000;
     VALIDATION_INTERVAL=3000;
     NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));
elif [[ $ARCH == *"gpt3_medium"* ]]; then
     LR=5e-4;
     NUM_STEPS=120000;
     NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));
     SAVE_INTERVAL_UPDATES=3000;
     VALIDATION_INTERVAL=2000;
     CLIP_NORM=0.1;
     UPDATE_FREQ=8;
elif [[ $ARCH == *"gpt3_large"* ]]; then
     LR=5e-4;
     NUM_STEPS=65000;
     NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));
     SAVE_INTERVAL_UPDATES=2000;
     VALIDATION_INTERVAL=1000;
     CLIP_NORM=0.1;
     UPDATE_FREQ=8;
elif [[ $ARCH == *"gpt3_xl"* ]]; then
     LR=5e-4;
     NUM_STEPS=50000;
     NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));
     SAVE_INTERVAL_UPDATES=2000;
     VALIDATION_INTERVAL=500;
     CLIP_NORM=0.1;
     UPDATE_FREQ=8;
elif [[ $ARCH == *"transformer_lm"* ]]; then
     TOKENS_PER_SAMPLE=1024;
     LR=5e-4;
     CLIP_NORM=0.1;
     UPDATE_FREQ=8;
     NUM_STEPS=600000;
     SAVE_INTERVAL_UPDATES=12000;
     VALIDATION_INTERVAL=6000;
     NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));
fi;

# $DATA_PARALLEL_GROUPS identifies which ranks we will synchronize over. "A,B C,D" means we will synchronize ranks A,B and synchronize ranks C,D.
if [[ $NUM_GPUS == "8" ]]; then
     if [[ $EXPERIMENT == *"dense"*  || $EXPERIMENT == *"domain_token"* ]]; then
          DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7";
     elif  [[ $EXPERIMENT == *"demix"* ]]; then
          DATA_PARALLEL_GROUPS="0 1 2 3 4 5 6 7";
     fi;
elif [[ $NUM_GPUS == "16" ]]; then
     if [[ $EXPERIMENT == *"dense"*  || $EXPERIMENT == *"domain_token"* ]]; then
          DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15";
     elif  [[ $EXPERIMENT == *"demix"* ]]; then
          DATA_PARALLEL_GROUPS="0,1 2,3 4,5 6,7 8,9 10,11 12,13 14,15";
     fi;
elif [[ $NUM_GPUS == "32" ]]; then
     if [[ $EXPERIMENT == *"dense"*  || $EXPERIMENT == *"domain_token"* ]]; then
          DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31";
     elif  [[ $EXPERIMENT == *"demix"* ]]; then
          DATA_PARALLEL_GROUPS="0,1,2,3 4,5,6,7 8,9,10,11 12,13,14,15 16,17,18,19 20,21,22,23 24,25,26,27 28,29,30,31";
     fi;
elif [[ $NUM_GPUS == "64" ]]; then
     if [[ $EXPERIMENT == *"dense"*  || $EXPERIMENT == *"domain_token"* ]]; then
     	DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63";
     elif  [[ $EXPERIMENT == *"demix"* ]]; then
	     DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7 8,9,10,11,12,13,14,15 16,17,18,19,20,21,22,23 24,25,26,27,28,29,30,31 32,33,34,35,36,37,38,39 40,41,42,43,44,45,46,47 48,49,50,51,52,53,54,55 56,57,58,59,60,61,62,63";
     fi;
elif [[ $NUM_GPUS == "128" ]]; then
     if [[ $EXPERIMENT == *"dense"*  || $EXPERIMENT == *"domain_token"* ]]; then
     	DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127";
     elif  [[ $EXPERIMENT == *"demix"* ]]; then
	     DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7 8,9,10,11,12,13,14,15 16,17,18,19,20,21,22,23 24,25,26,27,28,29,30,31 32,33,34,35,36,37,38,39 40,41,42,43,44,45,46,47 48,49,50,51,52,53,54,55 56,57,58,59,60,61,62,63 64,65,66,67,68,69,70,71 72,73,74,75,76,77,78,79 80,81,82,83,84,85,86,87 88,89,90,91,92,93,94,95 96,97,98,99,100,101,102,103 104,105,106,107,108,109,110,111 112,113,114,115,116,117,118,119 120,121,122,123,124,125,126,127";
     fi;
fi;

if [[ $EXPERIMENT == *"demix"* ]]; then
     srun --label python fairseq_cli/train.py     $DATA_PATH \
          --task multidomain_language_modeling \
          --multidomain-sampling-alpha 0.7 \
          --sample-break-mode none \
          --log-format simple  \
          --log-interval $LOG_INTERVAL    \
          --skip-invalid-size-inputs-valid-test     \
          --validate-interval-updates $VALIDATION_INTERVAL     \
          --save-interval-updates $SAVE_INTERVAL_UPDATES     \
          --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
          --arch $ARCH    \
	     --criterion desynchronized_cross_entropy     \
          --lr-scheduler polynomial_decay     \
          --num-workers 2 \
          --max-sentences $BATCH_SIZE \
          --no-epoch-checkpoints \
          --max-sentences-valid $BATCH_SIZE \
          --lr $LR              \
          --tokens-per-sample $TOKENS_PER_SAMPLE          \
          --optimizer adam \
          --adam-betas '(0.9, 0.95)'  \
          --adam-eps 10e-8 \
          --weight-decay 0.1 \
          --clip-norm $CLIP_NORM      \
          --max-update $NUM_STEPS     \
          --total-num-update $NUM_STEPS     \
          --warmup-updates $NUM_WARMUP_STEPS     \
          --update-freq $UPDATE_FREQ     \
          --save-dir ${SERIALIZATION_DIR}/demix_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}      \
          --batch-size-valid 2                        \
          --wandb-project $WANDB_PROJECT           \
          --valid-subset $valid_subset \
          --train-domains $domains  \
          --eval-domains $domains \
          --required-batch-size-multiple 1 \
          --memory-efficient-fp16 \
          --distributed-world-size $NUM_GPUS \
          --distributed-port $PORT \
          --desynchronize --domain-parallel \
          --ddp-backend no_c10d \
          --sync-type manual \
          --untie-parameters feedforward \
          --data-parallel-groups "${DATA_PARALLEL_GROUPS}" \
          --seed 1 \
          --all-gather-list-size 32000;
elif [[ $EXPERIMENT == *"unbalanced"* ]]; then
     srun --label python fairseq_cli/train.py     $DATA_PATH \
               --task multidomain_language_modeling \
               --sample-break-mode none \
               --log-format simple  \
               --log-interval $LOG_INTERVAL    \
               --skip-invalid-size-inputs-valid-test     \
               --validate-interval-updates $VALIDATION_INTERVAL     \
               --save-interval-updates $SAVE_INTERVAL_UPDATES     \
               --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
               --arch $ARCH    \
               --criterion desynchronized_cross_entropy     \
               --lr-scheduler polynomial_decay     \
               --num-workers 2 \
               --max-sentences $BATCH_SIZE \
               --no-epoch-checkpoints \
               --max-sentences-valid $BATCH_SIZE \
               --lr $LR              \
               --tokens-per-sample $TOKENS_PER_SAMPLE          \
               --optimizer adam \
               --adam-betas '(0.9, 0.95)'  \
               --adam-eps 10e-8 \
               --weight-decay 0.1 \
               --clip-norm $CLIP_NORM      \
               --max-update $NUM_STEPS     \
               --total-num-update $NUM_STEPS     \
               --warmup-updates $NUM_WARMUP_STEPS     \
               --update-freq $UPDATE_FREQ     \
               --save-dir ${SERIALIZATION_DIR}/unbalanced_dense_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}      \
               --batch-size-valid 2                        \
               --wandb-project $WANDB_PROJECT           \
               --valid-subset $valid_subset \
               --train-domains $domains  \
               --eval-domains $domains \
               --required-batch-size-multiple 1 \
               --memory-efficient-fp16 \
               --distributed-world-size $NUM_GPUS \
               --distributed-port $PORT \
               --all-gather-list-size 32000 \
               --unbalanced;
elif [[ $EXPERIMENT == *"dense"* ]]; then
     srun --label python fairseq_cli/train.py     $DATA_PATH \
               --task multidomain_language_modeling \
               --sample-break-mode none \
               --log-format simple  \
               --log-interval $LOG_INTERVAL    \
               --skip-invalid-size-inputs-valid-test     \
               --validate-interval-updates $VALIDATION_INTERVAL     \
               --save-interval-updates $SAVE_INTERVAL_UPDATES     \
               --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
               --arch $ARCH    \
               --criterion desynchronized_cross_entropy     \
               --lr-scheduler polynomial_decay     \
               --num-workers 2 \
               --max-sentences $BATCH_SIZE \
               --no-epoch-checkpoints \
               --max-sentences-valid $BATCH_SIZE \
               --lr $LR              \
               --tokens-per-sample $TOKENS_PER_SAMPLE          \
               --optimizer adam \
               --adam-betas '(0.9, 0.95)'  \
               --adam-eps 10e-8 \
               --weight-decay 0.1 \
               --clip-norm $CLIP_NORM      \
               --max-update $NUM_STEPS     \
               --total-num-update $NUM_STEPS     \
               --warmup-updates $NUM_WARMUP_STEPS     \
               --update-freq $UPDATE_FREQ     \
               --save-dir ${SERIALIZATION_DIR}/dense_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}      \
               --batch-size-valid 2                        \
               --wandb-project $WANDB_PROJECT           \
               --valid-subset $valid_subset \
               --train-domains $domains  \
               --eval-domains $domains \
               --required-batch-size-multiple 1 \
               --memory-efficient-fp16 \
               --distributed-world-size $NUM_GPUS \
               --distributed-port $PORT \
               --all-gather-list-size 32000;
elif [[ $EXPERIMENT == *"switch"* ]]; then
     srun --label python fairseq_cli/train.py     \
               $DATA_PATH     \
               --task multidomain_language_modeling     \
               --sample-break-mode none     \
               --log-format simple     \
               --log-interval $LOG_INTERVAL    \
               --skip-invalid-size-inputs-valid-test     \
               --validate-interval-updates $VALIDATION_INTERVAL     \
               --save-interval-updates $SAVE_INTERVAL_UPDATES     \
               --keep-interval-updates $KEEP_INTERVAL_UPDATES     \
               --no-epoch-checkpoints \
               --arch $ARCH    \
               --criterion moe_cross_entropy     \
               --lr-scheduler polynomial_decay     \
               --lr $LR              \
               --tokens-per-sample 1024          \
               --optimizer adam \
               --adam-betas '(0.9, 0.95)'  \
               --adam-eps 10e-8 \
               --weight-decay 0.1 \
               --num-workers 2 \
               --max-sentences $BATCH_SIZE \
               --max-sentences-valid $BATCH_SIZE \
               --clip-norm $CLIP_NORM      \
               --max-update $NUM_STEPS     \
               --total-num-update $NUM_STEPS     \
               --warmup-updates $NUM_WARMUP_STEPS     \
               --wandb-project $WANDB_PROJECT \
               --save-dir ${SERIALIZATION_DIR}/switch_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}         \
               --batch-size-valid 2                        \
               --train-domains $domains \
               --eval-domains $domains \
               --valid-subset $valid_subset \
               --required-batch-size-multiple 1 \
               --update-freq $UPDATE_FREQ \
               --fp16 \
               --fp16-no-flatten-grads \
               --moe-freq 2 \
               --moe-top1-expert \
               --moe-expert-count $NUM_GPUS \
               --moe-gating-use-fp32 \
               --moe-gate-loss-wt 0.01 \
               --moe-gate-loss-combine-method sum \
               --moe-second-expert-policy all \
               --distributed-world-size $NUM_GPUS \
               --distributed-port $PORT \
               --ddp-backend no_c10d \
               --all-gather-list-size 32000;
elif [[ $EXPERIMENT == *"gshard"* ]]; then
     srun --label python fairseq_cli/train.py     \
               $DATA_PATH     \
               --task multidomain_language_modeling     \
               --sample-break-mode none     \
               --log-format simple     \
               --log-interval $LOG_INTERVAL    \
               --skip-invalid-size-inputs-valid-test     \
               --validate-interval-updates $VALIDATION_INTERVAL     \
               --save-interval-updates $SAVE_INTERVAL_UPDATES     \
               --keep-interval-updates $KEEP_INTERVAL_UPDATES     \
               --no-epoch-checkpoints \
               --arch $ARCH    \
               --criterion moe_cross_entropy     \
               --lr-scheduler polynomial_decay     \
               --lr $LR               \
               --tokens-per-sample 1024          \
               --optimizer adam \
               --adam-betas '(0.9, 0.95)'  \
               --adam-eps 10e-8 \
               --weight-decay 0.1 \
               --num-workers 2 \
               --max-sentences $BATCH_SIZE \
               --max-sentences-valid $BATCH_SIZE \
               --clip-norm $CLIP_NORM      \
               --max-update $NUM_STEPS     \
               --total-num-update $NUM_STEPS     \
               --warmup-updates $NUM_WARMUP_STEPS     \
               --wandb-project $WANDB_PROJECT \
               --save-dir ${SERIALIZATION_DIR}/gshard_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}         \
               --batch-size-valid 2                        \
               --train-domains $domains \
               --eval-domains $domains \
               --valid-subset $valid_subset \
               --required-batch-size-multiple 1 \
               --update-freq $UPDATE_FREQ \
               --fp16 \
               --fp16-no-flatten-grads \
               --moe-freq 2 \
               --moe-expert-count $NUM_GPUS \
               --moe-gating-use-fp32 \
               --moe-gate-loss-wt 0.01 \
               --moe-gate-loss-combine-method sum \
               --moe-second-expert-policy all \
               --distributed-world-size $NUM_GPUS \
               --distributed-port $PORT \
               --ddp-backend no_c10d \
               --all-gather-list-size 32000;
elif [[ $EXPERIMENT == *"domain_token"* ]]; then
     # domain token
     srun --label python fairseq_cli/train.py     $DATA_PATH \
          --task multidomain_language_modeling \
          --sample-break-mode none \
          --log-format simple  \
          --log-interval $LOG_INTERVAL    \
          --skip-invalid-size-inputs-valid-test     \
          --validate-interval-updates $VALIDATION_INTERVAL     \
          --save-interval-updates $SAVE_INTERVAL_UPDATES     \
          --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
          --arch $ARCH    \
	     --criterion desynchronized_cross_entropy     \
          --lr-scheduler polynomial_decay     \
          --num-workers 2 \
          --max-sentences $BATCH_SIZE \
          --no-epoch-checkpoints \
          --max-sentences-valid $BATCH_SIZE \
          --lr $LR              \
          --tokens-per-sample $TOKENS_PER_SAMPLE          \
          --optimizer adam \
          --adam-betas '(0.9, 0.95)'  \
          --adam-eps 10e-8 \
          --weight-decay 0.1 \
          --clip-norm $CLIP_NORM      \
          --max-update $NUM_STEPS     \
          --total-num-update $NUM_STEPS     \
          --warmup-updates $NUM_WARMUP_STEPS     \
          --update-freq $UPDATE_FREQ     \
          --save-dir ${SERIALIZATION_DIR}/domain_token_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}      \
          --batch-size-valid 2                        \
          --wandb-project $WANDB_PROJECT           \
          --valid-subset $valid_subset \
          --train-domains $domains  \
          --eval-domains $domains \
          --required-batch-size-multiple 1 \
          --memory-efficient-fp16 \
          --distributed-world-size $NUM_GPUS \
          --distributed-port $PORT \
          --all-gather-list-size 32000 \
          --add-domain-token;
fi;
