export DATA_DIR=$(pwd)/data-bin/cc100
export DATA_BIN=${DATA_DIR}/shard0
NUM_DATA_SHARDS=40
for i in $(seq 1 $(($NUM_DATA_SHARDS-1))); do
    DATA_BIN="${DATA_BIN}:${DATA_DIR}/shard${i}";
done
# echo $DATA_BIN
export SERIALIZATION_DIR="/private/home/zeyuliu/proj/demix/8langs/demix_8_GPUs_transformer_lm_gpt3_small_8langs_all_shard"
# set NUM_EVALUATION_GPUS equal to the number of experts you'd like to ensemble.
export NUM_EVALUATION_GPUS=8;
export seen_lang_dir=${SERIALIZATION_DIR}/seen_langs
mkdir -p $seen_lang_dir
# alphabetically sorted: ['de_DE', 'en_XX', 'fr_XX', 'id_ID', 'ja_XX', 'ro_RO', 'ru_RU', 'zh_CN']
# first loop get the data
# for domain in en_XX fr_XX zh_CN ru_RU ja_XX id_ID ro_RO de_DE
# do
#     mkdir -p ${SERIALIZATION_DIR}/${domain}/
#     export DEV_POSTERIOR_OUTPUT="${SERIALIZATION_DIR}/${domain}/dev_posteriors.jsonl"
#     export DOMAIN=$domain
#     bash demix/mix_eval_lm.sh $NUM_EVALUATION_GPUS $DATA_BIN ${SERIALIZATION_DIR}/checkpoint_last-rank-0.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-1.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-2.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-3.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-4.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-5.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-6.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-7.pt $DOMAIN $DEV_POSTERIOR_OUTPUT estimate > ${SERIALIZATION_DIR}/${domain}/estimate_eval_len1024.log 2>&1
# done


for domain in en_XX fr_XX zh_CN ru_RU ja_XX # id_ID ro_RO de_DE
do  
    for ensemble_type in uniform_prior # simple_average cached_prior
    do
        export DEV_POSTERIOR_OUTPUT="${SERIALIZATION_DIR}/${domain}/dev_posteriors.jsonl"
        export DOMAIN=$domain
        export POSTERIOR=$(tail -n 1 $DEV_POSTERIOR_OUTPUT | jq -rc '.exp_avg_posterior | join(",")')
        bash demix/mix_eval_lm.sh $NUM_EVALUATION_GPUS $DATA_BIN  ${SERIALIZATION_DIR}/checkpoint_last-rank-0.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-1.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-2.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-3.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-4.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-5.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-6.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-7.pt $DOMAIN $DEV_POSTERIOR_OUTPUT eval $POSTERIOR $ensemble_type > ${SERIALIZATION_DIR}/${domain}/${ensemble_type}_eval_len1024.log 2>&1
    done
done