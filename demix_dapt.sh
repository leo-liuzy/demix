export DATA_DIR=$(pwd)/data-bin/cc100
export DATA_BIN=${DATA_DIR}/shard0
NUM_DATA_SHARDS=40
for i in $(seq 1 $(($NUM_DATA_SHARDS-1))); do
    DATA_BIN="${DATA_BIN}:${DATA_DIR}/shard${i}";
done
# echo $DATA_BIN
export SERIALIZATION_DIR="/private/home/zeyuliu/proj/demix/8langs/demix_8_GPUs_transformer_lm_gpt3_small_8langs_all_shard"
# set NUM_EVALUATION_GPUS equal to the number of experts you'd like to ensemble.
export NEW_DATA_BIN=$DATA_BIN
export NUM_EVALUATION_GPUS=8;
export new_lang_dir=$SERIALIZATION_DIR/unseen_langs
mkdir -p $new_lang_dir

# alphabetically sorted: ['de_DE', 'en_XX', 'fr_XX', 'id_ID', 'ja_XX', 'ro_RO', 'ru_RU', 'zh_CN']
for new_domain in nl_XX fy_NL ko_KR my_MM es_XX gl_ES pl_PL be_BY
do  
    echo "Language: ${new_domain}"
    mkdir -p ${new_lang_dir}/${new_domain}
    export DEV_POSTERIOR_OUTPUT="${new_lang_dir}/${new_domain}/dev_posteriors.jsonl"
    bash demix/mix_eval_lm.sh $NUM_EVALUATION_GPUS $NEW_DATA_BIN  ${SERIALIZATION_DIR}/checkpoint_last-rank-0.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-1.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-2.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-3.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-4.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-5.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-6.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-7.pt $new_domain $DEV_POSTERIOR_OUTPUT estimate > ${new_lang_dir}/${new_domain}/estimate_eval_len1024.log 2>&1
    export POSTERIOR=$(tail -n 1 $DEV_POSTERIOR_OUTPUT | jq -rc '.exp_avg_posterior | join(",")')
    python scripts/find_best_expert_for_unseen_langs.py $POSTERIOR
done
# set NUM_EVALUATION_GPUS equal to the number of experts you'd like to ensemble.
