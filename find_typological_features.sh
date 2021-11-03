# export DATA_DIR=$(pwd)/data-bin/cc100
# export DATA_BIN=${DATA_DIR}/shard0
# NUM_DATA_SHARDS=40
# for i in $(seq 1 $(($NUM_DATA_SHARDS-1))); do
#     DATA_BIN="${DATA_BIN}:${DATA_DIR}/shard${i}";
# done
# echo $DATA_BIN
export SERIALIZATION_DIR="8langs/demix_8_GPUs_transformer_lm_gpt3_small_8langs_all_shard"
# set NUM_EVALUATION_GPUS equal to the number of experts you'd like to ensemble.
export NEW_DATA_BIN=$DATA_BIN
export NUM_EVALUATION_GPUS=8;
export NEW_LANG_DIR=$SERIALIZATION_DIR/unseen_langs
export SEEN_LANGS=en_XX,fr_XX,zh_CN,ru_RU,ja_XX,id_ID,ro_RO,de_DE
# alphabetically sorted: ['de_DE', 'en_XX', 'fr_XX', 'id_ID', 'ja_XX', 'ro_RO', 'ru_RU', 'zh_CN']
# for NEW_DOMAIN in nl_XX fy_NL ko_KR my_MM es_XX gl_ES pl_PL be_BY
# do  
#     echo "Language: ${NEW_DOMAIN}"
#     export DEV_POSTERIOR_OUTPUT="${NEW_LANG_DIR}/${NEW_DOMAIN}/dev_posteriors.jsonl"
#     # bash demix/mix_eval_lm.sh $NUM_EVALUATION_GPUS $NEW_DATA_BIN  ${SERIALIZATION_DIR}/checkpoint_last-rank-0.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-1.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-2.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-3.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-4.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-5.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-6.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-7.pt $NEW_DOMAIN $DEV_POSTERIOR_OUTPUT estimate > ${NEW_LANG_DIR}/${NEW_DOMAIN}/estimate_eval_len1024.log 2>&1
#     export POSTERIOR=$(tail -n 1 $DEV_POSTERIOR_OUTPUT | jq -rc '.exp_avg_posterior | join(",")')
#     python scripts/find_best_matching_typological_feature.py --seen-languages $SEEN_LANGS --demix-posterior $POSTERIOR --target-language $NEW_DOMAIN --result-dir ${NEW_LANG_DIR}/${NEW_DOMAIN}
#     echo
# done
# set NUM_EVALUATION_GPUS equal to the number of experts you'd like to ensemble.
python scripts/find_best_matching_typological_feature.py --seen-languages $SEEN_LANGS --target-languages nl_XX,fy_NL,ko_KR,my_MM,es_XX,gl_ES,pl_PL,be_BY --unseen-langs-dir ${NEW_LANG_DIR}