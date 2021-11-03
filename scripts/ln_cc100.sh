# This repository requires folders to be in format of "$DATA_PATH/$DOMAIN/shards/shard_k/"
# cc100 is currently structured in "$DATA_PATH/shard_k/$LANGUAGE"

CC100_DATA_PATH="/datasets01/cc100-bin/072820/250"
TARGET_DATA_SOURCE="data-bin"
# mkdir -p $TARGET_DATA_SOURCE
TARGET_DATA_PATH="${TARGET_DATA_SOURCE}/cc100"
mkdir -p $TARGET_DATA_PATH

for shard_subdir in $CC100_DATA_PATH/*/     # list directories in the form "/tmp/dirname/"
do  
    # echo $shard_subdir
    shard_subdir=${shard_subdir%*/}      # remove the trailing "/"
    shard="${shard_subdir##*/}"    # print everything after the final "/"
    target_shard_dir=${TARGET_DATA_PATH}/$shard
    mkdir -p $target_shard_dir
    ln -sf $shard_subdir/dict.txt $target_shard_dir/dict.txt
    # echo $target_shard_dir
    for lang_code in gl_ES # nl_XX fy_NL ko_KR my_MM es_XX gl_ES pl_PL be_BY # en_XX fr_XX zh_CN ru_RU ja_XX id_ID ro_RO de_DE  # ${shard_subdir}/*/
    do
        # lang_dir=${lang_dir%*/} 
        lang_dir=$shard_subdir/$lang_code
        # echo $lang_dir
        lang_code=${lang_dir##*/}
        target_lang_dir=${target_shard_dir}/$lang_code
        # echo $lang_code
        echo $target_lang_dir
        mkdir -p $target_lang_dir
        ln -sf $lang_dir/dict.txt $target_lang_dir/dict.txt
        ln -sf $lang_dir/train.bin $target_lang_dir/train.bin
        ln -sf $lang_dir/train.idx $target_lang_dir/train.idx
        ln -sf $lang_dir/valid.bin $target_lang_dir/valid_${lang_code}.bin
        ln -sf $lang_dir/valid.idx $target_lang_dir/valid_${lang_code}.idx
        ln -sf $lang_dir/valid.bin $target_lang_dir/test_${lang_code}.bin
        ln -sf $lang_dir/valid.idx $target_lang_dir/test_${lang_code}.idx
    done
    # exit
done