#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import os
import re
from lang2vec import lang2vec
import torch
import json
from scipy.special import softmax
from scipy.spatial import distance as spatial_distance
from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd 

TYPOLOGICAL_FEATURES = list(lang2vec.FEATURE_SETS_DICT.keys())
TYPOLOGICAL_DISTANCE_METRICS = lang2vec.DISTANCES
LETTER_CODES = lang2vec.LETTER_CODES
DIVERGENCE_METRICS = ["jensen-shannon", "earth-mover"]
EPS=1e-8

def string2list(input_string, eps=1e-5):
    try:
        prob_dist = list(map(float, input_string.split(",")))
        assert -eps <= sum(prob_dist) - 1 <= eps
    except:
        raise ValueError(f"Invalid Input string: {input_string}")
    return prob_dist

def divergence_func_from_name(name):
    if name == "jensen-shannon":
        return spatial_distance.jensenshannon
    elif name == "earth-mover":
        return wasserstein_distance
    else:
        raise ValueError(f"Unsupported divergence metric: {name}")

def main(eps=EPS):
    parser = argparse.ArgumentParser(
        description="Tool to average the params of input checkpoints to "
        "produce a new checkpoint",
    )
    # fmt: off
    parser.add_argument('--seen-languages', required=True,
                        default='en_XX,fr_XX,zh_CN,ru_RU,ja_XX,id_ID,ro_RO,de_DE', type=str,
                        help='Comma seperated list of seen languages (e.g. language used on training).')
    parser.add_argument('--target-language', required=True,
                        help='Target language of interest')
    parser.add_argument('--demix-posterior', required=True,
                        help='posterior distribution from DEMix; Expected input is a comma seperated and valid probability distribution')
    # parser.add_argument('--divergence-metric', required=True,
    #                     help='metric used for measuring divergence between two probability distribution')
    parser.add_argument('--result-dir', help="This script will save intermediate search results in a table and"\
                        "save this table to the provided directory")
    # fmt: on
    args = parser.parse_args()
    print(args)
    assert os.path.exists(args.result_dir)
    assert len(args.seen_languages) > 0 and ',' in args.seen_languages
    seen_langs = sorted(args.seen_languages.split(','))
    uniform_dist = np.ones(len(seen_langs)) / len(seen_langs)
    demix_posterior = string2list(args.demix_posterior)
    results = []
    header = ["seen_langs", 'target_lang', 'demix_posterior', 'typological_distance_metric', 'typological_distance', 
              'typological_similarity', 'approximated_posterior', 'divergence_metric', 'divergence', 'divergence_with_uniform']
    assert all('_' in lang for lang in seen_langs)
    assert '_' in args.target_language
    seen_langs = list(map(lambda x: LETTER_CODES[x.split('_')[0]], seen_langs))
    # print(seen_langs)
    target_lang = LETTER_CODES[args.target_language.split('_')[0]]
        
    for divergence_metric in DIVERGENCE_METRICS:
        print(f"Using {divergence_metric} divergence metric:")
        divergence_func = divergence_func_from_name(divergence_metric)
        divergence_with_uniform = divergence_func(demix_posterior, uniform_dist)

        min_divergence = float('inf')
        min_typological_distance_metric = None
        min_approximated_posterior = None
        distance_metric2langs = {}
        
        for distance_metric in TYPOLOGICAL_DISTANCE_METRICS:
            distance_matrix = lang2vec.distance(distance_metric, [target_lang] + seen_langs)
            # first 0 is selecting the first row; 1 is truncating the target_lang
            if distance_metric
            distance = distance_matrix[0][1:]
            # print(f"{distance}: distance")
            similarity = 1 - distance + eps # turn distance measure to similarity measure since distance is in [0, 1], eps added for case of all-zero
            approximated_posterior = similarity / similarity.sum() # softmax(similarity)
            divergence = divergence_func(demix_posterior, approximated_posterior)
            if divergence < min_divergence:
                min_divergence = divergence
                min_typological_distance_metric = distance_metric
                min_approximated_posterior = approximated_posterior
            results.append([args.seen_languages, args.target_language, demix_posterior, distance_metric, distance,
                            similarity, approximated_posterior, divergence_metric, divergence, divergence_with_uniform])

        print(f"min_typological_distance_metric: {min_typological_distance_metric}")
        print(f"min_divergence: {min_divergence}")
        print(f"divergence_with_uniform: {divergence_with_uniform}")
        print(f"min_approximated_posterior: {min_approximated_posterior}")
        print(f"demix_posterior: {demix_posterior}")
        print()
    
    df = pd.DataFrame(results, columns=header)
    df.to_csv(f"{args.result_dir}/typological_search_results.tsv", sep="\t")
    






if __name__ == "__main__":
    main()
