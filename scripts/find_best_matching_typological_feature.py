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
import matplotlib.pyplot as plt
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
    parser.add_argument('--target-languages', required=True,
                        help='Comma seperated list of interested target language')
    # parser.add_argument('--demix-posterior', required=True,
    #                     help='posterior distribution from DEMix; Expected input is a comma seperated and valid probability distribution')
    # parser.add_argument('--divergence-metric', required=True,
    #                     help='metric used for measuring divergence between two probability distribution')
    parser.add_argument('--unseen-langs-dir', help="This script will save intermediate search results in a table and"\
                        "save this table to the provided directory")
    # fmt: on
    args = parser.parse_args()
    print(args)
    from pdb import set_trace; set_trace()
    assert os.path.exists(args.unseen_langs_dir)
    assert len(args.seen_languages) > 0 and ',' in args.seen_languages
    seen_langs = sorted(args.seen_languages.split(','))
    target_langs = sorted(args.target_languages.split(',')) # 
    demix_posteriors = []
    for target_lang in target_langs:
        dev_posterior_filepath = f"{args.unseen_langs_dir}/{target_lang}/dev_posteriors.jsonl"
        dev_posteriors = open(dev_posterior_filepath, "r").readlines()
        dev_posterior = json.loads(dev_posteriors[-1])['exp_avg_posterior']
        demix_posteriors.append(dev_posterior)
        
    uniform_dist = np.ones(len(seen_langs)) / len(seen_langs)
    # demix_posterior = string2list(args.demix_posterior)
    results = []
    header = ["seen_langs", 'target_lang', 'demix_posterior', 'typological_distance_metric', 'typological_distance', 
              'typological_similarity', 'approximated_posterior', 'divergence_metric', 'divergence', 'divergence_with_uniform']
    assert all('_' in lang for lang in seen_langs)
    assert all('_' in lang for lang in target_langs)
    # assert '_' in args.target_language
    seen_langs = list(map(lambda x: LETTER_CODES[x.split('_')[0]], seen_langs))
    # target_langs = list(map(lambda x: LETTER_CODES[x.split('_')[0]], target_langs))
        
    for divergence_metric in DIVERGENCE_METRICS:
        print(f"Using {divergence_metric} divergence metric:")
        divergence_func = divergence_func_from_name(divergence_metric)
        distance_metric2langs = {m: [] for m in TYPOLOGICAL_DISTANCE_METRICS}
        lang2hitmiss = {}
        for i, target_lang_fb in enumerate(target_langs[:2]):
            print(f"Target language: {target_lang_fb}")
            assert '_' in target_lang_fb  # fb lang code has a '_' in it
            target_lang = LETTER_CODES[target_lang_fb.split('_')[0]]
            demix_posterior = demix_posteriors[i]

            divergence_with_uniform = divergence_func(demix_posterior, uniform_dist)
            min_divergence = float('inf')
            min_typological_distance_metric = None
            min_approximated_posterior = None
            for distance_metric in TYPOLOGICAL_DISTANCE_METRICS:
                distance_matrix = lang2vec.distance(distance_metric, [target_lang] + seen_langs)
                # first 0 is selecting the first row; 1 is truncating the target_lang
                distance = distance_matrix[0][1:]
                # print(f"{distance}: distance")
                similarity = 1 - distance + eps # turn distance measure to similarity measure since distance is in [0, 1], eps added for case of all-zero
                approximated_posterior = similarity / similarity.sum() # softmax(similarity)
                divergence = divergence_func(demix_posterior, approximated_posterior)
                if divergence < min_divergence:
                    min_divergence = divergence
                    min_typological_distance_metric = distance_metric
                    min_approximated_posterior = approximated_posterior
                results.append([args.seen_languages, target_lang_fb, demix_posterior, distance_metric, distance,
                                similarity, approximated_posterior, divergence_metric, divergence, divergence_with_uniform])
            # here, count how many times one metric is best for one language
            distance_metric2langs[distance_metric].append(target_lang_fb)
            lang2hitmiss[target_lang_fb] = [distance_metric == min_typological_distance_metric for distance_metric in TYPOLOGICAL_DISTANCE_METRICS]

            print(f"min_typological_distance_metric: {min_typological_distance_metric}")
            print(f"min_divergence: {min_divergence}")
            print(f"divergence_with_uniform: {divergence_with_uniform}")
            print(f"min_approximated_posterior: {min_approximated_posterior}")
            print(f"demix_posterior: {demix_posterior}")
            print()
        # Different divergence metric leads to different best distance metric, so plot it for different divergence metric
        plt.clf()
        plt.title(f"Best distance metric by {divergence_metric} divergence")
        plt.bar(distance_metric2langs.keys(), [len(v) for v in distance_metric2langs.values()])
        plt.xlabel("Name of distance metric (in lang2vec)")
        plt.ylabel("No. of times seelcted as the best matching metric")
        plt.savefig(f"{args.unseen_langs_dir}/best_distance_metric_stats_by_{divergence_metric}.pdf", format="pdf")
        plt.show()

        # also have a scatter plot
        plt.clf()
        lang2hitmiss = pd.DataFrame(lang2hitmiss)
        lang2hitmiss = lang2hitmiss.transpose()
        # print(lang2hitmiss)
        lang2hitmiss.columns = TYPOLOGICAL_DISTANCE_METRICS
        plt.title(f"Best distance metric by {divergence_metric} divergence")
        heatmap = plt.imshow(lang2hitmiss)
        plt.xticks(range(len(lang2hitmiss.columns.values)), lang2hitmiss.columns.values, rotation='vertical')
        plt.yticks(range(len(lang2hitmiss.index)), lang2hitmiss.index)
        cbar = plt.colorbar(mappable=heatmap, ticks=[0, 1], orientation='vertical')  
        # vertically oriented colorbar
        cbar.ax.set_yticklabels(['Not best', 'Best']) 
        plt.xlabel("Name of distance metric (in lang2vec)")
        plt.ylabel("Name of unseen language")
        plt.savefig(f"{args.unseen_langs_dir}/best_distance_metric_heatmap_by_{divergence_metric}.pdf", format="pdf")
        plt.show()

    df = pd.DataFrame(results, columns=header)
    df.to_csv(f"{args.unseen_langs_dir}/typological_search_results.tsv", sep="\t")

if __name__ == "__main__":
    main()
