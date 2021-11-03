#!/usr/bin/env python
"""Helper script to find the best expert to use for novel domain
Expect input is a comma seperated and valid probability distribution
e.g. export POSTERIOR=$(tail -n 1 $DEV_POSTERIOR_OUTPUT | jq -rc '.exp_avg_posterior | join(",")')

"""
import sys

from argparse import Namespace  # noqa
from numpy import argmax

SEEN_LANGS = ['de_DE', 'en_XX', 'fr_XX', 'id_ID', 'ja_XX', 'ro_RO', 'ru_RU', 'zh_CN']
EPS=1e-5
THRESHOLD = 0.05
if __name__ == '__main__':
    input_string = sys.argv[1]
    print(f"Selecting probabilty at least {THRESHOLD}...")
    try:
        prob_dist = list(map(float, input_string.split(",")))
        assert -EPS <= sum(prob_dist) - 1 <= EPS
    except:
        raise ValueError("Invalid Input")
    indexed_prob_dist = enumerate(prob_dist)
    sorted_prob_dist = sorted(indexed_prob_dist, key=lambda x: x[1], reverse=True)
    # print(sorted_prob_dist)
    filtered_prob = filter(lambda x: x[1] >= THRESHOLD, sorted_prob_dist)
    named_prob = list(map(lambda x: (SEEN_LANGS[x[0]], x[1]), filtered_prob))
    print(named_prob)
    