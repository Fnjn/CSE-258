#!/usr/bin/env python3

import random

def split_data(X, Y, holdout_ratio, shuffle=False):
    m = len(X)
    m_holdout = int(m * holdout_ratio)

    if shuffle:
        r = zip(X, Y)
        random.shuffle(r)
        X, Y = zip(*r)

    return X[:-m_holdout], Y[:-m_holdout], X[-m_holdout:], Y[-m_holdout:]


def getSummary(data):
    X = [d['summary'] for d in data]
    Y = [d['rating'] for d in data]
    return X, Y
