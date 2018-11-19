#!/usr/bin/env python3

import random

def split_data(X, Y, holdout_ratio, shuffle=False):
    m = len(X)
    m_holdout = int(m * holdout_ratio)

    if shuffle:
        r = list(zip(X, Y))
        random.shuffle(r)
        X, Y = list(zip(*r))

    return X[:-m_holdout], Y[:-m_holdout], X[-m_holdout:], Y[-m_holdout:]


def getSummary(data):
    X = [d['summary'] for d in data]
    Y = [d['rating'] for d in data]
    return X, Y

def create_purchase_dataset(data, holdout_ratio):
    reviewer_item_pair = {}
    reviewer_list = []
    item_list = []

    for d in data:
        reviewer = d['reviewerID']
        item = d['itemID']
        pair = reviewer_item_pair.get(reviewer, [])
        pair.append(item)
        reviewer_item_pair[reviewer] = pair
        reviewer_list.append(reviewer)
        item_list.append(item)

    import random
    cnt = 0
    neg_pair = []

    while(cnt < 200000):
        reviewer = random.choice(reviewer_list)
        item = random.choice(item_list)
        if item not in reviewer_item_pair[reviewer]:
            neg_pair.append((reviewer, item))
            cnt += 1

    dataX = []
    dataY = []

    for d in data:
        dataX.append((d['reviewerID'], d['itemID']))
        dataY.append(1)

    dataX += neg_pair
    dataY += len(neg_pair) * [0]

    trainX, trainY, valX, valY = split_data(dataX, dataY, holdout_ratio, shuffle=True)
    return trainX, trainY, valX, valY
