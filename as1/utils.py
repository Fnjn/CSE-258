#!/usr/bin/env python3

import random
import numpy as np

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

class Mat:
    def __init__(self):
        self.user2index = {}
        self.item2index = {}
        self.n_user = 0
        self.n_item = 0

        self.user2item = {}
        self.item2user = {}
        self.all_rating = 0.
        self.all_cnt = 0

        self.all_samples = []

    def addEntry(self, entry):
        u = entry['reviewerID']
        i = entry['itemID']
        r = entry['rating']

        self.all_rating += r
        self.all_cnt += 1

        if u not in self.user2index:
            self.user2index[u] = self.n_user
            self.n_user += 1
        if i not in self.item2index:
            self.item2index[i] = self.n_item
            self.n_item += 1

        self.all_samples.append((self.user2index[u], self.item2index[i], r))

        if u not in self.user2item:
            self.user2item[u] = [(i,r)]
        else:
            self.user2item[u].append((i,r))

        if i not in self.item2user:
            self.item2user[i] = [(u,r)]
        else:
            self.item2user[i].append((u,r))

    def addEntries(self, entries):
        for e in entries:
            self.addEntry(e)
        self.all_avg = self.all_rating / self.all_cnt

    def ratingMat(self):
        self.R = np.zeros([self.n_user, self.n_item])
        for u, value in self.user2item.items():
            for v in value:
                i, r = v
                user_index = self.user2index[u]
                item_index = self.item2index[i]
                self.R[user_index, item_index] = r
