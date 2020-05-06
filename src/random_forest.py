#!/usr/bin/python3.6

from collections import defaultdict
import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

DEBUG = True
IGNORE = []


class RandomForest:

    def __init__(self, data, Y, num_trees=100):
        self.G = calc_global_counts(data)
        self.word_order = [x for x in self.G.keys()]
        self.pca = PCA(n_components=50)
        points = np.array([self.vectorize(doc) for doc in data])
        points = self.pca.fit_transform(points)
        
        self.model = RandomForestClassifier(n_estimators=num_trees)
        self.model.fit(points, Y)

    # Predict the class of the given document
    def predict_all(self, documents):
        points = [self.vectorize(doc) for doc in documents]
        points = self.pca.transform(points)
        results = self.model.predict(points)
        return results

    # Form a vector for the given document
    def vectorize(self, doc):
        vec = [0] * len(self.word_order)
        counts = get_doc_counts(doc)
        for i in range(len(self.word_order)):
            word = self.word_order[i]
            vec[i] = counts[word] * self.G[word]

        return vec



def calc_global_counts(documents):
    n = len(documents)
    counts = defaultdict(lambda: 0)

    for d in documents:
        words_seen = set()
        for word in d.split(' '):
            if word in IGNORE:
                continue
            elif word not in words_seen:
                words_seen.add(word)
                counts[word] = counts[word] + 1
    
    final_counts = defaultdict(lambda:0)

    for word in counts.keys():
        if 10 < counts[word] < (n / 2):
            final_counts[word] = math.log(n / counts[word])

    return final_counts


def get_doc_counts(doc):
    counts = defaultdict(lambda: 0)
    for word in doc.split(' '):
        counts[word] = counts[word] + 1
    return counts

