#!/usr/bin/python3.6

from collections import defaultdict
import math
import numpy as np
import os
from sklearn.decomposition import PCA
import warnings
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    warnings.filterwarnings("ignore",category=Warning)
    import tensorflow.keras as tf
    import tensorflow as tf2
    tf2.compat.v1.logging.set_verbosity(tf2.compat.v1.logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEBUG = True
IGNORE = []


class NeuralNet:

    def __init__(self, data, Y, h1, h2=0, h3=0):
        self.G = calc_global_counts(data)
        self.word_order = [x for x in self.G.keys()]
        
        Y = [(x * 2 - 1) for x in Y] # Keep 1 as 1, turn 0 into -1 to match tanh function
        self.pca = PCA(n_components=50)
        points = np.array([self.vectorize(doc) for doc in data])
        points = self.pca.fit_transform(points)

        self.model = make_model(h1, h2, h3)
        self.model.fit(points, Y, batch_size=100, epochs=3, verbose=0)

    # Predict the class of the given document
    def predict_all(self, documents):
        points = [self.vectorize(doc) for doc in documents]
        points = self.pca.transform(points)
        results = self.model.predict(points)
        results = [simplify(x) for x in results]
        return results

    # Form a vector for the given document
    def vectorize(self, doc):
        vec = [0] * len(self.word_order)
        counts = get_doc_counts(doc)
        for i in range(len(self.word_order)):
            word = self.word_order[i]
            vec[i] = counts[word] * self.G[word]

        return vec



def simplify(result):
    if result > 0:
        return 1
    else:
        return 0


def make_model(h1, h2, h3):
    model = tf.Sequential()
    model.add(tf.layers.Dense(input_shape=(50, ), units=h1, activation='tanh'))
    if h2 > 0:
        model.add(tf.layers.Dense(units=h2, activation='tanh'))
    if h3 > 0:
        model.add(tf.layers.Dense(units=h3, activation='tanh'))
    model.add(tf.layers.Dense(units=1, activation='tanh'))
    model.compile(loss='mse', metrics=['accuracy', tf.metrics.AUC()], optimizer='rmsprop')
    return model


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

