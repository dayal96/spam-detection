#!/usr/bin/python3.6

from collections import defaultdict
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DEBUG = True
IGNORE = []

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

def log(string):
    print(string)


class ClusterClassifier:

    def __init__(self, train_data, train_Y, num_clusters):
        self.G = calc_global_counts(train_data)
        self.word_order = [x for x in self.G.keys()]
        self.num_clusters = num_clusters
        points = np.array([self.vectorize(doc) for doc in train_data])
        self.pca = PCA(n_components=50)
        points = self.pca.fit_transform(points)
        self.kmeans = KMeans(n_clusters=num_clusters)
        self.kmeans.fit(points)
        centroids = self.kmeans.cluster_centers_
        self.centroids = self.classify_centroids(points, train_Y)

    # Classify centroids based on index
    def classify_centroids(self, points, classes):
        centroid_classes = [0] * self.num_clusters
        centroid_sizes = [0] * self.num_clusters

        for i in range(len(points)):
            point = points[i]
            closest = self.kmeans.predict([point])[0]
            centroid_classes[closest] += classes[i]
            centroid_sizes[closest] += 1

        for i in range(self.num_clusters):
            centroid_classes[i] = round(centroid_classes[i] / centroid_sizes[i])
        
        return centroid_classes

    # Predict the class of the given document
    def predict_all(self, documents):
        points = [self.vectorize(doc) for doc in documents]
        points = self.pca.transform(points)
        closest_cs = self.kmeans.predict(points)
        return [self.centroids[c] for c in closest_cs]

    # Form a vector for the given document
    def vectorize(self, doc):
        vec = [0] * len(self.word_order)
        counts = get_doc_counts(doc)
        for i in range(len(self.word_order)):
            word = self.word_order[i]
            vec[i] = counts[word] * self.G[word]

        return vec
    
