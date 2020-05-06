#!/usr/bin/python3.6

import ann
import cluster
import measure
import parsedata as pd
import random_forest as rf
import time

def create_combinations_ann():
    anns = []
    for h1 in [5, 10, 20]:    
        for h2 in [5, 10, 20]:
            for h3 in [0]:
                model_name = ("Neural Network (" + str(h1) + ", " + str(h2) +
                              ", " + str(h3) + ")")
                anns.append((ann_model_maker(h1, h2, h3), model_name))
    return anns



def ann_model_maker(h1, h2, h3):
    def specific_model_maker(train_data, train_Y):
        model = ann.NeuralNet(train_data, train_Y, h1=h1, h2=h2, h3=h3)
        return model
    return specific_model_maker

def forest_maker(n):
    def specific_forest_maker(train_data, train_Y):
        model = rf.RandomForest(train_data, train_Y, n)
        return model
    return specific_forest_maker

def cluster_50(train_data, train_Y):
    model = cluster.ClusterClassifier(train_data, train_Y, 50)
    return model

def cluster_100(train_data, train_Y):
    model = cluster.ClusterClassifier(train_data, train_Y, 100)
    return model


NEURAL_NETWORKS = create_combinations_ann()
CLUSTERS = [(cluster_50, "K-Means Cluster (50)"), (cluster_100, "K-Means Cluster (100)")]
FORESTS = [(forest_maker(50), "Random Forest (50)"),
           (forest_maker(100), "Random Forest (100)")]
CONSTRS = CLUSTERS + FORESTS + NEURAL_NETWORKS


DATASETS = [(pd.read_sms_spam_data, "SMS SPAM"),
            (pd.read_ling_spam, "LING SPAM"),
            (pd.read_tweet_spam_data, "TWEET SPAM")]

def simple_eval_all():
    for (d, d_name) in DATASETS:
        print("".join(['-']*80))
        print(d_name)
        print("".join(['-']*80))
        data, Y = d()
        for (c, c_name) in CONSTRS:
            print(c_name)
            metrics = measure.cross_validate(c, data, Y)
            measure.print_avg_metrics(metrics)
            print()
        print("".join(['-']*80))

def cross_data_eval():
    for i in range(len(DATASETS)):
        test = DATASETS[i]
        test_data, test_Y = test[0]()
        
        train = DATASETS[:i] + DATASETS[i+1:]
        train = [d[0]() for d in train]
        train_data = []
        train_Y = []

        for td, ty in train:
            train_data.extend(td)
            train_Y.extend(ty)

        print("".join(['-']*80))
        print("Held Out : " + test[1])
        print("Training Size : " + str(len(train_data)))
        print("Test Size : " + str(len(test_data)))
        print("".join(['-']*80))
        

        for (c, c_name) in CONSTRS:
            print(c_name)
            model = c(train_data, train_Y)
            metric = measure.evaluate(model, test_data, test_Y)
            measure.print_avg_metrics([metric])
            print()
        print("".join(['-']*80))

def all_data_eval():
    all_data = []
    all_Y = []

    for (d, d_name) in DATASETS:
        data, Y = d()
        all_data.extend(data)
        all_Y.extend(Y)
    
    print("".join(['-']*80))
    print("COMBINED DATASET CROSS-VALIDATION")
    print("".join(['-']*80))
    
    for (c, c_name) in CONSTRS:
        print(c_name)
        metrics = measure.cross_validate(c, all_data, all_Y)
        measure.print_avg_metrics(metrics)
        print()
    print("".join(['-']*80))

def main():
    start = time.time()
    simple_eval_all()
    print()
    print()
    cross_data_eval()
    print()
    print()
    all_data_eval()
    total_time = time.time() - start
    print()
    print()
    print("Time Taken : " + str(total_time) + " s")

main()
