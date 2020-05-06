#!/usr/bin/python3.6

import random

# Displays Recall, Precision, Neg Recall and Neg Precision
def print_avg_metrics(metrics):
    n = len(metrics)
    avg_metrics = metrics[0]

    for i in range(1, n):
        for key in avg_metrics.keys():
            avg_metrics[key] += metrics[i][key]
    
    recall = 0
    if avg_metrics['fp'] == 0 and avg_metrics['tp'] == 0:
        recall = 1
    else:
        recall = avg_metrics['tp'] / (avg_metrics['tp'] + avg_metrics['fp'])

    precision = 0
    if avg_metrics['fn'] == 0 and avg_metrics['tp'] == 0:
        precision = 1
    else:
        precision = avg_metrics['tp'] / (avg_metrics['tp'] + avg_metrics['fn'])
    
    neg_recall = 0
    if avg_metrics['fn'] == 0 and avg_metrics['tn'] == 0:
        neg_recall = 1
    else:
        neg_recall = avg_metrics['tn'] / (avg_metrics['tn'] + avg_metrics['fn'])

    neg_precision = 0
    if avg_metrics['fp'] == 0 and avg_metrics['tn'] == 0:
        neg_precision = 1
    else:
        neg_precision = avg_metrics['tn'] / (avg_metrics['tn'] + avg_metrics['fp'])

    print("Recall : " + str(recall))
    print("Precision : " + str(precision))
    print("Neg Recall : " + str(neg_recall))
    print("Neg Precision : " + str(neg_precision))
    print(avg_metrics)
    return avg_metrics

# Evaluate given classifier on given data and Y targets
def evaluate(model, data, Y):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    predictions = model.predict_all(data)
    
    for i in range(len(data)):
        prediction = predictions[i]
        
        if prediction == Y[i] == 1:
            tp += 1
        elif prediction == 1:
            fp += 1
        elif prediction == Y[i] == 0:
            tn += 1
        else:
            fn += 1

    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

# Apply cross_validation on the given classifier (trained using given constructor)
def cross_validate(constructor, data, Y):
    chunks = create_chunks(data, Y)
    metrics = []

    for i in range(len(chunks)):
        train_chunks = chunks[:i] + chunks[i+1:]
        train_data, train_Y = merge_chunks(train_chunks)
        test_data, test_Y = chunks[i]
        model = constructor(train_data, train_Y)
        metrics.append(evaluate(model, test_data, test_Y))
    
    return metrics

# Create 10 chunks out of given data and target rows
def create_chunks(data, Y):
    n = len(data)
    chunk_size = n // 10

    chunks = []
    cur_index = 0
    order = [x for x in range(len(Y))]
    random.shuffle(order)

    for i in range(10):
        chunk_data = []
        chunk_y = []
        for j in range(chunk_size):
            c_id = order[cur_index]
            chunk_data.append(data[c_id])
            chunk_y.append(Y[c_id])
            cur_index += 1
        
        while (i == 9 and cur_index < n):
            c_id = order[cur_index]
            chunk_data.append(data[c_id])
            chunk_y.append(Y[c_id])
            cur_index += 1
        
        chunks.append((chunk_data, chunk_y))
    
    return chunks

# Merge chunk into single data matrix and corresponding Y vector
def merge_chunks(chunks):
    data = []
    Y = []
    for (chunk_data, chunk_Y) in chunks:
        data.extend(chunk_data)
        Y.extend(chunk_Y)
    
    return (data, Y)

