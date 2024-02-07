import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluation_A (ann_file, prediction):

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for triplet in ann_file:
        if ann_file[triplet] == True and prediction[triplet] == True:
            true_positives += 1
        elif ann_file[triplet] == True and prediction[triplet] == False:
            false_negatives += 1
        elif ann_file[triplet] == False and prediction[triplet] == True:
            false_positives += 1

    prec = true_positives/(true_positives+false_positives)
    rec = true_positives/(true_positives+false_negatives)
    f_measure = 2*(prec*rec)/(prec+rec)
    
    return prec, rec, f_measure

def evaluation_B (ann_file, prediction):

    y_labels = np.array(list(ann_file.values()), dtype = float)
    y_pred = np.array(list(prediction.values()), dtype = float)

    precision = precision_score(y_labels, y_pred)
    recall = recall_score(y_labels, y_pred)
    fscore = f1_score(y_labels, y_pred)
    
    return precision, recall, fscore