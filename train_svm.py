# To train a SVM classifier based on the features extracted for Void

# According to the original paper, when training on ASVspoof 2017 dataset, the SVM RBF classifier is 
# trained with data in both training and developing sets and evaluated against the evaluation set.

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as ssig
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import math
import librosa
from sklearn import svm
import pickle
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import NearestCentroid
from feature_extraction import LinearityDegreeFeatures, HighPowerFrequencyFeatures, extract_lpcc


def audio_label(s):
    it = {b'genuine':1, b'spoof':0}
    return it[s]


if __name__ == '__main__':
    file_train = os.path.join(os.getcwd(), 'features_labels', 'feature_label_train.npy')
    # If all the features and labels are extracted and stored into './feature_label.npy', then directly load this file;
    # Otherwise, generate this file first:
    if os.path.isfile(file_train):
        # Load the extracted features and corresponding labels:
        data_train = np.load(file_train)
    else:
        print('Feature extraction has not been done. Please extract Void features using data_preparation.py!')

    # Now we have all the extracted features and labels loaded in 'data'.
    # The next step is to split the features and labels into x_data and y_data:
    x_train, y_train = np.split(data_train, indices_or_sections=(97,), axis=1)

    # Import development data for simple testing:
    file_dev = os.path.join(os.getcwd(), 'features_labels', 'feature_label_dev.npy')
    # If all the features and labels are extracted and stored into './feature_label.txt', then directly load this file;
    # Otherwise, generate this file first:
    if os.path.isfile(file_dev):
        # Load the extracted features and corresponding labels:
        data_dev = np.load(file_dev)
    else:
        print('Feature extraction has not been done. Please extract Void features using data_preparation.py!')
    # Load the Void features and corresponding labels:
    x_dev, y_dev = np.split(data_dev, indices_or_sections=(97,), axis=1)

    # Import evaluation data for testing:
    file_eval = os.path.join(os.getcwd(), 'features_labels', 'feature_label_eval.npy')
    # If all the features and labels are extracted and stored into './feature_label.txt', then directly load this file;
    # Otherwise, generate this file first:
    if os.path.isfile(file_eval):
        # Load the extracted features and corresponding labels:
        data_eval = np.load(file_eval)
    else:
        print('Feature extraction has not been done. Please extract Void features using data_preparation.py!')
    # Load the Void features and corresponding labels:
    x_eval, y_eval = np.split(data_eval, indices_or_sections=(97,), axis=1)

    # Define and train the SVM classifier:
    classifier = svm.SVR(kernel='rbf', gamma='auto', C=0.5, epsilon=0.1)
    #classifier = svm.SVC(C=1.8, kernel='rbf', gamma='auto', class_weight=None)
    classifier.fit(np.concatenate((x_train, x_dev)), np.concatenate((y_train, y_dev)).ravel())

    # WHEN USING SVC
    # Prediction result of SVM classifier on TRAIN set of ASVspoof 2017 v2 dataset:
    print("Results on TRAINING SET:")
    result_pred = classifier.predict(x_train)
    # Calculate TP, TN, FP, FN:
    # live-human as POSITIVE
    # spoof as NEGATIVE
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in np.arange(y_train.size):
        # The i-th predicted label:
        label_pred = result_pred[i]
        # The actual label for i-th sample:
        label_actual = y_train[i]
        if label_pred <= 0.5:
            if label_actual == 0:
                TN += 1
            else:
                FN += 1
        elif label_pred > 0.5:
            if label_actual == 1:
                TP += 1
            else:
                FP += 1
    FAR = FP / (FP + TN)
    FRR = FN / (FN + TP)
    print("#TruePositive =", TP)
    print("#FalsePositive =", FP)
    print("#TrueNegative =", TN)
    print("#FalseNegative =", FN)
    print("FAR =", FAR)
    print("FRR =", FRR, "\n")

    # Prediction result of SVM classifier on DEVELOPING set of ASVspoof 2017 v2 dataset:
    print("Results on DEVELOPING SET:")
    result_pred = classifier.predict(x_dev)
    # Calculate TP, TN, FP, FN:
    # live-human as POSITIVE
    # spoof as NEGATIVE
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in np.arange(y_dev.size):
        # The i-th predicted label:
        label_pred = result_pred[i]
        # The actual label for i-th sample:
        label_actual = y_dev[i]
        if label_pred <= 0.5:
            if label_actual == 0:
                TN += 1
            else:
                FN += 1
        elif label_pred > 0.5:
            if label_actual == 1:
                TP += 1
            else:
                FP += 1
    FAR = FP / (FP + TN)
    FRR = FN / (FN + TP)
    print("#TruePositive =", TP)
    print("#FalsePositive =", FP)
    print("#TrueNegative =", TN)
    print("#FalseNegative =", FN)
    print("FAR =", FAR)
    print("FRR =", FRR, "\n")

    # Prediction result of SVM classifier on EVALUATION set of ASVspoof 2017 v2 dataset:
    print("Results on EVALUATION SET:")
    result_pred = classifier.predict(x_eval)
    # Calculate TP, TN, FP, FN:
    # live-human as POSITIVE
    # spoof as NEGATIVE
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in np.arange(y_eval.size):
        # The i-th predicted label:
        label_pred = result_pred[i]
        # The actual label for i-th sample:
        label_actual = y_eval[i]

        #print("predict, actual:", label_pred, label_actual)

        if label_pred <= 0.5:
            if label_actual == 0:
                TN += 1
            else:
                FN += 1
        elif label_pred > 0.5:
            if label_actual == 1:
                TP += 1
            else:
                FP += 1
    FAR = FP / (FP + TN)
    FRR = FN / (FN + TP)
    print("#TruePositive =", TP)
    print("#FalsePositive =", FP)
    print("#TrueNegative =", TN)
    print("#FalseNegative =", FN)
    print("FAR =", FAR)
    print("FRR =", FRR, "\n")

    # If the SVM classifier is satisfactory, uncomment to save the model:
    
    s = pickle.dumps(classifier)
    save_name = os.path.join(os.getcwd(), 'models', 'svm.pkl')
    f = open(save_name, 'wb+')
    f.write(s)
    f.close()
    print("Model saved into " + save_name)




