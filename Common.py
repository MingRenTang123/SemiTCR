# -*- coding: utf-8 -*-
import random
import math
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from collections import Counter
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, roc_auc_score,classification_report
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV,StratifiedShuffleSplit
import gpflow

def feature(data):
    length = []
    for i in range(len(data)):
        length.append(len(data['cdr3b'][i]))
    max_length = max(length)
    hydrophobicity = {'A': 0.16, 'B': -3.14, 'C': 2.50, 'D': -2.49, 'E': -1.50, 'F': 5.00, 'G': -3.31,
                      'H': -4.63, 'I': 4.41, 'K': -5.00, 'L': 4.76, 'M': 3.23, 'N': -3.79, 'P': -4.92,
                      'Q': -2.76, 'R': -2.77, 'S': -2.85, 'T': -1.08, 'V': 3.02, 'W': 4.88, 'X': 4.59,
                      'Y': 2.00, 'Z': -2.13}
    hydrophobicity_noBXZ = {'A': 0.16,  'C': 2.50, 'D': -2.49, 'E': -1.50, 'F': 5.00, 'G': -3.31,
                      'H': -4.63, 'I': 4.41, 'K': -5.00, 'L': 4.76, 'M': 3.23, 'N': -3.79, 'P': -4.92,
                      'Q': -2.76, 'R': -2.77, 'S': -2.85, 'T': -1.08, 'V': 3.02, 'W': 4.88, 
                      'Y': 2.00}
    AA = []
    for k in hydrophobicity_noBXZ:
        AA.append(hydrophobicity_noBXZ[k])
        
    pos_aa, pos_basicity, pos_hydro, pos_helicity, pos_mutation, pos_pI = [[] for _ in range(6)]
    for sequence in data['cdr3b']:
        length = len(sequence)
        start_pos = -1 * (length // 2)
        pos_range = list(range(start_pos, start_pos + length)) if length % 2 == 1 else \
                list(range(start_pos, 0)) + list(range(0, start_pos + length + 1))

        pos_aa.append({'pos_{}_{}'.format(pos, aa): 1 for pos, aa in zip(pos_range, sequence)})
        pos_hydro.append({'pos_{}_hydrophobiciy'.format(pos): hydrophobicity[aa]
                            for pos, aa in zip(pos_range, sequence)})
        
    features_list = []
    features_list.append(pd.DataFrame.from_records(pos_hydro).fillna(0))
    
    feature_hydrophobiciy = features_list[0].iloc[[0]].values.tolist()
    
    train_data_hydrop = []
    for all_length in range(len(data)):
        data2_hydrop = []
        for j in range(len(AA)):
            feature_hydrophobiciy = features_list[0].iloc[[all_length]].values.tolist()
            for i in range(max_length):
                if(AA[j] != feature_hydrophobiciy[0][i]):
                    feature_hydrophobiciy[0][i] = 0
            data2_hydrop.append(feature_hydrophobiciy[0])
        train_data_hydrop.append(data2_hydrop)
        
    train_data_hydrop = np.array(train_data_hydrop)
    
    x_tridata = []
    
    num = max_length * 20
    for i in range(len(data)):
        x_tridata.append(train_data_hydrop[i].reshape(num))
    
    return np.array(x_tridata)

def sklearn_classes(data):
    classes = data['Epitope']
    classes = classes.values.tolist()
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(classes)
    return integer_encoded

class SemiTCR_TriTraining:
    def __init__(self, classifier):
        if sklearn.base.is_classifier(classifier):
            self.classifiers = [sklearn.base.clone(classifier) for i in range(3)]
        else:
            self.classifiers = [sklearn.base.clone(classifier[i]) for i in range(3)]

    def fit(self, L_X, L_y, U_X):
        LabelX, Labely = [[]] * 3, [[]] * 3
        isImprove = True
        self.count = 0
        for i in range(3):
            sample = sklearn.utils.resample(L_X, L_y)
            self.classifiers[i].fit(*sample)
        e = [0] * 3
        isUpdate = [False] * 3
        initial_e = [0.5] * 3
        initial_l = [0] * 3
        while isImprove:
            self.count += 1
            for i in range(3):
                j, k = np.delete(np.array([0, 1, 2]), i)
                isUpdate[i] = False
                e[i] = self.measure_error(L_X, L_y, j, k)
                if e[i] < initial_e[i]:
                    U_y_j = self.classifiers[j].predict(U_X)
                    U_y_k = self.classifiers[k].predict(U_X)
                    LabelX[i] = U_X[U_y_j == U_y_k]
                    Labely[i] = U_y_j[U_y_j == U_y_k]
                    if initial_l[i] == 0:
                        initial_l[i] = int(e[i] / (initial_e[i] - e[i]) + 1)
                    if initial_l[i] < len(Labely[i]):
                        if e[i] * len(Labely[i]) < initial_e[i] * initial_l[i]:
                            isUpdate[i] = True
                        elif initial_l[i] > e[i] / (initial_e[i] - e[i]):
                            L_index = np.random.choice(len(Labely[i]), int(initial_e[i] * initial_l[i] / e[i] - 1))
                            LabelX[i], Labely[i] = LabelX[i][L_index], Labely[i][L_index]
                            isUpdate[i] = True

            for i in range(3):
                if isUpdate[i]:
                    self.classifiers[i].fit(np.append(L_X, LabelX[i], axis=0), np.append(L_y, Labely[i], axis=0))
                    initial_e[i] = e[i]
                    initial_l[i] = len(Labely[i])

            if isUpdate == [False] * 3:
                isImprove = False
                
    def predict(self, X):
        pred = np.asarray([self.classifiers[i].predict(X) for i in range(3)])
        pred[0][pred[1]==pred[2]] = pred[1][pred[1]==pred[2]]
        return pred[0]
        
    def score(self, X, y):
        return sklearn.metrics.accuracy_score(y, self.predict(X))
        
    def measure_error(self, X, y, j, k):
        j_pred = self.classifiers[j].predict(X)
        k_pred = self.classifiers[k].predict(X)
        wrong_index =np.logical_and(j_pred != y, k_pred==j_pred)
        return sum(wrong_index)/sum(j_pred == k_pred)
    
path_dash_human = r"/home/bianhd/dash_human/"
a = os.walk(path_dash_human)
dash_human = []
for root, dirs, files in os.walk(path_dash_human):  
    print(root)
    print(dirs)
    dash_human.append(files)
file_dash_human = dash_human[0]
print(file_dash_human)
