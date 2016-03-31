from sklearn import svm
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.neural_network import MLPClassifier

import numpy as np

def PredictGrad1(trainData, labels, testData, est = 100, max_dep = 5, min_samples_spl = 1):
    clf = GradientBoostingClassifier(n_estimators = est, max_depth = max_dep, min_samples_split = min_samples_spl)
    clf.fit(trainData, labels)
    predict = clf.predict_proba(testData)
    return predict


def PredictGrad(trainData, labels, testData, est = 100, max_dep = 5, min_samples_spl = 1):
    clf = GradientBoostingClassifier(n_estimators = est, max_depth = max_dep, min_samples_split = min_samples_spl)  #   loss = 'exponential' - worther
    clf.fit(trainData, labels)
    predict = clf.predict_proba(testData)
    return predict[:,1]

'''
def PredictMLP(trainData, labels, testData):
    clf = MLPClassifier((100, 100))
    clf.fit(trainData, labels)
    clf.predict(testData)
    return predict

def PredictAda(trainData, labels, testData, est = 100):
    clf = AdaBoostClassifier(n_estimators=est)
    clf.fit(trainData, labels)
    predict = clf.predict_proba(testData)
    return predict[:,1]

'''
def PredictGauss(trainData, labels, testData):
    clf = GaussianNB()
    clf.fit(trainData, labels)
    predict = clf.predict(testData)
    #predict1 = clf.predict_proba(testData)
    return predict


def PredictRandomForest(trainData, labels, testData, dep = 20, n_est = 100, max_feat = 5):
    clf = RandomForestClassifier(max_depth = dep, n_estimators = n_est, max_features = max_feat)
    clf.fit(trainData, labels)
    predict = clf.predict_proba(testData)
    return predict[:,1] 


# ????? ?????
def PredictSVC(trainData, labels, testData):
    clf = SVC(gamma=0.2)
    clf.fit(trainData, labels)
    predict = clf.predict_proba(testData)
    return predict
