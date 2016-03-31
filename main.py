import Data
import Metrics
import numpy as np
import timeit
import algos
import threading
import Models
from sklearn.ensemble import ExtraTreesClassifier

from multiprocessing import Queue, Process, Pipe
from datetime import datetime
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss


import os

minValue = {0  : 0.17, 
            1  : 8.755, 
            2  : 0.046,
            3  : 0.043,
            4  : 4.186,
            5  : 0.492,
            6  : 0.258, 
            7  : 5.147,
            8  : 0.487,
            9  : 0.133,
            10 : 0.030, 
            11 : 0.056,
            12 : 1.208,
            13 : 1.9,
            14 : 0.017,
            15 : 0.267, 
            16 : 19.92,
            17 : 0.217,
            18 : 0.14,
            19 : 2.96, 
            20 : 10.513,
            21 : 14.371,
            22 : 0.003,
            23 : 0.852,
            24 : 0.357, 
            25 : 2.62,
            26 : 0.222,
            27 : 1.137,
            28 : 0.5,
            29 : 0.017,
            30 : 0.517,
            31 : 0.058,
            32 : 3.578, 
            33 : 0.001,
            34 : 0.834,
            35 : 5.091,
            36 : 6.125,
            37 : 4.808,
            38 : 0.974}



def Normalization(prediction):
    for i in range(0, len(prediction)):
        for j in range(0, len(prediction[i])):
            prediction[i][j] = max(prediction[i][j], minValue[j] * 0.001)
            
        s = sum(prediction[i])
        prediction[i] = prediction[i] / s

    return prediction



def CreateStrubArray(array):
    s = strubArray = [1/39] * len(array) * len(array[0])
    return s




def RunDifferentClassifiers(trainData, labels, testData, labels2):
    casesNum = 39

    print(" ")

    for i in range(0, casesNum):
        trueLabels = labels2[:,i]
        
        pred2 = Models.PredictGrad(trainData, labels[:, i], testData, est = 50)
        pred3 = Models.PredictGrad(trainData, labels[:, i], testData, est = 100)
        pred4 = Models.PredictGrad(trainData, labels[:, i], testData, est = 200)
        pred5 = Models.PredictGrad(trainData, labels[:, i], testData, est = 500)
        pred6 = Models.PredictGrad(trainData, labels[:, i], testData, est = 1000)
        pred7 = Models.PredictGrad(trainData, labels[:, i], testData, est = 2000)
        pred8 = Models.PredictRandomForest(trainData, labels[:, i], testData)
       

        #res1 = Metrics.QualityMetrics.LinearQuality(trueLabels, pred1)
        res2 = Metrics.QualityMetrics.LinearQuality(trueLabels, pred2)
        res3 = Metrics.QualityMetrics.LinearQuality(trueLabels, pred3)
        res4 = Metrics.QualityMetrics.LinearQuality(trueLabels, pred4)
        res5 = Metrics.QualityMetrics.LinearQuality(trueLabels, pred5)
        res6 = Metrics.QualityMetrics.LinearQuality(trueLabels, pred6)
        res7 = Metrics.QualityMetrics.LinearQuality(trueLabels, pred7)
        res8 = Metrics.QualityMetrics.LinearQuality(trueLabels, pred8)

        print(str(i) + " " + 
              #str(round(res1)) + " " + 
              str(round(res2)) + " " +
              str(round(res3)) + " " + str(round(res4)) + " " + str(round(res5)) + " " + 
              str(round(res6)) + " " + str(round(res7)) + " " + str(round(res8)))


def RunAllFeaturesAll(trainData, labels, testData):
    res = RunAllFeatures(trainData, labels, testData)

    Normalization(res)

    return res


'''
def RunAllFeatures1(trainData, labels, testData, conn = 0, featureFrom = 0, featureTo = 39):
    casesNum = featureTo - featureFrom
    fullprediction = [0] * casesNum

    algonum = algos.featureAlgos[0]
    algo = algos.allPossibleAlgorithms2[algonum[1]]
    pred = Models.PredictGrad1(trainData, labels, testData, est = 5, max_dep = algo[0],  min_samples_spl = algo[1])

    return pred
'''

def RunAllFeatures(trainData, labels, testData, conn = 0, featureFrom = 0, featureTo = 39, saveAndLoadWithFile = False):
    casesNum = featureTo - featureFrom
    fullprediction = [0] * casesNum

    for i in range(featureFrom, featureTo):
        fname = data.commonPath + "submission\\" + str(i) + ".npy"
        if os.path.isfile(fname) and saveAndLoadWithFile:
            fullprediction[i - featureFrom] = np.load(fname)
            print(str(dt.hour) + "." + str(dt.minute) + "." + str(dt.second) + "   model " + str(i) + " load from file." )
        else:
            algonum = algos.featureAlgos[i]
            algo = algos.allPossibleAlgorithms2[algonum[1]]

            pred = Models.PredictGrad(trainData, labels[:, i], testData, est = 500, max_dep = algo[0],  min_samples_spl = algo[1])
        
            fullprediction[i - featureFrom] = pred
            dt = datetime.now()
            print(str(dt.hour) + "." + str(dt.minute) + "." + str(dt.second) + "   model " + str(i) + " done." )

            if saveAndLoadWithFile:
                np.save(data.commonPath + "\\submission\\" + str(i) + ".npy", pred)


    fullprediction = np.array(fullprediction)
    if not conn == 0:
        conn.send(fullprediction)


    return fullprediction
    





def RunHalf():
    fullprediction = RunAllFeatures(data.TrainDataHalf1, data.LabelsDataHalf1, data.TrainDataHalf2, casesNum)

    strubPrediction = CreateStrubArray(fullprediction)

    labels = data.LabelsDataHalf2[:,0:casesNum]
    res1 = Metrics.QualityMetrics.LinearQuality(labels, fullprediction)
    res2 = Metrics.QualityMetrics.LinearQuality(labels, strubPrediction)

    print("result = " + str(res1))
    print("strub result = " + str(res2))

    return fullprediction



def RunFull():
    start = timeit.default_timer()
    result = RunAllFeatures(data.TrainData, data.Labels, data.TestData, 39)
    stop = timeit.default_timer()
    print("time elapsed: " + str(stop - start))
    return result



def PrintRes(res, trueLabels, header = ""):
    print("") 
    print(header) 

    if (len(res),len(res[0])) == (len(trueLabels),len(trueLabels[0])):
        for i in range(0, len(res[0])):
            res1 = Metrics.QualityMetrics.LinearQuality(trueLabels[:,i], res[:,i])
            res2 = Metrics.QualityMetrics.MLogLoss(trueLabels[:,i], res[:,i])
            res3 = log_loss(trueLabels[:,i], res[:,i])
            
            print("model: " + str(i) + " result: " + str(res1) + ",   " + str(res2) + ",   " + str(res3))
            
        print("") 
        res1 = Metrics.QualityMetrics.LinearQuality(trueLabels, res)
        res2 = Metrics.QualityMetrics.MLogLoss(trueLabels, res)
        res3 = log_loss(trueLabels, res)
        
        print("result full = " + str(res1) + ",   " + str(res2) + ",    " + str(res3))




def startWith4ThreadsWithProve(trainData, labels, testData, trueLabels):
    #res, p1, p2, p3, p4  = startWith4Threads(trainData, labels, testData) #, p1, p2, p3, p4
    res = startWith4Threads(trainData, labels, testData)

    PrintRes(res, trueLabels, "simple")

    return res


def startWith2ThreadsWithProve(trainData, labels, testData, trueLabels):
    res = startWith2Threads(trainData, labels, testData)
    PrintRes(res, trueLabels, "simple")
    return res


def startWith2Threads(trainData, labels, testData):
    print("startWith4Threads started")
    parent_conn1, child_conn1 = Pipe()
    parent_conn2, child_conn2 = Pipe()

    t1 = threading.Thread(target=RunAllFeatures, args = (trainData, labels, testData, child_conn1, 0, 20, ))
    t2 = threading.Thread(target=RunAllFeatures, args = (trainData, labels, testData, child_conn2, 20, 39, ))

    t1.start()
    t2.start()

    rr1 = parent_conn1.recv()
    rr2 = parent_conn2.recv()

    t1.join();
    t2.join();

    print("all tasks done")
                  
    rr1 = np.row_stack((rr1, rr2))
    
    print("result combined")

    res = np.transpose(rr1)
    
    pp = Normalization(res)

    print("normalization done")
    return pp



def startWith4Threads(trainData, labels, testData):
    print("startWith4Threads started")
    parent_conn1, child_conn1 = Pipe()
    parent_conn2, child_conn2 = Pipe()
    parent_conn3, child_conn3 = Pipe()
    parent_conn4, child_conn4 = Pipe()

    t1 = threading.Thread(target=RunAllFeatures, args = (trainData, labels, testData, child_conn1, 0, 9,  True))
    t2 = threading.Thread(target=RunAllFeatures, args = (trainData, labels, testData, child_conn2, 9, 19, True))
    t3 = threading.Thread(target=RunAllFeatures, args = (trainData, labels, testData, child_conn3, 19, 29, True))
    t4 = threading.Thread(target=RunAllFeatures, args = (trainData, labels, testData, child_conn4, 29, 39, True))

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    rr1 = parent_conn1.recv()
    rr2 = parent_conn2.recv()
    rr3 = parent_conn3.recv()
    rr4 = parent_conn4.recv()

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    print("all tasks done")
                  
    rr1 = np.row_stack((rr1, rr2))
    rr3 = np.row_stack((rr3, rr4))
    rr1 = np.row_stack((rr1, rr3))
    
    print("result combined")

    res = np.transpose(rr1)

    pp = Normalization(res)
    #pp = res

    print("normalization done")

    #return (res, p1, p2, p3, p4)
    return pp
    


def Feature_importances():
    ress = [0] * 39
    for i in range(0,39):
        clf = ExtraTreesClassifier()
        clf = clf.fit(data.TrainData, data.Labels[:,i])
        res = clf.feature_importances_
        print("model   " + str(i) + str(res));
        ress[i] = res
    
    print("res " + str(ress));



def RunFull():
    start = timeit.default_timer()
    result = RunAllFeatures(data.TrainData, data.Labels, data.TestData, 39)
    stop = timeit.default_timer()
    print("time elapsed: " + str(stop - start))
    return result





print("start")


data = Data.DataStorage()
data.LoadData(fromBinaryFile = True, filtering = False, UseStreets = True, UseStreetTypes = True)
dt = datetime.now()
print("start time {0}.{1}.{2}".format(dt.hour, dt.minute, dt.second))

#result = startWith2ThreadsWithProve(data.TrainDataHalf1, data.LabelsDataHalf1, data.TrainDataHalf2, data.LabelsDataHalf2)

#result = RunAllFeatures(data.TrainDataHalf1, data.LabelsDataHalf1, data.TrainDataHalf2, conn = 0, featureFrom = 0, featureTo = 39)
#res = RunAllFeatures1(data.TrainDataHalf1, data.LabelsNHalf1, data.TrainDataHalf2)
#PrintRes(result, data.LabelsDataHalf2, "simple")
#result = RunAllFeatures(data.TrainData, data.Labels, data.TestData, conn = 0, featureFrom = 0, featureTo = 39, saveAndLoadWithFile = True)

result = startWith4Threads(data.TrainData, data.Labels, data.TestData)
dt = datetime.now()
print("start time {0}.{1}.{2}".format(dt.hour, dt.minute, dt.second))

data.Save(result)
