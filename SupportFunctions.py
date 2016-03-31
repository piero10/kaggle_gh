from sklearn.preprocessing import label_binarize
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
import hashlib
import sys

tempratureAndRain = {1 : [10.7, 11.7],
                     2 : [12.2, 11.1],
                     3 : [12.8, 11.0],
                     4 : [13.4, 6.5],
                     5 : [14.2, 3.8],
                     6 : [15.3, 1.5],
                     7 : [15.7, 0.3],
                     8 : [16.4, 1.0],
                     9 : [17.1, 1.7],
                     10 : [16.4, 3.9],
                     11 : [13.7, 8.9],
                     12 : [10.9, 11.6]}


weekDays = {"Sunday" : 0, 
            "Monday" : 1,
            "Tuesday": 2, 
            "Wednesday" : 3, 
            "Thursday" : 4, 
            "Friday" : 5, 
            "Saturday" : 6}


districtions = {"BAYVIEW" : 0, 
                "NORTHERN" : 1,
                "INGLESIDE": 2, 
                "TARAVAL" : 3, 
                "MISSION" : 4, 
                "TENDERLOIN" : 5, 
                "RICHMOND" : 6,
                "CENTRAL" : 7, 
                "PARK" : 8,
                "SOUTHERN" : 9}


streestTypesDict = {"AV" : 0, 
                "BL" : 1,                
                "CR" : 2,                
                "CT" : 3,                
                "DR" : 4,                
                "LN" : 5,                 
                "PZ" : 6,                
                "PL" : 7,                
                "RD" : 8,                
                "ST" : 9,                
                "TR" : 10,                
                "HWY": 11,                
                "HY" : 12,                
                "WY" : 13,                
                "WAY": 14}



def LabelsSimpleFromLabels(labels):
    resN = [0] * len(labels)
    for i in range(0,len(labels)):
        for j in range(0,len(labels[i])):
            if labels[i,j] == 1:
                resN[i] = int(j)
                break

    return resN


'''
def Adress(adressColumn, hashLen = 100):
    streets = Streets(adressColumn, hashLen) 
    
    res = np.c_[res, streets]
    return res 
'''
    

def Streets(streetColumn, hashLen = 100):
    streets = np.empty((len(streetColumn), 2), dtype=object)
    streetsTogether = np.empty(len(streetColumn), dtype=object)

    for i in range(0, len(streetColumn)):
        street = [w for w in streetColumn[i].split("/")]
        
        streets[i, 0] = "None"
        streets[i, 1] = "None"

        if len(street) > 0:
            firstStreet = [st for st in street[0].split()]# and len(st) > 2] # check it !!!!!! seems len(st) > 2 isn't nessesary    and len(st) > 2 and not st == 'hwy' and not st == 'way'] 
            if len(firstStreet) > 0:
                streets[i, 0] = ''.join(firstStreet)

        if len(street) > 1:
            secondString = [st for st in street[1].split()]# if st.isupper() and len(st) > 2] # and len(st) > 2 and not st == 'hwy' and not st == 'way']
            if len(secondString) > 0:
                streets[i, 1] = ''.join(secondString)

        streetsTogether[i] = streets[i, 0] + " " + streets[i, 1]
        streetsTogether[i] = streetsTogether[i].lower()

    r1 = my_hashing(streetsTogether, hashLen)

    return r1



def StreetType(streetColumn):
    streetsTypesBin = np.zeros((len(streetColumn), 15), dtype=int)

    for i in range(0, len(streetColumn)):
        street = [w for w in streetColumn[i].upper().split("/")]
        

        tp = np.zeros(15, dtype=int)
        types = []
        if len(street) > 0:
            if len(street) > 0:
                types = [st for st in street[0].split() if st in streestTypesDict.keys()] # and len(st) > 2 and not st == 'hwy' and not st == 'way'] 

            if len(street) > 1:
                tp1 = [st for st in street[1].split() if st in streestTypesDict.keys()]
                types.extend(tp1)

            for j in types:
                tp[streestTypesDict[j]] = 1

        streetsTypesBin[i] = tp

    return streetsTypesBin






def my_hashing(features, N):
    x = [0] * N
    featuresBinarized = [0] * len(features)
    for i in range(len(features)):
        strings = features[i].split()
        featuresBinarized[i] = [0] * N
        for j in range(0, len(strings)):
            if strings[j] != "None":
                sha = abs(hash(strings[j]))
                featuresBinarized[i][sha % N] = 1

    return featuresBinarized



def Coordinates(coordinates):
    a = 0.0
    b = 0.0
    a_count = 0
    b_count = 0

    coord = [0] * len(coordinates)

    for i in range(0, len(coordinates)):
        if coordinates[i, 0] == coordinates[i, 0]:
            a += coordinates[i, 0]
            a_count += 1

        if coordinates[i, 1] == coordinates[i, 1]:
            b += coordinates[i, 1]
            b_count += 1

    ave0 = a / a_count
    ave1 = b / b_count 

    for i in range(0, len(coordinates)):
        if coordinates[i, 0] != coordinates[i, 0]:
            coordinates[i, 0] = ave0

        if coordinates[i, 1] != coordinates[i, 1]:
            coordinates[i, 1] = ave1

    coordinatesCircle = [0] * len(coordinates)
    for i in range(0, len(coordinates)):
        dist = math.sqrt((coordinates[i, 0] - ave0) * (coordinates[i, 0] - ave0) + (coordinates[i, 1] - ave1) * (coordinates[i, 1] - ave1))
        m = coordinates[i, 1] - ave1
        m = max(m, 0.001)
        angle = math.atan((coordinates[i, 0] - ave0) / m)

        coordinatesCircle[i] = [dist, angle, dist * angle]

    coordinates = np.c_[coordinates, coordinatesCircle]

    return coordinates



def PDDistriction(column):
    resColumns = [0] * len(column)
    for i in range(0, len(column)):
        resColumns[i] = [0] * 10
            
        if column[i] in districtions.keys():
            key = districtions[column[i]]
            resColumns[i][key] = 1

    return resColumns



def GetWeathere(month):
    res = tempratureAndRain[month]
    return res[0], res[1]



def IFCorner(adressColumn):
    res = [0] * len(adressColumn)

    for i in range(0, len(adressColumn)):
        if "/" in adressColumn:
            res[i] = 1

    return res



def LabelsBinarization(self, labels):
    la = label_binarize(labels, classes=self.allPosibleLabels)
    return la



def WeekDaysBinarization(column):
    column1 = [0] * len(column)
    for i in range(0, len(column)):
        r = 7
        if column[i] in weekDays.keys():
            r = weekDays[column[i]]

        column1[i] = r

    myset = set(column1)
    mm = list(myset)
    r1 = label_binarize(column1, classes=mm)
    r1 = r1[:,0:7]
    r1 = np.column_stack((r1, column1))

    weekDay = [0] * len(column1)

    for i in range(0, len(column1)):
        weekDay[i] = 0
        if column1[i] == 0 or column1[i] == 6:
            weekDay[i] = 1

    r1 = np.column_stack((r1, weekDay))

    return r1


  
def ToHoursDistribution(tm):
    res = [0] * 24

    hour = math.floor(tm / 60)
    mn = tm - hour * 60
        
    if hour > 0:
        hourBefore = hour - 1
    else:
        hourBefore = 23

    if hour < 23:
        hourAfter = hour + 1
    else:
        hourAfter = 0

    add = mn - 30

    res[hour] = 2 - np.abs(add) / 30
    res[hourAfter] = mn / 60 
    res[hourBefore] = 1 - res[hourAfter]
                
    return res