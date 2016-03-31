'''
# it works

import urllib, json

def geocode(addr):
    url = "http://maps.googleapis.com/maps/api/geocode/json?address=%s&sensor=false" %   (urllib.quote(addr.replace(' ', '+')))
    data = urllib.urlopen(url).read()
    info = json.loads(data).get("results")[0].get("geometry").get("location")

    return info

address = 'Constitution Ave NW & 10th St NW, Washington, DC'

r = geocode(address)
print "%s %s" % (r['lat'], r['lng'])
'''

import math 
import csv
from time import sleep
import pandas
import numpy as np
import requests
import json


def DistanceToTheNearestRoad(lat, lng):
    str1 = 'https://maps.googleapis.com/maps/api/directions/json?origin={0},{1}&destination=37.7,-122.0'.format(lat, lng)
    #print(str1)
    r = requests.get(str1, auth=('user', 'pass'))
    data = json.loads(r.text)
    
    lat1 = data["routes"][0]["legs"][0]["steps"][0]["start_location"]["lat"]
    lng1 = data["routes"][0]["legs"][0]["steps"][0]["start_location"]["lng"]
    
    res = math.sqrt((lat - lat1) * (lat - lat1) + (lng - lng1) * (lng - lng1))
    return res * 135000

    
    

dist = DistanceToTheNearestRoad(37.761295, -122.413870) # 100
print(dist)
'''
dist = DistanceToTheNearestRoad(37.761286, -122.412754) # 0
print(dist)
dist = DistanceToTheNearestRoad(37.535429, -122.476326) # 0
print(dist)
'''

df = np.load("C:\\python_kaggle\\crime_spyder_project\\dump_allData.npy")
print(df[0])
print(df[1])
print(len(df))

dat = np.c_[df[:,0], df[:,14:16]]
print(dat[0])

res= []* len(dat)
for i in range(0, len(dat)):
    dist = DistanceToTheNearestRoad(dat[0], dat[1])
    res = [i, dist]
    if (i / 10 == i // 10):
        np.save("C:\\python_kaggle\\crime_spyder_project\\dump_allData_nearest.npy", res)
        
    sleep(1)
