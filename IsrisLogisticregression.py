#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 18:04:00 2018

@author: risyadav
"""

import numpy as np 
from numpy import genfromtxt 
from sklearn import linear_model
import csv

file = genfromtxt("iris.txt", delimiter = ",", dtype = "str")
dic = {}
count = 0

for val in file:
    if val[4] not in dic:
        dic[val[4]] = count
        count +=1



for val in file:
    val[4] = dic[val[4]]

print(file)
trainingSet = file[:130]
testingSet = file[130:] 

trainingX = trainingSet[:,[0,1,2,3]]
trainingX = trainingX.astype(float)
trainingY = trainingSet[:,[4]]

testingX = testingSet[:,[0,1,2,3]]
testingX = testingX.astype(float)
testingY = testingSet[:,[4]]

lr = linear_model.LogisticRegression()
lr.fit(trainingX,trainingY)

print(testingX[12])

print(lr.predict([testingX[4]]))
print('real value is' + str(testingY[4]))

lr.score(testingX, testingY)*100




