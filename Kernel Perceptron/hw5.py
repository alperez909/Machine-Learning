import numpy as np
import itertools
import math
import copy
from sets import Set

# Make a list for each data file
def makeList(filename):
    dataPoint = []
    rArray = []
    with open(filename) as f:
        for line in f:
            # Read an individual line
            line =[x for x in line.split()]
            # A list that holds a list of data points
            rArray.append(line)
    rArray = np.array(rArray)
    return rArray

training = makeList('hw5train.txt')
test = makeList('hw5test.txt')


def subStrings(string,p):
    s = Set([])
    i = 0
    j = p
    while(j <= len(string)):
        s.add(string[i:j])
        i = i + 1
        j = i + p
    return s

s1 =  subStrings('abcda',1)
s2 =  subStrings('ab',1)

def numCommon(set1,set2):
    s = set1.intersection(set2)
    return len(s)

print numCommon(s1,s2)

print s1


def allSS(data,p):
    ls = []
    i = 0
    while(i < len(data)):
        ls.append( [ subStrings( (data[i][0]), p ), data[i][1] ] )
        i = i + 1
    return ls

def perceptron(data1,p):
    datac = copy.deepcopy(data1)
    features = np.delete(datac,len(datac[0])-1,1)
    labels = np.array([int(float(row[len(data1[0])-1])) for row in data1])
    k = np.array([features[0][0], labels[0]])
    k =  k.reshape(1,2)
    print k
    
    i = 1
    while( i < len(datac)):
        j = 0
        w = 0
        while( j < len(k)):
            s1 = subStrings(  k[0][j], p  )
            #print s1
            #print features[i]
            s2 = subStrings( features[i][0], p )
            #print s2
            w = w + labels[i]*numCommon( s1 , s2  )
            j = j + 1
        if ( w <= 0):
            k = np.append(k, [features[i][0], labels[i]])
            k = k.reshape(len(k),2)
        i = i + 1
    return k

print perceptron(training,1)


'''
def perceptron(data1,p):
    classifier = allSS(data1,p) # point a against all other points
    labels = np.array([int(float(row[len(data1[0])-1])) for row in data1])
    data = copy.deepcopy(data1)
    ls = []
    dot_product = 0
    
    i = 1
    while( i < len(classifier)):
        j = 0
        while( j < len(classifier)):
            dot_product = dot_product + numCommon(classifier[i][0], classifier[j][0])*labels[j]
            j  = j + 1
        print dot_product, i
        data[i][1] = str(math.copysign(1,dot_product))
        dot_product = 0
        i = i + 1
    return data

#training_ = perceptron(training,4)
'''

'''
def classify(data1,data2,p):
    classifier = allSS(data1,p) # all points in classifier against 1_d
    d = allSS(data2,p)
    d_ = copy.deepcopy(data2)
    labels1 = np.array([int(float(row[len(data1[0])-1])) for row in data1])
    sum = 0

    i = 0
    while(i < len(d)):
        j = 0
        while( j < len(classifier) ):
            sum = sum + numCommon(classifier[j][0], d[i][0])*labels1[j]
            j  = j + 1
        d_[i][1] = str(math.copysign(1,sum))
        sum = 0
        i = i + 1
    return d_
'''

#training_ =  classify(training,training,4)

def err(data1,data2):
    labels1 = np.array([int(float(row[len(data1[0])-1])) for row in data1])
    labels2 = np.array([int(float(row[len(data2[0])-1])) for row in data2])
    
    i = 0
    num_errors = 0
    while i < len(labels1):
        if labels1[i] != labels2[i]:
            num_errors = num_errors + 1
        i = i + 1
    return (num_errors/float(len(labels1)))

#err(training,training_)





