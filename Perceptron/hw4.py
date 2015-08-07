import numpy as np
import math
import copy
from collections import Counter

# Make a list for each data file
def makeList(filename):
    dataPoint = []
    rArray = []
    with open(filename) as f:
        for line in f:
            # Read an individual line
            line =[float(x) for x in line.split()]
            # A list that holds a list of data points
            rArray.append(line)
    rArray = np.array(rArray)
    return rArray

# Original Perceptron
def perceptron(data,passes,posLabel):
    datac = copy.deepcopy(data)
    # Decide passes of data
    datac = np.array(passes*(datac.tolist()))
    # Get labels and make them binary
    labels = np.array([int(row[len(datac[0])-1]) for row in datac])
    
    # Select first element in vector and set it to -1
    labels[labels!=posLabel] = -1
    # Set all the other elements in vector to 1
    labels[labels==posLabel] = 1
    # Get feature vectors
    features = np.delete(datac,len(datac[0])-1,1)

    
    #Initial w1
    w = np.zeros(len(features[0]))
    
    i = 0
    while i < len(features):
        if labels[i]*np.dot(w,features[i]) <= 0:
            w = w + labels[i]*features[i]
        i = i + 1
    return w

# Predict labels for the data given w classifier
def predictLabels(data,w):
    #labels = np.array([int(row[len(data[0])-1]) for row in data])
    datac = copy.deepcopy(data)
    features = np.delete(datac,len(datac[0])-1,1)

    i = 0
    while i < len(features):
        datac[i][len(datac[0])-1] = math.copysign(1,np.dot(w,features[i]))
        i = i + 1
    return datac

# Calculate error
def err(data1,data2,posLabel):
    labels1 = np.array([int(row[len(data1[0])-1]) for row in data1])
    labels1[labels1!=posLabel] = -1
    labels1[labels1==posLabel] = 1
    
    labels2 = np.array([int(row[len(data2[0])-1]) for row in data2])

    i = 0
    num_errors = 0
    while i < len(labels1):
        if labels1[i] != labels2[i]:
            num_errors = num_errors + 1
        i = i + 1
    return (num_errors/float(len(labels1)))

####################################################
def voted_avg_perc(data,passes,posLabel):
    datac = copy.deepcopy(data)
    # Decide passes of data
    datac = np.array(passes*(datac.tolist()))
    
    # Get labels and make them binary
    labels = np.array([int(row[len(datac[0])-1]) for row in datac])
    # Select first element in vector and set it to -1
    labels[labels!=posLabel] = -1
    # Set all the other elements in vector to 1
    labels[labels==posLabel] = 1

    # Get feature vectors
    features = np.delete(datac,len(datac[0])-1,1)

    #Initial w1
    w = np.zeros(len(features[0]))
    ws = np.array([])
    
    i = 0
    while i < len(features):
        if labels[i]*np.dot(w,features[i]) <= 0:
            w = w + labels[i]*features[i]
            c = 1
            ws = np.append(ws,w)
            ws = np.append(ws,c)
        else:
            cm = ws[len(ws)-1]
            cm = cm+1
            ws[len(ws)-1] = cm
        i = i + 1
    return ws.reshape(len(ws)/(len(w) + 1), len(w) + 1)

def predictLabels_v(data,ws):
    datac = copy.deepcopy(data)
    features = np.delete(datac,len(datac[0])-1,1)
    ci = np.array([row[len(ws[0])-1] for row in ws])
    wi = np.delete(ws,len(ws[0])-1,1)
    labels_predicted = np.array([])

    i = 0
    # Loop over features
    while i < len(features):
        j = 0
        # Loop over ws
        while j < len(ws):
            labels_predicted = np.append( labels_predicted , ci[j]*math.copysign(1,np.dot(wi[j],features[i])) )
            j = j + 1
        ''' IF WE NEED TO FIND MAJORITY
        # Majority
        c = Counter(labels_predicted)
        # Update sign
        datac[i][len(datac[0])-1] = c.most_common(1)[0][0]
        '''
        datac[i][len(datac[0])-1] = math.copysign(1,sum(labels_predicted))
        # Reset
        labels_predicted = np.array([])
        i = i + 1
    return datac


#######################################################
def predictLabels_avg(data,ws):
    datac = copy.deepcopy(data)
    features = np.delete(datac,len(datac[0])-1,1)
    ci = np.array([row[len(ws[0])-1] for row in ws])
    wi = np.delete(ws,len(ws[0])-1,1)
    labels_predicted = np.array([])

    wsum = np.zeros(len(wi[0]))
    j = 0
    while j < len(ws):
        wsum = ci[j]*wi[j] + wsum
        j = j + 1
    
    i = 0
    # Loop over features
    while i < len(features):
        j = 0
        # Update sign
        datac[i][len(datac[0])-1] = math.copysign(1,np.dot(wsum,features[i]))
        i = i + 1
    return datac

#####################################################
# Read training data
training = makeList('hw4atrain.txt')
# Read test data
test = makeList('hw4atest.txt')


#training = np.array([[1,1,2],[2,1,1]])
#test = np.array([[-1,-2,1], [4,2,2], [2,-5,1]])
#posLabel = 2
#ws = voted_avg_perc(training,1,posLabel)
#print predictLabels_v(training,ws)

#### Perceptron
posLabel = 0

w = perceptron(training,1,posLabel)
training_1 = predictLabels(training,w)
test_1 = predictLabels(test,w)

w = perceptron(training,2,posLabel)
training_2 = predictLabels(training,w)
test_2 = predictLabels(test,w)

w = perceptron(training,3,posLabel)
training_3 = predictLabels(training,w)
test_3 = predictLabels(test,w)

print "{ Preceptron, Pass = 1, training error:", err(training,training_1,posLabel), "}"
print "{ Preceptron, Pass = 1, test error:    ", err(test,test_1,posLabel), "}"
print "{ Preceptron, Pass = 2, training error:", err(training,training_2,posLabel), "}"
print "{ Preceptron, Pass = 2, test error:    ", err(test,test_2,posLabel), "}"
print "{ Preceptron, Pass = 3, training error:", err(training,training_3,posLabel), "}"
print "{ Preceptron, Pass = 3, test error:    ", err(test,test_3,posLabel), "}"

print ""

ws = voted_avg_perc(training,1,posLabel)
training_1 = predictLabels_v(training,ws)
test_1 = predictLabels_v(test,ws)

ws = voted_avg_perc(training,2,posLabel)
training_2 = predictLabels_v(training,ws)
test_2 = predictLabels_v(test,ws)

ws = voted_avg_perc(training,3,posLabel)
training_3 = predictLabels_v(training,ws)
test_3 = predictLabels_v(test,ws)

print "{ Voted Preceptron, Pass = 1, training error:", err(training,training_1,posLabel), "}"
print "{ Voted Preceptron, Pass = 1, test error:    ", err(test,test_1,posLabel), "}"
print "{ Voted Preceptron, Pass = 2, training error:", err(training,training_2,posLabel), "}"
print "{ Voted Preceptron, Pass = 2, test error:    ", err(test,test_2,posLabel), "}"
print "{ Voted Preceptron, Pass = 3, training error:", err(training,training_3,posLabel), "}"
print "{ Voted Preceptron, Pass = 3, test error:    ", err(test,test_3,posLabel), "}"


print ""

ws = voted_avg_perc(training,1,posLabel)
training_1 = predictLabels_avg(training,ws)
test_1 = predictLabels_avg(test,ws)

ws = voted_avg_perc(training,2,posLabel)
training_2 = predictLabels_avg(training,ws)
test_2 = predictLabels_avg(test,ws)

ws = voted_avg_perc(training,3,posLabel)
training_3 = predictLabels_avg(training,ws)
test_3 = predictLabels_avg(test,ws)

print "{ Averaged Preceptron, Pass = 1, training error:", err(training,training_1,posLabel), "}"
print "{ Averaged Preceptron, Pass = 1, test error:    ", err(test,test_1,posLabel), "}"
print "{ Averaged Preceptron, Pass = 2, training error:", err(training,training_2,posLabel), "}"
print "{ Averaged Preceptron, Pass = 2, test error:    ", err(test,test_2,posLabel), "}"
print "{ Averaged Preceptron, Pass = 3, training error:", err(training,training_3,posLabel), "}"
print "{ Averaged Preceptron, Pass = 3, test error:    ", err(test,test_3,posLabel), "}"

print ""
####################################################

# Read training data
training = makeList('hw4btrain.txt')
# Read test data
test = makeList('hw4btest.txt')

#training = [[1,2],[2,2],[3,3],[4,1],[5,0],[6,0],[7,0],[8,3]]

def ova_classifier(data,passes,pos_begin,pos_end):
    classifiers = perceptron(data,passes,pos_begin)
    pos_begin = pos_begin + 1
    
    while pos_begin <= pos_end:
        wi = perceptron(data,passes,pos_begin)
        classifiers = np.vstack((classifiers,wi))
        pos_begin = pos_begin + 1
    return classifiers

def predictLabels_ova(data,ws):
    datac = copy.deepcopy(data)
    features = np.delete(datac,len(datac[0])-1,1)
    predicted = 0

    i = 0
    while i < len(features):
        j = 0
        while j < len(ws):
            m = math.copysign(1,np.dot(ws[j],features[i]))
            #print "p:",predicted, "j:",j,"i:",i,"sign:",m
            if  1 == m:
                if predicted == 0:
                    datac[i][len(datac[0])-1] = j
                    predicted = predicted + 1
                else:
                    datac[i][len(datac[0])-1] = 10
                    break
            else:
                if predicted == 0:
                    datac[i][len(datac[0])-1] = 10
            j = j + 1
        predicted = 0
        i = i + 1
    return datac

def err_ova(data1,data2):
    labels1 = np.array([int(row[len(data1[0])-1]) for row in data1])
    labels2 = np.array([int(row[len(data2[0])-1]) for row in data2])
    
    i = 0
    num_errors = 0
    while i < len(labels1):
        if labels1[i] != labels2[i]:
            num_errors = num_errors + 1
        i = i + 1
    return (num_errors/float(len(labels1)))

# Return a list containing the count of each label
def countLables(data):
    count = 0
    all_counts = []
    labels = np.array([int(row[len(data[0])-1]) for row in data])
    i = 0
    # Loop over all labels
    while i < 10:
        j = 0
        count = 0
        # Loop every label in data and check if
        # it equals the i label
        while j < len(data):
            if i == labels[j]:
                count = count + 1
            j = j+1
        all_counts.append(count)
        i = i+1
    return all_counts

# Create the confusion matrix
def createMatrix(data1,data2):
    labels1 = np.array([int(row[len(data1[0])-1]) for row in data1])
    labels2 = np.array([int(row[len(data2[0])-1]) for row in data2])
    count_l = countLables(data1)
    #print count_l
    i = 0
    # A list of all C_ij
    all_cij = []
    # i rows
    while i < 11:
        j = 0
        # j columns
        while j < 10:
            k = 0
            c_ij = 0
            # Loop that counts all points that have label j
            # in test data and i in classified data
            while k < len(data1):
                if j == labels1[k] and i == labels2[k]:
                    c_ij = c_ij + 1
                k = k + 1
            # Add c_ij/Nj to list
            all_cij.append(c_ij/float(count_l[j]))
            j = j + 1
        i = i + 1
    # Transform list into matrix
    datar = np.array(all_cij)
    shape = (11,10)
    return datar.reshape(shape)


ws = ova_classifier(training,1,0,9)

r = predictLabels_ova(test,ws)

matr = createMatrix(test,r)
print matr

