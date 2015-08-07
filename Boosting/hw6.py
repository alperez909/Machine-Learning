import numpy as np
import copy
import math

def makeList(filename):
    dataPoint = []
    rArray = []
    with open(filename) as f:
        for line in f:
            # Read an individual line
            line =[int(x) for x in line.split()]
            # A list that holds a list of data points
            rArray.append(line)
    return np.array(rArray)

def makeList_(filename):
    rArray = []
    with open(filename) as f:
        for line in f:
            # Read an individual line
            line = line.split()
            # A list that holds a list of data points
            rArray.append(line)
    return np.array(rArray)

def boosting(data, t):
    # Copy of data
    datac = copy.deepcopy(data)
    # Initial D1
    d1 = 1/float(len(datac))
    
    # Labels
    labels = np.array([int(row[len(datac[0])-1]) for row in datac])
    # Feature points
    features = np.delete(datac,len(datac[0])-1,1)
    # D
    d = np.array(len(datac)* [d1])
    # ALL (h,a) pairs for each t
    all_pairs = np.array([])
    
    ti = 0
    while( ti < t ): # Loop over each time
        i = 0
        # Best h
        h = np.array([])
        # Highest accuracy value
        lowest_err = 1
        # Classifier i
        c_i = 0
        # Plus or Minus
        pom = 0
        while( i < len(features[0]) ): # Loop over each word
            x = 0
            # Classified label accumulator
            h_plus = np.array([])
            h_minus = np.array([])
            # Error of the  current classifier for plus/minus
            h_plus_err = 0
            h_minus_err = 0
            while( x < len(datac) ): # Loop over each x
                if (features[x][i] == 1):
                    h_plus = np.append(h_plus,1)
                    h_minus = np.append(h_minus,-1)
                else:
                    h_plus = np.append(h_plus,-1)
                    h_minus = np.append(h_minus,1)
                x = x + 1
            # Caluclate errors
            vp = ((h_plus + labels)/2)
            v = np.array([1 if x == 0 else 0 for x in vp])
            h_plus_err = np.dot( v , d )
            if( h_plus_err < lowest_err ):
                lowest_err = h_plus_err
                h = h_plus
                c_i = i
                pom = 1
            vm = ((h_minus + labels)/2)
            v = np.array([1 if x == 0 else 0 for x in vm])
            h_minus_err = np.dot( v , d )
            if( h_minus_err < lowest_err):
                lowest_err = h_minus_err
                h = h_minus
                c_i = i
                pom = -1
            i = i + 1
        # Error of best classifier
        vb = ((h + labels)/2)
        v = np.array([1 if x == 0 else 0 for x in vb])
        h_err = np.dot(v , d )
        # Alpha of best classifier
        a = 0.5*(math.log((1.0 - h_err)/h_err))
        # Update d's
        inc = 0
        while(inc < len(d)):
            d[inc] = d[inc]*math.exp(-1*a*labels[inc]*h[inc])
            inc = inc + 1
        s = sum(d)
        d = d/s
        # All pairs
        all_pairs = np.append(all_pairs,[c_i,pom,a])
        ti = ti + 1
    return np.reshape(all_pairs,(t,3))

def classify(training_data, data_classifying, t):
    # Train data
    classifier = boosting(training,t)
    datac = copy.deepcopy(data_classifying)
    
    # Labels
    labels = np.array([int(row[len(datac[0])-1]) for row in datac])
    # Feature points
    features = np.delete(datac,len(datac[0])-1,1)
    h = np.array([])
    
    x = 0
    while (x < len(datac)): # Loop over each x
        ti = 0
        sum = 0
        while (ti < t):
            if (features[x][int(classifier[ti][0])] == 1):
                if (int(classifier[ti][1]) == 1):
                    sum = sum + classifier[ti][2]*1
                else:
                    sum = sum + classifier[ti][2]*-1
            else:
                if (int(classifier[ti][1]) == 1):
                    sum = sum + classifier[ti][2]*-1
                else:
                    sum = sum + classifier[ti][2]*1
            ti = ti + 1
        h = np.append(h, math.copysign(1, sum))
        x = x + 1
    return h

# Calculate error
def err(data1,classified_labels):
    labels1 = np.array([int(row[len(data1[0])-1]) for row in data1])

    i = 0
    num_errors = 0
    while i < len(labels1):
        if labels1[i] != classified_labels[i]:
            num_errors = num_errors + 1
        i = i + 1
    return (num_errors/float(len(labels1)))

def rounds(data,dict,t):
    b = boosting(training,t)
    i = 0
    sign = ''
    while( i < t):
        if(int(b[i][1] == 1)):
           sign = '+'
        else:
           sign = '-'
        print 'Round:', i+1,'Word:',dict[int(b[i][0])][0],'h_plus/h_minus:',sign
        i = i+1

##################################
training = makeList('hw6train.txt')
test = makeList('hw6test.txt')
dictionary = makeList_('hw6dictionary.txt')
'''
classified_labels_train3 = classify(training,training,3)
classified_labels_test3 = classify(training,test,3)
print '{ t = 3 | training err = ', err(training,classified_labels_train3), ' }'
print '{ t = 3 | test err     = ', err(test,classified_labels_test3), ' }'

classified_labels_train7 = classify(training,training,7)
classified_labels_test7 = classify(training,test,7)
print '{ t = 7 | training err = ', err(training,classified_labels_train7), ' }'
print '{ t = 7 | test err     = ', err(test,classified_labels_test7), ' }'

classified_labels_train10 = classify(training,training,10)
classified_labels_test10 = classify(training,test,10)
print '{ t = 10 | training err = ', err(training,classified_labels_train10), ' }'
print '{ t = 10 | test err     = ', err(test,classified_labels_test10), ' }'

classified_labels_train15 = classify(training,training,15)
classified_labels_test15 = classify(training,test,15)
print '{ t = 15 | training err = ', err(training,classified_labels_train15), ' }'
print '{ t = 15 | test err     = ', err(test,classified_labels_test15), ' }'

classified_labels_train20 = classify(training,training,20)
classified_labels_test20 = classify(training,test,20)
print '{ t = 20 | training err = ', err(training,classified_labels_train20), ' }'
print '{ t = 20 | test err     = ', err(test,classified_labels_test20), ' }'
'''

rounds(training,dictionary,10)