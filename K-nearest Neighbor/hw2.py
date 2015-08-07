import numpy as np
import math
import copy

# Make a list for each data file
def makeList(filename):
    dataPoint = []
    rArray = []
    with open(filename) as f:
        for line in f:
            # Read an individual line
            line =[int(x) for x in line.split()]
            # Split line list into the elements that are a feature vector
            feature = line[:784]
            # Split line list into the elements that are a label
            label = line[784:]
            # Make a data point a list
            dataPoint.append(feature)
            dataPoint.append(label)
            # A list that holds a list of data points
            rArray.append(dataPoint)
            # Reset data point for the next line
            dataPoint=[]
    return rArray

# Calculate the distance between two data points for two seperte data files.
# Data indicies correspond to a data point at the index for their respective data file
def distance(data_1, data_2, data1_index, data2_index):
    x1 = np.array(data_1[data1_index][0])
    x2 = np.array(data_2[data2_index][0])
    # Subtracts (x1_i - x2_i)
    xs = x1-x2
    # Squares (x1_i - x2_i)^2
    s = np.square(xs)
    # Sums (x1_1 - x2_1)^2 + ... + (x1_n - x2_n)^2)
    f = sum(s)
    # Returns sqrt(x1_1 - x2_1)^2 + ... + (x1_n - x2_n)^2)
    return math.sqrt(f)

# Calculte the k nearest neighbor in data1 for a single point in data2
def knn_foreach(k, data1, data2, data2_index):
    i = 0
    # List to contain the distances from a single point to the a point, i.
    dists = []
    # Loop over all points in data1
    while(i<len(data1)):
        d = distance(data1,data2,i,data2_index)
        dists.append(d)
        i = i + 1
    # A list containing indicies of all distances between the single point in data2
    # and all the points in data1, in increasing order.
    sorted_index = sorted(range(len(dists)), key=lambda k: dists[k])
    i = 0
    nearest = []
    # Loop to find the k nearest points
    while(i < k):
        index = sorted_index[0]
        nearest = nearest + data1[index][1]
        sorted_index.pop(0)
        i=i+1
    # Find the label that appears the most often in its nearest neighbors
    maxV = max(set(nearest), key=nearest.count)
    # Update label
    data2[data2_index][1] = [maxV]

# Calculate the k nearest neighbor for all data points in data2
def knn_forall(k,data1,data2):
    i = 0
    while i < len(data2):
        knn_foreach(k,data1,data2,i)
        i = i+1

# Calculate the error of the k nearest neighbor
def error(k,data1,data2):
    # Compute the k-nearest neighbor
    data2_before=copy.deepcopy(data2)
    knn_forall(k,data1,data2)
    i = 0
    num_errors = 0
    # Loop to find where original data and
    # classified data differ
    while i < len(data2):
        if data2_before[i][1] != data2[i][1]:
            num_errors = num_errors + 1
        i = i + 1
    return (num_errors / float(len(data2)))

# Return a list containing the count of each label
def countLables(data):
    count = 0
    all_counts = []
    i = 0
    # Loop over all labels
    while i < 10:
        j = 0
        count = 0
        # Loop every label in data and check if
        # it equals the i label
        while j < len(data):
            if i in data[j][1]:
                count = count + 1
            j = j+1
        all_counts.append(count)
        i = i+1
    return all_counts

# Create the confusion matrix
def createMatrix(data):
    # A copy of data
    data_before = copy.deepcopy(data)
    # A list of the count of each label
    labels = countLables(data)
    # Nearest neighbor classifier
    knn_forall(3,training,data)
    i = 0
    # A list of all C_ij
    all_cij = []
    # i rows
    while i < 10:
        j = 0
        # j columns
        while j < 10:
            k = 0
            c_ij = 0
            # Loop that counts all points that have label j
            # in test data and i in classified data
            while k < len(data):
                if j in data_before[k][1] and i in data[k][1]:
                    c_ij = c_ij + 1
                k = k + 1
            # Add c_ij/Nj to list
            all_cij.append(c_ij/float(labels[j]))
            j = j + 1
        i = i + 1
    # Transform list into matrix
    data = np.array(all_cij)
    shape = (10,10)
    return data.reshape(shape)


training = makeList('hw2train.txt')
test = makeList('hw2test.txt')
validate = makeList('hw2validate.txt')

'''
# Training Errors
k1_training_error = error(1,training,training)
print k1_training_error

k3_training_error = error(3,training,training)
print k3_training_error

k5_training_error = error(5,training,training)
print k5_training_error

k11_training_error = error(11,training,training)
print k11_training_error

k16_training_error = error(16,training,training)
print k16_training_error

k21_training_error = error(21,training,training)
print k21_training_error

# Validation Errors
k1_validate_error = error(1,training,validate)
print k1_validate_error

k3_validate_error = error(3,training,validate)
print k3_validate_error

k5_validate_error = error(5,training,validate)
print k5_validate_error

k11_validate_error = error(11,training,validate)
print k11_validate_error

k16_validate_error = error(16,training,validate)
print k16_validate_error

k21_validate_error = error(21,training,validate)
print k21_validate_error

# Test Error
k1_test_error = error(1,training,test)
print k1_test_error

'''

# Confusion Matrix
confusion = createMatrix(test)
print confusion


