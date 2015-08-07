import numpy as np
import random
from collections import Counter
import copy
import scipy

class Node:
    """
    Tree node: left and right child + parent + threshold
    and feature pair (if any) + data set + label predicted
    (if any) + boolean to check if a nodes value has been
    predicted or that of both its children
    """
    def __init__(self, parent, threshold, data):
        self.left = None
        self.right = None
        self.feat = None
        self.parent = parent
        self.data = data
        self.vpredicted = None
        self.predicted = False
        self.threshold = threshold
    
    def insertLeft(self, threshold, data):
        self.left = Node(self, threshold, data)

    def insertRight(self, threshold, data):
        self.right = Node(self, threshold, data)

    def getData(self):
        return self.data
    
    def getParent(self):
        return self.parent

    def setPredicted(self,predicted):
        self.predicted = predicted

    def getPredicted(self):
        return self.predicted
    
    def setFeat(self,feat):
        self.feat = feat
    
    def getFeat(self):
        return self.feat

    def setThreshold(self,threshold):
        self.threshold = threshold

    def getThreshold(self):
        return self.threshold

    def set_vpredicted(self,vpredicted):
        self.vpredicted= vpredicted
    
    def get_vpredicted(self):
        return self.vpredicted


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

# Find the midpoint between two values in a feature
def midPoint(x, p1, p2):
    return (x[p2]+x[p1])/2.0

# Compute entropy of the data
def computeEntropy(data):
    # Lables of the data set
    labels = np.array([int(row[len(data[0])-1]) for row in data])
    # Total number of elements in the data
    total = float(len(data))
    # Frequency of each label
    frequency = (Counter(labels).items())
    
    i = 0
    h = 0.0
    while i < len(frequency):
        # Calculates the entropy
        h = ((frequency[i][1]/total)*np.log(frequency[i][1]/total)) + h
        i = i + 1
    return (h*-1)

# Split data into data less and data greater than threshold
def split_point(data, feature_num, threshold):
    # Hold less than and greater than values
    ldata = []
    gdata = []
    i = 0
    # Feature column vector
    feature = np.array([row[feature_num] for row in data])
    # Loop to find values greater and less than threshold
    while i < len(feature):
        if feature[i] <= threshold:
            ldata.append(data[i])
        else:
            gdata.append(data[i])
        i = i+1
    # Return two matricies of the split data
    return np.array([np.array(ldata), np.array(gdata)])

# Probability between two data sets
def prob(data1,data2):
    return len(data1)/float(len(data2))


# Create a decision tree from given data and node
def createTree(data, node):
    # Make node the currentNode
    currentNode = node

    # Labels and frequency of each label
    labels = np.array([int(row[len(data[0])-1]) for row in data])
    frequency = (Counter(labels).items())
    
    # Check if data has the same label for all data points.
    # If it does predict label
    if len(frequency) == 1:
        currentNode.setPredicted(True)
        currentNode.set_vpredicted(frequency[0][0])
        return

    # Array to contain the entropy of each threshold
    all_thresholds = []
    # Array
    pair = []
    # Feature
    feature_num = 0
    # Highest information gained
    highest_ig = 0
    # Feature containing the highest information gained
    highest_feature_num = 0
    # Threshold which gave the highest information gained
    t = 0
    # Loop over all features to find the best feature to split data
    while feature_num < len(data[0]) - 1:
        i = 0
        # Get feature from data
        feature = np.array([row[feature_num] for row in data])
        # Used to find possible thresholds
        feature_s = np.array(sorted(set(feature)))
        # Check if there is more than one data point in the feature vector.
        # Continue if there is
        if len (feature_s) > 1:
            # Loop over a single feature to find the highest information
            # gained within that feature
            while i < len(feature_s) - 1:
                # Threshold of possible split point
                threshold = midPoint(feature_s,i,i+1)
                # Split data into two data sets
                matricies = split_point(data, feature_num, threshold)
                # Calculate entropy of each side and add them together
                lentropy = computeEntropy(matricies[0])
                gentropy = computeEntropy(matricies[1])
                # p_less = number of elements in matrix[0]/ number of elements in feature
                p_lesser = prob(matricies[0], feature)
                p_greater = prob(matricies[1], feature)
                # Conditional Probability given threshold
                conditional_entropy = lentropy*(p_lesser) + gentropy*(p_greater)
                # Calculate entropy of original data and subtract from the entropy above which will give information gained
                oentropy = computeEntropy(data)
                # Information gained at the threshold
                ig = oentropy - conditional_entropy
                # IG and Threshold value pair.
                pair = [ig, threshold]
                # Append to a array with all ig and threshold pairs in order to compare and get the highest information gained
                all_thresholds.append(pair)
                # insert into all_thresholds
                i = i+1
        # The case when there is one data point in the feature
        else:
            ig = 0.0
            print feature_s[0], 'feature'
            pair = [ig, feature_s[0]]
            print pair, 'this is pair'
            all_thresholds.append(pair)
        # Compare every threshold from every feature and select the best one
        ig_cv = [row[0] for row in all_thresholds]
        index = ig_cv.index(max(ig_cv))
        if highest_ig < all_thresholds[index][0]:
            t = all_thresholds[index][1]
            highest_ig = all_thresholds[index][0]
            highest_feature_num = feature_num
        feature_num = feature_num + 1
    ############### set up node feature and threshold
    currentNode.setFeat(highest_feature_num)
    currentNode.setThreshold(t)
    ###############
    # Split at that point
    matricies = split_point(data, highest_feature_num, t)
    # Make the left child of root the left of the matricies returned by split
    currentNode.left = Node(currentNode, None, matricies[0])
    # Make the right child of root the right of the matricies returned by split
    currentNode.right = Node(currentNode, None, matricies[1])
    # Make the left child the currentNode
    currentNode = currentNode.left

    # repeat procedure in depth first order
    createTree(currentNode.data, currentNode)
    
    # When tree returns set current node to its parent
    currentNode = currentNode.parent

    # If at a parent node there are three cases.
    # 1. If root return
    # 2. If both children have predicted all labels move to the parent node
    # 3. If both children are not predicted move to right child
    if currentNode.left is not None and currentNode.right is not None:
        if currentNode.left.predicted is True and currentNode.right.predicted is True:
            if currentNode.parent is None:
                return
            else:
                currentNode = currentNode.parent
                createTree(currentNode.data, currentNode)
        else:
            currentNode = currentNode.right
            createTree(currentNode.data, currentNode)

# Predict label given a tree and data set
def predictLabel(data, tree):
    i = 0
    currentNode = tree
    # Loop over all feature row vecotrs
    while i < len(data):
        # Loop until feature row vector predicts a label
        while currentNode.threshold is not None or currentNode.feat is not None:
            if data[i][currentNode.feat] <= currentNode.threshold:
                currentNode = currentNode.left
            else:
                currentNode = currentNode.right
        # Get label predicted and set it in data
        if currentNode.vpredicted is not None:
            plabel = currentNode.vpredicted
            data[i][len(data[0])-1] = plabel
        # Set currentNode back to root
        currentNode = tree
        i = i + 1

# Calcualte error
def error(data, tree):
    # Labels before classification
    labelsBefore = np.array([int(row[len(data[0])-1]) for row in data])
    predictLabel(data,tree)
    # Labels after classification
    labelsAfter = np.array([int(row[len(data[0])-1]) for row in data])
    i = 0
    count = 0
    # Count the number labels that differ
    while i < len(labelsBefore):
        if labelsBefore[i] != labelsAfter[i]:
            count = count + 1
        i = i + 1
    # Calculate the error given the number of differences
    # and the length of the data set
    e = count/float(len(labelsBefore))
    print e

# Print tree to stdout.
# Represented in depth first order
global order
order = 0
def treeP(node):
    global order
    if order == 0:
        print 'Depth First'
    order = order+1
    if node.threshold is not None:
        print '%i: [(x%i <= %g), %i] ' %(order, node.feat, node.threshold, len(node.data))
    if node.vpredicted is not None:
        print '%i: [Predicted: %i, %i] ' %(order, node.vpredicted, len(node.data))
        return
    node1 = node.left
    node2 = node.right
    treeP(node1)
    treeP(node2)

#################################

# Read training data
training = makeList('hw3train.txt')
# Read test data
test = makeList('hw3test.txt')
# Set a root node with all data
root = Node(None, None, training)
# Create tree
createTree(training,root)

# Print Tree
treeP(root)

print ''

# Print test error
error(test,root)



