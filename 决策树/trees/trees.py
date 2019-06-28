from math import log
import operator

# 计算数据的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) # 数据集实例的个数
    labelCounts = {} # 用字典记录标签的种类和个数
    for featVec in dataSet:
        currentLabel = featVec[-1] # 数据最后就是标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries # 每种类别的概率
        shannonEnt -= prob * log(prob, 2) # 用公式计算香农熵
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 按照特定属性划分数据集
def splitDataSet(dataSet, axis, value):
    # dataSet要划分的数据集合,axis数据的哪一列属性,value属性值
    retDataSet = [] # 设置新的集合,防止改动原来的,要多次调用
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1 : ]) # 通过切片,将axis列的属性删除,得到新的数据
            retDataSet.append(reduceFeatVec) # 注意extend与append的区别
    return retDataSet

# 选择最好的划分方式,数据要求：由列表元素组成，且列表长度相同，最后一列是label
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 # 求第一行的长度，即属性个数(最后一行是label)
    baseEntropy = calcShannonEnt(dataSet) # 原数据的香农熵
    bestInfoGain = 0.0 # 最优的信息增益值
    bestFeature = -1 # 最优的feature编号
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] # 在dataSet中按行，放example[i]入featList，即取i列的属性放入list中
        uniqueVals = set(featList) # 创建无序不重复的list
        newEntropy = 0.0
        # 对每个唯一的特征值划分一次数据集，计算新熵值，并求和
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) 
            prob = len(subDataSet) / float(len(dataSet)) # 计算概率  !!!不是很理解
            newEntropy += prob * calcShannonEnt(subDataSet) # 求信息熵
        infoGain = baseEntropy - newEntropy # 信息增益
        # 取最优解和相应的索引值
        if (infoGain > bestInfoGain): 
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel : {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
