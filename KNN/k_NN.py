from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


def creatDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 距离度量，欧式距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5

    # 将距离排序，从大到小
    sortedDistIndicies = distances.argsort()
    # 选取前k个最短距离
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount

# 将文本数据转化为numpy
def file2matrix(filename):
    fr = open(filename) # 打开文件
    
    numberOfLines = len(fr.readlines()) # 获取文件中的行数
    returnMat = zeros((numberOfLines, 3)) # 生成对应的矩阵
    classLabelVector = [] # 标签
    index = 0
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip() # 删除字符串头尾指定字符，默认空格或换行，strip([chars])
        listFromLine = line.split('\t') # 以'\t'切割字符 str.split(str = "", num = string.count(str))
        returnMat[index, :] = listFromLine[0:3] # 每列的属性
        classLabelVector.append(int(listFromLine[-1])) # label
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


if __name__ == '__main__':
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    print(normMat)
    print(ranges, minVals)

