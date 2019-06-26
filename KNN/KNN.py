from numpy import *
import operator

# 将文本数据转化为numpy
def file2matrix(filename):
    fr = open(filename) # 打开文件
    numberOfLines = len(fr.readlines()) # 获取文件中的行数
    returnMat = zeros((numberOfLines, 3)) # 生成对应的矩阵
    classLabelVector = [] # 标签
    index = 0
    fr = open(filename) # 不知道为啥不重新打开就错误
    for line in fr.readlines():
        line = line.strip() # 删除字符串头尾指定字符，默认空格或换行，strip([chars])
        listFromLine = line.split('\t') # 以'\t'切割字符 str.split(str = "", num = string.count(str))
        returnMat[index, :] = listFromLine[0:3] # 每列的属性
        classLabelVector.append(int(listFromLine[-1])) # label
        index += 1
    return returnMat, classLabelVector
        
def autoNorm(dataSet):
    # 计算每种属性最大值，最小值，范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0) 
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1)) # 生成与最小值之差组成的矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1)) # 将最小值之差除以范围组成的矩阵
    return normDataSet, ranges, minVals

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

def datingClassTest():
    hoRatio = 0.1 # 设置测试数据比例，
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt") # 从文件中加载数据
    normMat, ranges, minVals = autoNorm(datingDataMat) # 归一化数据
    m = normMat.shape[0] # 表示矩阵的行数，即矩阵的维数
    numTestVecs = int(m * hoRatio) # 设置测试样本数量
    print("numTestVecs = ", numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 对数据进行测试
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is : %d " % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is : %f" (errorCount / float(numTestVecs)))
    print(errorCount)

def classifyPerson():
    resultList = ['not at all', 'in small does', 'in large does']
    percentTats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent filer miles earned per year ?"))
    iceCream = float(input("liters of ice cream consumed per year ?"))
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print("you will probably like this person :" , resultList[classifierResult -1])

if __name__ == '__main__':
    datingClassTest()
  