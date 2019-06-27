from numpy import *
from os import listdir

# 将32*32的图像转化为1*1024的向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])
    return returnVect

def handWritingClassTest():
        # 导入训练数据
        hwLabels =[]
        trainingFileList = listdir('trainingDigits') # 加载目录
        m = len(trainingFileList)
        trainingMat = zeros((m, 1024))
        for i in range(m):
                fileNameStr = trainingFileList[i]
                fileStr = fileNameStr.split('.')[0] # 除去文件扩展名 .txt
                classNumStr = fileStr.split('_')[0] # 获取标签
                hwLabels.append(int(classNumStr))
                trainingMat[i, :] = img2vector('trainingDigits/%s' % (fileNameStr))
        
        # 导入测试数据
        testFileList = listdir('testDigits')
        errorCount = 0.0
        mTest = len(testFileList)
        for i in range(mTest):
                fileNameStr = testFileList[i]
                fileStr = fileNameStr.split('.')[0]
                classNumStr = fileStr.split('_')[0]
                vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
                classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
                print("the classifier came back is: %d , the real answer is: %d" % (classifierResult, classNumStr[i]))
                if (classifierResult != classNumStr):
                        errorCount += 1.0
        print("the total number of error is: %d,\n the total error rate is: %f" % (errorCount, float(errorCount / mTest)))

def classify0(inX, dataMat, dataLabels, k):
        dataMatSize = dataMat.shape[0] 
        diffMat = tile(inX, (dataMatSize, 1)) - dataMat
        sqDiffMat = diffMat ** 2
        sqDistacnes = sqDiffMat.sum(axis = 1)
        distances = sqDiffMat ** 0.5

        sortedDistIndicies = distances.argsort()
        classCount = {}
# 不知道index 怎么找到原来的label， wrong!!!
        for i in range(k):
                voteIlabel = dataLabels[sortedDistIndicies[i]]
                classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
        return sortedClassCount[0][0]

if __name__ == '__main__':
        handWritingClassTest()
