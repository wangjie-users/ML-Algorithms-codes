'''
created in 2017/12/18
用逻辑回归预测疝气病马的死亡率
@author: Jie Wang

'''

import logRegress
from numpy import *

#分类函数：分为两类
def classifyVector(inX, weights):
    prob = logRegress.sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

#训练模型及测试
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21): #从0到20，不包括最后21号元素
            lineArr.append(float(currLine[i]))
        trainingSet .append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = logRegress.RandomGradAsent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is :%f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is:%f" % (numTests, errorSum/float(numTests)))

if __name__ == '__main__':
    multiTest()
