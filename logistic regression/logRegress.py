'''
Created in 2017/12/15
Logistic Regression Algorithms
@author: Jie Wang

'''
from numpy import *
import matplotlib.pyplot as plt

# import numpy
def loadDataset (): #读取数据，并分开存储
    dataMat = []
    labelMat = []
    fr = open("testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat
# if __name__ == '__main__':
#     print(loadDataset())

def sigmoid(inX):#用sigmoid函数转化
    return 1.0/ (1+ exp(-inX))

#更新回归系数（基于极大似然法和梯度上升法）
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)#转为numpy矩阵
    labelMat = mat(classLabels).transpose()#将行向量转化为列向量，方便计算
    m,n = shape(dataMatrix)#返回矩阵的维度
    alpha = 0.001 #步长
    maxCycles = 500 #迭代次数
    weights = ones((n,1)) #创建每个元素为1的n行1列的矩阵，即一个列向量，作用是初始化回归系数为1
    for k in range(maxCycles):#每次更新时都需要遍历整个数据集，算法复杂度太高
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error#由最大似然法和梯度上升法求得该迭代公式
    return weights

#改进后的更新回归系数方法0（基于极大似然法和随机梯度上升法）,分类准确率较低
def RandomGradAsent0 (dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

#改进后的更新回归系数方法1（基于样本随机选择和alpha动态减少机制，同时保证了准确率和算法复杂度）
def RandomGradAsent1 (dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i) +0.01 #alpha每次迭代都要进行调整
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])#每次选取该值后，要将其删除
    return weights

#作图
def plotBestFit(weights):
    dataMat, labelMat =loadDataset()
    dataArr = array(dataMat)#将numpy矩阵转化为数组
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)#将画布分为1x1，并取第一部分
    #ax = fig.add_subplot(222)#图像是2x2的，且当前选中第二部分
    ax.scatter(xcord1, ycord1 ,s=30, c='red' ,marker='s')#把训练集中的所有点描出来
    ax.scatter(xcord2, ycord2, s=30, c='green', )
    x = arange(-3.0, 3.0, 0.1)#筛选出分割点
    y = (-weights[0]-weights[1]*x)/weights[2]#筛选出分割点；weights是一个矩阵，必须转化为数组。故在最后用getA()转化
    ax.plot(x, y)#连线
    plt.xlabel('X1')#设置标签
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    dataArr, labelArr=loadDataset()
    # print(gradAscent(dataArr , labelArr))

    weight1 = gradAscent(dataArr, labelArr)
    plotBestFit(weight1.getA()) #矩阵通过getA()将自身返回成一个n维数组对象

    weight2 = RandomGradAsent0(array(dataArr), labelArr)
    plotBestFit(weight2)

    weight3 = RandomGradAsent1(array(dataArr),labelArr)
    plotBestFit(weight3)