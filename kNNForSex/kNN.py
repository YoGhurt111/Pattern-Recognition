# coding=utf-8
from scipy import stats
import pylab
import math
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib
import operator
from numpy import *

def loadData(path):
    dataList = []
    with open(path,'r') as f:
        for item in f.readlines():
            dataList.append(map(float,item.split()[:3]))
    sampleLen = len(dataList)
    return dataList,sampleLen

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 读取矩阵第一维度的长度4
    # tile函数将矩阵inX重复(4,1)次，形成4x1的矩阵，分别计算出测试点与样本中各点的坐标距离之差
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # axis=1代表将矩阵的每一行向量相加，并得出1xN的矩阵
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # 返回数组值从小到大的索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # classCount.iteritems()返回字典的迭代对象，operator.itemgetter(1)代表取第二个数，也就是字典的value
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def getDataArray():
    dataOfBoy, lenthOfBoy = loadData("boy.txt")
    labelsOfBoy = ['A' for i in range(lenthOfBoy)]
    # x = [dataOfBoy[i][1] for i in range(lenthOfBoy)]
    # y = [dataOfBoy[i][0] for i in range(lenthOfBoy)]
    # plt.scatter(x, y, s=x*500, c=u'r', marker=u'*')
    dataOfGirl, lenthOfGirl = loadData("girl.txt")
    labelsOfGirl = ['B' for i in range(lenthOfGirl)]
    # x = [dataOfGirl[i][1] for i in range(lenthOfGirl)]
    # y = [dataOfGirl[i][0] for i in range(lenthOfGirl)]
    # plt.scatter(x, y, s=x*500, c=u'g', marker=u'o')
    # plt.show()
    group = []
    group.extend(dataOfBoy)
    group.extend(dataOfGirl)
    group = array(group)
    # print group
    labels = []
    labels.extend(labelsOfBoy)
    labels.extend(labelsOfGirl)
    # print labels
    return group, labels

def judge(data, lenth, label, group, labels):
    rightNum = 0.0
    for i in range(lenth):
        if classify0(data[i], group, labels, 1) == label[i]:
            rightNum = rightNum + 1
    print rightNum/lenth


dataOfBoy, lenthOfBoy = loadData("boynew.txt")
labelsOfBoy = ['A' for i in range(lenthOfBoy)]
dataOfGirl, lenthOfGirl = loadData("girlnew.txt")
labelsOfGirl = ['B' for i in range(lenthOfGirl)]
group, labels = getDataArray()
print "男生的正确率："
judge(dataOfBoy, lenthOfBoy, labelsOfBoy, group, labels)
print "女生的正确率："
judge(dataOfGirl, lenthOfGirl, labelsOfGirl, group, labels)

