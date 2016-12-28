# coding=utf-8
import numpy as np
from scipy import stats
import pylab
import math
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib

def loadData(path1,path2):
    dataList = []
    with open(path1,'r') as f:
        for item in f.readlines():
            dataList.append(map(float,item.split()[:2]))
    classList = [1]*len(dataList)
    with open(path2,'r') as f:
        for item in f.readlines():
            dataList.append(map(float,item.split()[:2]))
            classList.append(0)
    return dataList,classList

def getParam(DataList,ClassList):
    boyData = np.mat([DataList[i] for i in range(len(DataList)) if ClassList[i] == 1])  # n*2
    girlData = np.mat([DataList[i] for i in range(len(DataList)) if ClassList[i] == 0])
    meanOfBoy = np.mat((boyData.T[0].mean(),boyData.T[1].mean())).T  # 2*1 类均值向量mi
    meanOfGirl = np.mat((girlData.T[0].mean,girlData.T[1].mean(0))).T
    covOfBoy = np.mat(np.cov(boyData.T))  # 2*2 类内离散度矩阵
    covOfGirl = np.mat(np.cov(girlData.T))
    Sw = np.mat(covOfBoy + covOfGirl)  # 2*2 总类内离散度
    w = Sw.I * (meanOfGirl - meanOfBoy)  # 2*1 最优投影方向
    w0 = (meanOfBoy+meanOfGirl)/2
    return w,w0

def classify(data,w,w0):
    if w.T*(data-0.5*(m1+m2)) >= math.log((1-priopriP1)/priopriP1):
        return 1
    else:
        return 0

def judge(dataList,w,m1,m2,classList):
    rightNum = 0.0
    for i in range(dataList):
        if classify(dataList2[i],w,m1,m2,0.5) == classList[i]:
            rightNum = rightNum + 1
    print "正确率是"+rightNum/len(classList)


dataList1,classList1 = loadData("boy.txt","girl.txt")
w1,mean1,mean0 = getParam(dataList1,classList1)
dataList2,classList2 = loadData("boynew.txt","girlnew.txt")
judge(dataList2,w1,mean1,mean0,classList2)


