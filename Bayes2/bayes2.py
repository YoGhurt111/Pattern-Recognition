# coding=utf-8
import numpy as np
from scipy import stats
import pylab
import math
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib


def loadData(path):
    dataList = []
    with open(path, 'r') as f:
        for item in f.readlines():
            dataList.append(map(float, item.split()[:2]))
    sampleLen = len(dataList)
    return dataList,sampleLen

def getParam(DataList):
    sampleData = np.mat([DataList[i] for i in range(len(DataList))])  # n*2
    meanOfSample = np.mat((sampleData.T[0].mean(),sampleData.T[1].mean())).T  # 2*1
    covOfSample = np.mat(np.cov(sampleData.T))  # 2*2
    detOfSample = np.linalg.det(covOfSample.T)
    return meanOfSample,covOfSample,detOfSample

def classify(data,mean1,mean0,cov1,cov0,det1,det0,priopriP1):
    temp1 = np.mat(data).T-mean1  # 2x1
    temp0 = np.mat(data).T-mean0
    p1 = (1/(2*math.pi*(det1**0.5)))*(math.e**float(-0.5*np.dot(np.dot(temp1.T,np.mat(cov1).I),temp1)))
    p0 = (1/(2*math.pi*(det0**0.5)))*(math.e**float(-0.5*np.dot(np.dot(temp0.T,np.mat(cov0).I),temp0)))
    #p1 = -math.log(2*math.pi*(det1**0.5))-0.5*(np.dot(np.dot((np.mat(data).T-mean1).T,np.mat(cov1).I),np.mat(data).T-mean1))
    #p0 = -math.log(2*math.pi*(det0**0.5))-0.5*(np.dot(np.dot((np.mat(data).T-mean0).T,np.mat(cov0).I),np.mat(data).T-mean0))
    if p1*priopriP1 >= p0*(1-priopriP1):
        return 1
    else:
        return 0

def judge(dataListForBoy,dataListForGirl,mean1,mean0,cov1,cov0,det1,det0,testLen1,testLen0):
    rightNum1 = 0.0
    rightNum0 = 0.0
    for i in range(len(dataListForBoy)):
        if classify(dataListForBoy[i],mean1,mean0,cov1,cov0,det1,det0,0.5) == 1:
            rightNum1 = rightNum1 + 1
    print "判断男生正确率是",rightNum1/testLen1
    for i in range(len(dataListForGirl)):
        if classify(dataListForGirl[i],mean1,mean0,cov1,cov0,det1,det0,0.5) == 0:
            rightNum0 = rightNum0 + 1
    print "判断女生正确率是",rightNum0/testLen0


dataList1,lenth1 = loadData("boy.txt")
dataList0,lenth0 = loadData("girl.txt")
mean1,cov1,det1 = getParam(dataList1)
#print mean1,cov1,det1
mean0,cov0,det0 = getParam(dataList0)
testList1,testLen1 = loadData("boynew.txt")
testList0,testLen0 = loadData("girlnew.txt")
judge(testList1,testList0,mean1,mean0,cov1,cov0,det1,det0,testLen1,testLen0)


