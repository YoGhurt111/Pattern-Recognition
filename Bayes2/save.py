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
    boyLen = len(dataList)
    with open(path2,'r') as f:
        for item in f.readlines():
            dataList.append(map(float,item.split()[:2]))
    girlLen = len(dataList)-boyLen
    return dataList,classList

def getParam(DataList,ClassList):
    boyData = np.mat([DataList[i] for i in range(len(DataList)) if ClassList[i] == 1])  # n*2
    girlData = np.mat([DataList[i] for i in range(len(DataList)) if ClassList[i] == 0])
    meanOfBoy = np.mat((boyData.T[0].mean(),boyData.T[1].mean())).T  # 2*1
    meanOfGirl = np.mat((girlData.T[0].mean(),girlData.T[1].mean())).T
    covOfBoy = np.mat(np.cov(boyData.T))  # 2*2
    covOfGirl = np.mat(np.cov(girlData.T))
    detOfBoy = np.linalg.det(covOfBoy.T)
    detOfGirl = np.linalg.det(covOfGirl.T)
    return meanOfBoy,meanOfGirl,covOfBoy,covOfGirl,detOfBoy,detOfGirl

def classify(data,mean1,mean0,cov1,cov0,det1,det0,priopriP1):
    p1 = math.log(1/(2*math.pi*(det1**0.5)))-0.5*(np.dot(np.dot((np.mat(data).T-mean1).T,np.mat(cov1).T),np.mat(data).T-mean1))
    p0 = math.log(1/(2*math.pi*(det0**0.5)))-0.5*(np.dot(np.dot((np.mat(data).T-mean0).T,np.mat(cov0).T),np.mat(data).T-mean0))
    if p1*priopriP1 >= p0*(1-priopriP1):
        return 1
    else:
        return 0

def judge(datalist,mean1,mean0,cov1,cov0,det1,det0,classlist):
    rightNum = 0.0
    for i in range(len(datalist)):
        if classify(datalist[i],mean1,mean0,cov1,cov0,det1,det0,0.3) == classlist[i]:
            rightNum = rightNum + 1
    print "正确率是",rightNum/len(classlist)


dataList1,classList1 = loadData("boy.txt","girl.txt")
mean1,mean0,cov1,cov0,det1,det0 = getParam(dataList1,classList1)
dataList2,classList2 = loadData("boynew.txt","girlnew.txt")
judge(dataList2,mean1,mean0,cov1,cov0,det1,det0,classList2)


