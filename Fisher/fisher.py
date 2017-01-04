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
    with open(path,'r') as f:
        for item in f.readlines():
            dataList.append(map(float,item.split()[:2]))
    sampleLen = len(dataList)
    return dataList,sampleLen

def getParam(DataList):
    sampleData = np.mat([DataList[i] for i in range(len(DataList))])  # n*2
    meanOfSample = np.mat((sampleData.T[0].mean(),sampleData.T[1].mean())).T  # 2*1
    covOfSample = np.mat(np.cov(sampleData.T))  # 2*2
    return meanOfSample,covOfSample

def classify(data,w,w0):
    if float(np.dot(w.T,np.mat(data).T)) > w0:
        return 1
    else:
        return 0

def judge(dataListForBoy,dataListForGirl,w,w0,testLen1,testLen0):
    rightNum1 = 0.0
    rightNum0 = 0.0
    for i in range(len(dataListForBoy)):
        if classify(dataListForBoy[i],w,w0) == 1:
            rightNum1 = rightNum1 + 1
    print "判断男生正确率是",rightNum1/testLen1
    for i in range(len(dataListForGirl)):
        if classify(dataListForGirl[i],w,w0) == 0:
            rightNum0 = rightNum0 + 1
    print "判断女生正确率是",rightNum0/testLen0

def plotROC(items, title, xlable, ylable, legend):
    for points, color in items:
        x = [i[0] for i in points]
        y = [i[1] for i in points]
        pylab.plot(x,y, color)
    pylab.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.grid(True)
    plt.legend(legend, loc='upper right')
    pylab.show()


dataList1,lenth1 = loadData("boy.txt")
dataList0,lenth0 = loadData("girl.txt")
prioriP1 = 0.7
mean1,cov1 = getParam(dataList1)
mean0,cov0 = getParam(dataList0)
Sw = np.mat(cov1+cov0)
w = np.dot(Sw.I, (mean1 - mean0))
m1 = np.dot(w.T,mean1)
m0 = np.dot(w.T,mean0)
w0 =float((m1+m0)/2 + math.log(prioriP1/(1-prioriP1))/(lenth0+lenth1-2))
testList1,testLen1 = loadData("boynew.txt")
testList0,testLen0 = loadData("girlnew.txt")
# 散点图
x = [dataList1[i][0] for i in range(lenth1)]
y = [dataList1[i][1] for i in range(lenth1)]
plt.scatter(x, y, s=x*500, c=u'r', marker=u'*')
x = [dataList0[i][0] for i in range(lenth0)]
y = [dataList0[i][1] for i in range(lenth0)]
plt.scatter(x, y, s=x*500, c=u'g', marker=u'o')
# 决策面
x = np.linspace(155, 175, 50)
y = np.array(((w0 - w[0] * x) / w[1]).T)
plt.plot(x, y)
plt.show()
# ROC
points = []
for a in np.linspace(float(m0), float(m1), 10):
    err1 = 0.0
    err0 = 0.0
    for i in range(lenth1):
        if classify(testList1[i], w, a) != 1:
            err1 = err1 + 1
    err1Rate = err1 / lenth1
    for i in range(lenth0):
        if classify(testList0[i], w, a) != 0:
            err0 = err0 + 1
    err0Rate = err0 / lenth0
    points.append([err0Rate, err1Rate])
plotROC([[points, 'y']], 'ROC Curves', "girl's error rate", "boy's error rate", ('Fisher', ))
judge(testList1, testList0, w, w0, testLen1, testLen0)



