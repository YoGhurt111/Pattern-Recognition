# coding: utf-8

import numpy as np
from scipy import stats
import pylab
import math
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib

def plotROC(items,title, xlable, ylable, legend):
    for points, color in items:
        x = [i[0] for i in points]
        y = [i[1] for i in points]
        pylab.plot(x,y, color)
    pylab.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.grid(True)
    plt.legend(legend, loc='upper right')
    x = np.linspace(0, max(x), 100)
    plt.plot(x, x)
    pylab.show()

def loadDataSet():
    DataSet = []
    with open('boy.txt', 'r') as f:
        for item in f.readlines():
            DataSet.append(map(float, item.split()[:2]))
    classLables = [1]*len(DataSet)
    with open('girl.txt', 'r') as f:
        for item in f.readlines():
            DataSet.append(map(float, item.split()[:2]))
            classLables.append(0)
    #print DataSet
    return DataSet, classLables

def loadTestSet():
    DataSet = []
    with open('boynew.txt', 'r') as f:
        for item in f.readlines():
            DataSet.append(map(float, item.split()[:2]))
    classLables = [1]*len(DataSet)
    with open('girlnew.txt', 'r') as f:
        for item in f.readlines():
            DataSet.append(map(float, item.split()[:2]))
            classLables.append(0)
    return DataSet, classLables

def getParam(dataSet, classLables):
    boyData = np.mat([dataSet[i] for i in range(len(dataSet)) if classLables[i]==1])  # n*2
    girlData  = np.mat([dataSet[i] for i in range(len(dataSet)) if classLables[i]==0])
    mean1 = np.mat((boyData.T[0].mean(), boyData.T[1].mean())).T  # 2*1
    # print mean1
    mean0 = np.mat((girlData.T[0].mean(), girlData.T[1].mean())).T
    print mean0
    cov1 = np.mat(np.cov(boyData.T))  # 2*2
    #print cov1
    cov0 = np.mat(np.cov(girlData.T))
    Sw = np.mat(cov0 + cov1)
    #print Sw
    w = Sw.I * (mean0 - mean1)
    #print w
    m1 = np.linalg.det(w.T * mean1)
    #print m1
    m0 = np.linalg.det(w.T * mean0)
    return w, m0, m1, cov0, cov1, (m0 + m1) / 2

def classify(data, w, w0):
    if w.T * data > w0:
        return 0
    else:
        return 1

def pdf(x, pw, mu, cov):
    x = np.matrix(x)
    return pw * np.exp(-1*np.dot(np.dot((x-mu),cov.I),(x.T-mu.T))/2)/np.sqrt(np.linalg.det(cov))

def classifyB(x, pw, mu, cov, a):
    p=[1,1]
    for i in range(2):
        p[i] = pdf(x, pw[i], mu[i], cov[i])
    if(p[0]-p[1]> a):
        return 0
    else:
        return 1

pw = {0:0.5,1:0.5}
dataSet, classLables = loadDataSet()
w, m0, m1,cov0, cov1, w0 = getParam(dataSet, classLables)
for i in range(len(dataSet)):
    plt.plot(dataSet[i][0], dataSet[i][1], 'ro' if classLables[i]==0 else 'bo')
# x = np.linspace(145, 190, 100)
# y = np.array((w[0] * x / w[1]).T)
# plt.plot(x, y)
x = np.linspace(145, 190, 100)
y = np.array(((w0 - w[0] * x )/ w[1]).T)
plt.plot(x, y)

# delta = 0.1
# x = np.arange(145, 190, delta)
# y = np.arange(-20, 140, delta)
# X, Y = np.meshgrid(x, y)
# Z0 = mlab.bivariate_normal(X, Y, m0[0, 0], m0[1, 0], cov0[0, 0], cov0[1, 1], cov0[0, 1])
# Z1 = mlab.bivariate_normal(X, Y, m1[0, 0], m1[1, 0], cov1[0, 0], cov1[1, 1], cov1[0, 1])
# Z = 10000*(Z1 - Z0)
# print Z[:10]
# CS = plt.contour(X, Y, Z0)
# plt.clabel(CS, inline=1, fontsize=6)

plt.show()