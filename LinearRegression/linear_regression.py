# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 以男生的身高和体重做鞋码的回归分析
def loadData(filePath):
    dataMat = []
    labelMat = []
    with open(filePath, 'r') as f:
        for item in f.readlines():
            dataMat.append(map(float, item.split()[:2]))
            labelMat.append(float(item.split()[-1]))
    return np.mat(dataMat), np.mat(labelMat).T


def standRegres(xMat, yMat):
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) != 0:
        w = xTx.I * xMat.T * yMat
        return w


dMat, lMat = loadData("E:\Pattern-Recognition\Bayes2\\boynew.txt")
ws = standRegres(dMat, lMat)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dMat[:, 0].flatten().A[0], dMat[:, 1].flatten().A[0], lMat[:, 0].flatten().A[0])
xCopy = dMat.copy()
xCopy.sort(0)
yGuess = xCopy * ws
ax.plot_trisurf(xCopy[:, 0].flatten().A[0], xCopy[:, 1].flatten().A[0], yGuess.flatten().A[0])
plt.show()
