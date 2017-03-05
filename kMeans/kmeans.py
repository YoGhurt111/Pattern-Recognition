from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return mat(dataMat)


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean
    return centroids, clusterAssment






dataMat = loadDataSet("all.txt")
myCentroids, clustAssing = kMeans(dataMat, 2)
ax = plt.subplot(111, projection='3d')
ax.scatter(myCentroids[-1, 0], myCentroids[-1, 1], myCentroids[-1, 2], 'filled', s=150,  c='g')
ax.scatter(myCentroids[-2, 0], myCentroids[-2, 1], myCentroids[-2, 2], 'filled', s=150, c='r')
for line in dataMat:
    if distEclud(line, myCentroids[-2]) < distEclud(line, myCentroids[-1]):
        ax.scatter(line[:, 0], line[:, 1], line[:, 2], marker='*', c='r')
    else:
        ax.scatter(line[:, 0], line[:, 1], line[:, 2], marker='^', c='g')
plt.show()