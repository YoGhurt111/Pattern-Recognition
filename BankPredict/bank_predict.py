# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import naive_bayes
from sklearn import svm
from sklearn import neighbors



def numerical(matrix):
    for row in matrix:
        row[1] = col1_numerical(row[1])
        row[2] = col2_numerical(row[2])
        row[3] = col3_numerical(row[3])
        row[4] = col4_numerical(row[4])
        row[5] = col5_numerical(row[5])
        row[6] = col6_numerical(row[6])
        row[7] = col7_numerical(row[7])
        row[8] = col8_numerical(row[8])
        row[9] = col9_numerical(row[9])
        row[14] = col14_numerical(row[14])
        row[20] = col20_numerical(row[20])
    return matrix


def col1_numerical(str):
    value = {
        'admin.': 0,
        'blue-collar': 1,
        'entrepreneur': 2,
        'housemaid': 3,
        'management': 4,
        'retired': 5,
        'self-employed': 6,
        'services': 7,
        'student': 8,
        'technician': 9,
        'unemployed': 10,
        'unknown': 11
    }
    return value[str]


def col2_numerical(str):
    value = {
        'divorced': 0,
        'married': 1,
        'single': 2,
        'unknown': 3
    }
    return value[str]


def col3_numerical(str):
    value = {
        'basic.4y': 0,
        'basic.6y': 1,
        'basic.9y': 2,
        'high.school': 3,
        'illiterate': 4,
        'professional.course': 5,
        'university.degree': 6,
        'unknown': 7
    }
    return value[str]


def col4_numerical(str):
    value = {
        'no': 0,
        'yes': 1,
        'unknown': 2
    }
    return value[str]


def col5_numerical(str):
    value = {
        'no': 0,
        'yes': 1,
        'unknown': 2
    }
    return value[str]


def col6_numerical(str):
    value = {
        'no': 0,
        'yes': 1,
        'unknown': 2
    }
    return value[str]


def col7_numerical(str):
    value = {
        'cellular': 0,
        'telephone': 1
    }
    return value[str]

def col8_numerical(str):
    value = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12
    }
    return value[str]


def col9_numerical(str):
    value = {
        'mon': 1,
        'tue': 2,
        'wed': 3,
        'thu': 4,
        'fri': 5
    }
    return value[str]


def col14_numerical(str):
   value = {
       'failure': 0,
       'nonexistent': 1,
       'success': 2
   }
   return value[str]


def col20_numerical(str):
    value = {
        'yes': 0,
        'no': 1
    }
    return value[str]


def naive_bayes_classifier():
    model = naive_bayes.MultinomialNB(alpha=0.01)
    return model


def knn_classifier(train_x, train_y):
    model = neighbors.KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


def svm_classifier(train_x, train_y):
    model = svm.SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


def get_results(narray):
    results = []
    for row in narray:
        results.append(row[20])
    return results


def pca(dataMat, topNfeat=9999999):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)  # 对特征值进行从小到大的排列
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def draw(dataMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='*', s=80, c='green')
    plt.show()



# 训练数据处理
df = pd.read_csv('bank-additional-full.csv')
narray = np.array(numerical(df.values), dtype=np.float)
results = get_results(narray)
trainData = np.delete(narray, [20], axis=1)
print np.shape(trainData)
lowDMat, reconMat = pca(trainData, 2)
print np.shape(lowDMat)
print np.shape(reconMat)
draw(lowDMat, results)
# 测试数据处理




