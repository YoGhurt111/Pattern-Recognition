# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn import metrics
from sklearn import preprocessing
from scipy import interp
from sklearn import tree
from sklearn.cross_validation import StratifiedKFold



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


def Decision_Tree_classifier(train_x, train_y):
    model = tree.DecisionTreeClassifier(max_depth=5)
    model.fit(train_x, train_y)
    return model


def knn_classifier(train_x, train_y):
    model = neighbors.KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


def logistic_regression_classifier(train_x, train_y):
    model = LogisticRegression(penalty='l2')
    model.fit(trainData, trainResults)
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
    return lowDDataMat


def draw(dataMat, labels):
    ax=plt.subplot(111, projection='3d')
    failData = []
    sccessData = []
    i = 0
    for flag in labels:
        if flag == 1.0:
            failData.append([dataMat[i, 0], dataMat[i, 1], dataMat[i, 2]])
        else:
            sccessData.append([dataMat[i, 0], dataMat[i, 1], dataMat[i, 2]])
        i += 1
        if i == 200:
            break;
    failData = np.array(failData)
    sccessData = np.array(sccessData)
    ax.scatter(failData[:, 0], failData[:, 1], failData[:, 2], marker=u'*', c=u'red')
    ax.scatter(sccessData[:, 0], sccessData[:, 1], sccessData[:, 2], marker=u'^',  c=u'green')
    plt.show()


def plot_ROC(classifier, x, y, n_folds=5):
    #使用6折交叉验证，并且画ROC曲线
    cv = StratifiedKFold(y, n_folds=n_folds)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(7, 7))
    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
        # Compute ROC curve and area the curve
        #通过roc_curve()函数，求出fpr和tpr，以及阈值
        fpr, tpr, thresholds = metrics.roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)            #对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
        mean_tpr[0] = 0.0                                 #初始处为0
        roc_auc = metrics.auc(fpr, tpr)
        #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    #画对角线
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)                     #在mean_fpr100个点，每个点处插值插值多次取平均
    mean_tpr[-1] = 1.0                         #坐标最后一个点为（1,1）
    mean_auc = metrics.auc(mean_fpr, mean_tpr)        #计算平均AUC值
    #画平均ROC曲线
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# 训练数据处理
df = pd.read_csv('bank-additional-full.csv')
narray = np.array(numerical(df.values), dtype=np.float)
trainResults = np.array(get_results(narray))
trainData = np.delete(narray, [20], axis=1)
print '训练样本的个数和每个样本的特征数：', np.shape(trainData)
lowDMat_train = pca(trainData, 3)
print '经过PCA降维后的样本个数和每个样本的特征数：', np.shape(lowDMat_train)
draw(lowDMat_train, trainResults)
trainData = preprocessing.normalize(trainData)

# 获取模型
knn_model = knn_classifier(trainData, trainResults)
decision_tree_model = Decision_Tree_classifier(trainData, trainResults)
logistic_regression_model = logistic_regression_classifier(trainData, trainResults)

# 测试数据处理
df = pd.read_csv('bank-additional.csv')
narray = np.array(numerical(df.values), dtype=np.float)
testResults = np.array(get_results(narray))
testData = preprocessing.normalize(np.delete(narray, [20], axis=1))


# KNN正确率计算
knn_predict = knn_model.predict(testData)
knn_accuracy = metrics.accuracy_score(testResults, knn_predict)
print '在同等数据量的情况下，KNN算法的正确率是：',  knn_accuracy
plot_ROC(knn_model, testData, np.array(testResults))
# 决策树正确率计算
decision_tree_predict = decision_tree_model.predict(testData)
decision_tree_accuracy = metrics.accuracy_score(testResults, decision_tree_predict)
print '在同等数据量的情况下，决策树算法的正确率是：', decision_tree_accuracy
plot_ROC(decision_tree_model, testData, np.array(testResults))
# Logistic回归正确率计算
logistic_regression_predict = logistic_regression_model.predict(testData)
logistic_regression_accuracy = metrics.accuracy_score(testResults, logistic_regression_predict)
plot_ROC(logistic_regression_model, testData, np.array(testResults))
print '在同等数据量的情况下，Logistic回归算法的正确率是：', logistic_regression_accuracy





