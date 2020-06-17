from pygraphembedding.datasets import DatasetLoader
from pygraphembedding.preprocess import ColorPreprocessor
from pygraphembedding.preprocess import SimplePreprocessor
from pygraphembedding.utils import minusmean
from pygraphembedding.algorithms import MFA
from pygraphembedding.algorithms import PCA
from pygraphembedding.algorithms import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from imutils import paths
import os

PATH = "D:\\Python\\datasets\\att_faces"

imagePaths = []

for clazz in os.listdir(PATH):
    for img in os.listdir(PATH + "\\" + clazz):
        imagePaths.append(PATH + "\\" + clazz + "\\" + img)

# 读取数据集
sp = SimplePreprocessor(height=56, width=46)
cp = ColorPreprocessor()
dataloader = DatasetLoader(preprocessors=[sp, cp])
(data, labels) = dataloader.load(imagePaths, 50)

# 数据减去平均值
# data = minusmean(data)

# 对标签进行编码
le = LabelEncoder()
labels = le.fit_transform(labels)

# 划分数据集
(trainX, testX, trainY, testY) = train_test_split(data, labels, train_size=0.3, random_state=22, stratify=labels)

N = trainX.shape[0]
Nc = len(np.unique(labels))

MODEL = {
    "pca": PCA(n_neighbors=N-Nc),
    "lda": LDA(),
    "mfa": MFA(intraK=2, interK=50, n_neighbors=40)
}

# 使用模型训练
# model = MODEL["pca"]
# model = MODEL["lda"]
model = MODEL["mfa"]
trainX = trainX.reshape(trainX.shape[0], trainX.shape[1] * trainX.shape[2])
testX = testX.reshape(testX.shape[0], testX.shape[1] * testX.shape[2])

# 先进行PCA
# preModel = PCA(N-Nc, svd_solver='auto')
# trainX = preModel.fit_transform(trainX)
# testX = preModel.transform(testX)

# model.fit(trainX)
model.fit(trainX, trainY)

projectedTestX = model.transform(testX)
projectedTrainX = model.transform(trainX)
projectedTrainX = np.array(projectedTrainX, dtype=float)
projectedTestX = np.array(projectedTestX, dtype=float)

# 使用knn计算准确率
knnModel = KNeighborsClassifier(n_neighbors=5, weights='distance')
knnModel.fit(projectedTrainX, trainY)
predY = knnModel.predict(projectedTestX)

# 评价模型
# print(classification_report(testY, predY, target_names=le.classes_))
print(accuracy_score(testY, predY))