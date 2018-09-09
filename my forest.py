import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


def split(feature, slices):
    """this function can split data for different algorithms"""
    length = len(feature)
    each_slice = length//slices
    for i in range(1, slices+1):
        yield feature[each_slice*(i-1):each_slice*i]


# read data
address = os.path.join(os.path.realpath(os.getcwd()), '1.csv')
raw_data = pd.read_csv(address)
data = np.array(raw_data, dtype=np.object_)[..., 1:]
del raw_data

# split features and labels
features = data[..., :len(data[0])-1]
labels = data[..., len(data[0])-1]
del data

# split train and test data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=1)
labels_test = np.array(labels_test, dtype=int)
labels_train = np.array(labels_train, dtype=int)
del features, labels

features = np.array(list(split(features_train, 3)))
labels = np.array(list(split(labels_train, 3)))
del features_train, labels_train

predict = []
# make tree and predict test labels
clf = tree.DecisionTreeClassifier()
clf.fit(features[0], labels[0])
tree_predict = clf.predict(features_test)
predict.append(tree_predict)

# make knn and predict with it
knn = KNeighborsClassifier()
knn.fit(features[1], labels[1])
knn_predict = knn.predict(features_test)
predict.append(knn_predict)

# make regression and predict with it
reg = linear_model.LinearRegression()
reg.fit(features[2], labels[2])
regression_predict = reg.predict(features_test)
predict.append(regression_predict)

main_predict = np.abs(np.round(KMeans(n_clusters=1).fit(np.array(predict)).cluster_centers_))
predict = np.array(main_predict[0], dtype=int)

score = accuracy_score(labels_test, predict)
print(score)
