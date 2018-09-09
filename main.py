import pandas as pd
import numpy as np
import time
import os
import sys
from sklearn.model_selection import train_test_split as split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

# set the first time
t1 = time.time()

# read csv file
address = os.path.join(os.path.realpath(os.getcwd()), '1.csv')
raw_data = pd.read_csv(address)
data = np.array(raw_data, dtype=object)[..., 1:]

# delete raw_data
del raw_data

# split Healthy and sick person
sick = []
healthy = []
for i in data:
    if i[len(i)-1] == 1:
        sick.append(i)
    else:
        healthy.append(i)
sick = np.array(sick)[..., :len(sick[0])-1]
healthy = np.array(healthy)[..., :len(healthy[0])-1]

# find center of healthy and sick person
sick = KMeans(n_clusters=1).fit(sick)
healthy = KMeans(n_clusters=1).fit(healthy)
healthy = healthy.cluster_centers_
sick = sick.cluster_centers_

# find the weight of data
average = KMeans(n_clusters=1).fit(np.array([healthy[0], sick[0]])).cluster_centers_
weight = KMeans(n_clusters=1).fit([(abs(average-sick))[0], (abs(average-healthy))[0]]).cluster_centers_

# split features and labels
features = data[..., :len(data[0])-1] * weight
labels = data[..., len(data[0])-1]

# delete data
del data

# split test data and train data
features_train, features_test, labels_train, labels_test = split(features, labels, random_state=7076)

# delete features and labels
del features, labels


# make regression and fit train data and predict
regression = linear_model.LinearRegression()
regression.fit(features_train, labels_train)
predict_labels_regression = np.abs(np.round(regression.predict(X=features_test)))
del regression

# make Regressor tree and predict labels
tree_regressor = tree.DecisionTreeRegressor()
tree_regressor.fit(features_train, labels_train)
predict_labels_tree_regressor = tree_regressor.predict(features_test)
del tree_regressor

# make classifer tree and predict labels
tree_classifer = tree.DecisionTreeClassifier()
tree_classifer.fit(features_train, np.array(labels_train, dtype=int))
predict_labels_tree_classifer = tree_classifer.predict(features_test)
del tree_classifer

# predict label with KNN
KNN = KNeighborsClassifier()
KNN.fit(features_train, np.array(labels_train, dtype=int))
predict_labels_KNN = KNN.predict(features_test)
del KNN

# make one prediction from all of predicts
predict_labels = np.round(KMeans(n_clusters=1).fit([predict_labels_regression
                                                    , predict_labels_tree_classifer
                                                    , predict_labels_tree_regressor
                                                    , predict_labels_KNN]).cluster_centers_)


# change the type of labels_test
labels_test = np.array(labels_test, dtype=int)

# set the second time
t2 = time.time()

# test the predicts labels
score = accuracy_score(labels_test, predict_labels[0])

# print the Time elapsed and its score
print('Score: ', score)
print('Time: ', t2-t1)

# exit
sys.exit()
