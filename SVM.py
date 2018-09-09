import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split as split
from sklearn import svm
from sklearn.metrics import accuracy_score

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
features_train, features_test, labels_train, labels_test = split(features, labels, random_state=1)
labels_test = np.array(labels_test, dtype=int)
labels_train = np.array(labels_train, dtype=int)
del features, labels

C = 1.0
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))

i = 0
models[i].fit(features_train, labels_train)
prediction = models[i].predict(features_test)

score = accuracy_score(labels_test, prediction)
print(score)
