import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as split
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

# make forest and predict labels
forest = RandomForestClassifier(max_depth=7, random_state=0)
forest.fit(features_train, labels_train)
prediction = forest.predict(features_test)

# test the predict labels
score = accuracy_score(labels_test, prediction)
print(score)
