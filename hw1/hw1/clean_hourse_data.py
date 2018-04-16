# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from utils import Database

from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

train = pd.read_csv('./input/train.csv')
labels = train["SalePrice"]
test = pd.read_csv('./input/test.csv')
data = pd.concat([train, test], ignore_index=True)
data = data.drop("SalePrice", 1)
ids = test["Id"]

# Count the number of NaNs each column has.
nans = pd.isnull(data).sum()
nans = nans[nans > 0].sort_values(ascending=False)
print('delete ', nans.index)

# Remove id and columns with more than a thousand missing values
data = data.drop("Id", 1)
data = data.drop("Alley", 1)
data = data.drop("Fence", 1)
data = data.drop("MiscFeature", 1)
data = data.drop("PoolQC", 1)
data = data.drop("FireplaceQu", 1)

all_columns = data.columns.values
non_categorical = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1",
                   "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
                   "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea",
                   "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
                   "ScreenPorch", "PoolArea", "MiscVal"]

categorical = [value for value in all_columns if value not in non_categorical]
#  One Hot Encoding and nan transformation
data = pd.get_dummies(data)

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
data = imp.fit_transform(data)

# Log transformation
data = np.log(data)
labels = np.log(labels)

# Change -inf to 0 again
data[data == -np.inf] = 0
# Split traing and test
data_offset = np.average(data, axis=0)
data -= data_offset

labels_offset = np.average(labels, axis=0)
labels -= labels_offset

train = data[:1460]
test = data[1460:]

val = train[:292]
val_label = labels[:292]

train = train[292:]
train_label = labels[292:]

db = Database('./input/cleaned.h5')
db['val'] = val
db['train'] = train
db['val_label'] = val_label
db['train_label'] = train_label
db.close()

print(train.shape, val.shape)

results = {}


def test_model(clf):
    cv = KFold(n_splits=5, shuffle=True, random_state=45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(clf, train, train_label, cv=cv, scoring=r2)
    scores = [r2_val_score.mean()]
    return scores


clf = linear_model.LinearRegression()
results["Linear"] = test_model(clf)

for epsilon in [1., 1.35, 10, 100,1000]:
    clf = linear_model.HuberRegressor(epsilon=epsilon)
    results[f'huber.{epsilon}'] = test_model(clf)

print(results)
