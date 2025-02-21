from imports import Cifar10 as cif
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

import numpy as np

def split_cifar_data():
    data = cif()

    x_train = np.array(data.train_data.data).reshape(len(data.train_data), -1) / 255
    y_train = np.array(data.train_data.targets) 
    
    x_test = np.array(data.test_data.data).reshape(len(data.test_data), -1) / 255
    y_test = np.array(data.test_data.targets)
    
    return x_train, y_train, x_test, y_test

def regression_tree(x_train, y_train):
    model = tree.DecisionTreeRegressor(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42)
    model.fit(x_train, y_train)
    return model

def predict(model, x_test):
    y_test = model.predict(x_test)
    return y_test

def get_model_metrics(model: tree.DecisionTreeRegressor):
    max_depth = model.get_depth()
    total_leaves = model.get_n_leaves()
    return max_depth, total_leaves

def get_test_metrics(y_test, y_predict):
    mse = mean_squared_error(y_test, y_predict)
    r2s = r2_score(y_test, y_predict)
    return mse, r2s

def classification_tree(x_train, y_train):
    model = tree.DecisionTreeClassifier(
        max_depth = 10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        criterion='gini'
    )
    model.fit(x_train, y_train)
    return model

def accuracy_class(y_test, y_pred):
    return accuracy_score(y_test, y_pred)