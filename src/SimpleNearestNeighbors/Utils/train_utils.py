from copy import deepcopy

import numpy as np
import pandas as pd

from preprocessing_utils import normalize_dataset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


    
"""
In this function we will evaluate a model with StratifiedKFold
cross-validation by returning the train and validation accuracy 
score.

Preprocessing_fn should take 4 arguments(if sepcified). The train 
and test instances and labels.
"""
def cross_validate_model(model, instances, labels, preprocessing_fn=None, n_splits=10):
    stratified_k_fold = StratifiedKFold(n_splits)

    avg_train_acc = 0.0
    avg_val_acc   = 0.0

    for train_indexes, val_indexes in stratified_k_fold.split(instances, labels):
        train_instances, train_labels = instances.iloc[train_indexes, :], labels.iloc[train_indexes]
        val_instances,   val_labels   = instances.iloc[val_indexes, :],   labels.iloc[val_indexes]

        # Preprocess dataset
        if preprocessing_fn != None:
            train_instances, train_labels, \
            val_instances,   val_labels      = preprocessing_fn(train_instances, \
                                                                train_labels, \
                                                                val_instances, \
                                                                val_labels)
        # Train the model
        model.fit(train_instances, train_labels)

        # Evaluate the model
        train_preds = model.predict(train_instances)
        val_preds   = model.predict(val_instances)

        avg_train_acc += accuracy_score(train_labels, train_preds)

        avg_val_acc   += accuracy_score(val_labels, val_preds)
    
    avg_train_acc /= n_splits

    avg_val_acc   /= n_splits

    return avg_train_acc, avg_val_acc



"""
In this function we will train the given model on the given
train data and then we will return the accuracy
score that we have when evaluating the (now trained) model
on the given test data. 
"""
def train_and_evaluate_model_on_test_set(model, train_instances, train_labels,
                                        test_instances, test_labels, preprocessing_fn=None):
    trained_model   = deepcopy(model)

    train_instances = deepcopy(train_instances)
    train_labels    = deepcopy(train_labels)

    test_instances  = deepcopy(test_instances)
    test_labels     = deepcopy(test_labels)

    # Preprocess dataset
    if preprocessing_fn != None:
        train_instances, train_labels, \
        test_instances,  test_labels     = preprocessing_fn(train_instances, \
                                                            train_labels, \
                                                            test_instances, \
                                                            test_labels)

    # Start training and evaluation
    trained_model.fit(train_instances, train_labels)

    test_preds  = trained_model.predict(test_instances)

    test_acc = accuracy_score(test_labels, test_preds)

    return trained_model, test_acc

    