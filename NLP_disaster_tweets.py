import os

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import f1_score

# Importing the datasets
data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

"""
This the code for my ongoing participation in a kaggle contest on NLP. This is a very well known contest called 
"disaster tweets". Given a dataset you need to classify it into tweets that are about a natural disaster and tweets
that are not. 
This code is an attempt at using sci kit's pipeline approach training three different models using a semi-supervised
method. 
Ultimately, in its current version we build a big grill to run through the parameters in order to achieve a more precise
model. 
"""


# Functions written for this project
def eval_and_print_metrics(clf, X_train, y_train, X_test, y_test):
    print("Number of training samples:", len(X_train))
    print("Unlabeled samples in training set:",
          sum(1 for x in y_train if x == -1))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Micro-averaged F1 score on test set: "
          "%0.3f" % f1_score(y_test, y_pred, average='micro'))
    print("-" * 10)
    print()


# Simplified training and evaluation function
def simple_eval(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Micro-averaged F1 score on test set: "
          "%0.3f" % f1_score(y_test, y_pred, average='micro'))
    print("-" * 10)


def pretrain_models(sdg_params: dict, vectorizer_params: dict):
    # Supervised Pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(**vectorizer_params)),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(**sdg_params)),
    ])
    # SelfTraining Pipeline
    st_pipeline = Pipeline([
        ('vect', CountVectorizer(**vectorizer_params)),
        ('tfidf', TfidfTransformer()),
        ('clf', SelfTrainingClassifier(SGDClassifier(**sdg_params), verbose=False)),
    ])
    # LabelSpreading Pipeline
    ls_pipeline = Pipeline([
        ('vect', CountVectorizer(**vectorizer_params)),
        ('tfidf', TfidfTransformer()),
        # LabelSpreading does not support dense matrices
        ('todense', FunctionTransformer(lambda x: x.todense())),
        ('clf', LabelSpreading()),
    ])
    return pipeline, st_pipeline, ls_pipeline


def training_and_evaluation(pipeline, st_pipeline, ls_pipeline):
    print("Supervised SGDClassifier on 100% of the data:")
    simple_eval(pipeline, X_train, y_train, X_test, y_test)
    print("simple_eval SGDClassifier on 20% of the training data:")
    simple_eval(pipeline, X_20, y_20, X_test, y_test)
    print("SelfTrainingClassifier on 20% of the training data (rest "
          "is unlabeled):")
    simple_eval(st_pipeline, X_train, y_train, X_test, y_test)

    if 'CI' not in os.environ:
        # LabelSpreading takes too long to run in the online documentation
        print("LabelSpreading on 20% of the data (rest is unlabeled):")
        simple_eval(ls_pipeline, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    sdg_params = {}
    vectorizer_params = {}
    X, y = data.text, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # select a mask of 20% of the train dataset
    y_mask = np.random.rand(len(y_train)) < 0.2

    # X_20 and y_20 are the subset of the train dataset indicated by the mask
    X_20, y_20 = map(list, zip(*((x, y)
                                 for x, y, m in zip(X_train, y_train, y_mask) if m)))
    # unmask a part of the dataset
    y_train[~y_mask] = -1
    # print("SelfTrainingClassifier on 20% of the training data (rest "
    # "is unlabeled):")

    # We define the parameters that will go into the grill.
    alpha_parameters = [1e-3, 1e-4, 1e-5, 1e-6]
    penalties = ['l2', 'l1']
    ngram_ranges = [(1, 2), (1, 1), (2, 2)]
    min_dfs = [1, 2, 3, 4, 5]
    max_dfs = [0.6, 0.8, 0.9, 1]
    # Run models into a very big loop.
    for val in alpha_parameters:
        for penal in penalties:
            for range in ngram_ranges:
                for min in min_dfs:
                    for max in max_dfs:
                        sdg_params = dict(alpha=val, penalty=penal, loss='log')
                        vectorizer_params = dict(ngram_range=range, min_df=min, max_df=max)
                        # Rellenar sdg_params y vectorizer_params
                        pipeline, st_pipeline, ls_pipeline = pretrain_models(sdg_params, vectorizer_params)
                        #we keep track of the iterations through this print.
                        print("these are the parameters used alpha = {}, ngram_ranges = {}, min_dfs = {}, max_dfs= {}".format(
                            val,
                            penal,
                            min,
                            max))
                        training_and_evaluation(pipeline, st_pipeline, ls_pipeline)
