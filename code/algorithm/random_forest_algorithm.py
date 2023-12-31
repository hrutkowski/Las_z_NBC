import pandas as pd
import numpy as np
from sklearn import metrics
from id3_classifier import ID3
from nbc_classifier import NBC


class RandomForest:

    def __init__(self, n):
        self.n = n

    def fit(self, X_train, y_train):
        ...

    def predict(self, X_test):
        ...

