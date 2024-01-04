import pandas as pd
import numpy as np
from sklearn import metrics


class NBC:

    def __init__(self, alpha=None):

        if alpha is None:
            self.alpha = 1.0
        else:
            self.alpha = alpha

    def fit(self, X_train, y_train):
        self.Xtrain = X_train
        self.ytrain = y_train

        # wyciaganie labeli i mozliwych wartosci klasy
        classColumn = y_train.keys().tolist()[0]       # label klasy
        classValues = y_train[classColumn].unique()    # unikalne wartości klasy
        self.classColumn = classColumn
        self.classValues = classValues

        # nazwy atrybutów
        attributes = X_train.keys().tolist()
        self.attributes = attributes

        # laczanie atrybutów z klasami
        df = X_train.join(y_train)      #dataframe z przemieszanych rekordów - połączone atrybuty z klasą

        # liczba unikalnych wartosci dla atrybutów, np. {'RI': 151, 'Na': 118, 'Mg': 82, 'Al': 103, 'Si': 116, 'K': 60, 'Ca': 123, 'Ba': 27, 'Fe': 30}
        self.numberOfValuesPerAttribute = {}

        for attr in attributes:
            self.numberOfValuesPerAttribute[attr] = X_train[attr].nunique()

        # prawdpodobienstwa warunkowe    np. {(2, 'RI', 1.5159): 0.010484927916120578, (2, 'RI', 1.51934): 0.003931847968545216,.. }
        self.conditionalPropabilities = {}

        # prawdopodobeinstwo nalezenia do klasy
        self.pClass = {}

        # liczba wystąpień klasy
        self.nClassOccurences = {}

        numberOfAttributesPerValue = self.numberOfValuesPerAttribute
        alpha = self.alpha

        # creating poropabilsitic model

        for classValue in classValues:
            self.nClassOccurences[classValue] = df[classColumn].value_counts()[classValue]
            self.pClass[classValue] = (self.nClassOccurences[classValue] + alpha) / (
                    len(df.index) + alpha * len(classValues))
            dfClass = df[df[classColumn] == classValue]
            for attr in attributes:
                for attrVal in df[attr].unique().tolist():
                    n = len(dfClass[dfClass[attr] == attrVal])
                    self.conditionalPropabilities[(classValue, attr, attrVal)] = (n + alpha) / (
                            len(dfClass) + alpha * numberOfAttributesPerValue[attr])


    def predict(self, X_test):
        listOfPredictions = []
        for i in range(len(X_test)):
            currPrediction = self._predict(X_test.iloc[[i]].values.flatten().tolist())
            listOfPredictions.append(currPrediction)
        return listOfPredictions

    def _predict(self, row):
        classValues = self.classValues
        attributes = self.attributes
        conditionalPropabilities = self.conditionalPropabilities
        numOfValPerAttr = self.numberOfValuesPerAttribute
        alpha = self.alpha

        p = float('-inf')
        predictedClass = 'undefined'

        for classValue in classValues:
            pClass = self.pClass[classValue]
            for i in range(len(attributes)):
                if (classValue, attributes[i], row[i]) in conditionalPropabilities:
                    pClass = pClass * conditionalPropabilities[(classValue, attributes[i], row[i])]
                else:
                    nvals = float(numOfValPerAttr[attributes[i]])
                    pClass = pClass * alpha / (self.nClassOccurences[classValue] + nvals * alpha)
            if pClass > p:
                predictedClass = classValue
                p = pClass

        return predictedClass

    def score(self, X_test, y_test):
        yList = y_test.values.flatten().tolist()
        prediction = self.predict(X_test)

        correctPredictions = 0

        for i in range(len(yList)):
            if yList[i] == prediction[i]:
                correctPredictions = correctPredictions + 1

        return correctPredictions / len(yList)
