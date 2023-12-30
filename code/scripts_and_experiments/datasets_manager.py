import pandas as pd
from typing import Tuple

def get_dataset_corona() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("datasets/exams.csv")

    df = df.drop(['Test_date', 'Corona'], axis=1)

    X = df.drop(['Ind_ID', 'Test_date', 'Corona'], axis=1)
    y = df['Corona']

    return X, y

def get_dataset_divorce() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("datasets/divorce.csv")

    X = df.drop(['Divorce_Y_N'], axis=1)
    y = df['Divorce_Y_N']

    return X, y

def get_dataset_glass() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("datasets/glass.csv")

    X = df.drop(['Type'], axis=1)
    y = df['Type']

    return X, y

def get_dataset_loan_approval_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("datasets/loan_approval_dataset.csv")

    df = df.drop(['loan_id'], axis=1)

    X = df.drop(['loan_status'], axis=1)
    y = df['loan_status']

    return X, y