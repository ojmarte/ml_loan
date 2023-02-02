from typing import List, Any, Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report

import shutil
import re
import os

import boto3
import botocore


def remove_symbols(col: pd.Series, symbol: str, replace_symbol: Any = None) -> [pd.Series, int]:
    res_col = col.str.replace(symbol, replace_symbol)
    null_count = res_col.isnull().sum()

    return [res_col, null_count]


def fill_empty_values(col: pd.Series, fill_values: List[Any], probabilities: List[float], type_inference: Any) -> \
        [pd.Series, int]:
    res_col = col.fillna(np.random.choice(fill_values, p=probabilities)).astype(type_inference)
    null_count = res_col.isnull().sum()

    return [res_col, null_count]


def encode_feature_values(col: pd.Series, encode_dict: Dict[str, int]) -> [pd.Series, int]:
    res_col = col.map(encode_dict)
    null_count = res_col.isnull().sum()

    return [res_col, null_count]


def sum_features(feat_to_sum: List[pd.Series], fill_val: Any = None) -> [pd.Series, int]:
    res_col = pd.Series([])

    for feat in feat_to_sum:
        res_col.add(feat, fill_value=fill_val)

    null_count = res_col.isnull().sum()

    return [res_col, null_count]


def apply_distribution_transformation(col: pd.Series, transformation_to_apply: str = 'log') -> [pd.Series, int]:
    if transformation_to_apply in ['log', 'sqrt', 'reciprocal']:
        transformation = getattr(np, transformation_to_apply)

        res_col = transformation(col)
        null_count = res_col.isnull().sum()

        return [res_col, null_count]


def train_test_and_evaluate_model(ML_lib: str, package_name: str | None, algorithm_name: str, cv_split: int,
                                  X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 123) \
        -> [pd.DataFrame,
            pd.DataFrame,
            pd.Series,
            pd.Series, Any,
            float, Any,
            List[float]]:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)

    if ML_lib == 'xgboost':
        algorithm = getattr(ML_lib, algorithm_name)
    else:
        algorithm = getattr(ML_lib, f'{package_name}.{algorithm_name}')

    model = algorithm()
    model.fit(X_train, y_train)

    acc = model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test)

    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)

    conf_matrix = confusion_matrix(y_test, y_pred)

    cross_val = cross_val_score(model, X, y, cv=cv_split)

    return [X_train, X_test, y_train, y_test, model, acc, conf_matrix, cross_val]


def tune_logistic_regression(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 123,
                             cv_split: int = 5, hyper_params: List[Any] = None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)

    model = LogisticRegression(hyper_params)

    model.fit(X_train, y_train)

    acc = model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_pred, y_pred)

    cross_val = cross_val_score(model, X, y, cv=cv_split)

    return [X_train, X_test, y_train, y_test, model, acc, conf_matrix, cross_val]

def download_s3_train_data(PATH, BUCKET_NAME, KEY, FILENAME):
    s3_client = boto3.client('s3')
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    models = []
    
    for s3_object in bucket.objects.all():
        for key in bucket.objects.all():
            x = re.search("^data/*", key.key)
            if x:
                models.append(key.key)
    
    FOLDER = models[models.index(''.join([KEY, FILENAME]))]
    print(FOLDER)
    
    try:
        s3_client.download_file(BUCKET_NAME, FOLDER, FILENAME)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
    
    DIR_NAME = f'{PATH}/data'
    
    if not os.path.isdir(DIR_NAME):
        os.mkdir(DIR_NAME, 0o777)
        
    src_path = f"{PATH}/{FILENAME}"
    dst_path = f'{PATH}/data/{FILENAME}'
    shutil.move(src_path, dst_path)

def uploud_s3_model(PATH, BUCKET_NAME, KEY):
    client = boto3.client('s3')
    entries = os.listdir(f'{PATH}/data')
    filename = [value if re.search('^dt_classifier_acc_*', value) else '' for value in entries][-1]

    client.upload_file(f"{PATH}/data/{filename}", BUCKET_NAME, f'{KEY}{filename}')

def remove_data_dir(PATH):
    shutil.rmtree(PATH)
