import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import warnings
import re
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import FunctionTransformer

warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)


def preprocess_part_1(df):
    df.loc[df['max_power']==' bhp', 'max_power'] = np.nan
    for col in ['mileage', 'engine', 'max_power']:
        mask = df[col].notna()
        df.loc[mask, col] = df.loc[mask, col].apply(lambda x: x.split()[0])
        df[col] = df[col].astype(float)
    return df

def get_num(s):
    pattern = '\d+((\.\d+)|(\,\d+))?'
    nums = [float(i.group().replace(',', '')) for i in re.finditer(pattern, s)]
    first, last = nums[0], nums[-1]
    if 'kgm' in s.lower():
        first*=9.81
    return first, last

def preprocess_part_2(df):
    df['max_torque_rpm'] = df['torque']
    for ind, col in enumerate(['torque', 'max_torque_rpm']):
        mask = df[col].notna()
        df.loc[mask, col] = df.loc[mask, col].apply(lambda x: get_num(x)[ind])
        df[col] = df[col].astype(float)
    return df

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_col):
        self.num_col = num_col
        self.est = SimpleImputer(strategy='median')

    def fit(self, X, y=None):
        self.est.fit(X[self.num_col])
        return self

    def transform(self, X):
        num_df = pd.DataFrame(self.est.transform(X[self.num_col]), columns=self.num_col)
        other_df = X[[col for col in X.columns if col not in self.num_col]]
        return pd.concat((num_df, other_df), axis=1)[X.columns]

def preprocess_part_4(df):
    for col in ['engine', 'seats']:
        df[col] = df[col].astype(int)
    return df

def preprocess_part_5(df):
    df['seats'] = df['seats'].astype(str).apply(lambda x: x + ' seats')
    return df

class NumTransformer_2(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.est_num = StandardScaler()

    def fit(self, X, y=None):
        self.est_num.fit(X)
        return self

    def transform(self, X):
        num_df = pd.DataFrame(self.est_num.transform(X), columns=X.columns)
        return num_df

class CatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.est_cat = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)

    def fit(self, X, y=None):
        self.est_cat.fit(X)
        return self

    def transform(self, X):
        categories = [i for j in self.est_cat .categories_ for i in j[1:]]
        cat_df = pd.DataFrame(self.est_cat.transform(X), columns=categories)
        return cat_df

def preprocess_part_6(df):
    df['brand'] = df.name.apply(lambda x: x.split()[0])
    df['power_per_liter'] = df.max_power / df.engine
    df['rpm_per_torque'] = df.max_torque_rpm / df.torque
    df['year_squared'] = df.year**2
    df['driven_per_year'] = df.km_driven / (datetime.now().year - df.year)
    return df

with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)


app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    print(item)
    item_df = pd.DataFrame([item.model_dump()])
    prediction = np.round(np.exp(pipeline.predict(item_df)), 2)
    return prediction


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    items_df = pd.DataFrame([item.dict() for item in items])
    predictions = np.round(np.exp(pipeline.predict(items_df)), 2)
    return predictions

