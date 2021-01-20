# 特徴量
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# Use Numpy [may cause Out-of-Memory (OOM) error]
def rolling_window(a, shape):  # rolling window for 2D array
    #次元数が増える
    #a(1次元に対して、先の２つの配列を入れるイメージ)
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.squeeze(np.lib.stride_tricks.as_strided(a, shape = s, strides = strides), axis = 1)
    
def median_fillna(df:pd.DataFrame):
    #欠損値に対して、各カラムごとに中央値を埋め込む
    #背景, featureにはNANが多いから
    # https://www.kaggle.com/wongguoxuan/eda-pca-xgboost-classifier-for-beginners
    train_median = df.median()
    df = df.fillna(train_median)

    return df, train_median

def feature_pca(train_x:pd.DataFrame, n_components = 50 , scaler= None):
    # Before we perform PCA, we need to normalise the features so that they have zero mean and unit variance
    # https://www.kaggle.com/wongguoxuan/eda-pca-xgboost-classifier-for-beginners
    if(scaler == None):
        scaler = StandardScaler()
        scaler.fit(train_x)

    train_x_norm = scaler.transform(train_x)

    pca = PCA(n_components=n_components).fit(train_x_norm)
    train_x_transform = pca.transform(train_x_norm)

    return train_x_transform, scaler

# We impute the missing values with the medians
def fillna_npwhere(array, values):
    # numpyにした状態でNULLがあるとき、valuesで穴埋めをするときに使う
    # https://www.kaggle.com/wongguoxuan/eda-pca-xgboost-classifier-for-beginners
    if np.isnan(array.sum()):
        array = np.where(np.isnan(array), values, array)
    return array

