import sys

from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import label_binarize
import pandas as pd


def discrete_to_binary_labels(df, c):
    bl = label_binarize(df.tolist(),classes = c)
    return bl

def train(df):
    
    feature_column_names = ['Country','Type','Affected','Killed']
    if len(feature_column_names) == 0:
        print >>sys.stderr, "feature_column_names are empty"
        sys.exit()
    x = []
    for fn in feature_column_names:
        if fn == 'Type' or fn == 'Country' or fn == 'Sub_Type':
            if len(x) == 0:
                x = discrete_to_binary_labels(df[fn],df[fn].unique())
            else:
                x = np.column_stack([x, discrete_to_binary_labels(df[fn], df[fn].unique())])
        elif fn == 'Affected' or fn =='Killed' or fn == 'Cost':
            if len(x) == 0:
                x = np.array(df[fn].tolist())
            else:
                x = np.column_stack([x, np.array(df[fn].tolist())])


    y = df['Impact_area'].tolist()
    
    
    regr = linear_model.LinearRegression()

    regr.fit(x, y)


    print("Standard error: %.2f"
          % np.sqrt(1 / (float)((x.shape[0]) - 2) * np.sum((regr.predict(x) - y) ** 2)))
    
    model = (regr, df)

    return model

def predict(model, df):

    feature_column_names = ['Country','Type','Affected','Killed']

    if len(feature_column_names) == 0:
        print >>sys.stderr, "feature_column_names are empty"
        sys.exit()
    x = []
    for fn in feature_column_names:
        if True in df[fn].isnull().tolist():
            print >>sys.stderr, fn + " contains missing values"
            sys.exit()
        if fn == 'Type' or fn == 'Country' or fn == 'Sub_Type':
            if len(x) == 0:
                x = discrete_to_binary_labels(df[fn],model[1][fn].unique())
            else:
                x = np.column_stack([x, discrete_to_binary_labels(df[fn], model[1][fn].unique())])
        elif fn == 'Affected' or fn =='Killed' or fn == 'Cost':
            if len(x) == 0:
                x = np.array(df[fn].tolist())
            else:
                x = np.column_stack([x, np.array(df[fn].tolist())])
    return model[0].predict(x)