#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: run.py
# Author: ZK Lei
# Function: 1)readin train data -> 2)data preprocess -> 3)training model ->
#           4)readin test data  -> 5)test model

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as XGB
import matplotlib.pyplot as plt

from config import *
from data_extract import DataExtract

def match_data(x_df, y_df):
    matched_df = pd.merge(x_df, y_df, on=["time"])
    return matched_df

def train_test(x, y, x_test):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=2018)
    xgb_instance = XGB.XGBRegressor(
             #max_depth=3,
             #min_child_weight=1,
             #gamma=0.5,
             #subsample=0.6,
             #colsample_bytree=0.6,
             #scale_pos_weight=1,
             #reg_alpha=0,
             #reg_lambda=1,
             seed=2018
            )
    xgb_instance.fit(x_train, y_train)
    XGB.plot_importance(xgb_instance)
    plt.show()
    y_test = xgb_instance.predict(x_test)
    #assert error
    return y_test

if __name__ == "__main__":
    # ----- data preprocess begin -----
    # read train data
    x_data = DataExtract(x_train_path)
    y1_data = DataExtract(y_train_path, 0)
    y2_data = DataExtract(y_train_path, 1)
    y3_data = DataExtract(y_train_path, 2)

    # feature normalized
    matched_df = x_data.df
    names = matched_df.columns.tolist()
    names.remove("time")
    for name in names:
        matched_df[name] = (matched_df[name] - matched_df[name].min())/(matched_df[name].max() - matched_df[name].min())

    # remove dup
    matched_df.drop_duplicates()

    # match data with labels
    matched_df = match_data(matched_df, y1_data.df)
    matched_df = match_data(matched_df, y2_data.df)
    matched_df = match_data(matched_df, y3_data.df)
    
    # change matched_df`s feature name
    cnt = 0
    for name in names:
        matched_df.rename(columns={name:str(cnt)}, inplace=True)
        cnt += 1
    
    # get y1, y2, y3
    y1 = matched_df.pop("y1")
    y2 = matched_df.pop("y2")
    y3 = matched_df.pop("y3")
    
    # drop unused feature
    matched_df.pop("time")
    
    # x_test
    x_test_df = DataExtract(x_test_path).df
    for name in names:
        x_test_df[name] = (x_test_df[name] - x_test_df[name].min())/(x_test_df[name].max() - x_test_df[name].min())
    times_col = x_test_df.pop("time")
    cnt = 0
    for name in names:
        x_test_df.rename(columns={name:str(cnt)}, inplace=True)
        cnt += 1
    
    # ----- data preprocess end -----
    
    # ----- training_test begin -----
    y1_test = train_test(matched_df, y1, x_test_df)
    y2_test = train_test(matched_df, y2, x_test_df)
    y3_test = train_test(matched_df, y3, x_test_df)
    
    print y1
    print y1_test
    
    
    # ----- training_test end -----

    # ----- save into file begin -----
    y1_test_df = pd.DataFrame({"time":times_col, "y1":y1_test})
    f = open(raw_submit_path, 'r')
    f_new = open('./new_submit.txt', 'w')
    
    cnt = 0
    line = f.readline().rstrip("\n")
    f_new.write(line + "\n")
    while line != "Y2":
        line = f.readline().rstrip("\n")
        if line == "Y2":
            break
        while times_col[cnt] != pd.Timestamp(line):
            cnt += 1
        f_new.write(line + "\t" + str(y1_test[cnt]) + "\n")
    
    cnt = 0
    f_new.write(line + "\n")
    while line != "Y3":
        line = f.readline().rstrip("\n")
        if line == "Y3":
            break
        while times_col[cnt] != pd.Timestamp(line):
            cnt += 1
        f_new.write(line + "\t" + str(y2_test[cnt]) + "\n")
    
    cnt = 0
    f_new.write(line + "\n")
    while line != "iron":
        line = f.readline().rstrip("\n")
        if line == "iron":
            break
        while times_col[cnt] != pd.Timestamp(line):
            cnt += 1
        f_new.write(line + "\t" + str(y3_test[cnt]) + "\n")
    
    f_new.write(line + "\n")
    while line:
        line = f.readline().rstrip("\n")
        if not line:
            break
        f_new.write(line + "\t" + str(0) + "\n")
    
    f.close()
    f_new.close()
    # ----- save into file end -----

    # appendix
    '''
    parameters= [{'learning_rate':[0.1],'n_estimators':[100]}]
    xgb_instance = GridSearchCV(XGB.XGBRegressor(
             max_depth=3,
             min_child_weight=1,
             gamma=0.5,
             subsample=0.6,
             colsample_bytree=0.6,
             scale_pos_weight=1,
             reg_alpha=0,
             reg_lambda=1,
             seed=2018
            ), 
            param_grid=parameters) 
    xgb_instance.fit(x_train, y1_train)
    '''
    
