#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: data_split.py

import pandas as pd
from config import *
from data_extract import DataExtract

def match_data(x_df, y_df):
    matched_df = pd.merge(x_df, y_df, on=["time"])
    return matched_df

if __name__ == "__main__":
    # read train data
    x_data = DataExtract(x_train_path)
    y1_data = DataExtract(y_train_path, 0)
    y2_data = DataExtract(y_train_path, 1)
    y3_data = DataExtract(y_train_path, 2)

    # match data with labels
    matched_df = match_data(x_data.df, y1_data.df)
    matched_df = match_data(matched_df, y2_data.df)
    matched_df = match_data(matched_df, y3_data.df)
    
    # split data
    
    # save into files
    matched_df.drop(["time"], axis=1, inplace=True) # delete time-colume
    matched_df.to_excel(matched_data_path)
