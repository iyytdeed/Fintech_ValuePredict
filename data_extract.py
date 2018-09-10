#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: data_extract.py

import pandas as pd
from config import *

class DataExtract:
    def __init__(self, file_path, sheet_name=0):
        self.file_path = file_path
        df = pd.read_excel(self.file_path, sheet_name)
        self.df = df
        self.df = self.df.fillna(method="bfill")    # fill with back
        self.df = self.df.fillna(method='ffill')    # fill with front
        self.fields = df.columns.tolist()
    
    def get_df(self):
        return self.df
    
    def get_fileds(self):
        return self.fields
    
    '''
    @property
    def x_train(self):
        return self.data[].tolist()
    '''

if __name__ == "__main__":
    x_data = DataExtract(x_train_path)
    x_data.fillNan()
    print x_data.df

