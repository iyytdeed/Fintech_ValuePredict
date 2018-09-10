#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: data_split.py

import pandas as pd

df_ori = pd.read_table('./submit.txt')
df_ori['Pred'] = 0
print df_ori
df_ori.to_csv('./new_submit.txt', sep='\t', index=False)
