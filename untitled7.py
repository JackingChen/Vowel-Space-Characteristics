#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 20:18:03 2023

@author: jack
"""

import pandas as pd


class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None
    def _func_proposed(self,val,omega):
        return np.float64(val) / omega

    def _func1(self,val):
        return 1127 *  math.log(np.float64(val) / 700 + 1)
    def func1(self,val):  #為了解決lambda function不能被serialize的問題
        return self._func1(val)
    def _func2(self,val):
        return 21.4 *  math.log(0.00437*np.float64(val)+ 1)
    def func2(self,val):
        return self._func2(val)
    def _func3(self,val):
        return 26.81 *  (np.float64(val) / (1960 + np.float64(val))) - 0.53
    def func3(self,val):
        return self._func3(val)
    def _func4(self,val, sex):
        if sex == 1:
            return 26.81 *  (np.float64(val) / (1960 + np.float64(val))) - 0.53
        elif sex == 2:
            return (26.81 *  (np.float64(val) / (1960 + np.float64(val))) - 0.53) - 1
        else:
            raise ValueError()
    def func4(self,val, sex):
        return self._func4(val,sex)
    def _func7(self,val, epsilon=1e-7):
        if val ==0:  #epsilon是拿來處理log 0 = 無限大的問題
            val=epsilon
        return math.log(np.float64(val))   
    def func7(self,val):
        return self._func7(val) 
    def _func10(self,val, Fmax):
        return np.float64(val) /  Fmax
    def func10(self,val):
        return self._func10(val) 
    def apply_function(self, df, func, column=['F1','F2']):
        new_df = pd.DataFrame()
        new_df[column] = df[column].applymap(func)
        # 将未指定的列复制到新的DataFrame中
        for col in df.columns:
            if col not in  column:
                new_df[col] = df[col]
        return new_df
n=Normalizer()
# 創建範例DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [6, 7, 8, 9, 10],
    'C': [11, 12, 13, 14, 15]
}

df = pd.DataFrame(data)

# 取得每個欄位的最大值
max_values = df.max()


import pandas as pd

# 創建範例DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [6, 7, 8, 9, 10],
    'C': [11, 12, 13, 14, 15]
}

df = pd.DataFrame(data)

# 計算每個欄位的平均值
A_bar = df['A'].mean()
B_bar = df['B'].mean()

# 定義除法函式
def divide_column_by_bar(column, bar):
    return column / bar if bar != 0 else column

# 將A的column除以A_bar，B的column除以B_bar
df['A'] = df['A'].apply(divide_column_by_bar, args=(A_bar,))
df['B'] = df['B'].apply(divide_column_by_bar, args=(B_bar,))

print(df)

