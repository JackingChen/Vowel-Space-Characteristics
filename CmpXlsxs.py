#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:39:03 2023

@author: jack
"""

import pandas as pd


def readXlsx(file_prefix = "TASLPTABLE-ClassFusion_Norm",in_path="/media/jack/workspace/DisVoice/RESULTS/"):
    file_extensions = ['func1', 'func2', 'func3', 'func4', 'func7', 'proposed']
    data_dict = {}
    
    for extension in file_extensions:
        file_name = f"{file_prefix}[{extension}].xlsx"
        try:
            df = pd.read_excel(f"{in_path}{file_name}")  # 讀取Excel檔案
            data_dict[extension] = df  # 將資料存入字典中，以extension作為鍵值
        except FileNotFoundError:
            print(f"找不到檔案: {file_name}")
    return data_dict
    


fusion_dict=readXlsx(file_prefix = "TASLPTABLE-ClassFusion_Norm")
# Basefeat_dict=readXlsx(file_prefix = "TASLPTABLE-ClassBaseFeat_Norm")
function_list=list(fusion_dict.keys())

Inspect_cols=['ASDTD/SVC','f1']

diff_data_dict={}
for i in range(len(function_list)):
    for j in range(i + 1, len(function_list)):
        function1 = function_list[i]
        function2 = function_list[j]
        data1 = fusion_dict.get(function1)[Inspect_cols]
        data2 = fusion_dict.get(function2)[Inspect_cols]
        print(f"Comparing {function1} and {function2}:")
        diff_data=data1-data2
        diff_data_dict[f"{function1} - {function2}"]=diff_data
        print(f"Data 1: {data1}")
        print(f"Data 2: {data2}")
        print("------")