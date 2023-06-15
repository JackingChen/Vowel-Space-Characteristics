#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 13:18:17 2023

@author: jack
"""

import pandas as pd
import argparse
import os
def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser()
    parser.add_argument('--Normalize_way', default='proposed',
                            help='')

    args = parser.parse_args()
    return args
args = get_args()
Result_path="RESULTS/"

ResultOut_path = Result_path+"/Merged_xlsx/"
if not os.path.exists(ResultOut_path):
    os.makedirs(ResultOut_path)

# 讀取第一個檔案
file1 = Result_path+"/"+f"TASLPTABLE-ClassBaseFeat_Norm[{args.Normalize_way}].xlsx"
df1 = pd.read_excel(file1)

# 讀取第二個檔案
file2 = Result_path+"/"+f"TASLPTABLE-ClassFusion_Norm[{args.Normalize_way}].xlsx"
df2 = pd.read_excel(file2)

merged_df = pd.concat([df1, df2], axis=0)
merged_df.to_excel(ResultOut_path+"/"+f"TASLPTABLE-Classification_Norm[{args.Normalize_way}].xlsx")
print(f"generated at  {ResultOut_path}TASLPTABLE-Classification_Norm[{args.Normalize_way}].xlsx ")


# =============================================================================
# 
# =============================================================================

# 讀取第一個檔案
file1 = Result_path+"/"+f"TASLPTABLE-RegressBaseFeat_Norm[{args.Normalize_way}].xlsx"
df1 = pd.read_excel(file1)

# 讀取第二個檔案
file2 = Result_path+"/"+f"TASLPTABLE-RegressFusion_Norm[{args.Normalize_way}].xlsx"
df2 = pd.read_excel(file2)




merged_df = pd.concat([df1, df2], axis=0)
merged_df.to_excel(ResultOut_path+f"TASLPTABLE-Regression_Norm[{args.Normalize_way}].xlsx")
print(f"generated at  {ResultOut_path}TASLPTABLE-Regression_Norm[{args.Normalize_way}].xlsx ")