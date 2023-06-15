#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 17:37:15 2023

@author: jack
"""

import pandas as pd
import argparse
import os
from addict import Dict
def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser()
    parser.add_argument('--Normalize_way', default='proposed',
                            help='')

    args = parser.parse_args()
    return args
args = get_args()

ReadPath='/media/jack/workspace/DisVoice/RESULTS/Merged_xlsx/'
Normalized_ways_lst=['func1','func2','func3','func4','func7','func10','func13','func14','func15',
                     'func16','func17','proposed']
NormalizedNewways_lst = [n for n in Normalized_ways_lst if n != 'proposed']

fusion_indexes=[
'static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation-LOC_columns+DEP_columns+LOCDEP_Trend_D_cols+LOCDEP_Syncrony_cols',
'static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation-LOC_columns+DEP_columns+LOCDEP_Trend_D_cols',
'static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation-LOC_columns+DEP_columns+LOCDEP_Syncrony_cols',
'static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation-LOC_columns+DEP_columns',
'static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation-Inter-Vowel Dispersion',
'static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation-formant dependency',
'static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation-GC[VSC]\\textsubscript{inv}',
'static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation-Syncrony[VSC]',
'static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation-GC[P]\\textsubscript{part}']

dfResults_dict=Dict()
for Normalize_way in Normalized_ways_lst:
    file1 = ReadPath+"/"+f"TASLPTABLE-Regression_Norm[{Normalize_way}].xlsx"
    df1 = pd.read_excel(file1,index_col='Unnamed: 0')
    df1 = df1.drop('Unnamed: 0.1', axis=1)
    df1.loc[fusion_indexes,['MSE', 'pear', 'spear', 'CCC']] = df1.loc[fusion_indexes,'ADOS_C/SVR (MAE/pear/spear/CCC)'].str.split('/', expand=True).astype(float).values
    # df1.loc[fusion_indexes,['MSE', 'pear', 'spear', 'CCC']] = df1.loc[fusion_indexes,'ADOS_C/SVR (MAE/pear/spear/CCC)'].str.split('/', expand=True)
    df1.drop('ADOS_C/SVR (MAE/pear/spear/CCC)', axis=1, inplace=True)
    dfResults_dict['Regression'][Normalize_way]=df1
    
    
    file2 = ReadPath+"/"+f"TASLPTABLE-Classification_Norm[{Normalize_way}].xlsx"
    df2 = pd.read_excel(file2,index_col='Unnamed: 0')
    df2 = df2.drop('Unnamed: 0.1', axis=1)
 
    
    dfResults_dict['Classification'][Normalize_way]=df2
    
difffResults_dict=Dict()
for Normalize_way in NormalizedNewways_lst:
    for RstType in dfResults_dict.keys():
        difffResults_dict[RstType] = dfResults_dict[RstType][Normalize_way] - dfResults_dict[RstType]['proposed']
        
        
df=dfResults_dict[RstType][Normalize_way]


