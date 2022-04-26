#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 15:41:44 2022

@author: jackchen
"""

import os, glob
import pandas as pd
import re
from addict import Dict
import articulation.HYPERPARAM.FeatureSelect as FeatSel
# =============================================================================
'''

    This code is to find the optimized KNN parameters (knn weights, knn neighbors, Reorder_type)

'''
# =============================================================================
# Result_dict_all={}
# max_num=0
# max_sum=0
# max_notautismnASD=0
# xlsxfiles=glob.glob("Result_Interaction/*.xlsx")
# for file in xlsxfiles:
#     nameOfFile=os.path.basename(file).replace(".xlsx","")
#     df_result_file=pd.read_excel(file, index_col=0)
#     df_result_file_NotautismandASD_n_TDColumn=df_result_file[df_result_file.index.str.contains(r'(?=.*NotautismandASD.*)(?=.*TD.*)')]
#     # df_result_file_NotautismandASD_n_TDColumn=df_result_file[df_result_file.index.str.contains(r'(?=.*_Autism.*)(?=.*TD.*)')]
#     if max_notautismnASD < df_result_file_NotautismandASD_n_TDColumn.max().max():
#         print("Max prediction value ", df_result_file_NotautismandASD_n_TDColumn.max().max())
#         print("Is from", file)
#         max_notautismnASD =  df_result_file_NotautismandASD_n_TDColumn.max().max()

        
#     if max_num < df_result_file.max().max():
#         # print("Max prediction value ", df_result_file.max().max())
#         # print("Is from", file)
#         max_num =  df_result_file.max().max()
#     if max_sum < df_result_file.sum().sum():
#         # print("Max prediction value ", df_result_file.sum().sum())
#         # print("Is from", file)
#         max_sum =  df_result_file.sum().sum()
#     Result_dict_all[nameOfFile]=df_result_file
# print("ultimate max num=", max_notautismnASD)

# =============================================================================
# 
'''

    Check results from Classification fusion results

'''
['df_feature_lowMinimal_CSS',
  'df_feature_moderatehigh_CSS',
  'df_feature_Notautism_TC',
  'df_feature_ASD_TC',
  'df_feature_NotautismandASD_TC',
  'df_feature_Autism_TC',
  'df_feature_Notautism_TS',
  'df_feature_ASD_TS',
  'df_feature_NotautismandASD_TS',
  'df_feature_Autism_TS',
  'df_feature_Notautism_TSC',
  'df_feature_ASD_TSC',
  'df_feature_NotautismandASD_TSC',
  'df_feature_Autism_TSC']

manual_selected_settingXlsxFile='Classification_distance_2_DKIndividual'
With_staticphonation_analysis_cols_bool=False
# =============================================================================


    

Top_TManagedResult=Dict()
Top_TManagedResult_bag=Dict()
path="RESULTS/ADDed_UttFeat/"
xlsxfiles = [path+f for f in os.listdir(path) if re.search(r'Classification_(uniform|distance)_([0-9])_(DKIndividual|DKcriteria).xlsx', f)]
for file in xlsxfiles:
    nameOfFile=os.path.basename(file).replace(".xlsx","")
    df_result_file=pd.read_excel(file, index_col=0)
    
    df_ManagedResult=pd.DataFrame()
    df_ManagedResult_noPhonation_columns=pd.DataFrame()
    All_experiments=list(df_result_file.index)
    for i,experiment in enumerate(All_experiments):
        exp_pair_str, feature_module_str=experiment.split(" >> ")
        feature_module_str=feature_module_str.split("-")[0]
        
        df_ManagedResult.loc[exp_pair_str,feature_module_str]=df_result_file.loc[experiment].values[0]
        # Not including phonation columns
        if 'Phonation_columns' not in feature_module_str:
            df_ManagedResult_noPhonation_columns.loc[exp_pair_str,feature_module_str]=df_result_file.loc[experiment].values[0]

    Top_TManagedResult_bag[nameOfFile]=df_ManagedResult.T
    
    
    
     
    
    if With_staticphonation_analysis_cols_bool==True:
        df_ManagedResult_T=df_ManagedResult.T 
        Selected_analysis=df_ManagedResult_T
    else: 
        df_ManagedResult_noPhonation_columns_T=df_ManagedResult_noPhonation_columns.T   
        Selected_analysis=df_ManagedResult_noPhonation_columns_T
    
    
    df_Topcollect=pd.DataFrame()
    for cols in Selected_analysis.columns:
        
        df_tmp=Selected_analysis[cols].sort_values(ascending=False).iloc[:15]
        df_tmp_storage=pd.DataFrame([df_tmp.index.values,df_tmp.values,['-']*len(df_tmp.index)])
        df_tmp_storage.index=[cols,'val','sep']
        df_Topcollect=df_Topcollect.append(df_tmp_storage)
        
    df_Topcollect_T=df_Topcollect.T
    Top_TManagedResult[nameOfFile]=df_Topcollect_T  # The one you want to analyze

df_choosen_knnParameter_dynPhonation=Top_TManagedResult_bag[manual_selected_settingXlsxFile].T.loc[:,FeatSel.dynamic_feature_phonation]
df_choosen_knnParameter_dynLOC=Top_TManagedResult_bag[manual_selected_settingXlsxFile].T.loc[:,FeatSel.dynamic_feature_LOC]

df_choosen_knnParameter_staticLOCDEP=Top_TManagedResult_bag[manual_selected_settingXlsxFile].T.loc[:,FeatSel.static_feautre_LOC]
df_choosen_knnParameter_staticPhonation=Top_TManagedResult_bag[manual_selected_settingXlsxFile].T.loc[:,FeatSel.static_feautre_phonation]

df_fusionBettecols=Top_TManagedResult[manual_selected_settingXlsxFile]

# Sample_template=df_ManagedResult.T
# for k,v in Top_TManagedResult_bag.items():
#     print(v)
#     print((Sample_template - v).sum().sum())



Inspect_rows=[
    'TD vs df_feature_NotautismandASD_TC',
    'TD vs df_feature_Autism_TC',
    'TD vs df_feature_NotautismandASD_TS',
    'TD vs df_feature_Autism_TS',
    ]


Inspect_module_columns=[
'Phonation_columns',
'LOC_columns',
'DEP_columns',
'LOCDEP_columns',
'Phonation_Proximity_cols',
'Phonation_Trend_D_cols',
'Phonation_Trend_K_cols',
'Phonation_Convergence_cols',
'Phonation_Syncrony_cols',
'LOCDEP_Proximity_cols',
'LOCDEP_Trend_D_cols',
'LOCDEP_Trend_K_cols',
'LOCDEP_Convergence_cols',
'LOCDEP_Syncrony_cols'
]


df_feature_NotautismandASD_TC_larger=\
    df_ManagedResult.loc['TD vs df_feature_NotautismandASD_TC'][df_ManagedResult.loc['TD vs df_feature_NotautismandASD_TC'] > 0.799]
df_feature_Autism_TC_larger=\
    df_ManagedResult.loc['TD vs df_feature_Autism_TC'][df_ManagedResult.loc['TD vs df_feature_Autism_TC'] > 0.785]
df_feature_NotautismandASD_TS_larger=\
    df_ManagedResult.loc['TD vs df_feature_NotautismandASD_TS'][df_ManagedResult.loc['TD vs df_feature_NotautismandASD_TS'] > 0.747]
df_feature_Autism_TS_larger=\
    df_ManagedResult.loc['TD vs df_feature_Autism_TS'][df_ManagedResult.loc['TD vs df_feature_Autism_TS'] > 0.734]
        
df_Inspect_phonation_proximity_cols=df_ManagedResult.loc[Inspect_rows,'Phonation_Proximity_cols']
df_Inspect_phonation_proximity_cols=df_ManagedResult.loc[Inspect_rows,'LOCDEP_Trend_K_cols']

df_Inspect_module_cols=df_ManagedResult.loc[Inspect_rows,Inspect_module_columns]


aaa=ccc
# =============================================================================
# 
'''

    Check results from Regression fusion results

'''
manual_selected_settingXlsxFile='Regression_distance_2_DKIndividual'
Top_TManagedResult=Dict()
Top_TManagedResult_bag=Dict()
path="RESULTS/ADDed_UttFeat/"
xlsxfiles = [path+f for f in os.listdir(path) if re.search(r'Regression_(uniform|distance)_([0-9])_(DKIndividual|DKcriteria).xlsx', f)]
for file in xlsxfiles:
    nameOfFile=os.path.basename(file).replace(".xlsx","")
    df_result_file=pd.read_excel(file, index_col=0)
    
    
    
    
    
    df_dict={}
    for col in df_result_file.columns:
        task=col.split(" ")[0]
        df_result_file[['{}_R2'.format(task), '{}_pear'.format(task), '{}_spear'.format(task)]]=df_result_file[col].str.split("/",expand=True)
    
    df_managed_result=df_result_file.iloc[:,1:].astype(float)
    df_managed_result_IndexKlayer2=df_managed_result.copy()
    df_managed_result_IndexKlayer2.index=[c.split("-")[1] for c in df_managed_result.index.values]

    
    Top_TManagedResult_bag[nameOfFile]=df_managed_result_IndexKlayer2
    
df_fusionRegressioncols=Top_TManagedResult_bag[manual_selected_settingXlsxFile]
# df_choosen_knnParameter_dynPhonation=df_fusionRegressioncols.loc[FeatSel.dynamic_feature_phonation]
# df_choosen_knnParameter_dynLOC=df_fusionRegressioncols.loc[FeatSel.dynamic_feature_LOC]
# df_choosen_knnParameter_staticLOCDEP=df_fusionRegressioncols.loc[FeatSel.static_feautre_LOC]
# df_choosen_knnParameter_staticPhonation=df_fusionRegressioncols.loc[FeatSel.static_feautre_phonation]

df_Uttprosody_LOCDEP_columns_dict={}
df_Uttprosody_LOCDEP_columns=pd.DataFrame()
Utt_exp_idxs=[]
for metirc_col in ['ADOS_C/SVR_R2', 'ADOS_C/SVR_pear', 'ADOS_C/SVR_spear']:
    for utt_cols in FeatSel.Utt_feature:
        df_Uttprosody_LOCDEP_columns.loc[utt_cols,'']=df_fusionRegressioncols.loc[utt_cols,metirc_col]
        for locdep_cols in ['LOC_columns','DEP_columns','LOCDEP_columns']:
            index_query_str=utt_cols+"+"+locdep_cols
            Utt_exp_idxs.append(index_query_str)
            df_Uttprosody_LOCDEP_columns.loc[utt_cols,locdep_cols]=df_fusionRegressioncols.loc[index_query_str,metirc_col]
    df_Uttprosody_LOCDEP_columns_dict[metirc_col]=df_Uttprosody_LOCDEP_columns

df_fusionRegressioncols.loc[Utt_exp_idxs]


# Sample_template=df_managed_result
# for k,v in Top_TManagedResult_bag.items():
#     print((Sample_template - v).sum().sum())


