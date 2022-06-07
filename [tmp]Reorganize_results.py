#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 15:41:44 2022

@author: jackchen
"""

import os, glob
import pandas as pd
import numpy as np
import re
from addict import Dict
import articulation.HYPERPARAM.FeatureSelect as FeatSel
# =============================================================================
'''

    This code is to reorganize the prediction results and facilitate generating 
    nice tables 

'''

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
['TD vs df_feature_lowMinimal_CSS', 'TD vs df_feature_moderatehigh_CSS',
       'TD vs df_feature_NotautismandASD_TC', 'TD vs df_feature_Autism_TC',
       'TD vs df_feature_NotautismandASD_TS', 'TD vs df_feature_Autism_TS',
       'TD vs df_feature_NotautismandASD_TSC', 'TD vs df_feature_Autism_TSC']
exp_pair1=['TD vs df_feature_lowMinimal_CSS', 'TD vs df_feature_moderatehigh_CSS']
exp_pair2=['TD vs df_feature_NotautismandASD_TC', 'TD vs df_feature_Autism_TC']
exp_pair3=['TD vs df_feature_NotautismandASD_TS', 'TD vs df_feature_Autism_TS']
exp_pair4=['TD vs df_feature_NotautismandASD_TSC', 'TD vs df_feature_Autism_TSC']
Exp_pairs=[exp_pair1,exp_pair2,exp_pair3,exp_pair4]
With_staticphonation_analysis_cols_bool=False
# =============================================================================


    

# =============================================================================
'''

    Reading from classification results

'''
path="RESULTS/Fusion_result/Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation/"
# =============================================================================
manual_selected_settingXlsxFile= path+'Classification_distance_3_DKIndividual.xlsx'
file = manual_selected_settingXlsxFile



nameOfFile=os.path.basename(file).replace(".xlsx","")
df_result_file=pd.read_excel(file, index_col=0)
All_experiments=list(df_result_file.index)

CriteriaSiftedResult=pd.DataFrame([])
CriteriaSifted_noPhonation_columns_dict=pd.DataFrame([])
df_ManagedResult_noPhonation_columns=pd.DataFrame()
data_dict={}
for i,experiment in enumerate(All_experiments):
    exp_pair_str, feature_module_str=experiment.split(" >> ")
    CriteriaSiftedResult.loc[exp_pair_str,feature_module_str]=df_result_file.loc[experiment].values[0]
    
    # Not including phonation columns
    # if 'Phonation_columns' not in feature_module_str and 'Phonation_columns' not in load_data_type:
    if 'Phonation_columns' not in feature_module_str:
        CriteriaSifted_noPhonation_columns_dict.loc[exp_pair_str,feature_module_str]=df_result_file.loc[experiment].values[0]
CriteriaSiftedResult.loc['Average']=CriteriaSiftedResult.mean(axis=0)
Aaa=CriteriaSiftedResult.T.sort_values(by='Average',ascending=False)

'''

    Reading from regression results

'''
manual_selected_settingXlsxFile= path+'Regression_distance_3_DKIndividual.xlsx'
file = manual_selected_settingXlsxFile

nameOfFile=os.path.basename(file).replace(".xlsx","")
df_result_file=pd.read_excel(file, index_col=0)

Regress_column_name='ADOS_C/SVR (MSE/pear/spear)'
df_result_file[Regress_column_name]

columns_sel=['MSE','pear','spear']


df_result_dicts=Dict()
for cols in df_result_file.columns:
    df_Managed_result=pd.DataFrame([],columns=columns_sel)
    for idx in df_result_file.index:
        exp_str=cols.split(" ")[0]
        # df_Managed_result.name=exp_str
        feature_module_str=idx.split("-")[-1]
        result_str=df_result_file.loc[idx,cols]
        result_lst= [float(r) for r in result_str.split("/")]
        
        df_Managed_result.loc[feature_module_str]=result_lst
        df_result_dicts[exp_str]=df_Managed_result














Top_CriteriaSifted_noPhonation_columns_Result[nameOfFile]=CriteriaSifted_noPhonation_columns_dict
Top_CriteriaSiftedResult[nameOfFile]=CriteriaSiftedResult
            
    
Top_collected_largerthan_dfs=Dict()


CSS_rows=['TD vs df_feature_lowMinimal_CSS','TD vs df_feature_moderate_CSS','TD vs df_feature_high_CSS']
TC_rows=['TD vs df_feature_NotautismandASD_TC','TD vs df_feature_Autism_TC']

Top_df_CriteriaSifted_columns=Dict()
for nameOfFile in Top_CriteriaSifted_noPhonation_columns_Result.keys():
    df_CriteriaSifted_columns={}
    df_columns=Top_CriteriaSifted_noPhonation_columns_Result[nameOfFile]
    df_CriteriaSifted_columns['CSS_rows']=df_columns.loc[CSS_rows]
    df_CriteriaSifted_columns['TC_rows']=df_columns.loc[TC_rows]
    
    
    def Analyze_Tasks_greaterthan_CompareCol(df_columns,compare_col='Phonation_Proximity_cols'):
        Top_collected_CSScols_largerthan_dfs=Dict()
        df_columns.loc['Average']=df_columns.mean()
        # Key_colms=[k for k in df_columns.keys() if compare_col in k]
        Key_colms=[k for k in df_columns.keys()]
        for K_c in Key_colms:
            # df_datacompare_raw=df_columns[K_c]
            # df_datacompare_raw.query('{0} > {1}'.format(K_c,compare_col))
            # df_compete_result=df_datacompare_raw[df_datacompare_raw[K_c] > df_datacompare_raw[compare_col]]
            # df_compete_result=df_columns[df_columns[K_c] > df_columns[compare_col]]
            # if (df_columns.loc[:,K_c] > df_columns.loc[:,compare_col]).any():
            if (df_columns.loc['Average',K_c] > df_columns.loc['Average',compare_col]).any():
                if nameOfFile not in Top_collected_CSScols_largerthan_dfs.keys():
                    Top_collected_CSScols_largerthan_dfs[nameOfFile]=df_columns.loc[:,[K_c,compare_col]]
                else:
                    Top_collected_CSScols_largerthan_dfs[nameOfFile]=pd.concat([df_columns.loc[:,K_c],\
                                                                                Top_collected_CSScols_largerthan_dfs[nameOfFile]],axis=1)
        df_top_collected_CSScols_largerthan_dfs=pd.DataFrame.from_dict(Top_collected_CSScols_largerthan_dfs[nameOfFile],orient='columns')
        return df_top_collected_CSScols_largerthan_dfs
    Top_df_CriteriaSifted_columns[nameOfFile]=df_CriteriaSifted_columns



Top_GreaterThan_result_dict=Dict()
for nameOfFile in Top_df_CriteriaSifted_columns.keys():
    df_CriteriaSifted_noPhonation_columns=Top_df_CriteriaSifted_columns[nameOfFile]
    for key in df_CriteriaSifted_columns.keys():
        Top_GreaterThan_result_dict[key][nameOfFile]=Analyze_Tasks_greaterthan_CompareCol(df_CriteriaSifted_noPhonation_columns[key],compare_col='Phonation_Proximity_cols')        
        # Top_GreaterThan_result_dict['{0}_{1}'.format(nameOfFile,key)]=Analyze_Tasks_greaterthan_CompareCol(df_CriteriaSifted_noPhonation_columns['CSS_rows'],compare_col='Phonation_Proximity_cols')        



# for data_type in p.keys():
#     p[data_type]
#     print(p[data_type]['Phonation_Proximity_cols'])


df_choosen_knnParameter_dynPhonation=Top_CriteriaSiftedResult[manual_selected_settingXlsxFile].loc[:,FeatSel.dynamic_feature_phonation]
df_choosen_knnParameter_dynLOC=Top_CriteriaSiftedResult[manual_selected_settingXlsxFile].loc[:,FeatSel.dynamic_feature_LOC]

df_choosen_knnParameter_staticLOCDEP=Top_CriteriaSiftedResult[manual_selected_settingXlsxFile].loc[:,FeatSel.static_feautre_LOC]
df_choosen_knnParameter_staticPhonation=Top_CriteriaSiftedResult[manual_selected_settingXlsxFile].loc[:,FeatSel.static_feautre_phonation]

Top_df_choosen_knnParameter_LOCseries=Dict()
for m_s_sXlsx in Top_CriteriaSiftedResult.keys():
    Top_df_choosen_knnParameter_LOCseries[m_s_sXlsx]=Top_CriteriaSiftedResult[m_s_sXlsx].loc[CSS_rows,FeatSel.dynamic_feature_LOC+FeatSel.static_feautre_LOC].T



Top_df_choosen_dict=Dict()
Top_df_choosen_dict['dynPhonation']=df_choosen_knnParameter_dynPhonation
Top_df_choosen_dict['dynLOC']=df_choosen_knnParameter_dynLOC
Top_df_choosen_dict['staticLOCDEP']=df_choosen_knnParameter_staticLOCDEP
Top_df_choosen_dict['staticPhonation']=df_choosen_knnParameter_staticPhonation

# df_fusionBettecols=Top_TManagedResult[manual_selected_settingXlsxFile]

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


