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
import articulation.HYPERPARAM.PaperNameMapping as PprNmeMp

from itertools import combinations, product

def Swap2PaperName(feature_rawname,PprNmeMp):
    if feature_rawname in PprNmeMp.Paper_name_map.keys():
        featurename_paper=PprNmeMp.Paper_name_map[feature_rawname]
        feature_keys=featurename_paper
    else: 
        feature_keys=feature_rawname
    return feature_keys
def Inverse_Swap2PaperName(feature_rawname,PprNmeMp):
    if feature_rawname in PprNmeMp.Inverse_Paper_name_map.keys():
        featurename_paper=PprNmeMp.Inverse_Paper_name_map[feature_rawname]
        feature_keys=featurename_paper
    else: 
        feature_keys=feature_rawname
    return feature_keys
# =============================================================================
'''

    This code is to reorganize the prediction results and facilitate generating 
    nice tables 

'''

SecondLvl_strmapp={
    # 'Vowel Dispersion+formant dependency':'vowel space characteristics(VSC)'
    'Vowel Dispersion+formant dependency':'VSC'
    }
# =============================================================================
# 
'''

    Check results from Classification fusion results

'''
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
''' Columns_comb3 = All possible feature combination + phonation_proximity_col :: Main analysis 1-2'''
''' Columns_comb5 = All possible feature combination + Phonation_Syncrony_cols :: Main analysis 1-2'''
''' Columns_comb5 = All possible feature combination + Phonation_Syncrony_cols :: Main analysis 2'''
path="RESULTS/Fusion_result/Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation/"
# path="RESULTS/Fusion_result/feat_comb3/"
# =============================================================================
manual_selected_settingXlsxFile= path+'Classification_distance_3_DKIndividual.xlsx'
file = manual_selected_settingXlsxFile



nameOfFile=os.path.basename(file).replace(".xlsx","")
df_result_file=pd.read_excel(file, index_col=0)

def SomeFix():
    reconstructed_idx_lst=[]
    for idx in  df_result_file.index:
        exp_str, feature=idx.split(" >> ")
        OriginalName_feat=Inverse_Swap2PaperName(feature,PprNmeMp)
        reconstructed_idx=exp_str+" >> "+OriginalName_feat
        reconstructed_idx_lst.append(reconstructed_idx)
    df_result_file.index=reconstructed_idx_lst
    df_result_file.to_excel('RESULTS/Fusion_result/Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation/Classification_distance_3_DKIndividual.xlsx')
    # df_best_result_allThreeClassifiers.to_excel(Result_path+"/"+"Classification_{knn_weights}_{knn_neighbors}_{Reorder_type}.xlsx".format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,Reorder_type=Reorder_type))




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
df_ManagedResult_classification=CriteriaSiftedResult.T.sort_values(by='Average',ascending=False)
# ['TD vs df_feature_lowMinimal_CSS >> LOC_columns+DEP_columns+LOCDEP_Trend_K_cols+Phonation_Convergence_cols+Phonation_Syncrony_cols', 'ASDTD'],
# ['TD vs df_feature_moderate_CSS >> LOC_columns+DEP_columns+LOCDEP_Trend_D_cols+LOCDEP_Trend_K_cols+Phonation_Proximity_cols', 'ASDTD'],
# ['TD vs df_feature_high_CSS >> LOCDEP_Trend_D_cols+Phonation_Proximity_cols', 'ASDTD'],

# ['TD vs df_feature_lowMinimal_CSS >> Phonation_Convergence_cols+Phonation_Syncrony_cols', 'ASDTD'],
# ['TD vs df_feature_moderate_CSS >> Phonation_Proximity_cols', 'ASDTD'],
# ['TD vs df_feature_high_CSS >> Phonation_Proximity_cols', 'ASDTD'],

exp_str='TD vs df_feature_lowMinimal_CSS >> LOC_columns+DEP_columns+LOCDEP_Trend_K_cols+Phonation_Convergence_cols+Phonation_Syncrony_cols'
df_result_file.loc[exp_str]

# Index_arrangement_lst=[
#     'Vowel Dispersion',
#     'LOC_columns_Intra',
#     'formant dependency',
#     'Proximity[phonation]',
#     'Convergence[phonation]',
#     'Syncrony[phonation]',
#     'Trend[phonation]_d',
#     'Trend[phonation]_k',
#     'Proximity[LOCDEP]',
#     'Convergence[LOCDEP]',
#     'Syncrony[LOCDEP]',
#     'Trend[LOCDEP]_d',
#     'Trend[LOCDEP]_k',
#     ]

# df_ManagedResult_classification=df_ManagedResult_classification.loc[Index_arrangement_lst]




# 疊加feature
# 第一步：製造想要的combination
CombFeatureCol_dict3=Dict()
CombPhonationCol=Dict()
# LOC_list = ['LOCDEP_Trend_D_cols', 'LOCDEP_Trend_K_cols', 'LOC_columns+DEP_columns']
LOC_list = FeatSel.dynamic_feature_LOC+FeatSel.static_feautre_LOC
# Phonation_list = ['Phonation_Trend_D_cols','Phonation_Trend_K_cols', 'Phonation_Syncrony_cols']
Phonation_list = ['Phonation_Trend_D_cols', 'Phonation_Syncrony_cols']
LOC_combs=[]
for L in range(1, len(LOC_list)+1):
    for subset in combinations(LOC_list, L):
        tmp_lst=[]
        for S in subset:
            tmp_lst+=S.split('+')
        LOC_combs.append(tmp_lst)
Phonation_combs=[]
for L in range(1, len(LOC_list)+1):
    for subset in combinations(Phonation_list, L):
        tmp_lst=[]
        for S in subset:
            tmp_lst+=S.split('+')
        Phonation_combs.append(tmp_lst)

for LOC_lst, Phonation_lst in list(product(Phonation_combs,LOC_combs)):
    CombFeatureCol_dict3[   '+'.join(LOC_lst)+"&"+'+'.join(Phonation_lst)    ]=LOC_lst+Phonation_lst
for Phonation_lst in Phonation_combs:
    CombPhonationCol['+'.join(Phonation_lst)]=Phonation_lst

CombFeatureCol=CombFeatureCol_dict3




# 第二步：比對df_ManagedResult_regression裡面的index所含有的feautre和我們想要找的實驗組合，然後把他裝到新的dataframe
Str2SortedLst=Dict()
FeatCombStr_lst=list(df_ManagedResult_classification.index)
df_layer1_fusion_results=pd.DataFrame([],columns=df_ManagedResult_classification.columns)
df_layer1_fusion_results_originalName=pd.DataFrame([],columns=df_ManagedResult_classification.columns)
df_layer1_fusion_results_FeatCombStr=pd.DataFrame([],columns=df_ManagedResult_classification.columns)
df_phonation_fusion=pd.DataFrame([],columns=df_ManagedResult_classification.columns)
df_phonation_originalName_fusion=pd.DataFrame([],columns=df_ManagedResult_classification.columns)
for FSL in FeatCombStr_lst:
    Str2SortedLst[FSL]=sorted(FSL.split("+"))
    for keys, values in CombPhonationCol.items():     
        PaperNamekeys='+'.join([Swap2PaperName(k,PprNmeMp) for k in re.split('\+',keys)])
        # if len (Str2SortedLst[FSL]) ==1 and len (values) ==1:
            
            # print(''.join(sorted(values)), ''.join(Str2SortedLst[FSL]))
        if ''.join(Str2SortedLst[FSL]) == ''.join(sorted(values)):
            print('columns match for ', FSL)
            print(''.join(sorted(values)), ''.join(Str2SortedLst[FSL]))
            for c in df_ManagedResult_classification.columns:
                df_phonation_fusion.loc[PaperNamekeys,c]=df_ManagedResult_classification.loc[FSL,c]
                df_phonation_fusion.loc[FSL,c]=df_ManagedResult_classification.loc[FSL,c]

    
    
    for keys, values in CombFeatureCol.items():        
        # PaperNamekeys='+'.join([Swap2PaperName(k,PprNmeMp) for k in re.split('\+|\&',keys)])
        PaperNamekeys='+'.join([Swap2PaperName(k,PprNmeMp) for k in re.split('\+',keys)])
        if ''.join(Str2SortedLst[FSL]) == ''.join(sorted(values)):
            # print('columns match for ', FSL)
            for c in df_ManagedResult_classification.columns:
                df_layer1_fusion_results.loc[PaperNamekeys,c]=df_ManagedResult_classification.loc[FSL,c]
                df_layer1_fusion_results_originalName.loc[keys,c]=df_ManagedResult_classification.loc[FSL,c]
                df_layer1_fusion_results_FeatCombStr.loc[FSL,c]=df_ManagedResult_classification.loc[FSL,c]



idx_order=[
'Inter-VSC',
'Syncrony[VSC]',
'Mod[VSC]_{d}',
'Mod[VSC]_{k}',
'Mod[VSC]_{d}+Inter-VSC',
'Mod[VSC]_{d}+Syncrony[VSC]',
'Mod[VSC]_{d}+Mod[VSC]_{k}',
'Mod[VSC]_{k}+Inter-VSC',
'Mod[VSC]_{k}+Syncrony[VSC]',
'Syncrony[VSC]+Inter-VSC',
'Mod[VSC]_{d}+Mod[VSC]_{k}+Inter-VSC',
'Mod[VSC]_{d}+Mod[VSC]_{k}+Syncrony[VSC]',
'Mod[VSC]_{d}+Syncrony[VSC]+Inter-VSC',
'Mod[VSC]_{k}+Syncrony[VSC]+Inter-VSC',
'Mod[VSC]_{d}+Mod[VSC]_{k}+Syncrony[VSC]+Inter-VSC',
]

col_order=[
'Convergence[P]',
'Syncrony[P]',
'Proximity[P]',
'Convergence[P]+Syncrony[P]',
'Proximity[P]+Convergence[P]',
'Proximity[P]+Syncrony[P]',
'Proximity[P]+Convergence[P]+Syncrony[P]',
]

Classification_fusion_result_dict=Dict()
for c in df_ManagedResult_classification.columns:
    Classification_fusion_result_dict[c]=pd.DataFrame()
    for idx in sorted(df_layer1_fusion_results.index):
        row,col=idx.split("&")
        PaperNamerow='+'.join([Swap2PaperName(k,PprNmeMp) for k in re.split('\+',row)])
        PaperNamecol='+'.join([Swap2PaperName(k,PprNmeMp) for k in re.split('\+',col)])
        for keys,values in SecondLvl_strmapp.items():
            PaperNamerow=PaperNamerow.replace(keys,values)
        Classification_fusion_result_dict[c].loc[PaperNamerow,PaperNamecol]=df_layer1_fusion_results.loc[idx,c]
        Classification_fusion_result_originalKey_dict[c].loc[PaperNamerow,PaperNamecol]=df_layer1_fusion_results.loc[idx,c]
    
    Classification_fusion_result_dict[c]=Classification_fusion_result_dict[c].loc[idx_order,col_order]

#%%
# 第一步：制定規則（什麼樣的feautre attribute包含哪些features）
Feature_category_rule1=Dict()

Feature_category_rule1['joint_feature_phonation']=['Phonation_Proximity_cols',
                                                      'Phonation_Convergence_cols',
                                                      'Phonation_Syncrony_cols']
Feature_category_rule1['individual_feature_phonation']=['Phonation_Trend_D_cols',
                                                        'Phonation_Trend_K_cols',]

Feature_category_rule1['joint_feature_LOCDEP']=['LOCDEP_Proximity_cols',
                                                  'LOCDEP_Convergence_cols',
                                                  'LOCDEP_Syncrony_cols']

Feature_category_rule1['individual_feature_LOCDEP']=[ 'LOCDEP_Trend_D_cols',
                                                              'LOCDEP_Trend_K_cols',]

Feature_category_rule2=Dict()
Feature_category_rule2['vowel space characteristics(VSC)']=['LOC_columns','DEP_columns','LOC_columns_Intra']
Feature_category_rule2['Conversation[P]']=FeatSel.dynamic_feature_phonation
Feature_category_rule2['Conversation[VSC]']=FeatSel.dynamic_feature_LOC

# 第二步：做出實驗字串到裡面含有什麼feature的mapping
CombFeatureCol_dict=Dict()
CombFeatureCol_dict['vowel space characteristics(VSC)+Phonation_Proximity_cols']=Feature_category_rule2['vowel space characteristics(VSC)']+['Phonation_Proximity_cols']
CombFeatureCol_dict['Conversation[VSC]+Phonation_Proximity_cols']=Feature_category_rule2['Conversation[VSC]']+['Phonation_Proximity_cols']

for feat in Feature_category_rule2['vowel space characteristics(VSC)']:
    CombFeatureCol_dict[feat+'+Phonation_Proximity_cols']=[feat]+['Phonation_Proximity_cols']
for feat in Feature_category_rule2['Conversation[VSC]']:
    CombFeatureCol_dict[feat+'+Phonation_Proximity_cols']=[feat]+['Phonation_Proximity_cols']

CombFeatureCol_dict2=Dict()
CombFeatureCol_dict2['vowel space characteristics(VSC)+Phonation_Syncrony_cols']=Feature_category_rule2['vowel space characteristics(VSC)']+['Phonation_Syncrony_cols']
CombFeatureCol_dict2['Conversation[VSC]+Phonation_Syncrony_cols']=Feature_category_rule2['Conversation[VSC]']+['Phonation_Syncrony_cols']

for feat in Feature_category_rule2['vowel space characteristics(VSC)']:
    CombFeatureCol_dict2[feat+'+Phonation_Syncrony_cols']=[feat]+['Phonation_Syncrony_cols']
for feat in Feature_category_rule2['Conversation[VSC]']:
    CombFeatureCol_dict2[feat+'+Phonation_Syncrony_cols']=[feat]+['Phonation_Syncrony_cols']

if path=="RESULTS/Fusion_result/feat_comb3/":
    CombFeatureCol=CombFeatureCol_dict
elif path=="RESULTS/Fusion_result/feat_comb5/":
    CombFeatureCol=CombFeatureCol_dict2
else:
    raise ValueError("Wrong variable: path", path)


# 第一步：製造想要的combination
CombFeatureCol_dict3=Dict()
LOC_list = ['LOCDEP_Trend_D_cols', 'LOCDEP_Trend_K_cols', 'LOCDEP_Syncrony_cols','LOC_columns+DEP_columns']
LOC_list = FeatSel.dynamic_feature_LOC+FeatSel.static_feature_LOC
Phonation_list = ['Phonation_Convergence_cols', 'Phonation_Trend_K_cols']
LOC_combs=[]
for L in range(1, len(LOC_list)+1):
    for subset in combinations(LOC_list, L):
        tmp_lst=[]
        for S in subset:
            tmp_lst+=S.split('+')
        LOC_combs.append(tmp_lst)
Phonation_combs=[]
for L in range(1, len(LOC_list)+1):
    for subset in combinations(Phonation_list, L):
        tmp_lst=[]
        for S in subset:
            tmp_lst+=S.split('+')
        Phonation_combs.append(tmp_lst)
for LOC_lst, Phonation_lst in list(product(LOC_combs,Phonation_combs)):
    CombFeatureCol_dict3[   '+'.join(LOC_lst)+"&"+'+'.join(Phonation_lst)    ]=LOC_lst+Phonation_lst

CombFeatureCol=CombFeatureCol_dict3

# 第二步：比對df_ManagedResult_regression裡面的index所含有的feautre和我們想要找的實驗組合，然後把他裝到新的dataframe
Str2SortedLst=Dict()
FeatCombStr_lst=list(df_ManagedResult_regression.index)
df_layer1_fusion_results=pd.DataFrame([],columns=df_ManagedResult_regression.columns)
for FSL in FeatCombStr_lst:
    Str2SortedLst[FSL]=sorted(FSL.split("+"))
    for keys, values in CombFeatureCol.items():        
        # PaperNamekeys='+'.join([Swap2PaperName(k,PprNmeMp) for k in re.split('\+|\&',keys)])
        PaperNamekeys='+'.join([Swap2PaperName(k,PprNmeMp) for k in re.split('\+',keys)])
        if ''.join(Str2SortedLst[FSL]) == ''.join(sorted(values)):
            print('columns match for ', FSL)
            for c in df_ManagedResult_regression.columns:
                df_layer1_fusion_results.loc[PaperNamekeys,c]=df_ManagedResult_regression.loc[FSL,c]


# 第三步：把他manage成LOC row跟phonation column的dataframe
Regression_fusion_result_dict=Dict()
for c in df_ManagedResult_regression.columns:
    Regression_fusion_result_dict[c]=pd.DataFrame()
    for idx in df_layer1_fusion_results.index:
        row,col=idx.split("&")
        PaperNamerow='+'.join([Swap2PaperName(k,PprNmeMp) for k in re.split('\+',row)])
        PaperNamecol='+'.join([Swap2PaperName(k,PprNmeMp) for k in re.split('\+',col)])
        for keys,values in SecondLvl_strmapp.items():
            PaperNamerow=PaperNamerow.replace(keys,values)
        Regression_fusion_result_dict[c].loc[PaperNamerow,PaperNamecol]=df_layer1_fusion_results.loc[idx,c]
# 第三步：比對df_ManagedResult_classification裡面的index所含有的feautre和我們想要找的實驗組合，然後把他裝到新的dataframe
Str2SortedLst=Dict()
FeatCombStr_lst=list(df_ManagedResult_classification.index)
df_layer1_fusion_results=pd.DataFrame([],columns=df_ManagedResult_classification.columns)
for FSL in FeatCombStr_lst:
    Str2SortedLst[FSL]=sorted(FSL.split("+"))
    for keys, values in CombFeatureCol.items():        
        PaperNamekeys='+'.join([Swap2PaperName(k,PprNmeMp) for k in re.split('\+',keys)])
        if ''.join(Str2SortedLst[FSL]) == ''.join(sorted(values)):
            print('columns match for ', FSL)
            for c in df_ManagedResult_classification.columns:
                df_layer1_fusion_results.loc[PaperNamekeys,c]=df_ManagedResult_classification.loc[FSL,c]

# Index_arrangement_fusion_lst=[ # articulation feature +Syncrony[P]
#     'vowel space characteristics(VSC)+Syncrony[P]',
#     'Conversation[VSC]+Syncrony[P]',
#     'Inter-Vowel Dispersion+Syncrony[P]',
#     'Intra-Vowel Dispersion+Syncrony[P]',
#     'formant dependency+Syncrony[P]',
#     'Proximity[VSC]+Syncrony[P]',
#     'Convergence[VSC]+Syncrony[P]',
#     'Syncrony[VSC]+Syncrony[P]',
#     'Modulation[VSC]_{d}+Syncrony[P]',
#     'Modulation[VSC]_{k}+Syncrony[P]',    
# ]


Index_arrangement_fusion_lst=[ # articulation feature +Syncrony[P]
    'vowel space characteristics(VSC)+Proximity[P]',
    'Conversation[VSC]+Proximity[P]',
    'Inter-Vowel Dispersion+Proximity[P]',
    'Intra-Vowel Dispersion+Proximity[P]',
    'formant dependency+Proximity[P]',
    'Proximity[VSC]+Proximity[P]',
    'Convergence[VSC]+Proximity[P]',
    'Syncrony[VSC]+Proximity[P]',
    'Modulation[VSC]_{d}+Proximity[P]',
    'Modulation[VSC]_{k}+Proximity[P]',
    ]
df_layer1_fusion_results.loc['formant dependency+Proximity[P]',]
df_layer1_fusion_results=df_layer1_fusion_results.loc[Index_arrangement_fusion_lst]
#%%
# !!!!!
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
    exp_str=cols.split(" ")[0]
    df_Managed_result=pd.DataFrame([],columns=columns_sel)
    for idx in df_result_file.index:
        
        # df_Managed_result.name=exp_str
        feature_module_str=idx.split("-")[-1]
        result_str=df_result_file.loc[idx,cols]
        result_lst= [float(r) for r in result_str.split("/")]
        df_Managed_result.loc[feature_module_str]=result_lst
    
    df_result_dicts[exp_str]=df_Managed_result
df_ManagedResult_regression=df_result_dicts['ADOS_C/SVR']


# 第一步：製造想要的combination
CombFeatureCol_dict3=Dict()
LOC_list = ['LOCDEP_Trend_D_cols', 'LOCDEP_Trend_K_cols', 'LOCDEP_Syncrony_cols','LOC_columns+DEP_columns']
Phonation_list = ['Phonation_Convergence_cols', 'Phonation_Trend_K_cols']
LOC_combs=[]
for L in range(1, len(LOC_list)+1):
    for subset in combinations(LOC_list, L):
        tmp_lst=[]
        for S in subset:
            tmp_lst+=S.split('+')
        LOC_combs.append(tmp_lst)
Phonation_combs=[]
for L in range(1, len(LOC_list)+1):
    for subset in combinations(Phonation_list, L):
        tmp_lst=[]
        for S in subset:
            tmp_lst+=S.split('+')
        Phonation_combs.append(tmp_lst)
for LOC_lst, Phonation_lst in list(product(LOC_combs,Phonation_combs)):
    CombFeatureCol_dict3[   '+'.join(LOC_lst)+"&"+'+'.join(Phonation_lst)    ]=LOC_lst+Phonation_lst

CombFeatureCol=CombFeatureCol_dict3

# 第二步：比對df_ManagedResult_regression裡面的index所含有的feautre和我們想要找的實驗組合，然後把他裝到新的dataframe
Str2SortedLst=Dict()
FeatCombStr_lst=list(df_ManagedResult_regression.index)
df_layer1_fusion_results=pd.DataFrame([],columns=df_ManagedResult_regression.columns)
for FSL in FeatCombStr_lst:
    Str2SortedLst[FSL]=sorted(FSL.split("+"))
    for keys, values in CombFeatureCol.items():        
        # PaperNamekeys='+'.join([Swap2PaperName(k,PprNmeMp) for k in re.split('\+|\&',keys)])
        PaperNamekeys='+'.join([Swap2PaperName(k,PprNmeMp) for k in re.split('\+',keys)])
        if ''.join(Str2SortedLst[FSL]) == ''.join(sorted(values)):
            print('columns match for ', FSL)
            for c in df_ManagedResult_regression.columns:
                df_layer1_fusion_results.loc[PaperNamekeys,c]=df_ManagedResult_regression.loc[FSL,c]


# 第三步：把他manage成LOC row跟phonation column的dataframe
Regression_fusion_result_dict=Dict()
for c in df_ManagedResult_regression.columns:
    Regression_fusion_result_dict[c]=pd.DataFrame()
    for idx in df_layer1_fusion_results.index:
        row,col=idx.split("&")
        PaperNamerow='+'.join([Swap2PaperName(k,PprNmeMp) for k in re.split('\+',row)])
        PaperNamecol='+'.join([Swap2PaperName(k,PprNmeMp) for k in re.split('\+',col)])
        for keys,values in SecondLvl_strmapp.items():
            PaperNamerow=PaperNamerow.replace(keys,values)
        Regression_fusion_result_dict[c].loc[PaperNamerow,PaperNamecol]=df_layer1_fusion_results.loc[idx,c]



for keys in df_result_dicts.keys():
    df_result_dicts[keys]=df_result_dicts[keys].loc[Index_arrangement_lst]




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


