#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 15:49:14 2022

@author: jackchen
"""
import os, glob, sys
import pickle
import argparse
from addict import Dict
import numpy as np
import pandas as pd
from articulation.HYPERPARAM import phonewoprosody, Label

from articulation.HYPERPARAM.PeopleSelect import SellectP_define
import articulation.HYPERPARAM.FeatureSelect as FeatSel
import articulation.HYPERPARAM.PaperNameMapping as PprNmeMp
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr,pearsonr 
from sklearn import neighbors
from metric import Evaluation_method 
import scipy.stats as stats

from articulation.HYPERPARAM.PlotFigureVars import *

# =============================================================================
def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--inpklpath', default='/media/jack/workspace/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--label_choose_lst', default=['ADOS_C'],
                            help=['ADOS_C','dia_num'])
    parser.add_argument('--dataset_role', default='TD_DOCKID',
                            help='[TD_DOCKID_emotion | ASD_DOCKID_emotion | kid_TD | kid88]')
    parser.add_argument('--knn_weights', default='uniform',
                            help='path of the base directory')
    parser.add_argument('--knn_neighbors', default=2,  type=int,
                            help='path of the base directory')
    parser.add_argument('--Inspect_roles', default=['D','K'],
                            help='')
    parser.add_argument('--Reorder_type', default='DKIndividual',
                            help='[DKIndividual, DKcriteria]')
    parser.add_argument('--Normalize_way', default='func15',
                            help='')
    
    args = parser.parse_args()
    return args


args = get_args()
pklpath=args.inpklpath
label_choose_lst=args.label_choose_lst # labels are too biased
dataset_role=args.dataset_role
knn_weights=args.knn_weights
knn_neighbors=args.knn_neighbors
Reorder_type=args.Reorder_type
# Randseed=args.Randseed
outpklpath=args.inpklpath+"/Session_formants_people_vowel_feat/"

if args.Normalize_way=='None':
    args.Normalize_way=None

# =============================================================================
'''

    Analysis area

'''
# =============================================================================
import seaborn as sns
from pylab import text
# dfFormantStatisticpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles'
dfFormantStatisticpath=os.getcwd()
def Swap2PaperName(feature_rawname,PprNmeMp):
    if feature_rawname in PprNmeMp.Paper_name_map.keys():
        featurename_paper=PprNmeMp.Paper_name_map[feature_rawname]
        feature_keys=featurename_paper
    else: 
        feature_keys=feature_rawname
    return feature_keys
# feat='Syncrony_measure_of_variance_phonation_DKIndividual' #Dynamic features phonation
# feat='Syncrony_measure_of_variance_DKIndividual' #Dynamic features LOCDEP
feat='Formant_AUI_tVSAFCRFvals'
def Add_label(df_formant_statistic,Label,label_choose='ADOS_S'):
    for people in df_formant_statistic.index:
        bool_ind=Label.label_raw['name']==people
        df_formant_statistic.loc[people,label_choose]=Label.label_raw.loc[bool_ind,label_choose].values
    return df_formant_statistic

# =============================================================================
    
# df_formant_statistic77_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/{name}_{role}.pkl'.format(name=feat,role='KID_FromASD_DOCKID')
df_formant_statistic77_path=dfFormantStatisticpath+f'/Features/ClassificationMerged_dfs/{args.Normalize_way}/ASD_DOCKID/static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation.pkl'
df_feature_ASD=pickle.load(open(df_formant_statistic77_path,'rb'))
# df_formant_statistic_ASDTD_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/{name}_{role}.pkl'.format(name=feat,role='KID_FromTD_DOCKID')
df_formant_statistic_ASDTD_path=dfFormantStatisticpath+f'/Features/ClassificationMerged_dfs/{args.Normalize_way}/TD_DOCKID/static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation.pkl'
if not os.path.exists(df_formant_statistic_ASDTD_path) or not os.path.exists(df_formant_statistic77_path):
    raise FileExistsError
df_feature_TD=pickle.load(open(df_formant_statistic_ASDTD_path,'rb'))


# ADD label

# TMP change back soon
df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='C+S')
df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='ADOS_cate_CSS')
df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='ADOS_cate_C')
df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='ADOS_cate_S')
df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='VCI')
df_feature_TD=Add_label(df_feature_TD,Label,label_choose='VCI')
df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='VIQ')
df_feature_TD=Add_label(df_feature_TD,Label,label_choose='VIQ')
df_feature_TD=Add_label(df_feature_TD,Label,label_choose='ADOS_SC')

# for l in Label.label_raw.columns:
#     if Label.label_raw[l].dtype != 'O':
#         df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose=l)
#         df_feature_TD=Add_label(df_feature_TD,Label,label_choose=l)

# create different ASD cohort
filter_Minimal_TCSS=df_feature_ASD['ADOS_cate_CSS']==0
filter_low_TCSS=df_feature_ASD['ADOS_cate_CSS']==1
filter_moderate_TCSS=df_feature_ASD['ADOS_cate_CSS']==2
filter_high_TCSS=df_feature_ASD['ADOS_cate_CSS']==3

filter_Notautism_TC=df_feature_ASD['ADOS_cate_C']==0
filter_ASD_TC=df_feature_ASD['ADOS_cate_C']==1
filter_Autism_TC=df_feature_ASD['ADOS_cate_C']==2

filter_Notautism_TS=df_feature_ASD['ADOS_cate_S']==0
filter_ASD_TS=df_feature_ASD['ADOS_cate_S']==1
filter_Autism_TS=df_feature_ASD['ADOS_cate_S']==2

df_feature_Minimal_CSS=df_feature_ASD[filter_Minimal_TCSS]
df_feature_low_CSS=df_feature_ASD[filter_low_TCSS]
df_feature_moderate_CSS=df_feature_ASD[filter_moderate_TCSS]
df_feature_high_CSS=df_feature_ASD[filter_high_TCSS]
df_feature_lowMinimal_CSS=df_feature_ASD[filter_low_TCSS | filter_Minimal_TCSS]
df_feature_moderatehigh_CSS=df_feature_ASD[filter_moderate_TCSS | filter_high_TCSS]

df_feature_Notautism_TC=df_feature_ASD[filter_Notautism_TC]
df_feature_ASD_TC=df_feature_ASD[filter_ASD_TC]
df_feature_NotautismandASD_TC=df_feature_ASD[filter_Notautism_TC | filter_ASD_TC]
df_feature_Autism_TC=df_feature_ASD[filter_Autism_TC]

df_feature_Notautism_TS=df_feature_ASD[filter_Notautism_TS]
df_feature_ASD_TS=df_feature_ASD[filter_ASD_TS]
df_feature_NotautismandASD_TS=df_feature_ASD[filter_Notautism_TS | filter_ASD_TS]
df_feature_Autism_TS=df_feature_ASD[filter_Autism_TS]


# Identify quadrant1 and quadrant3 samples
TaskQuadrant_dict=Dict()
TaskQuadrant_dict['moderateTask']['quadrant1']=['2017_01_20_01_243_1','2017_02_08_02_023_1']
TaskQuadrant_dict['moderateTask']['quadrant3']=['2017_07_20_TD_emotion','2021_01_23_5841_1(醫生鏡頭對焦到前面了)_emotion',\
                                                '2021_03_15_5874_1(醫生鏡頭模糊，醫生聲音雜訊大)_emotion']
TaskQuadrant_dict['highTask']['quadrant1']=[]
TaskQuadrant_dict['highTask']['quadrant3']=['2017_08_29_TD_emotion','2018_05_19_5593_1_emotion',\
                                                '2020_10_17_5633_1_emotion','2021_01_29_5843_1(醫生鏡頭模糊)_emotion']

    
# prepare for moderate Q1 Q3 instances
Moderate_Q1Q3_idxs=TaskQuadrant_dict['moderateTask']['quadrant1']+TaskQuadrant_dict['moderateTask']['quadrant3']
df_ModerateTask=pd.concat([df_feature_moderate_CSS,df_feature_TD],axis=0)
df_ModerateTask_Q1Q3=df_ModerateTask.loc[Moderate_Q1Q3_idxs].copy()
df_ModerateTask_exclude_Q1Q3=df_ModerateTask.drop(Moderate_Q1Q3_idxs).copy()
df_ModerateTask_Q1Q3['Selected']='Mispredicted samples'
df_ModerateTask_exclude_Q1Q3['Selected']='Other samples'
df_ModerateTask_postprocess=pd.concat([df_ModerateTask_Q1Q3,df_ModerateTask_exclude_Q1Q3],axis=0)
df_ModerateTask_postprocess_PprNme=df_ModerateTask_postprocess.copy()
df_ModerateTask_postprocess_PprNme.columns=[Swap2PaperName(col,PprNmeMp).replace("[$","[").replace("$]","]") for col in df_ModerateTask_postprocess.columns]


High_Q1Q3_idxs=TaskQuadrant_dict['highTask']['quadrant1']+TaskQuadrant_dict['highTask']['quadrant3']
df_HighTask=pd.concat([df_feature_high_CSS,df_feature_TD],axis=0)
df_HighTask_Q1Q3=df_HighTask.loc[High_Q1Q3_idxs].copy()
df_HighTask_exclude_Q1Q3=df_HighTask.drop(High_Q1Q3_idxs).copy()
df_HighTask_Q1Q3['Selected']=1
df_HighTask_exclude_Q1Q3['Selected']=0
df_HighTask_postprocess=pd.concat([df_HighTask_Q1Q3,df_HighTask_exclude_Q1Q3],axis=0)

# Plot moderate special samples
x_str='Trend[meanF0_var(A:,i:,u:)]_d'
y_str='between_variance_norm(A:,i:,u:)'

x_PprNme_str=Swap2PaperName(x_str,PprNmeMp).replace("[$","[").replace("$]","]")
y_PprNme_str=Swap2PaperName(y_str,PprNmeMp).replace("[$","[").replace("$]","]")

# =============================================================================
# 這邊檢查feature值
Inspect_columns=[
'Proximity[between_covariance_norm(A:,i:,u:)]',
'Proximity[between_variance_norm(A:,i:,u:)]',
'Proximity[within_covariance_norm(A:,i:,u:)]',
'Proximity[within_variance_norm(A:,i:,u:)]',
'Proximity[total_covariance_norm(A:,i:,u:)]',
]
# =============================================================================
Inspectdf_feature_lowMinimal_CSS=df_feature_lowMinimal_CSS[Inspect_columns]
Inspectdf_feature_moderatehigh_CSS=df_feature_moderatehigh_CSS[Inspect_columns]
Inspectdf_feature_high_CSS=df_feature_high_CSS[Inspect_columns]
Inspectdf_feature_TD=df_feature_TD[Inspect_columns]

#%%
# =============================================================================
# Scatter plot the distribution
# =============================================================================

# sellect_people_define=SellectP_define()
# SevereASD_age_sex_match=sellect_people_define.SevereASD_age_sex_match_ver2
# MildASD_age_sex_match=sellect_people_define.MildASD_age_sex_match_ver2
# TD_normal_ver2=sellect_people_define.TD_normal_ver2

# df_formant_statistic_agesexmatch_ASDSevere=df_feature_ASD.copy().loc[SevereASD_age_sex_match]
# df_formant_statistic_agesexmatch_ASDMild=df_feature_ASD.copy().loc[MildASD_age_sex_match]
# df_formant_statistic_TD_normal=df_feature_TD.copy().loc[TD_normal_ver2]

TopTop_data_lst=[]
# TopTop_data_lst.append(['df_formant_statistic_agesexmatch_ASDSevere','df_formant_statistic_TD_normal'])
# TopTop_data_lst.append(['df_formant_statistic_agesexmatch_ASDMild','df_formant_statistic_TD_normal'])
# TopTop_data_lst.append(['df_formant_statistic_agesexmatch_ASDMild','df_formant_statistic_agesexmatch_ASDSevere'])
''' Notice, ASD should be on the left '''

# TopTop_data_lst.append(['df_feature_ASD','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_low_CSS','df_feature_TD'])
TopTop_data_lst.append(['df_feature_lowMinimal_CSS','df_feature_TD'])
TopTop_data_lst.append(['df_feature_moderate_CSS','df_feature_TD'])
# TopTop_data_lst.append(['df_ModerateTask_Q1Q3','df_ModerateTask_exclude_Q1Q3'])
# TopTop_data_lst.append(['df_HighTask_Q1Q3','df_HighTask_exclude_Q1Q3'])
TopTop_data_lst.append(['df_feature_high_CSS','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_lowMinimal_CSS','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_moderatehigh_CSS','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_lowMinimal_CSS','df_feature_moderatehigh_CSS'])
# TopTop_data_lst.append(['df_feature_lowMinimal_CSS','df_feature_moderatehigh_CSS','df_feature_TD'])
Inspect_label='VIQ'
print(df_feature_lowMinimal_CSS[Inspect_label].mean(), '+-', df_feature_lowMinimal_CSS[Inspect_label].std())
print(df_feature_moderate_CSS[Inspect_label].mean(), '+-', df_feature_moderate_CSS[Inspect_label].std())
print(df_feature_high_CSS[Inspect_label].mean(), '+-', df_feature_high_CSS[Inspect_label].std())
print(df_feature_TD[Inspect_label].mean(), '+-', df_feature_TD[Inspect_label].std())
# TopTop_data_lst.append(['df_feature_low_CSS','df_feature_moderate_CSS'])
# TopTop_data_lst.append(['df_feature_moderate_CSS','df_feature_high_CSS'])
# TopTop_data_lst.append(['df_feature_low_CSS','df_feature_high_CSS'])
# TopTop_data_lst.append(['df_feature_lowMinimal_CSS','df_feature_moderate_CSS'])
# TopTop_data_lst.append(['df_feature_lowMinimal_CSS','df_feature_high_CSS'])

# TopTop_data_lst.append(['df_feature_Notautism_TC','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_ASD_TC','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_NotautismandASD_TC','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_Autism_TC','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_NotautismandASD_TC','df_feature_Autism_TC'])
# TopTop_data_lst.append(['df_feature_NotautismandASD_TC','df_feature_Autism_TC','df_feature_TD'])

# TopTop_data_lst.append(['df_feature_Notautism_TS','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_ASD_TS','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_NotautismandASD_TS','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_Autism_TS','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_NotautismandASD_TS','df_feature_Autism_TS'])


# TopTop_data_lst.append(['df_feature_Notautism_TS','df_feature_ASD_TS'])
# TopTop_data_lst.append(['df_feature_ASD_TS','df_feature_Autism_TS'])
# TopTop_data_lst.append(['df_feature_Notautism_TS','df_feature_Autism_TS'])

# self_specify_cols=[
#     'FCR2',
    # 'VSA2',
    # 'between_covariance_norm(A:,i:,u:)', 
    # 'between_variance_norm(A:,i:,u:)',
    # 'within_covariance_norm(A:,i:,u:)', 
    # 'within_variance_norm(A:,i:,u:)',
    # 'total_covariance_norm(A:,i:,u:)', 
    # 'total_variance_norm(A:,i:,u:)',
    # 'sam_wilks_lin_norm(A:,i:,u:)', 
    # 'pillai_lin_norm(A:,i:,u:)',
    # 'hotelling_lin_norm(A:,i:,u:)', 
    # 'roys_root_lin_norm(A:,i:,u:)',
    # 'Between_Within_Det_ratio_norm(A:,i:,u:)',
    # 'Between_Within_Tr_ratio_norm(A:,i:,u:)',
    # 'pear_12',
    # 'spear_12',
    # 'kendall_12',
    # 'dcorr_12'
    # ]
self_specify_cols=[
# 'Trend[VSA2]_d',
# 'Trend[FCR2]_d',
# 'Trend[between_covariance_norm(A:,i:,u:)]_d',
# 'Trend[between_variance_norm(A:,i:,u:)]_d',
# 'Trend[within_covariance_norm(A:,i:,u:)]_d',
# 'Trend[within_variance_norm(A:,i:,u:)]_d',
# 'Trend[total_covariance_norm(A:,i:,u:)]_d',
# 'Trend[total_variance_norm(A:,i:,u:)]_d',
# 'Trend[sam_wilks_lin_norm(A:,i:,u:)]_d',
# 'Trend[pillai_lin_norm(A:,i:,u:)]_d',
# 'Trend[hotelling_lin_norm(A:,i:,u:)]_d',
# 'Trend[roys_root_lin_norm(A:,i:,u:)]_d',
# 'Trend[Between_Within_Det_ratio_norm(A:,i:,u:)]_d',
# 'Trend[Between_Within_Tr_ratio_norm(A:,i:,u:)]_d', 
# 'Trend[pear_12]_d',
# 'Trend[spear_12]_d',
# 'Trend[kendall_12]_d',
# 'Trend[dcorr_12]_d'
]


FeatureSet_lst=[
    'LOCDEP_Proximity_cols',
  ]
self_specify_cols=[]
for FSL in FeatureSet_lst:
    self_specify_cols+=getattr(FeatSel, FSL)

# FeatureSet_lst=['Trend[Vowel_dispersion_inter__vowel_centralization]_d','Trend[Vowel_dispersion_inter__vowel_dispersion]_d',\
#                 'Trend[Vowel_dispersion_intra]_d','Trend[formant_dependency]_d'] #Trend[LOCDEP]d + Proximity[phonation]
# FeatureSet_lst=['Vowel_dispersion_inter__vowel_centralization','Vowel_dispersion_inter__vowel_dispersion',\
#                 ]     #Inter vowel dispersion + Syncrony[phonation]
# FeatureSet_lst=['Trend[Vowel_dispersion_inter__vowel_centralization]_d','Trend[Vowel_dispersion_inter__vowel_dispersion]_d',\
#                 'Trend[Vowel_dispersion_intra]_d','Trend[formant_dependency]_d',
#                 'Trend[Vowel_dispersion_inter__vowel_centralization]_k','Trend[Vowel_dispersion_inter__vowel_dispersion]_k',\
#                 'Trend[Vowel_dispersion_intra]_k','Trend[formant_dependency]_k',
#                 'Vowel_dispersion_inter__vowel_centralization','Vowel_dispersion_inter__vowel_dispersion','formant_dependency'\
#                 ]

# self_specify_cols=[]
# for col in FeatureSet_lst:
#     self_specify_cols+=FeatSel.CategoricalName2cols[col]


Parameters=df_feature_ASD.columns
if len(self_specify_cols) > 0:
    inspect_cols=self_specify_cols
else:
    inspect_cols=Parameters



plot=False
SWAPname_bool=True
Record_dict=Dict()
All_cmp_dict=Dict()
for Top_data_lst in TopTop_data_lst:
    Record_dict[' vs '.join(Top_data_lst)]=pd.DataFrame(index=inspect_cols)
    # All_cmp_dict[' vs '.join(Top_data_lst)]=pd.DataFrame(index=inspect_cols)
    All_cmp_dict[' vs '.join(Top_data_lst)]=pd.DataFrame()
    import warnings
    warnings.filterwarnings("ignore")
    for columns in inspect_cols:
        if SWAPname_bool==True:
            columns_papername=Swap2PaperName(columns,PprNmeMp)
        else:
            columns_papername=columns
        # =============================================================================
        if plot:
            fig, ax = plt.subplots()
        # =============================================================================
        data=[]
        dataname=[]
        for dstr in Top_data_lst:
            dataname.append(dstr)
            data.append(vars()[dstr])
        # =============================================================================
        if plot:
            for i,d in enumerate(data):
                # ax = sns.distplot(d[columns], ax=ax, kde=False)
                ax = sns.distplot(d[columns], ax=ax, label=Top_data_lst)
                title='{0}'.format('Inspecting feature ' + columns)
                plt.title( title )
            fig.legend(labels=dataname)  
        # =============================================================================
        for tests in [stats.f_oneway, stats.ttest_ind]:
            test_results=tests(vars()[Top_data_lst[0]][columns],vars()[Top_data_lst[1]][columns])
            p_val=test_results[1]

            if tests == stats.f_oneway:
                mean_difference=vars()[Top_data_lst[0]][columns].median() - vars()[Top_data_lst[1]][columns].median()
                All_cmp_dict[' vs '.join(Top_data_lst)].loc[columns_papername,'FTest'+' - '.join(Top_data_lst)]=np.round(mean_difference,3)
                All_cmp_dict[' vs '.join(Top_data_lst)].loc[columns_papername,'FTest'+'p']=np.round(p_val,3)
            elif tests == stats.ttest_ind:
                mean_difference=vars()[Top_data_lst[0]][columns].mean() - vars()[Top_data_lst[1]][columns].mean()
                All_cmp_dict[' vs '.join(Top_data_lst)].loc[columns_papername,'TTest'+' - '.join(Top_data_lst)]=np.round(mean_difference,3)
                All_cmp_dict[' vs '.join(Top_data_lst)].loc[columns_papername,'TTest'+'p']=np.round(p_val,3)
            if p_val < 0.05:
                print('Testing Feature: ',columns_papername)
                print(mean_difference , np.round(test_results[1],6))
                if tests == stats.f_oneway:
                    Record_dict[' vs '.join(Top_data_lst)].loc[columns_papername,'FTest'+' - '.join(Top_data_lst)]=mean_difference
                    Record_dict[' vs '.join(Top_data_lst)].loc[columns_papername,'FTest'+'p']=p_val
                if tests == stats.ttest_ind:
                    Record_dict[' vs '.join(Top_data_lst)].loc[columns_papername,'TTest'+' - '.join(Top_data_lst)]=mean_difference
                    Record_dict[' vs '.join(Top_data_lst)].loc[columns_papername,'TTest'+'p']=p_val
        # =============================================================================
        if plot:
            addtext='{0}/({1})'.format(np.round(mean_difference,3),np.round(p_val,3))
            text(0.9, 0.9, addtext, ha='center', va='center', transform=ax.transAxes)
            addtextvariable='{0} vs {1}'.format(Top_data_lst[0],Top_data_lst[1])
            text(0.9, 0.6, addtextvariable, ha='center', va='center', transform=ax.transAxes)
        # =============================================================================
    warnings.simplefilter('always')

dynamic_cols_sel='' #This means do not specify any columns
# Record_certainCol_dict={}
df_CertainCol_T=pd.DataFrame()
for test_name, values in All_cmp_dict.items():
    
    ASD_group_name=test_name.split(" vs ")[0].replace('df_feature_','')
    
    data_T=values.loc[values.index.str.startswith(dynamic_cols_sel),values.columns.str.startswith("TTest")]
    for idx in data_T.index:
        # for col in data_T.columns:
        # Fill in direction: ASD < TD or ASD >= TD
        if data_T.loc[idx,data_T.columns[0]]<0:
            df_CertainCol_T.loc[idx,ASD_group_name]='ASD $<$ TD'
        elif data_T.loc[idx,data_T.columns[0]]>0:
            df_CertainCol_T.loc[idx,ASD_group_name]='ASD $>$ TD'
        else:
            df_CertainCol_T.loc[idx,ASD_group_name]='ASD $=$ TD'
        
        if data_T.loc[idx,data_T.columns[1]]>0.05:
            df_CertainCol_T.loc[idx,ASD_group_name+'_p']='n.s.'
        else:
            df_CertainCol_T.loc[idx,ASD_group_name+'_p']=data_T.loc[idx,data_T.columns[1]]
# df_CertainCol_T=df_CertainCol_T.T







def Record_dict2Record_dictdf():
    # 舊的function 先放著
    dynamic_cols_sel='' #This means do not specify any columns
    # Record_certainCol_dict={}
    df_CertainCol_U=pd.DataFrame()
    df_CertainCol_T=pd.DataFrame()
    for test_name, values in Record_dict.items():
        
        data_T=values.loc[values.index.str.startswith(dynamic_cols_sel),values.columns.str.startswith("TTest")]
        data_U=values.loc[values.index.str.startswith(dynamic_cols_sel),values.columns.str.startswith("UTest")]
        # Utest results
        if len(data_U.columns) == 2:
            df_feat=data_U.iloc[:,0]
            df_feat.columns=[test_name]
        elif len(data_U.columns) == 0:
            df_feat=data_U
            df_feat[test_name]=np.nan
        df_CertainCol_U=pd.concat([df_CertainCol_U,df_feat],axis=1)
        
        # Ttest results
        
        if len(data_T.columns) == 2:
            df_feat=data_T.iloc[:,0]
            df_feat.columns=[test_name]
        elif len(data_T.columns) == 0:
            df_feat=data_T
            df_feat[test_name]=np.nan
    
        df_CertainCol_T=pd.concat([df_CertainCol_T,df_feat],axis=1)
    df_CertainCol_U=df_CertainCol_U.T
    df_CertainCol_T=df_CertainCol_T.T


#Clear Record_dict
Record_cleaned_dict={}
for keys, values in Record_dict.items():
    Record_cleaned_dict[keys]=values.dropna(thresh=2,axis=0)
    
    
    
    
# =============================================================================
'''

    Regression area

'''    
# =============================================================================

feat='Formant_AUI_tVSAFCRFvals'

# =============================================================================

# df_formant_statistic77_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/{name}_{role}.pkl'.format(name=feat,role='ASD_DOCKID')
df_feature_staticLOCDEP=pickle.load(open('articulation/Pickles/Session_formants_people_vowel_feat/Formant_AUI_tVSAFCRFvals_KID_FromASD_DOCKID.pkl','rb'))
df_feature_dynamicLOCDEP=pickle.load(open('articulation/Pickles/Session_formants_people_vowel_feat/Syncrony_measure_of_variance_DKIndividual_ASD_DOCKID.pkl','rb'))
df_feature_dynamicLOCDEP=Add_label(df_feature_ASD,Label,label_choose='ADOS_C')
df_feature_dynamicLOCDEP=Add_label(df_feature_ASD,Label,label_choose='ADOS_S')
# df_feature_dynamicPhonation=pickle.load(open('articulation/Pickles/Session_formants_people_vowel_feat/Syncrony_measure_of_variance_phonation_DKIndividual_ASD_DOCKID.pkl','rb'))

N=0
Eval_med=Evaluation_method()
label_correlation_choose_lst=['ADOS_C']

staticLOCDEP_cols=FeatSel.LOC_columns + FeatSel.DEP_columns
dynamicPhonation_cols=FeatSel.Phonation_Trend_D_cols + FeatSel.Phonation_Trend_K_cols + FeatSel.Phonation_Proximity_cols + FeatSel.Phonation_Convergence_cols + FeatSel.Phonation_Syncrony_cols
dynamicLOCDEP_cols=FeatSel.LOCDEP_Trend_D_cols + FeatSel.LOCDEP_Trend_K_cols + FeatSel.LOCDEP_Proximity_cols + FeatSel.LOCDEP_Convergence_cols + FeatSel.LOCDEP_Syncrony_cols





Aaad_Correlation_staticLOCDEP=Eval_med.Calculate_correlation(label_correlation_choose_lst,\
                                                    df_feature_ASD,\
                                                    N=2,columns=staticLOCDEP_cols,constrain_sex=-1, constrain_module=-1,\
                                                    feature_type='Session_formant')
#To swap to paper name
for Label_choose in label_correlation_choose_lst:
    Aaad_Correlation_staticLOCDEP[Label_choose].index=[Swap2PaperName(feature_rawname,PprNmeMp) for feature_rawname in Aaad_Correlation_staticLOCDEP[Label_choose].index]
    
    
Aaad_Correlation_dynamicLOCDEP=Eval_med.Calculate_correlation(label_correlation_choose_lst,\
                                                    df_feature_ASD,\
                                                    N=0,columns=dynamicLOCDEP_cols,constrain_sex=-1, constrain_module=-1,\
                                                    feature_type='')
for Label_choose in label_correlation_choose_lst:
    Aaad_Correlation_dynamicLOCDEP[Label_choose].index=[Swap2PaperName(feature_rawname,PprNmeMp) for feature_rawname in Aaad_Correlation_dynamicLOCDEP[Label_choose].index]
Aaad_Correlation_dynamicPhonation=Eval_med.Calculate_correlation(label_correlation_choose_lst,\
                                                    df_feature_ASD,\
                                                    N=0,columns=dynamicPhonation_cols,constrain_sex=-1, constrain_module=-1,\
                                                    feature_type='')
for Label_choose in label_correlation_choose_lst:
    Aaad_Correlation_dynamicPhonation[Label_choose].index=[Swap2PaperName(feature_rawname,PprNmeMp) for feature_rawname in Aaad_Correlation_dynamicPhonation[Label_choose].index]    



# =============================================================================
# 看哪個feature跟label有correlation
# =============================================================================
# dynamic_cols_sel='' #This means do not specify any columns
# # Record_certainCol_dict={}
# df_Regression_T=pd.DataFrame()
# for label_corr_str in  label_correlation_choose_lst:
    
#     Correlation_result_lst=[
#     Aaad_Correlation_staticLOCDEP[label_corr_str],
#     Aaad_Correlation_dynamicLOCDEP[label_corr_str],
#     Aaad_Correlation_dynamicPhonation[label_corr_str],
#     ]
#     InfoCorr_all=pd.concat(Correlation_result_lst,axis=0)
#     # for test_name, values in All_cmp_dict.items():
#     values=InfoCorr_all
#     # ASD_group_name=test_name.split(" vs ")[0].replace('df_feature_','')
#     # 先只看pearson的值
#     data_P=values.loc[values.index.str.startswith(dynamic_cols_sel),values.columns.str.startswith("pearson")]
#     for idx in data_P.index:
#         # for col in data_P.columns:
#         # Fill in direction: ASD < TD or ASD >= TD
#         df_Regression_T.loc[idx,data_P.columns[0]]=data_P.loc[idx,data_P.columns[0]]

#         if data_P.loc[idx,data_P.columns[1]]>0.05:
#             df_Regression_T.loc[idx,data_P.columns[1]]='n.s.'
#         else:
#             df_Regression_T.loc[idx,data_P.columns[1]]=data_P.loc[idx,data_P.columns[1]]
# # idx = df_Regression_T["pearson_p"].apply(lambda x: type(x) == np.float64)
# Significant_idxs = df_Regression_T["pearson_p"].apply(lambda x: type(x) != str)
# df_Regression_significant_T=df_Regression_T.loc[Significant_idxs]



#%%
# =============================================================================
'''

    Plot area

'''
from sklearn import neighbors
from Syncrony import Syncrony
PhoneOfInterest_str='A:,i:,u:'
features=[
    'between_covariance_norm(A:,i:,u:)',
     'between_variance_norm(A:,i:,u:)',
     'within_covariance_norm(A:,i:,u:)',
     'within_variance_norm(A:,i:,u:)',
     'total_covariance_norm(A:,i:,u:)',
    ]
# col從features裡面選
col='between_covariance_norm(A:,i:,u:)'
Inspect_people=list(df_feature_lowMinimal_CSS.index) #df_feature_lowMinimal_CSS, df_feature_moderate_CSS, df_feature_high_CSS

Colormap_role_dict=Dict()
Colormap_role_dict['D']='orange'
Colormap_role_dict['K']='blue'
df_person_segment_feature_DKIndividual_Topdict={}
# =============================================================================

df_person_segment_feature_DKIndividual_dict_TD=pickle.load(open(outpklpath+"df_person_segment_feature_DKIndividual_dict_{0}_{1}.pkl".format('TD_DOCKID', 'formant'),"rb"))
df_person_segment_feature_DKIndividual_dict_ASD=pickle.load(open(outpklpath+"df_person_segment_feature_DKIndividual_dict_{0}_{1}.pkl".format('ASD_DOCKID', 'formant'),"rb"))

df_person_segment_feature_DKIndividual_Topdict['TD']=df_person_segment_feature_DKIndividual_dict_TD
df_person_segment_feature_DKIndividual_Topdict['ASD']=df_person_segment_feature_DKIndividual_dict_ASD

def DrawFuncDK(segment_info_Topdict,Inspect_people,participant='TD'):
    # =============================================================================
    # 畫function_DK
    # 生出df_syncrony_measurement_dict
    # =============================================================================
    
    syncrony=Syncrony()
    PhoneOfInterest_str=''
    MinNumTimeSeries=knn_neighbors+1
    Inspect_roles=args.Inspect_roles
    
    segment_info_dict=segment_info_Topdict[participant]
    df_basic_additional_info=syncrony._Add_additional_info(segment_info_dict,Label,label_choose_lst,\
                                                  Inspect_roles, MinNumTimeSeries=MinNumTimeSeries,PhoneOfInterest_str=PhoneOfInterest_str)
    df_syncrony_measurement_dict=Dict()
    Knn_aggressive_mode=True

    Col_continuous_function_DK=syncrony.KNNFitting(segment_info_dict,\
                col, args.Inspect_roles,\
                knn_weights=knn_weights,knn_neighbors=knn_neighbors,MinNumTimeSeries=MinNumTimeSeries,\
                st_col_str='IPU_st', ed_col_str='IPU_ed', aggressive_mode=Knn_aggressive_mode)


    df_syncrony_measurement_col=syncrony._calculate_features_col(Col_continuous_function_DK,col)
    if df_syncrony_measurement_col.isna().any().any():
        print("The columns with Nan is ", col)
        
    df_syncrony_measurement_dict[col]=df_syncrony_measurement_col
    # =============================================================================
    # 換名字
    # =============================================================================
    score_column='Proximity[{}]'.format(col)
    # score_column='Trend[{0}]{suffix}'.format(col,suffix='_k')
    score_df=df_syncrony_measurement_col
    score_cols=[score_column]
    functionDK_people=Col_continuous_function_DK
    st_col_str='IPU_st'
    ed_col_str='IPU_ed'
    #%%
    # =============================================================================
    '''
        
        Plot function
        畫出來

    '''
    # Inputs: 
    # score_df
    # segment_info_dict
    # args.Inspect_roles
    # =============================================================================
    for people in list(score_df.sort_values(by=score_column).index):
        if len(Inspect_people) !=0:
            if people not in Inspect_people:
                continue
        df_person_segment_feature_role_dict=segment_info_dict[people]
        fig, ax = plt.subplots()
        for role_choose in args.Inspect_roles:
            df_dynVals=df_person_segment_feature_role_dict[role_choose][col]
            # remove outlier that is falls over 3 times of the std
            df_dynVals_deleteOutlier=df_dynVals[(np.abs(stats.zscore(df_dynVals)) < 3)]
            df_stidx=df_person_segment_feature_role_dict[role_choose][st_col_str]
            df_edidx=df_person_segment_feature_role_dict[role_choose][ed_col_str]
            recWidth=df_dynVals_deleteOutlier.min()/100
            # recWidth=0.0005
            for x_1 , x_2, y in zip(df_stidx.values ,df_edidx.values,df_dynVals_deleteOutlier.values):            
                ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,recWidth,color=Colormap_role_dict[role_choose],alpha=0.5))
            
            # add an totally overlapped rectangle but it will show the label
            ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,recWidth,color=Colormap_role_dict[role_choose],label=role_choose,alpha=0.5))
            
            
            plt.plot(functionDK_people[people][role_choose],color=Colormap_role_dict[role_choose])
        ax.autoscale()
        plt.title(f"{participant}--{col}")
        plt.legend()
        
        score=score_df.loc[people,score_cols]
        print("\n".join(["{}, {}: {}".format(people, idx,v) for idx, v in zip(score.index.values,np.round(score.values,3))]))
        info_arr=["{}: {}".format(idx,v) for idx, v in zip(score.index.values,np.round(score.values,3))]
        addtext='\n'.join(info_arr)
        x0, xmax = plt.xlim()
        y0, ymax = plt.ylim()
        data_width = xmax - x0
        data_height = ymax - y0
        # text(x0/0.1 + data_width * 0.004, -data_height * 0.002, addtext, ha='center', va='center')
        text(0, -0.1,addtext, ha='center', va='center', transform=ax.transAxes)
        
        plt.show()
        fig.clf()


DrawFuncDK(segment_info_Topdict=df_person_segment_feature_DKIndividual_Topdict,
           Inspect_people=Inspect_people,
           participant='ASD')
DrawFuncDK(segment_info_Topdict=df_person_segment_feature_DKIndividual_Topdict,
           Inspect_people=[],
           participant='TD')

aaa=ccc
# =============================================================================
# 畫function_DK
# 生出df_syncrony_measurement_dict
# =============================================================================
syncrony=Syncrony()
PhoneOfInterest_str=''
MinNumTimeSeries=knn_neighbors+1
Inspect_roles=args.Inspect_roles
df_basic_additional_info=syncrony._Add_additional_info(df_person_segment_feature_DKIndividual_dict,Label,label_choose_lst,\
                                              Inspect_roles, MinNumTimeSeries=MinNumTimeSeries,PhoneOfInterest_str=PhoneOfInterest_str)
df_syncrony_measurement_dict=Dict()
Knn_aggressive_mode=True

Col_continuous_function_DK=syncrony.KNNFitting(df_person_segment_feature_DKIndividual_dict,\
            col, args.Inspect_roles,\
            knn_weights=knn_weights,knn_neighbors=knn_neighbors,MinNumTimeSeries=MinNumTimeSeries,\
            st_col_str='IPU_st', ed_col_str='IPU_ed', aggressive_mode=Knn_aggressive_mode)


df_syncrony_measurement_col=syncrony._calculate_features_col(Col_continuous_function_DK,col)
if df_syncrony_measurement_col.isna().any().any():
    print("The columns with Nan is ", col)
    
df_syncrony_measurement_dict[col]=df_syncrony_measurement_col
# =============================================================================
# 換名字
# =============================================================================
score_column='Proximity[{}]'.format(col)
# score_column='Trend[{0}]{suffix}'.format(col,suffix='_k')
score_df=df_syncrony_measurement_col
score_cols=[score_column]
functionDK_people=Col_continuous_function_DK
st_col_str='IPU_st'
ed_col_str='IPU_ed'
#%%
# =============================================================================
'''
    
    Plot function
    畫出來

'''
# Inputs: 
# score_df
# df_person_segment_feature_DKIndividual_dict
# args.Inspect_roles
# =============================================================================
for people in list(score_df.sort_values(by=score_column).index):
    df_person_segment_feature_role_dict=df_person_segment_feature_DKIndividual_dict[people]
    fig, ax = plt.subplots()
    for role_choose in args.Inspect_roles:
        df_dynVals=df_person_segment_feature_role_dict[role_choose][col]
        # remove outlier that is falls over 3 times of the std
        df_dynVals_deleteOutlier=df_dynVals[(np.abs(stats.zscore(df_dynVals)) < 3)]
        df_stidx=df_person_segment_feature_role_dict[role_choose][st_col_str]
        df_edidx=df_person_segment_feature_role_dict[role_choose][ed_col_str]
        recWidth=df_dynVals_deleteOutlier.min()/100
        # recWidth=0.0005
        for x_1 , x_2, y in zip(df_stidx.values ,df_edidx.values,df_dynVals_deleteOutlier.values):            
            ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,recWidth,color=Colormap_role_dict[role_choose],alpha=0.5))
        
        # add an totally overlapped rectangle but it will show the label
        ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,recWidth,color=Colormap_role_dict[role_choose],label=role_choose,alpha=0.5))
        
        
        plt.plot(functionDK_people[people][role_choose],color=Colormap_role_dict[role_choose])
    ax.autoscale()
    plt.title(col)
    plt.legend()
    
    score=score_df.loc[people,score_cols]
    info_arr=["{}: {}".format(idx,v) for idx, v in zip(score.index.values,np.round(score.values,3))]
    addtext='\n'.join(info_arr)
    x0, xmax = plt.xlim()
    y0, ymax = plt.ylim()
    data_width = xmax - x0
    data_height = ymax - y0
    # text(x0/0.1 + data_width * 0.004, -data_height * 0.002, addtext, ha='center', va='center')
    text(0, -0.1,addtext, ha='center', va='center', transform=ax.transAxes)
    
    plt.show()
    fig.clf()

aaa=ccc
                