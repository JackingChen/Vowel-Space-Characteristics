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
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr,pearsonr 

from metric import Evaluation_method 
# =============================================================================
def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--inpklpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--label_choose_lst', default=['ADOS_C'],
                            help=['ADOS_C','dia_num'])
    parser.add_argument('--dataset_role', default='TD_DOCKID',
                            help='[TD_DOCKID_emotion | ASD_DOCKID_emotion | kid_TD | kid88]')
    args = parser.parse_args()
    return args


args = get_args()
pklpath=args.inpklpath
label_choose_lst=args.label_choose_lst # labels are too biased
dataset_role=args.dataset_role
# Randseed=args.Randseed
outpklpath=args.inpklpath+"/Session_formants_people_vowel_feat/"
# =============================================================================
'''

    Analysis area

'''
# =============================================================================
import seaborn as sns
from pylab import text
# dfFormantStatisticpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles'
dfFormantStatisticpath=os.getcwd()

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
df_formant_statistic77_path=dfFormantStatisticpath+'/Features/ClassificationMerged_dfs/distance_3_DKIndividual/ASD_DOCKID/static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation.pkl'
df_feature_ASD=pickle.load(open(df_formant_statistic77_path,'rb'))
# df_formant_statistic_ASDTD_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/{name}_{role}.pkl'.format(name=feat,role='KID_FromTD_DOCKID')
df_formant_statistic_ASDTD_path=dfFormantStatisticpath+'/Features/ClassificationMerged_dfs/distance_3_DKIndividual/TD_DOCKID/static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation.pkl'
if not os.path.exists(df_formant_statistic_ASDTD_path) or not os.path.exists(df_formant_statistic77_path):
    raise FileExistsError
df_feature_TD=pickle.load(open(df_formant_statistic_ASDTD_path,'rb'))


# ADD label
df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='ADOS_cate_CSS')
df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='ADOS_cate_C')
df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='ADOS_cate_S')
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
TopTop_data_lst.append(['df_feature_high_CSS','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_lowMinimal_CSS','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_moderatehigh_CSS','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_lowMinimal_CSS','df_feature_moderatehigh_CSS'])
# TopTop_data_lst.append(['df_feature_lowMinimal_CSS','df_feature_moderatehigh_CSS','df_feature_TD'])



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
'Trend[between_covariance_norm(A:,i:,u:)]_d',
'Trend[between_variance_norm(A:,i:,u:)]_d',
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

# self_specify_cols=FeatSel.Phonation_Trend_D_cols + FeatSel.Phonation_Trend_K_cols + FeatSel.Phonation_Proximity_cols + FeatSel.Phonation_Convergence_cols + FeatSel.Phonation_Syncrony_cols
# self_specify_cols=FeatSel.LOCDEP_Trend_D_cols + FeatSel.LOCDEP_Trend_K_cols + FeatSel.LOCDEP_Proximity_cols + FeatSel.LOCDEP_Convergence_cols + FeatSel.LOCDEP_Syncrony_cols

if len(self_specify_cols) > 0:
    inspect_cols=self_specify_cols
# else:
#     inspect_cols=Parameters
import scipy.stats as stats

plot=True
Record_dict=Dict()
All_cmp_dict=Dict()
for Top_data_lst in TopTop_data_lst:
    Record_dict[' vs '.join(Top_data_lst)]=pd.DataFrame(index=inspect_cols)
    All_cmp_dict[' vs '.join(Top_data_lst)]=pd.DataFrame(index=inspect_cols)
    import warnings
    warnings.filterwarnings("ignore")
    for columns in inspect_cols:
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
        for tests in [stats.mannwhitneyu, stats.ttest_ind]:
            test_results=tests(vars()[Top_data_lst[0]][columns],vars()[Top_data_lst[1]][columns])
            p_val=test_results[1]

            if tests == stats.mannwhitneyu:
                mean_difference=vars()[Top_data_lst[0]][columns].median() - vars()[Top_data_lst[1]][columns].median()
                All_cmp_dict[' vs '.join(Top_data_lst)].loc[columns,'UTest'+' - '.join(Top_data_lst)]=mean_difference
                All_cmp_dict[' vs '.join(Top_data_lst)].loc[columns,'UTest'+'p']=p_val
            elif tests == stats.ttest_ind:
                mean_difference=vars()[Top_data_lst[0]][columns].mean() - vars()[Top_data_lst[1]][columns].mean()
                All_cmp_dict[' vs '.join(Top_data_lst)].loc[columns,'TTest'+' - '.join(Top_data_lst)]=mean_difference
                All_cmp_dict[' vs '.join(Top_data_lst)].loc[columns,'TTest'+'p']=p_val
            if p_val < 0.05:
                print('Testing Feature: ',columns)
                print(mean_difference , np.round(test_results[1],6))
                if tests == stats.mannwhitneyu:
                    Record_dict[' vs '.join(Top_data_lst)].loc[columns,'UTest'+' - '.join(Top_data_lst)]=mean_difference
                    Record_dict[' vs '.join(Top_data_lst)].loc[columns,'UTest'+'p']=p_val
                if tests == stats.ttest_ind:
                    Record_dict[' vs '.join(Top_data_lst)].loc[columns,'TTest'+' - '.join(Top_data_lst)]=mean_difference
                    Record_dict[' vs '.join(Top_data_lst)].loc[columns,'TTest'+'p']=p_val
        # =============================================================================
        if plot:
            addtext='{0}/({1})'.format(np.round(mean_difference,3),np.round(p_val,3))
            text(0.9, 0.9, addtext, ha='center', va='center', transform=ax.transAxes)
            addtextvariable='{0} vs {1}'.format(Top_data_lst[0],Top_data_lst[1])
            text(0.9, 0.6, addtextvariable, ha='center', va='center', transform=ax.transAxes)
        # =============================================================================
    warnings.simplefilter('always')

# Record_certainCol_dict={}
# df_CertainCol=pd.DataFrame()
# for test_name, values in Record_dict.items():
#     df_proximity=values.loc[values.index.str.startswith("Proximity"),:].iloc[:,0]
#     df_proximity.columns=[test_name]
#     df_CertainCol=pd.concat([df_CertainCol,df_proximity],axis=1)
# df_CertainCol=df_CertainCol.T


# dynamic_cols_sel='Convergence'
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
df_feature_staticLOCDEP=pickle.load(open('/homes/ssd1/jackchen/DisVoice/articulation/Pickles/Session_formants_people_vowel_feat/Formant_AUI_tVSAFCRFvals_KID_FromASD_DOCKID.pkl','rb'))
df_feature_dynamicLOCDEP=pickle.load(open('/homes/ssd1/jackchen/DisVoice/articulation/Pickles/Session_formants_people_vowel_feat/Syncrony_measure_of_variance_DKIndividual_ASD_DOCKID.pkl','rb'))
df_feature_dynamicLOCDEP=Add_label(df_feature_ASD,Label,label_choose='ADOS_C')
df_feature_dynamicLOCDEP=Add_label(df_feature_ASD,Label,label_choose='ADOS_S')
df_feature_dynamicPhonation=pickle.load(open('/homes/ssd1/jackchen/DisVoice/articulation/Pickles/Session_formants_people_vowel_feat/Syncrony_measure_of_variance_phonation_DKIndividual_ASD_DOCKID.pkl','rb'))

N=0
Eval_med=Evaluation_method()
label_correlation_choose_lst=['ADOS_C']

staticLOCDEP_cols=FeatSel.LOC_columns + FeatSel.DEP_columns
dynamicPhonation_cols=FeatSel.Phonation_Trend_D_cols + FeatSel.Phonation_Trend_K_cols + FeatSel.Phonation_Proximity_cols + FeatSel.Phonation_Convergence_cols + FeatSel.Phonation_Syncrony_cols
dynamicLOCDEP_cols=FeatSel.LOCDEP_Trend_D_cols + FeatSel.LOCDEP_Trend_K_cols + FeatSel.LOCDEP_Proximity_cols + FeatSel.LOCDEP_Convergence_cols + FeatSel.LOCDEP_Syncrony_cols

Aaad_Correlation_staticLOCDEP=Eval_med.Calculate_correlation(label_correlation_choose_lst,\
                                                    df_feature_staticLOCDEP,\
                                                    N=2,columns=staticLOCDEP_cols,constrain_sex=-1, constrain_module=-1,\
                                                    feature_type='Session_formant')
Aaad_Correlation_dynamicLOCDEP=Eval_med.Calculate_correlation(label_correlation_choose_lst,\
                                                    df_feature_dynamicLOCDEP,\
                                                    N=0,columns=dynamicLOCDEP_cols,constrain_sex=-1, constrain_module=-1,\
                                                    feature_type='')
Aaad_Correlation_dynamicPhonation=Eval_med.Calculate_correlation(label_correlation_choose_lst,\
                                                    df_feature_dynamicPhonation,\
                                                    N=0,columns=dynamicPhonation_cols,constrain_sex=-1, constrain_module=-1,\
                                                    feature_type='')
aaa=ccc
#%%
# =============================================================================
'''

    Plot area

'''
# =============================================================================
from sklearn import neighbors
df_POI_person_segment_DKIndividual_feature_dict_TD=pickle.load(open(outpklpath+"df_POI_person_segment_DKIndividual_feature_dict_{0}_{1}.pkl".format('TD_DOCKID', 'phonation'),"rb"))
df_POI_person_segment_DKIndividual_feature_dict_ASD=pickle.load(open(outpklpath+"df_POI_person_segment_DKIndividual_feature_dict_{0}_{1}.pkl".format('ASD_DOCKID', 'phonation'),"rb"))

st_col_str='IPU_st'  #It is related to global info
ed_col_str='IPU_ed'  #It is related to global info
Inspect_roles=args.Inspect_roles
df_person_segment_feature_DKIndividual_dict=df_POI_person_segment_DKIndividual_feature_dict_ASD['A:,i:,u:']['segment']
p_1=Inspect_roles[0]
p_2=Inspect_roles[1]
df_syncrony_measurement=pd.DataFrame()
col = 'meanF0_mean(A:,i:,u:)'
Colormap_role_dict=Dict()
Colormap_role_dict['D']='orange'
Colormap_role_dict['K']='blue'
knn_weights='distance'
# knn_weights='uniform'
knn_neighbors=2
MinNumTimeSeries=knn_neighbors+1
PhoneOfInterest_str = 'A:,i:,u:'
functionDK_people=Dict()
for people in df_person_segment_feature_DKIndividual_dict.keys():
    if len(df_person_segment_feature_DKIndividual_dict[people][p_1])<MinNumTimeSeries or len(df_person_segment_feature_DKIndividual_dict[people][p_2])<MinNumTimeSeries:
        continue
    df_person_segment_feature_role_dict=df_person_segment_feature_DKIndividual_dict[people]
    
    RESULT_dict={}
    # kNN fitting
    Totalendtime=min([df_person_segment_feature_role_dict[role][ed_col_str].values[-1]  for role in Inspect_roles])
    Mintimeserieslen=min([len(df_person_segment_feature_role_dict[role])  for role in Inspect_roles])
    T = np.linspace(0, Totalendtime, int(Totalendtime))[:, np.newaxis]
    
    RESULT_dict['timeSeries_len[{}]'.format(PhoneOfInterest_str)]=Mintimeserieslen
    for label_choose in label_choose_lst:
        RESULT_dict[label_choose]=Label.label_raw[Label.label_raw['name']==people][label_choose].values[0]
    functionDK={}
    for role_choose in Inspect_roles:
        df_dynVals=df_person_segment_feature_role_dict[role_choose][col]
        # remove outlier that is falls over 3 times of the std
        df_dynVals_deleteOutlier=df_dynVals[(np.abs(stats.zscore(df_dynVals)) < 3)]
        df_stidx=df_person_segment_feature_role_dict[role_choose][st_col_str]
        df_edidx=df_person_segment_feature_role_dict[role_choose][ed_col_str]
        
        
        Mid_positions=[]
        for x_1 , x_2, y in zip(df_stidx.values ,df_edidx.values,df_dynVals_deleteOutlier.values):            
            # ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,0.5,color=Colormap_role_dict[role_choose]))
            start_time=x_1
            end_time=x_2
            mid_time=(start_time+end_time)/2
            Mid_positions.append(mid_time)    
        
        recWidth=df_dynVals_deleteOutlier.min()
        # add an totally overlapped rectangle but it will show the label
        # ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,recWidth,color=Colormap_role_dict[role_choose],label=role_choose))
        knn = neighbors.KNeighborsRegressor(knn_neighbors, weights=knn_weights)
        X, y=np.array(Mid_positions).reshape(-1,1), df_dynVals_deleteOutlier
        try:
            y_ = knn.fit(X, y.values).predict(T)
        except ValueError:
            print("Problem people happen at ", people, role_choose)
            print("df_dynVals", df_dynVals)
            print("==================================================")
            print("df_dynVals_deleteOutlier", df_dynVals_deleteOutlier)
            raise ValueError
        functionDK[role_choose]=y_
        # plt.plot(y_,color=Colormap_role_dict[role_choose],alpha=0.5)
    # ax.autoscale()
    # plt.title(col)
    # plt.legend()
    # plt.show()
    # fig.clf()
    functionDK_people[people]=functionDK
    
    proximity=-np.abs(np.mean(functionDK['D'] - functionDK['K']))
    D_t=-np.abs(functionDK['D']-functionDK['K'])

    time=T.reshape(-1)
    Convergence=pearsonr(D_t,time)[0]
    Trend_D=pearsonr(functionDK['D'],time)[0]
    Trend_K=pearsonr(functionDK['K'],time)[0]
    delta=[-15, -10, -5, 0, 5, 10, 15]        
    syncron_lst=[]
    for d in delta:
        if d < 0: #ex_ d=-15
            f_d_shifted=functionDK['D'][-d:]
            f_k_shifted=functionDK['K'][:d]
        elif d > 0: #ex_ d=15
            f_d_shifted=functionDK['D'][:-d]
            f_k_shifted=functionDK['K'][d:]
        else: #d=0
            f_d_shifted=functionDK['D']
            f_k_shifted=functionDK['K']
        syncron_candidate=pearsonr(f_d_shifted,f_k_shifted)[0]
        
        syncron_lst.append(syncron_candidate)
    syncrony=syncron_lst[np.argmax(np.abs(syncron_lst))]
    
    RESULT_dict['Proximity[{}]'.format(col)]=proximity
    RESULT_dict['Trend[{}]_d'.format(col)]=Trend_D
    RESULT_dict['Trend[{}]_k'.format(col)]=Trend_K
    RESULT_dict['Convergence[{}]'.format(col)]=Convergence
    RESULT_dict['Syncrony[{}]'.format(col)]=syncrony
    
    df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict,orient='index').T
    df_RESULT_list.index=[people]
    df_syncrony_measurement=df_syncrony_measurement.append(df_RESULT_list)
    
# score_column='Proximity[{}]'.format(col)
score_column='Proximity[{0}]{suffix}'.format(col,suffix='')
score_df=df_syncrony_measurement
score_cols=[score_column]

# =============================================================================
'''
    
    Plot function

'''
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
        # recWidth=df_dynVals_deleteOutlier.min()/100
        recWidth=0.05
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