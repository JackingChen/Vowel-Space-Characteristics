#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 17:38:27 2021

@author: jackchen
"""
import os, glob, sys
import pickle
import numpy as np
import argparse
import pandas as pd
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
import seaborn as sns
import matplotlib.pyplot as plt
from addict import Dict
import itertools
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from itertools import combinations
from pylab import text

required_path_app = '/homes/ssd1/jackchen/DisVoice/articulation'  # for WER module imported in metric
sys.path.append(required_path_app)
from HYPERPARAM import phonewoprosody, Label


def criterion_filter(df_formant_statistic,N=10,\
                     constrain_sex=-1, constrain_module=-1,constrain_agemax=-1,constrain_ADOScate=-1,constrain_agemin=-1,\
                     evictNamelst=[],feature_type='Session_formant'):
    if feature_type == 'Session_formant':
        filter_bool=np.logical_and(df_formant_statistic['u_num']>N,df_formant_statistic['a_num']>N)
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['i_num']>N)
    elif feature_type == 'Syncrony_formant':
        filter_bool=df_formant_statistic['timeSeries_len']>N
    else:
        filter_bool=pd.Series([True]*len(df_formant_statistic),index=df_formant_statistic.index)
    if constrain_sex != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['sex']==constrain_sex)
    if constrain_module != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['Module']==constrain_module)
    if constrain_agemax != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['age']<=constrain_agemax)
    if constrain_agemin != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['age']>=constrain_agemin)
    if constrain_ADOScate != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['ADOS_cate']==constrain_ADOScate)
        
    if len(evictNamelst)>0:
        for name in evictNamelst:
            filter_bool.loc[name]=False
    # get rid of nan values
    filter_bool=np.logical_and(filter_bool,~df_formant_statistic.isna().T.any())
    return df_formant_statistic[filter_bool]

def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--dfFormantStatisticpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--ResultsOutpath', default='RESULTS/DF_FEATURECHECK_ASDvsTD/',
                        help='')
    parser.add_argument('--epoch', default='1',
                        help='')
    parser.add_argument('--label_choose_lst', default=['ADOS_C'],
                            help='path of the base directory')
    parser.add_argument('--Stat_med_str_VSA', default='mean',
                            help='path of the base directory')
    parser.add_argument('--Inspect_features', default=['F1','F2'],
                            help='')
    args = parser.parse_args()
    return args


args = get_args()
dfFormantStatisticpath=args.dfFormantStatisticpath
label_choose_lst=args.label_choose_lst
Stat_med_str_VSA=args.Stat_med_str_VSA
Inspect_features=args.Inspect_features

required_path_app = '/mnt/sdd/jackchen/egs/formosa/s6/local'  # for WER module imported in metric
sys.path.append(required_path_app)
from metric import Evaluation_method     

Eval_med=Evaluation_method()
# =============================================================================
'''

    Data preparation

'''
class Feature_Getter:
    def __init__(self, dfFormantStatisticpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',\
                       Stat_med_str_VSA='mean',\
                       Inspect_features=['F1','F2']):
        self.InPath=dfFormantStatisticpath
        self.Stat_med_str_VSA=Stat_med_str_VSA
        self.Inspect_features=['F1','F2']
        self.N=3
        self.SevereASD_age_sex_match=[
            '2016_03_05_01_079fu',
            '2017_08_15_01_413',
            '2017_01_21_01_242_1',
            '2015_12_06_01_045_1',
            '2017_05_21_01_292_1',
            '2017_07_25_01_172',
            '2016_10_22_01_161_fu',
            '2017_08_07_01_227',
            '2017_07_24_01_217_1',
            '2015_12_27_01_058fu',
            # '2016_10_21_01_202_1',
            ]
        self.MildASD_age_sex_match=['2017_08_11_01_300_1',
                '2017_08_15_01_413',
                '2016_03_05_01_079fu',
                '2016_07_06_01_078_1',
                '2016_11_27_01_207_1',
                '2016_01_22_02_083_1',
                '2017_05_21_01_292_1',
                '2016_10_22_01_161_fu',
                '2017_08_07_01_227',
                '2015_12_13_01_153',
                '2016_10_21_01_202_1',
                ]
        self.TD_normal=['2020_07_31_5595_1_emotion', '2021_01_23_5843_1(醫生鏡頭模糊)_emotion',
       '2020_10_24_5819_1_emotion', '2020_07_30_5818_1_emotion',
       '2021_01_09_5840_1_emotion', '2020_10_17_5633_1_emotion',
       '2021_01_29_5843_1(醫生鏡頭模糊)_emotion',
       '2021_02_03_5854_1(醫生鏡頭模糊)_emotion',
       '2021_01_23_5841_1(醫生鏡頭對焦到前面了)_emotion',
       '2021_02_05_5856_4(醫生鏡頭模糊)_emotion',
       '2021_01_29_5848_4(醫生鏡頭模糊)_emotion']
    def read_Sessionfeature(self,feat='Formant_AUI_tVSAFCRFvals'):
        df_formant_statistic77_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/{name}_{role}.pkl'.format(name=feat,role='kid88')
        self.df_feature_ASD=pickle.load(open(df_formant_statistic77_path,'rb'))
        df_formant_statistic_ASDTD_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/{name}_{role}.pkl'.format(name=feat,role='kid_TD')
        if not os.path.exists(df_formant_statistic_ASDTD_path) or not os.path.exists(df_formant_statistic77_path):
            raise FileExistsError
        self.df_feature_TD=pickle.load(open(df_formant_statistic_ASDTD_path,'rb'))
        return self.df_feature_ASD, self.df_feature_TD
    
    def read_Syncronyfeature(self,feat='Syncrony_measure_of_variance'):
        df_formant_statistic77_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/{name}_{role}.pkl'.format(name=feat,role='ASD_DOCKID')
        self.df_feature_ASD=pickle.load(open(df_formant_statistic77_path,'rb'))
        df_formant_statistic_ASDTD_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/{name}_{role}.pkl'.format(name=feat,role='TD_DOCKID')
        if not os.path.exists(df_formant_statistic_ASDTD_path) or not os.path.exists(df_formant_statistic77_path):
            raise FileExistsError
        self.df_feature_TD=pickle.load(open(df_formant_statistic_ASDTD_path,'rb'))
        return self.df_feature_ASD, self.df_feature_TD
    def read_durStrlenSpeedfeature(self,feat='df_dur_strlen_speed'):
        df_dur_strlen_speed_ASD77_path='Features/Other/{name}_{role}.pkl'.format(name=feat,role='ASD')
        df_dur_strlen_speed_TD_path='Features/Other/{name}_{role}.pkl'.format(name=feat,role='TD')
        self.df_dur_strlen_speed_ASD=pickle.load(open(df_dur_strlen_speed_ASD77_path,'rb'))
        self.df_dur_strlen_speed_TD=pickle.load(open(df_dur_strlen_speed_TD_path,'rb'))
        return self.df_dur_strlen_speed_ASD, self.df_dur_strlen_speed_TD
    def read_dfPersonSegmentFeatureDict(self,feat='formant'):
        df_person_segment_feature_dict_ASD77_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/df_person_segment_feature_dict_{role}_{feat}.pkl'.format(feat=feat,role='ASD_DOCKID')
        df_person_segment_feature_dict_TD_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/df_person_segment_feature_dict_{role}_{feat}.pkl'.format(feat=feat,role='TD_DOCKID')
        self.df_person_segment_feature_dict_ASD=pickle.load(open(df_person_segment_feature_dict_ASD77_path,'rb'))
        self.df_person_segment_feature_dict_TD=pickle.load(open(df_person_segment_feature_dict_TD_path,'rb'))
        return self.df_person_segment_feature_dict_ASD, self.df_person_segment_feature_dict_TD
    def read_dfPersonSegmentBasicInfoFeatureDict(self,feat='df_syncronyBasicInfo'):
        df_person_segment_feature_dict_ASD77_path='Features/Other/'+"df_person_segment_feature_dict_{0}_{1}.pkl".format('ASD', feat)
        df_person_segment_feature_dict_TD_path='Features/Other/'+"df_person_segment_feature_dict_{0}_{1}.pkl".format('TD', feat)
        self.df_person_segmentBasicInfo_feature_dict_ASD=pickle.load(open(df_person_segment_feature_dict_ASD77_path,'rb'))
        self.df_person_segmentBasicInfo_feature_dict_TD=pickle.load(open(df_person_segment_feature_dict_TD_path,'rb'))
        return self.df_person_segment_feature_dict_ASD, self.df_person_segment_feature_dict_TD
    def read_dfPersonPOISegmentFeatureDict(self,feat='phonation'):
        df_person_segment_feature_dict_ASD77_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/df_POI_person_segment_feature_dict_{role}_{feat}.pkl'.format(feat=feat,role='ASD_DOCKID')
        df_person_segment_feature_dict_TD_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/df_POI_person_segment_feature_dict_{role}_{feat}.pkl'.format(feat=feat,role='TD_DOCKID')
        self.df_person_segment_feature_dict_ASD=pickle.load(open(df_person_segment_feature_dict_ASD77_path,'rb'))
        self.df_person_segment_feature_dict_TD=pickle.load(open(df_person_segment_feature_dict_TD_path,'rb'))
        return self.df_person_segment_feature_dict_ASD, self.df_person_segment_feature_dict_TD
    
    
    
    def Get_labels_choosen(self,insp_people,Label,label_cols):
        df_labels_lsts=pd.DataFrame([],columns=label_cols)
        for people in insp_people:
            bool_ind=Label.label_raw['name']==people
            df_labels_lsts.loc[people,label_cols]=Label.label_raw.loc[bool_ind,label_cols].values
        return df_labels_lsts
    def _update_label(self,label):
        self.Label=label

    
# =============================================================================
'''  T-Test ASD vs TD''' 
df_formant_statistic_77=pd.DataFrame()
df_formant_statistic_TD=pd.DataFrame()
feat_getter=Feature_Getter(dfFormantStatisticpath,\
                           Stat_med_str_VSA=Stat_med_str_VSA,\
                           Inspect_features=Inspect_features)

feat_getter._update_label(Label)
label_cols=Label.label_choose
df_labels_ageMatchSevere=feat_getter.Get_labels_choosen(feat_getter.SevereASD_age_sex_match,Label,label_cols)
df_labels_ageMatchMild=feat_getter.Get_labels_choosen(feat_getter.MildASD_age_sex_match,Label,label_cols)
df_labels_TDnormal=feat_getter.Get_labels_choosen(feat_getter.TD_normal,Label,label_cols)

df_Session_formant_statistic_ASD, df_Session_formant_statistic_TD=feat_getter.read_Sessionfeature('Formant_AUI_tVSAFCRFvals')
df_Session_phonation_statistic_ASD, df_Session_phonation_statistic_TD=feat_getter.read_Sessionfeature('Phonation_meanvars')
df_Session_formant_statistic_TD['dia_num']=100


df_Syncrony_formant_statistic_ASD, df_Syncrony_formant_statistic_TD=feat_getter.read_Syncronyfeature('Syncrony_measure_of_variance')
df_Syncrony_phonation_statistic_ASD, df_Syncrony_phonation_statistic_TD=feat_getter.read_Syncronyfeature('Syncrony_measure_of_variance_phonation')
# df_Syncrony_phonation_statistic_ASD=df_Syncrony_phonation_statistic_ASD.dropna()
# df_Syncrony_phonation_statistic_TD=df_Syncrony_phonation_statistic_TD.dropna()
df_dur_strlen_speed_ASD, df_dur_strlen_speed_TD=feat_getter.read_durStrlenSpeedfeature('df_speedlenBasicInfo')
df_person_segment_feature_dict_ASD_formant, df_person_segment_feature_dict_TD_formant=feat_getter.read_dfPersonSegmentFeatureDict('formant')
df_person_segment_feature_dict_ASD_syncronyBasicInfo, df_person_segment_feature_dict_TD_syncronyBasicInfo=feat_getter.read_dfPersonSegmentBasicInfoFeatureDict('syncronyBasicInfo')
df_POI_person_segment_feature_dict_ASD_phonation, df_POI_person_segment_feature_dict_TD_phonation=feat_getter.read_dfPersonPOISegmentFeatureDict('phonation')

df_formant_statistic_77=pd.concat([df_formant_statistic_77,df_Session_formant_statistic_ASD],axis=1)
df_formant_statistic_TD=pd.concat([df_formant_statistic_TD,df_Session_formant_statistic_TD],axis=1)

# df_formant_statistic_77=pd.concat([df_formant_statistic_77,df_Session_phonation_statistic_ASD],axis=1)
# df_formant_statistic_TD=pd.concat([df_formant_statistic_TD,df_Session_phonation_statistic_TD],axis=1)

df_formant_statistic_77=pd.concat([df_formant_statistic_77,df_Syncrony_formant_statistic_ASD],axis=1)
df_formant_statistic_TD=pd.concat([df_formant_statistic_TD,df_Syncrony_formant_statistic_TD],axis=1)


# df_formant_statistic_77=pd.concat([df_formant_statistic_77,df_dur_strlen_speed_ASD],axis=1)
# df_formant_statistic_TD=pd.concat([df_formant_statistic_TD,df_dur_strlen_speed_TD],axis=1)

df_formant_statistic_77=pd.concat([df_formant_statistic_77,df_Syncrony_phonation_statistic_ASD],axis=1) #Contains nan
df_formant_statistic_TD=pd.concat([df_formant_statistic_TD,df_Syncrony_phonation_statistic_TD],axis=1)

df_formant_statistic_77=pd.concat([df_formant_statistic_77,df_dur_strlen_speed_ASD],axis=1) #Contains nan
df_formant_statistic_TD=pd.concat([df_formant_statistic_TD,df_dur_strlen_speed_TD],axis=1)

df_formant_statistic_77=df_formant_statistic_77.loc[:,~df_formant_statistic_77.columns.duplicated()]
df_formant_statistic_TD=df_formant_statistic_TD.loc[:,~df_formant_statistic_TD.columns.duplicated()]


if '2015_12_07_02_003' in df_formant_statistic_77.index:
    df_formant_statistic_77= df_formant_statistic_77.drop(index='2015_12_07_02_003') 

def Add_numSums(df_formant_statistic):  
    df_formant_statistic['u_num+i_num+a_num']=df_formant_statistic['u_num'] +\
                                            df_formant_statistic['i_num'] +\
                                            df_formant_statistic['a_num']
    return df_formant_statistic

df_formant_statistic_77=Add_numSums(df_formant_statistic_77)
df_formant_statistic_TD=Add_numSums(df_formant_statistic_TD)


SevereASD_age_sex_match=feat_getter.SevereASD_age_sex_match
MildASD_age_sex_match=feat_getter.MildASD_age_sex_match

def DropColsContainsNan(df_formant_statistic,PeopleOfInterest):
    df_formant_statistic_POI=df_formant_statistic.loc[PeopleOfInterest]
    lst=[]
    for col in df_formant_statistic_POI.columns:
        if df_formant_statistic_POI[col].isnull().values.any():
            lst.append(col)
    if len(lst) > 0:
        print('dropped columns: ', lst)
        df_formant_statistic=df_formant_statistic.drop(columns=lst)
    return df_formant_statistic


    


df_formant_statistic_77=DropColsContainsNan(df_formant_statistic_77,SevereASD_age_sex_match)
df_formant_statistic_77=DropColsContainsNan(df_formant_statistic_77,MildASD_age_sex_match)

df_formant_statistic=df_formant_statistic_77
def Add_label(df_formant_statistic,Label,label_choose='ADOS_S'):
    for people in df_formant_statistic.index:
        bool_ind=Label.label_raw['name']==people
        df_formant_statistic.loc[people,label_choose]=Label.label_raw.loc[bool_ind,label_choose].values
    return df_formant_statistic

df_dur_strlen_speed_ASD=Add_label(df_dur_strlen_speed_ASD,Label,label_choose='ADOS_C')
df_dur_strlen_speed_TD=Add_label(df_dur_strlen_speed_TD,Label,label_choose='ADOS_C')

df_formant_statistic_77=Add_label(df_formant_statistic_77,Label,label_choose='ADOS_S')
df_formant_statistic_77=Add_label(df_formant_statistic_77,Label,label_choose='ADOS_cate')
df_formant_statistic_77=Add_label(df_formant_statistic_77,Label,label_choose='AA2')
df_formant_statistic_TD=Add_label(df_formant_statistic_TD,Label,label_choose='ADOS_S')

Corr_formant_statistic_77=df_formant_statistic_77.corr()
Corr_formant_statistic_77.loc['AA2','ADOS_C']

''' Load DF that describes cases not qualified (ex: one example might be defined in articulation ) '''
ManualCondition=Dict()
suffix='.xlsx'
condfiles=glob.glob('articulation/Inspect/condition/*'+suffix)
for file in condfiles:
    df_cond=pd.read_excel(file)
    name=os.path.basename(file).replace(suffix,"")
    ManualCondition[name]=df_cond['Unnamed: 0'][df_cond['50%']==True]


# sex=-1
# module=-1
# agemax=-1
# agemin=-1
# ADOScate=-1
# N=1

# df_formant_statistic_77=criterion_filter(df_formant_statistic_77,\
#                                         constrain_sex=sex,constrain_module=module,N=N,constrain_agemax=agemax,constrain_agemin=agemin,constrain_ADOScate=ADOScate,\
#                                         evictNamelst=[],feature_type='Syncrony_formant')
# df_formant_statistic_TD=criterion_filter(df_formant_statistic_TD,constrain_sex=sex,constrain_module=module,N=N,constrain_agemax=agemax,feature_type='Syncrony_formant')





df_formant_statistic_77['ASDTD']=1
df_formant_statistic_TD['ASDTD']=2
df_formant_statistic_all=df_formant_statistic_77.append(df_formant_statistic_TD)
df_formant_statistic_all=df_formant_statistic_all[~df_formant_statistic_all[['Module','sex']].isna().any(axis=1)]
# =============================================================================
'''

    Simple t-test, U-test area

'''
# =============================================================================
DynamicDurlenFeats=['Divergence[Average_POIduration]',
 'Divergence[Average_POIduration]_split',
 'Divergence[Average_POIduration]_var_p1',
 'Divergence[Average_POIduration]_var_p2',
 'Divergence[Average_uttduration]',
 'Divergence[Average_uttduration]_split',
 'Divergence[Average_uttduration]_var_p1',
 'Divergence[Average_uttduration]_var_p2',
 'Divergence[Average_uttwordSpeed]',
 'Divergence[Average_uttwordSpeed]_split',
 'Divergence[Average_uttwordSpeed]_var_p1',
 'Divergence[Average_uttwordSpeed]_var_p2',
 'Divergence[Average_uttwordlengeth]',
 'Divergence[Average_uttwordlengeth]_split',
 'Divergence[Average_uttwordlengeth]_var_p1',
 'Divergence[Average_uttwordlengeth]_var_p2',
 'Divergence[Average_NumberOfPhones]',
 'Divergence[Average_NumberOfPhones]_split',
 'Divergence[Average_NumberOfPhones]_var_p1',
 'Divergence[Average_NumberOfPhones]_var_p2',
 'Divergence[Average_uttphoneSpeed]',
 'Divergence[Average_uttphoneSpeed]_split',
 'Divergence[Average_uttphoneSpeed]_var_p1',
 'Divergence[Average_uttphoneSpeed]_var_p2',
 'Divergence[Segment_wordvariety]',
 'Divergence[Segment_wordvariety]_split',
 'Divergence[Segment_wordvariety]_var_p1',
 'Divergence[Segment_wordvariety]_var_p2',
 'Divergence[Segment_CtxPvariety]',
 'Divergence[Segment_CtxPvariety]_split',
 'Divergence[Segment_CtxPvariety]_var_p1',
 'Divergence[Segment_CtxPvariety]_var_p2']
DurlenFeat=['dur', 'strlen', 'speed', 'totalword', 'Average_POIDur',
       'Average_phoneSpeed', 'PhoneVariety']
base_variance_features=[
# 'FCR',
# 'VSA1',
# 'sam_wilks_lin_norm(A:,i:,u:)',
# 'pillai_lin_norm(A:,i:,u:)',
# 'between_covariance_norm(A:,i:,u:)',
'within_covariance_norm(A:,i:,u:)',
# 'sam_wilks_lin(A:,i:,u:)',
# 'pillai_lin(A:,i:,u:)',
# 'between_covariance(A:,i:,u:)',
# 'within_covariance(A:,i:,u:)',
]
# base_variance_features=[
#     'sam_wilks_lin_norm(A:,i:,u:)',
#     'hotelling_lin_norm(A:,i:,u:)',
#     'roys_root_lin_norm(A:,i:,u:)',
#     'pillai_lin_norm(A:,i:,u:)',
#     ]
base_phonation_features=['localJitter_mean(A:,i:,u:)',
'localJitter_mean(u:)',
'localJitter_mean(i:)',
'localJitter_mean(A:)',
'localabsoluteJitter_mean(A:,i:,u:)',
'localabsoluteJitter_mean(u:)',
'localabsoluteJitter_mean(i:)',
'localabsoluteJitter_mean(A:)',
'localShimmer_mean(A:,i:,u:)',
'localShimmer_mean(u:)',
'localShimmer_mean(i:)',
'localShimmer_mean(A:)',
'stdevF0_mean(A:,i:,u:)',
'stdevF0_mean(u:)',
'stdevF0_mean(i:)',
'stdevF0_mean(A:)',
    ]
dummy_feature=['u_num+i_num+a_num']
# base_features=base_variance_features + base_phonation_features + dummy_feature
# base_features=base_variance_features + base_phonation_features
base_features=base_variance_features
# Syncrony_measres=['Divergence[{0}]','Divergence[{0}]_split','Divergence[{0}]_var_p1','Divergence[{0}]_var_p2']
Syncrony_measres=['Divergence[{0}]','Divergence[{0}]_var_p1','Divergence[{0}]_var_p2']
# Syncrony_measres=['Average[{0}]','Average[{0}]_split','Average[{0}]_var_p1','Average[{0}]_var_p2']
def CombineMeasurefeat(Syncrony_measres,base_features):
    lst=[]
    for basefeat in base_features:
        for sync_M in Syncrony_measres:
            lst.append(sync_M.format(basefeat))
    return lst
Measurement_basefeature=CombineMeasurefeat(Syncrony_measres,base_features)


Divergence_phonation_columns=[
 'Divergence[intensity_mean_mean(A:,i:,u:)]',
 'Divergence[meanF0_mean(A:,i:,u:)]',
 'Divergence[stdevF0_mean(A:,i:,u:)]',
 'Divergence[intensity_mean_var(A:,i:,u:)]',
 'Divergence[meanF0_var(A:,i:,u:)]',
 'Divergence[stdevF0_var(A:,i:,u:)]',
 
 'Divergence[intensity_mean_mean(A:,i:,u:)]_var_p1',
 'Divergence[meanF0_mean(A:,i:,u:)]_var_p1',
 'Divergence[stdevF0_mean(A:,i:,u:)]_var_p1',
 'Divergence[intensity_mean_var(A:,i:,u:)]_var_p1',
 'Divergence[meanF0_var(A:,i:,u:)]_var_p1',
 'Divergence[stdevF0_var(A:,i:,u:)]_var_p1',
 
 'Divergence[intensity_mean_mean(A:,i:,u:)]_var_p2',
 'Divergence[meanF0_mean(A:,i:,u:)]_var_p2',
 'Divergence[stdevF0_mean(A:,i:,u:)]_var_p2', 
 'Divergence[intensity_mean_var(A:,i:,u:)]_var_p2', 
 'Divergence[meanF0_var(A:,i:,u:)]_var_p2', 
 'Divergence[stdevF0_var(A:,i:,u:)]_var_p2',
]

Divergence_phonation_columns_p1=[
 'Divergence[intensity_mean_mean(A:,i:,u:)]_var_p1',
 # 'Divergence[intensity_mean_mean(A:,i:,u:)]_var_p2',
 'Divergence[meanF0_mean(A:,i:,u:)]_var_p1',
 # 'Divergence[meanF0_mean(A:,i:,u:)]_var_p2',
 'Divergence[stdevF0_mean(A:,i:,u:)]_var_p1',
 # 'Divergence[stdevF0_mean(A:,i:,u:)]_var_p2',
 'Divergence[intensity_mean_var(A:,i:,u:)]_var_p1',
 # 'Divergence[intensity_mean_var(A:,i:,u:)]_var_p2',
 'Divergence[meanF0_var(A:,i:,u:)]_var_p1',
 # 'Divergence[meanF0_var(A:,i:,u:)]_var_p2',
 'Divergence[stdevF0_var(A:,i:,u:)]_var_p1',
 # 'Divergence[stdevF0_var(A:,i:,u:)]_var_p2',
]


base_features=['localabsoluteJitter_mean(A:,i:,u:)',
'localabsoluteJitter_mean(u:)',
'localabsoluteJitter_mean(i:)',
'localabsoluteJitter_mean(A:)',
'localShimmer_mean(A:,i:,u:)',
'localShimmer_mean(u:)',
'localShimmer_mean(i:)',
'localShimmer_mean(A:)',]

# base_features=['localabsoluteJitter_mean(A:,i:,u:)',
# 'localShimmer_mean(A:,i:,u:)',
# ]


lst=[]
for sync_M in Syncrony_measres:
    for basefeat in base_features:    
        lst.append(sync_M.format(basefeat))


print(" Simple between group test (t-test, u-test)")

# =============================================================================
'''

    Choose parameters

'''
# Parameters=list(df_Syncrony_phonation_statistic_ASD.columns)
# Parameters=list(df_Syncrony_formant_statistic_ASD.columns) + list(df_Session_formant_statistic_ASD.columns)
# Parameters=list(df_Syncrony_phonation_statistic_ASD.columns) + list(df_Session_phonation_statistic_ASD.columns)
Parameters=Measurement_basefeature
# Parameters=list(df_Syncrony_formant_statistic_ASD.columns)
# Parameters=df_Session_formant_statistic_ASD.columns
# =============================================================================
'''

    To Get each group data and show the mean/median group data

''' 
filter_boy=df_formant_statistic_77['sex']==1
filter_girl=df_formant_statistic_77['sex']==2
filter_M3=df_formant_statistic_77['Module']==3
filter_M4=df_formant_statistic_77['Module']==4

filter_Notautism=df_formant_statistic_77['ADOS_cate']==0
filter_ASD=df_formant_statistic_77['ADOS_cate']==1
filter_Autism=df_formant_statistic_77['ADOS_cate']==2

filter_AD=df_formant_statistic_77['dia_num']==0
filter_AS=df_formant_statistic_77['dia_num']==1
filter_HFA=df_formant_statistic_77['dia_num']==2

df_formant_statistic_77_boy=df_formant_statistic_77[filter_boy]
df_formant_statistic_77_girl=df_formant_statistic_77[filter_girl]
df_formant_statistic_77_M3boy=df_formant_statistic_77[filter_M3 & filter_boy]
df_formant_statistic_77_M4boy=df_formant_statistic_77[filter_M4 & filter_boy]
df_formant_statistic_77_M3girl=df_formant_statistic_77[filter_M3 & filter_girl]
df_formant_statistic_77_M4girl=df_formant_statistic_77[filter_M4 & filter_girl]
df_formant_statistic_77_M3=df_formant_statistic_77[filter_M3 ]
df_formant_statistic_77_M4=df_formant_statistic_77[filter_M4 ]
df_formant_statistic_77_Notautism=df_formant_statistic_77[filter_Notautism]
df_formant_statistic_77_ASD=df_formant_statistic_77[filter_ASD]
df_formant_statistic_77_Autism=df_formant_statistic_77[filter_Autism]
df_formant_statistic_77_AD=df_formant_statistic_77[filter_AD]
df_formant_statistic_77_AS=df_formant_statistic_77[filter_AS]
df_formant_statistic_77_HFA=df_formant_statistic_77[filter_HFA]


filter_boy=df_formant_statistic_TD['sex']==1
filter_girl=df_formant_statistic_TD['sex']==2
filter_M3=df_formant_statistic_TD['Module']==3
filter_M4=df_formant_statistic_TD['Module']==4

df_formant_statistic_TD_boy=df_formant_statistic_TD[filter_boy]
df_formant_statistic_TD_girl=df_formant_statistic_TD[filter_girl]
df_formant_statistic_TD_M3boy=df_formant_statistic_TD[filter_M3 & filter_boy]
df_formant_statistic_TD_M4boy=df_formant_statistic_TD[filter_M4 & filter_boy]
df_formant_statistic_TD_M3girl=df_formant_statistic_TD[filter_M3 & filter_girl]
df_formant_statistic_TD_M4girl=df_formant_statistic_TD[filter_M4 & filter_girl]
df_formant_statistic_TD_M3=df_formant_statistic_TD[filter_M3 ]
df_formant_statistic_TD_M4=df_formant_statistic_TD[filter_M4 ]
df_formant_statistic_TD_normal=df_formant_statistic_TD[np.logical_and(\
            df_formant_statistic_TD['ADOS_cate']==0,df_formant_statistic_TD['ADOS_C'].notnull())]


df_formant_statistic_agematchASDSevere=pd.DataFrame()
df_formant_statistic_agematchASDmild=pd.DataFrame()
df_bagForAgeMatch_severe, df_bagForAgeMatch_mild, df_bagForAgeMatch_ASD\
    =df_formant_statistic_77.copy(), df_formant_statistic_77.copy(), df_formant_statistic_77.copy()
for idx in df_formant_statistic_TD_normal.index:
    age = df_formant_statistic_TD_normal.loc[idx]['age']
    df_agematch_ASD_severe=df_bagForAgeMatch_severe[df_bagForAgeMatch_severe['age']==age]
    df_agematch_ASD_mild=df_bagForAgeMatch_mild[df_bagForAgeMatch_mild['age']==age]
    
    df_agematch_ASD_severe=df_agematch_ASD_severe.sort_values('ADOS_C').iloc[[-1],:]
    df_agematch_ASD_mild=df_agematch_ASD_mild.sort_values('ADOS_C').iloc[[0],:]
    
    df_formant_statistic_agematchASDSevere=df_formant_statistic_agematchASDSevere.append(df_agematch_ASD_severe)
    df_formant_statistic_agematchASDmild=df_formant_statistic_agematchASDmild.append(df_agematch_ASD_mild)
    
    
    #remove from bag
    df_bagForAgeMatch_severe=df_bagForAgeMatch_severe.drop(index=df_agematch_ASD_severe.index)
    df_bagForAgeMatch_mild=df_bagForAgeMatch_mild.drop(index=df_agematch_ASD_mild.index)
df_formant_statistic_agematchASDSeverenMild=pd.concat([df_formant_statistic_agematchASDSevere,df_formant_statistic_agematchASDmild])


df_formant_statistic_ASDagematch=pd.DataFrame()
TD_age_set=set(df_formant_statistic_TD_normal['age'])
for age in TD_age_set:
    df_agematch_ASD=df_bagForAgeMatch_ASD[df_bagForAgeMatch_ASD['age']==age]
    df_formant_statistic_ASDagematch=df_formant_statistic_ASDagematch.append(df_agematch_ASD)
    

df_formant_statistic_agesexmatch_ASDSevere=df_formant_statistic_77.copy().loc[SevereASD_age_sex_match]
df_formant_statistic_agesexmatch_ASDMild=df_formant_statistic_77.copy().loc[MildASD_age_sex_match]

''' select some columns to inspect '''
''' Should remove soon, this is just for temp inspection '''
inspect_columns=['Average[within_covariance_norm(A:,i:,u:)]_p1',
 'Average[within_covariance_norm(A:,i:,u:)]_p2']

df_formant_statistic_agesexmatch_ASDSevere_inspect=df_formant_statistic_agesexmatch_ASDSevere[inspect_columns]
df_formant_statistic_agesexmatch_ASDMild_inspect=df_formant_statistic_agesexmatch_ASDMild[inspect_columns]
df_formant_statistic_TD_normal_inspect=df_formant_statistic_TD_normal[inspect_columns]

inspect_columns_kid=['Average[within_covariance_norm(A:,i:,u:)]_p2']
df_kids_statustics_compare_inspect=pd.concat([df_formant_statistic_agesexmatch_ASDSevere[inspect_columns_kid],\
                                              df_formant_statistic_agesexmatch_ASDMild[inspect_columns_kid],\
                                              df_formant_statistic_TD_normal[inspect_columns_kid]],axis=1).reset_index(drop=True)

for df_role in ['df_formant_statistic_agesexmatch_ASDSevere_inspect',\
                'df_formant_statistic_agesexmatch_ASDMild_inspect',\
                'df_formant_statistic_TD_normal_inspect']:
    vars()[df_role]['within_covariance_norm(doc - kid)'] =\
        vars()[df_role]['Average[within_covariance_norm(A:,i:,u:)]_p1'] - \
            vars()[df_role]['Average[within_covariance_norm(A:,i:,u:)]_p2']

        

        

df_formant_statistic_agesexmatch_ASDSevere=Add_numSums(df_formant_statistic_agesexmatch_ASDSevere)
df_formant_statistic_agesexmatch_ASDMild=Add_numSums(df_formant_statistic_agesexmatch_ASDMild)
df_formant_statistic_TD_normal=Add_numSums(df_formant_statistic_TD_normal)


dfFormantStatisticFractionpath=dfFormantStatisticpath+'/Fraction'
if not os.path.exists(dfFormantStatisticFractionpath):
    os.makedirs(dfFormantStatisticFractionpath)

pickle.dump(df_formant_statistic_agesexmatch_ASDSevere,open(dfFormantStatisticFractionpath+'/df_formant_statistic_agesexmatch_ASDSevere.pkl','wb'))
pickle.dump(df_formant_statistic_agesexmatch_ASDMild,open(dfFormantStatisticFractionpath+'/df_formant_statistic_agesexmatch_ASDMild.pkl','wb'))
pickle.dump(df_formant_statistic_TD_normal,open(dfFormantStatisticFractionpath+'/df_formant_statistic_TD_normal.pkl','wb'))
pickle.dump(df_formant_statistic_77_Notautism,open(dfFormantStatisticFractionpath+'/df_formant_statistic_77_Notautism.pkl','wb'))
pickle.dump(df_formant_statistic_77_ASD,open(dfFormantStatisticFractionpath+'/df_formant_statistic_77_ASD.pkl','wb'))
pickle.dump(df_formant_statistic_77_Autism,open(dfFormantStatisticFractionpath+'/df_formant_statistic_77_Autism.pkl','wb'))


df_average_stdValues=pd.DataFrame([],columns=df_formant_statistic_agesexmatch_ASDSevere.columns)

df_average_stdValues.loc['df_formant_statistic_agesexmatch_ASDSevere_MEAN']=df_formant_statistic_agesexmatch_ASDSevere.mean()
df_average_stdValues.loc['df_formant_statistic_agesexmatch_ASDMild_MEAN']=df_formant_statistic_agesexmatch_ASDMild.mean()
df_average_stdValues.loc['df_formant_statistic_TD_normal_MEAN']=df_formant_statistic_TD_normal.mean()

df_average_stdValues.loc['df_formant_statistic_agesexmatch_ASDSevere_MEDIAN']=df_formant_statistic_agesexmatch_ASDSevere.median()
df_average_stdValues.loc['df_formant_statistic_agesexmatch_ASDMild_MEDIAN']=df_formant_statistic_agesexmatch_ASDMild.median()
df_average_stdValues.loc['df_formant_statistic_TD_normal_MEDIAN']=df_formant_statistic_TD_normal.median()

df_average_stdValues.loc['df_formant_statistic_agesexmatch_ASDSevere_STD']=df_formant_statistic_agesexmatch_ASDSevere.std()
df_average_stdValues.loc['df_formant_statistic_agesexmatch_ASDMild_STD']=df_formant_statistic_agesexmatch_ASDMild.std()
df_average_stdValues.loc['df_formant_statistic_TD_normal_STD']=df_formant_statistic_TD_normal.std()



ChosenASDs_mean=['df_formant_statistic_agesexmatch_ASDSevere_MEAN','df_formant_statistic_agesexmatch_ASDMild_MEAN','df_formant_statistic_TD_normal_MEAN',]
ChosenASDs_median=['df_formant_statistic_agesexmatch_ASDSevere_MEDIAN','df_formant_statistic_agesexmatch_ASDMild_MEDIAN','df_formant_statistic_TD_normal_MEDIAN',]


FeatureChosenmeans=df_average_stdValues.loc[ChosenASDs_mean,Measurement_basefeature].T
FeatureChosenmedians=df_average_stdValues.loc[ChosenASDs_median,Measurement_basefeature].T




df_Labels_average_stdValues=pd.DataFrame([],columns=df_labels_ageMatchSevere.columns)
MutualConversationLabels=['AA8','BB9']
Communication_labels=['AA1',
 'AA2',
 'AA3',
 'AA4',
 'AA5',
 'AA6',
 'AA7',
 'AA8',
 'AA9',
 'BB9']
df_Labels_average_stdValues.loc['df_labels_ageMatchSevere_MEAN']=df_labels_ageMatchSevere.mean()
df_Labels_average_stdValues.loc['df_labels_ageMatchMild_MEAN']=df_labels_ageMatchMild.mean()
df_Labels_average_stdValues.loc['df_labels_TDnormal_MEAN']=df_labels_TDnormal.mean()
df_Labels_average_stdValues.loc['df_labels_ageMatchSevere_MEDIAN']=df_labels_ageMatchSevere.median()
df_Labels_average_stdValues.loc['df_labels_ageMatchMild_MEDIAN']=df_labels_ageMatchMild.median()
df_Labels_average_stdValues.loc['df_labels_TDnormal_MEDIAN']=df_labels_TDnormal.median()
df_Labels_average_stdValues.loc['df_labels_ageMatchSevere_STD']=df_labels_ageMatchSevere.std()
df_Labels_average_stdValues.loc['df_labels_ageMatchMild_STD']=df_labels_ageMatchMild.std()
df_Labels_average_stdValues.loc['df_labels_TDnormal_STD']=df_labels_TDnormal.std()
ChosenASDs_mean=['df_labels_ageMatchSevere_MEAN','df_labels_ageMatchMild_MEAN','df_labels_TDnormal_MEAN',]
ChosenASDs_median=['df_labels_ageMatchSevere_MEDIAN','df_labels_ageMatchMild_MEDIAN','df_labels_TDnormal_MEDIAN',]
LabelChosenmeans=df_Labels_average_stdValues.loc[ChosenASDs_mean,Communication_labels].T
LabelChosenmedians=df_Labels_average_stdValues.loc[ChosenASDs_median,Communication_labels].T


TopTop_data_lst=[]
# TopTop_data_lst.append(['df_formant_statistic_77','df_formant_statistic_TD_normal'])
# TopTop_data_lst.append(['df_formant_statistic_agematchASDSevere','df_formant_statistic_TD_normal'])
# TopTop_data_lst.append(['df_formant_statistic_agematchASDmild','df_formant_statistic_TD_normal'])
# TopTop_data_lst.append(['df_formant_statistic_agematchASDSeverenMild','df_formant_statistic_TD_normal'])
# TopTop_data_lst.append(['df_formant_statistic_ASDagematch','df_formant_statistic_TD_normal'])

TopTop_data_lst.append(['df_formant_statistic_agesexmatch_ASDSevere','df_formant_statistic_TD_normal'])
TopTop_data_lst.append(['df_formant_statistic_agesexmatch_ASDMild','df_formant_statistic_TD_normal'])
TopTop_data_lst.append(['df_formant_statistic_agesexmatch_ASDMild','df_formant_statistic_agesexmatch_ASDSevere'])
# TopTop_data_lst.append(['df_formant_statistic_agesexmatch_ASDSevere','df_formant_statistic_agesexmatch_ASDMild','df_formant_statistic_TD_normal'])

# TopTop_data_lst.append(['df_formant_statistic_77_Notautism','df_formant_statistic_77_Autism'])
# TopTop_data_lst.append(['df_formant_statistic_77_ASD','df_formant_statistic_77_Autism'])
# TopTop_data_lst.append(['df_formant_statistic_77_Notautism','df_formant_statistic_77_ASD'])
# TopTop_data_lst.append(['df_formant_statistic_77_AD','df_formant_statistic_77_AS'])
# TopTop_data_lst.append(['df_formant_statistic_77_AS','df_formant_statistic_77_HFA'])
# TopTop_data_lst.append(['df_formant_statistic_77_AD','df_formant_statistic_77_HFA'])

# TopTop_data_lst.append(['df_labels_ageMatchSevere','df_labels_TDnormal'])
# TopTop_data_lst.append(['df_labels_ageMatchMild','df_labels_TDnormal'])
# TopTop_data_lst.append(['df_labels_ageMatchSevere','df_labels_ageMatchMild'])



# TopTop_data_lst.append(['df_formant_statistic_77_M3','df_formant_statistic_TD_M3'])
# TopTop_data_lst.append(['df_formant_statistic_77_M4','df_formant_statistic_TD_M4'])
# TopTop_data_lst.append(['df_formant_statistic_77_boy','df_formant_statistic_TD_boy'])
# TopTop_data_lst.append(['df_formant_statistic_77_girl','df_formant_statistic_TD_girl'])
# TopTop_data_lst.append(['df_formant_statistic_77_M3boy','df_formant_statistic_TD_M3boy'])
# TopTop_data_lst.append(['df_formant_statistic_77_M3girl','df_formant_statistic_TD_M3girl'])
# TopTop_data_lst.append(['df_formant_statistic_77_M4boy','df_formant_statistic_TD_M4boy'])
# TopTop_data_lst.append(['df_formant_statistic_77_M4girl','df_formant_statistic_TD_M4girl'])

self_specify_cols=['between_covariance(A:,i:,u:)', 
                   'between_variance(A:,i:,u:)', 
                   'within_covariance(A:,i:,u:)', 
                   'within_variance(A:,i:,u:)',
                   'pillai_lin(A:,i:,u:)',
                   'sam_wilks_lin(A:,i:,u:)']
# self_specify_cols=['AA1',
#  'AA2',
#  'AA3',
#  'AA4',
#  'AA5',
#  'AA6',
#  'AA7',
#  'AA8',
#  'AA9',
#  'BB1',
#  'BB2',
#  'BB4',
#  'BB5',
#  'BB6',
#  'BB7',
#  'BB8',
#  'BB9',
#  'BB10',
#  'ADOS_C',
#  'ADOS_S',
#  'ADOS_SC']
# label_data=Label.label_raw.copy()
# label_data.index=label_data['name']
# label_data=label_data[self_specify_cols]
# self_specify_cols=['Divergence[between_variance(A:,i:,u:)]', 'Divergence[between_covariance(A:,i:,u:)]_split',
#                    'Divergence[hotelling_B]']
# self_specify_cols=Measurement_basefeature
# self_specify_cols=['Divergence[within_covariance(A:,i:,u:)]_var_p2']
# self_specify_cols=['Divergence[within_covariance_norm(A:,i:,u:)]',
# 'Divergence[within_covariance_norm(A:,i:,u:)]_var_p1',
# 'Divergence[within_covariance_norm(A:,i:,u:)]_split',
# 'Divergence[within_covariance(A:,i:,u:)]_split',
# 'Divergence[within_covariance(A:,i:,u:)]_var_p2',
# 'Divergence[between_covariance(A:,i:,u:)]_split',
# 'Divergence[total_covariance(A:,i:,u:)]_split',]

# self_specify_cols=[
# 'Divergence[within_covariance(A:,i:,u:)]_split',
# 'Divergence[within_covariance(A:,i:,u:)]_var_p2',
# 'Divergence[within_covariance(A:,i:,u:)]',
# ]
# self_specify_cols=['Divergence[localShimmer_mean(A:,i:,u:)]']
# self_specify_cols=['AA1',
#   'AA2',
#   'AA3',
#   'AA4',
#   'AA5',
#   'AA6',
#   'AA7',
#   'AA8',
#   'AA9',
#   'BB1',
#   'BB2',
#   'BB4',
#   'BB5',
#   'BB6',
#   'BB7',
#   'BB8',
#   'BB9',
#   'BB10']
# self_specify_cols=[
#     'Average[within_covariance_norm(A:,i:,u:)]_p1',
#     'Average[within_covariance_norm(A:,i:,u:)]_p2',
#     'Average[within_covariance_norm(A:,i:,u:)]_split',
#     'Average[within_covariance_norm(A:,i:,u:)]'
#     ]
# self_specify_cols=Measurement_basefeature
# self_specify_cols=DurlenFeat
# self_specify_cols=['hotelling_lin(A:,i:,u:)']
# self_specify_cols=[
#     'Divergence[within_covariance_norm(A:,i:,u:)]_var_p2',
#     'Divergence[within_covariance_norm(A:,i:,u:)]_var_p1',
#     'Divergence[within_covariance_norm(A:,i:,u:)]_split',
#     'Divergence[within_covariance_norm(A:,i:,u:)]'
#     ]
# self_specify_cols=['Divergence[sam_wilks_lin_norm(A:,i:,u:)]',
#   'Divergence[sam_wilks_lin_norm(A:,i:,u:)]_split',
#   'Divergence[sam_wilks_lin_norm(A:,i:,u:)]_var_p1',
#   'Divergence[sam_wilks_lin_norm(A:,i:,u:)]_var_p2',
#    'Divergence[hotelling_lin_norm(A:,i:,u:)]',
#    'Divergence[hotelling_lin_norm(A:,i:,u:)]_split',
#    'Divergence[hotelling_lin_norm(A:,i:,u:)]_var_p1',
#    'Divergence[hotelling_lin_norm(A:,i:,u:)]_var_p2',
#    'Divergence[between_covariance_norm(A:,i:,u:)]',
#    'Divergence[between_covariance_norm(A:,i:,u:)]_split',
#    'Divergence[between_covariance_norm(A:,i:,u:)]_var_p1',
#    'Divergence[between_covariance_norm(A:,i:,u:)]_var_p2',
#    'Divergence[within_covariance_norm(A:,i:,u:)]',
#    'Divergence[within_covariance_norm(A:,i:,u:)]_split',
#    'Divergence[within_covariance_norm(A:,i:,u:)]_var_p1',
#    'Divergence[within_covariance_norm(A:,i:,u:)]_var_p2'
#    ]
# self_specify_cols=[
#     'AA8',
#     'BB9',
#     ]
# self_specify_cols=[
# 'Divergence[within_covariance_norm(A:,i:,u:)]_var_p1',\
# 'Divergence[within_covariance_norm(A:,i:,u:)]_var_p2',\
# 'Divergence[within_covariance_norm(A:,i:,u:)]'
# ]
if len(self_specify_cols) > 0:
    inspect_cols=self_specify_cols
else:
    inspect_cols=Parameters

Record_dict=Dict()
All_cmp_dict=Dict()
for Top_data_lst in TopTop_data_lst:
    Record_dict[' vs '.join(Top_data_lst)]=pd.DataFrame(index=inspect_cols)
    All_cmp_dict[' vs '.join(Top_data_lst)]=pd.DataFrame(index=inspect_cols)
    import warnings
    warnings.filterwarnings("ignore")
    for columns in inspect_cols:
        # =============================================================================
        fig, ax = plt.subplots()
        # =============================================================================
        data=[]
        dataname=[]
        for dstr in Top_data_lst:
            dataname.append(dstr)
            data.append(vars()[dstr])
        # =============================================================================
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
            # if np.isnan(p_val):
            #     aaa=ccc
            
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
        addtext='{0}/({1})'.format(np.round(mean_difference,3),np.round(p_val,3))
        text(0.9, 0.9, addtext, ha='center', va='center', transform=ax.transAxes)
        addtextvariable='{0} vs {1}'.format(Top_data_lst[0],Top_data_lst[1])
        text(0.9, 0.6, addtextvariable, ha='center', va='center', transform=ax.transAxes)
        # =============================================================================
    warnings.simplefilter('always')

# =============================================================================
'''

    Correlation manual

'''
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
  
# =============================================================================



Feature_collect=Dict()
Feature_collect['df_Session_formant_statistic_ASD'].feature=df_Session_formant_statistic_ASD
Feature_collect['df_Session_phonation_statistic_ASD'].feature=df_Session_phonation_statistic_ASD
Feature_collect['df_Syncrony_formant_statistic_ASD'].feature=df_Syncrony_formant_statistic_ASD
Feature_collect['df_Syncrony_phonation_statistic_ASD'].feature=df_Syncrony_phonation_statistic_ASD
Feature_collect['df_labels_ageMatchSevere'].feature=df_labels_ageMatchSevere
Feature_collect['df_labels_ageMatchMild'].feature=df_labels_ageMatchMild
Feature_collect['df_labels_TDnormal'].feature=df_labels_TDnormal
Feature_collect['df_dur_strlen_speed_ASD'].feature=df_dur_strlen_speed_ASD



Feature_collect['df_Session_formant_statistic_ASD'].columns=intersection(df_formant_statistic_77.columns,df_Session_formant_statistic_ASD.columns)
Feature_collect['df_Session_phonation_statistic_ASD'].columns=intersection(df_formant_statistic_77.columns,df_Session_phonation_statistic_ASD.columns)
Feature_collect['df_Syncrony_formant_statistic_ASD'].columns=intersection(df_formant_statistic_77.columns,df_Syncrony_formant_statistic_ASD.columns)
Feature_collect['df_Syncrony_phonation_statistic_ASD'].columns=intersection(df_formant_statistic_77.columns,df_Syncrony_phonation_statistic_ASD.columns)
Feature_collect['df_labels_ageMatchSevere'].columns=df_labels_ageMatchSevere.columns
Feature_collect['df_labels_ageMatchMild'].columns=df_labels_ageMatchMild.columns
Feature_collect['df_labels_TDnormal'].columns=df_labels_TDnormal.columns
Feature_collect['df_dur_strlen_speed_ASD'].columns=df_dur_strlen_speed_ASD.columns

Correlation_dict=Dict()
Filtered_feat=Dict()
for feature_strs in Feature_collect.keys():
    features = Feature_collect[feature_strs].feature
    columns = Feature_collect[feature_strs].columns
    
    
    N=2
    if 'df_Session' in feature_strs:
        feat_type='Session_formant' 
        features['u_num+i_num+a_num']=features['u_num'] +features['i_num'] + features['a_num']
        columns +=['u_num+i_num+a_num']       
        if 'df_Session_phonation' in feature_strs:
            N=0
    elif 'df_Syncrony' in feature_strs:
        feat_type='Syncrony_formant'
    else:
        feat_type=''
        
    features_filt=criterion_filter(features,N=N,evictNamelst=[],feature_type=feat_type)
    Filtered_feat[feature_strs]=features_filt
    
    
    correlations_ASD=Eval_med.Calculate_correlation(label_choose_lst,features_filt,N,columns,feature_type=feat_type)['ADOS_C']
    
    Correlation_dict[feature_strs]=correlations_ASD

self_specify_cols=[
'FCR',
'VSA1',
'between_covariance(A:,i:,u:)',
'between_variance(A:,i:,u:)',
'within_covariance(A:,i:,u:)',
'within_variance(A:,i:,u:)',
'sam_wilks_lin(A:,i:,u:)',
'pillai_lin(A:,i:,u:)',
]
Chosen_correla=Correlation_dict['df_Session_formant_statistic_ASD'].loc[self_specify_cols]

# df_bag=pd.DataFrame()
# for idx in Chosen_correla.index:
#     v_p = Chosen_correla.loc[idx,'pearson_p']
#     val = np.round(Chosen_correla.loc[idx,'pearsonr'],3)
#     p_val=Addsuffix(v_p)
#     val_str = str(val) + p_val
#     df_bag.loc[idx,'pearsonr']=val_str
    
#     v_p = Chosen_correla.loc[idx,'spearman_p']
#     val = np.round(Chosen_correla.loc[idx,'spearmanr'],3)
#     p_val=Addsuffix(v_p)
#     val_str = str(val) + p_val
#     df_bag.loc[idx,'spearmanr']=val_str
    
#     df_bag.loc[idx,'R2']=Chosen_correla.loc[idx,'R2']
    

# =============================================================================
''' Nice Table area 

Automatically generate nice table from Collected data dictionary

'''

ASDSevere_TDnormal='df_formant_statistic_agesexmatch_ASDSevere vs df_formant_statistic_TD_normal'
ASDMild_TDnormal='df_formant_statistic_agesexmatch_ASDMild vs df_formant_statistic_TD_normal'


# ASDSevere_TDnormal='df_labels_ageMatchSevere vs df_labels_TDnormal'
# ASDMild_TDnormal='df_labels_ageMatchMild vs df_labels_TDnormal'


KeyColumnMapp=Dict()
df_sources_dict=Dict()

mean_str='TTest'
median_str='UTest'
KeyColumnMapp['mean_str']=mean_str
KeyColumnMapp['median_str']=median_str
KeyColumnMapp['mean_strp']=mean_str + 'p'
KeyColumnMapp['median_strp']=median_str + 'p'
KeyColumnMapp['ASDSev_mean_str']=mean_str+ASDSevere_TDnormal.replace('vs','-')
KeyColumnMapp['ASDSev_median_str']=median_str+ASDSevere_TDnormal.replace('vs','-')
KeyColumnMapp['ASDMil_mean_str']=mean_str+ASDMild_TDnormal.replace('vs','-')
KeyColumnMapp['ASDMil_median_str']=median_str+ASDMild_TDnormal.replace('vs','-')


def Addsuffix(val):
    dagger='†'
    if val <0.1 and val > 0.05:
        return dagger
    elif val < 0.05 and val >= 0.01:
        return '*'
    elif val < 0.01 and val >= 0.001:
        return '**'
    elif val < 0.001:
        return '***'
    elif val > 0.1:
        return ''
    else:
        raise ValueError('not a valid p value')
def Add_dataframe(idx,col,KeyColumnMapp,df_source,df_Table):
    # col='ASDSev_mean'
    col_str=col+'_str'
    data_col=KeyColumnMapp[col_str]
    pool_med=col.split('_')[-1]
    pool_med_str=pool_med+'_str'
    # df_Table.loc[idx,col]=str(df_source.loc[idx,data_col])
    p_val=df_source.loc[idx,KeyColumnMapp[pool_med_str + 'p'] ]
    suffix=Addsuffix(p_val)
    df_Table.loc[idx,col] = str(np.round(df_source.loc[idx,data_col],3)) + suffix
    return df_Table
def Get_FeatureGrps(idx,Feature_collect):
    for Featgrps in Feature_collect.keys():
        # print(idx in Feature_collect[Featgrps].columns)
        if idx in Feature_collect[Featgrps].columns:
            return Featgrps
    raise KeyError('feature not existing in groups')
# =============================================================================
df_Table=pd.DataFrame([],columns=['ASDSev_median','ASDSev_mean','ASDMil_median','ASDMil_mean','pearson','spearman'])
ASDTD_cmps_lst=[col for col in  df_Table.columns if 'ASD' in col]

df_severenTD=All_cmp_dict[ASDSevere_TDnormal]
df_mildnTD=All_cmp_dict[ASDMild_TDnormal]
df_sources_dict['ASDSev']=df_severenTD
df_sources_dict['ASDMil']=df_mildnTD

bookeep_dict=Dict()
bookeepDF_dict=Dict()
count=0
for idx in df_severenTD.index:
    for col in ASDTD_cmps_lst:
        role_str=col.split('_')[0]
        df_source=df_sources_dict[role_str]
        df_Table=Add_dataframe(idx,col,KeyColumnMapp,df_source,df_Table).copy()
        # col_str=col+'_str'
        # data_col=KeyColumnMapp[col_str]
        # pool_med=col.split('_')[-1]
        # pool_med_str=pool_med+'_str'
        # # df_Table.loc[idx,col]=str(df_source.loc[idx,data_col])
        # p_val=df_source.loc[idx,KeyColumnMapp[pool_med_str + 'p'] ]
        # suffix=Addsuffix(p_val)
        # df_Table.loc[idx,col] = str(df_source.loc[idx,data_col]) + suffix
        # # print(df_Table['FCR',''])
        
    bookeepDF_dict[count]=df_Table
    count+=1
        # if col == 'ASDMil_mean' and idx == 'between_covariance(A:,i:,u:)':
        #     if '*' in df_Table.loc['between_covariance(A:,i:,u:)','ASDMil_mean']:
        #         aaa=ccc
        # if 'between_covariance(A:,i:,u:)' in df_Table.index:
        #     if type(df_Table.loc['between_covariance(A:,i:,u:)','ASDMil_mean']) != str:
        #         df_Table.loc['between_covariance(A:,i:,u:)','ASDMil_mean'] = str(df_Table.loc['between_covariance(A:,i:,u:)','ASDMil_mean'])
        #     if '*' in df_Table.loc['between_covariance(A:,i:,u:)','ASDMil_mean']:
        #         aaa=ccc
    feat_grp=Get_FeatureGrps(idx,Feature_collect)
    
    for corre_col in ['pearson','spearman']:
        r_val_str=corre_col + 'r'
        p_val_str=corre_col + '_p'
        
        df_source=Correlation_dict[feat_grp]
        p_val=df_source.loc[idx,p_val_str]
        suffix=Addsuffix(p_val)
        df_Table.loc[idx,corre_col] = str(np.round(df_source.loc[idx,r_val_str],3)) + suffix

assert 0==1
# =============================================================================
'''

    Plot area: box plot with significant level
    
    The experiment2 in TBME2021: Discriminative analysis of dyadic vowel space dynamics
    
    

'''

from statannot import add_stat_annotation
y_axis_name_map={}
y_axis_name_map['Divergence[pillai_lin_norm(A:,i:,u:)]']='$Div(Norm(Pillai))$'
y_axis_name_map['Divergence[pillai_lin_norm(A:,i:,u:)]_var_p1']='$Inc(Norm(Pillai))_{inv}$'
y_axis_name_map['Divergence[pillai_lin_norm(A:,i:,u:)]_var_p2']='$Inc(Norm(Pillai))_{part}$'
y_axis_name_map['Divergence[within_covariance_norm(A:,i:,u:)]']='$Div(Norm(WCC))$'
y_axis_name_map['Divergence[within_covariance_norm(A:,i:,u:)]_var_p1']='$Inc(Norm(WCC))_{inv}$'
y_axis_name_map['Divergence[within_covariance_norm(A:,i:,u:)]_var_p2']='$Inc(Norm(WCC))_{part}$'
##############################################################################


# column_selected=[
#   'Divergence[pillai_lin_norm(A:,i:,u:)]',
#   'Divergence[pillai_lin_norm(A:,i:,u:)]_var_p1',
#   'Divergence[pillai_lin_norm(A:,i:,u:)]_var_p2',
#   'Divergence[within_covariance_norm(A:,i:,u:)]',
#   'Divergence[within_covariance_norm(A:,i:,u:)]_var_p1',
#   'Divergence[within_covariance_norm(A:,i:,u:)]_var_p2'
#  ]

column_selected=[
  
  # 'Divergence[pillai_lin_norm(A:,i:,u:)]_var_p1',
  # 'Divergence[pillai_lin_norm(A:,i:,u:)]_var_p2',
  'Divergence[within_covariance_norm(A:,i:,u:)]_var_p1',
  'Divergence[within_covariance_norm(A:,i:,u:)]',
  'Divergence[pillai_lin_norm(A:,i:,u:)]',
  # 'Divergence[within_covariance_norm(A:,i:,u:)]_var_p2'
 ]


df_ASDTDpairs_total=df_formant_statistic_agesexmatch_ASDSevere.copy()
df_ASDTDpairs_total['ASDgrpTD']='$ASD_{Severe}$'
dftmp=df_formant_statistic_agesexmatch_ASDMild.copy()
dftmp['ASDgrpTD']='$ASD_{mild}$'
df_ASDTDpairs_total=df_ASDTDpairs_total.append(dftmp)
dftmp2=df_formant_statistic_TD_normal.copy()
dftmp2['ASDgrpTD']='TD'
df_ASDTDpairs_total=df_ASDTDpairs_total.append(dftmp2)

df_ASDTDpairs_total = df_ASDTDpairs_total.rename(y_axis_name_map,axis='columns')

plt.rcParams["figure.figsize"] = (9,5)

# base_num='16'
base_num='13'
for i, c_s in enumerate(column_selected):
    plt_claimnum=base_num + str(i+1)
    plt.subplot(int(plt_claimnum))
    
    
    
    sns.set(style="whitegrid")
    df = df_ASDTDpairs_total
    
    x = "ASDgrpTD"
    # y = "Divergence[within_covariance_norm(A:,i:,u:)]"
    y = y_axis_name_map[c_s]
    order = ['$ASD_{Severe}$', '$ASD_{mild}$', 'TD']
    ax = sns.boxplot(data=df, x=x, y=y, order=order)
    ax, test_result_list = add_stat_annotation(ax, data=df, x=x, y=y, order=order,
                        box_pairs=[("$ASD_{Severe}$", "$ASD_{mild}$"), ("$ASD_{Severe}$", "TD"), ("$ASD_{mild}$", "TD")],
                        test='Mann-Whitney', text_format='star', loc='outside', comparisons_correction=None, verbose=2)
    # ax.set_ylabel('')    
    ax.set_xlabel('')
plt.tight_layout()

# =============================================================================
'''

    Plot area: box plot of Norm(WCC) time series
    
    Discussion in TBME2021: Discriminative analysis of dyadic vowel space dynamics
    
    
'''

feat='within_covariance_norm(A:,i:,u:)'
peoplesets=feat_getter.MildASD_age_sex_match

people2num={p:i for i,p in enumerate(peoplesets)}
y_axis_name_map['within_covariance_norm(A:,i:,u:)']='$Norm(WCC)$'
RoleMapp={'D':'investigator','K':'participant'}


plt.rcParams["figure.figsize"] = (9,6)
df_data_top=pd.DataFrame([])
for people in peoplesets:
    for role in df_person_segment_feature_dict_ASD_formant[people].keys():
        df_data=df_person_segment_feature_dict_ASD_formant[people][role][[feat]]
        
        df_data['people']=people2num[people]
        df_data['role']=RoleMapp[role]
        if people in peoplesets:
            df_data_top=pd.concat([df_data_top,df_data],axis=0)
df_data_top.columns=[y_axis_name_map[feat],'people','role'] #Rename feature to the term in TBME paper
plt.ylim(0,1.1e10)

## Generate plot
# sns.boxplot(data=df_data_top, x=y_axis_name_map[feat], y='people', hue="role", orient="h")
sns.boxplot(data=df_data_top, x='people', y=y_axis_name_map[feat], hue="role", orient="v")

## Generate table
df_avg=pd.DataFrame([])
# df_avg=pd.DataFrame([],columns=[feat])

for role_i in ['investigator','participant']:
    for people in range(len(people2num)):
        df_selected_averaged_data=df_data_top[(df_data_top['role'] == role_i) & (df_data_top['people'] == people)].mean()
        df_avg.loc[people,role_i]=df_selected_averaged_data.iloc[0]



# df_data_top=pd.DataFrame([])
# for people in peoplesets:
#     for role in df_person_segment_feature_dict_TD_formant[people].keys():
#         df_data=df_person_segment_feature_dict_TD_formant[people][role][[feat]]
#         df_data['people']=people
#         df_data['role']=role
#         if people in peoplesets:
#             df_data_top=pd.concat([df_data_top,df_data],axis=0)
# sns.boxplot(data=df_data_top, x='people', y=feat, hue="role")

# =============================================================================
'''

    Manual area

'''
# =============================================================================



ASDSevere_data_lst=['df_formant_statistic_agematchASDSevere','df_formant_statistic_TD_normal']
ASDSevereMild_data_lst=['df_formant_statistic_agematchASDSeverenMild','df_formant_statistic_TD_normal']
ASDagematch_data_lst=['df_formant_statistic_ASDagematch','df_formant_statistic_TD_normal']
ASDagesexmatchSevere_data_lst=['df_formant_statistic_agesexmatch_ASDSevere','df_formant_statistic_TD_normal']
ASDagesexmatchMild_data_lst=['df_formant_statistic_agematchASDmild','df_formant_statistic_TD_normal']
# Inspect Record_dict
# Select features that Severe vs TD is stgnificant and see if the mild ASD has the same polarity with severe ASD
def Digin_Record_dict(Record_dict, Target_data_lst, Cmp_data_lst, Regess_result_dict=[]):
    if len(Regess_result_dict) >0:
        df_Feature_check_target=pd.DataFrame([],columns=['target', 'compare', 'corr_ASD'])
    else:
        df_Feature_check_target=pd.DataFrame([],columns=['target', 'compare'])
    for idx in Record_dict[' vs '.join(Target_data_lst)].index:
        df_target=Record_dict[' vs '.join(Target_data_lst)].loc[idx]
        df_auxilrary=All_cmp_dict[' vs '.join(Cmp_data_lst)].loc[idx]
        
        if not np.isnan(df_target['UTestp']):
            print(df_auxilrary['UTest' + ' - '.join(Cmp_data_lst)], df_target['UTest'+' - '.join(Target_data_lst)])
            if len(Regess_result_dict) >0:
                df_Feature_check_target.loc[idx]=[df_auxilrary['UTest' + ' - '.join(Cmp_data_lst)], df_target['UTest'+' - '.join(Target_data_lst)], Regess_result_dict['spearmanr']['__ASD'].loc[idx]]
            else:
                df_Feature_check_target.loc[idx]=[df_auxilrary['UTest' + ' - '.join(Cmp_data_lst)], df_target['UTest'+' - '.join(Target_data_lst)]]
    return df_Feature_check_target

df_Feature_check_severe = Digin_Record_dict(Record_dict, Target_data_lst = ASDSevere_data_lst, Cmp_data_lst = ASDagesexmatchSevere_data_lst).sort_index()
# df_Feature_check_mild = Digin_Record_dict(Record_dict, Target_data_lst = ASDagesexmatchSevere_data_lst, Cmp_data_lst = ASDSevere_data_lst).sort_index()
df_Feature_check_severemild = Digin_Record_dict(Record_dict, Target_data_lst = ASDSevereMild_data_lst, Cmp_data_lst = ASDSevere_data_lst).sort_index()
df_Feature_check_agematch = Digin_Record_dict(Record_dict, Target_data_lst = ASDagematch_data_lst, Cmp_data_lst = ASDSevere_data_lst).sort_index()
df_Feature_check_agesexmatchSevere = Digin_Record_dict(Record_dict, Target_data_lst = ASDagesexmatchSevere_data_lst, Cmp_data_lst = ASDagesexmatchSevere_data_lst).sort_index()
df_Feature_check_agesexmatchMild = Digin_Record_dict(Record_dict, Target_data_lst = ASDagesexmatchMild_data_lst, Cmp_data_lst = ASDagesexmatchSevere_data_lst).sort_index()
with open(args.ResultsOutpath + 'severe{}.txt'.format(args.epoch), 'w') as f_severe,\
    open(args.ResultsOutpath + 'mild{}.txt'.format(args.epoch), 'w') as f_mild,\
    open(args.ResultsOutpath + 'severemild{}.txt'.format(args.epoch), 'w') as f_severemild,\
    open(args.ResultsOutpath + 'agematch{}.txt'.format(args.epoch), 'w') as f_agematch,\
    open(args.ResultsOutpath + 'agesexmatch{}.txt'.format(args.epoch), 'w') as f_agesexmatch:
    print('df_Feature_check_severe:', df_Feature_check_severe, file=f_severe)
    print('df_Feature_check_mild:', df_Feature_check_mild, file=f_mild)
    print('df_Feature_check_severemild:', df_Feature_check_severemild, file=f_severemild)
    print('df_Feature_check_agematch:', df_Feature_check_agematch, file=f_agematch)
    print('df_Feature_check_agesexmatch:', df_Feature_check_agesexmatchSevere, file=f_agesexmatch)

print(df_Feature_check_agesexmatchSevere)



basic_columns=['u_num', 'a_num', 'i_num', 'ADOS_C', 'dia_num', 'sex', 'age', 'Module','timeSeries_len']
insp_column=basic_columns + ['Divergence[within_covariance(A:,i:,u:)]_var_p2']


Aa_ASD=df_formant_statistic_agesexmatch_ASDSevere[insp_column]
Aa_TD=df_formant_statistic_TD_normal[insp_column]

# Inspect_columns=[('within_covariance_norm(A:,i:,u:)','Divergence[within_covariance_norm(A:,i:,u:)]')]
Inspect_columns=[('within_covariance_norm(A:,i:,u:)',['Divergence[within_covariance_norm(A:,i:,u:)]_var_p1',\
                                                      'Divergence[within_covariance_norm(A:,i:,u:)]_var_p2',\
                                                      'Divergence[within_covariance_norm(A:,i:,u:)]'])]
# Inspect_columns=[('sam_wilks_lin_norm(A:,i:,u:)',['Divergence[sam_wilks_lin_norm(A:,i:,u:)]_var_p1',\
#                                                       'Divergence[sam_wilks_lin_norm(A:,i:,u:)]_var_p2',\
#                                                       'Divergence[sam_wilks_lin_norm(A:,i:,u:)]'])]
# Inspect_columns=[('pillai_lin_norm(A:,i:,u:)',['Divergence[pillai_lin_norm(A:,i:,u:)]_var_p1',\
#                                                       'Divergence[pillai_lin_norm(A:,i:,u:)]_var_p2',\
#                                                       'Divergence[pillai_lin_norm(A:,i:,u:)]'])]

score_df_columns=[]
def Plot_Timeseries(df_formant_statistic, df_person_segment_feature,Inspect_columns, score_df, feat_type='formant', showDist=False):
    
    # df_formant_statistic, df_person_segment_feature,Inspect_columns, score_df=\
    #     df_formant_statistic_agesexmatchASDMild_sorted,df_person_segment_feature_dict_ASD,Inspect_columns,df_formant_statistic_all
    plt.rcParams["figure.figsize"] = (6,6)
    fig=plt.figure()
    for people in df_formant_statistic.index:
        # df_person_segment_feature_dict_TD
        for cols, score_cols in Inspect_columns: 
            if feat_type == 'formant':
                Dict_df_ASD=df_person_segment_feature[people]
            else:
                pattern=r"\(.*\)"
                bag=re.findall(pattern, cols)
                assert len(bag) == 1
                phone_str=bag[0]
                phone=phone_str[phone_str.find('(')+1:phone_str.find(')')]
                Dict_df_ASD=df_person_segment_feature[phone]['segment'][people]
            df_ASD_d=Dict_df_ASD['D'][cols]
            df_ASD_k=Dict_df_ASD['K'][cols]
            df_ASD_d.name="Investigator"
            df_ASD_k.name="Participant"
            df_d_k_abs = (df_ASD_d - df_ASD_k).abs()
            df_d_k_abs.name="doc - kid"
            if showDist:
                df_ASD_dk=pd.concat([df_ASD_d,df_ASD_k,df_d_k_abs],axis=1)
            else:
                df_ASD_dk=pd.concat([df_ASD_d,df_ASD_k],axis=1)
            sns.lineplot(data=df_ASD_dk,linewidth = 6)
            title='{0}\n{1}'.format('ASD ' + people, 'Basefeat: ' + cols)
            
            plt.title( title )
            plt.grid(False)
            plt.legend(fontsize='x-large', title_fontsize='40')
            
            score=score_df.loc[people,score_cols]
            info_arr=["{}: {}".format(idx,v) for idx, v in zip(score.index.values,np.round(score.values,3))]
            addtext='\n'.join(info_arr)
            x0, xmax = plt.xlim()
            y0, ymax = plt.ylim()
            data_width = xmax - x0
            data_height = ymax - y0
            text(x0 + data_width * 0.4, -data_height * 0.2, addtext, ha='center', va='center')
            plt.show()
            plt.clf()

phonation_pattern=['shimmer','jitter','stdevF0']

for Insp_col in Inspect_columns:
    Inp_c=Insp_col[0]
    def checkphonation_feat(Inp_c):
        cond=False
        for pho_patt in phonation_pattern:
            if pho_patt in Inp_c.lower():
                cond = cond or True
        return cond
    if checkphonation_feat(Inp_c):
        feat_type='phonation'
        df_person_segment_feature_dict_ASD=df_POI_person_segment_feature_dict_ASD_phonation
        df_person_segment_feature_dict_TD=df_POI_person_segment_feature_dict_TD_phonation
    else:
        feat_type='formant'
        df_person_segment_feature_dict_ASD=df_person_segment_feature_dict_ASD_formant
        df_person_segment_feature_dict_TD=df_person_segment_feature_dict_TD_formant
        
    



    df_formant_statistic_agesexmatchASDMild_sorted=df_formant_statistic_agesexmatch_ASDMild.sort_values(Inspect_columns[0][-1])
    Plot_Timeseries(df_formant_statistic_agesexmatchASDMild_sorted,df_person_segment_feature_dict_ASD,Inspect_columns,df_formant_statistic_all,\
                    feat_type=feat_type)
    


    df_formant_statistic_agesexmatchASDSevere_sorted=df_formant_statistic_agesexmatch_ASDSevere.sort_values(Inspect_columns[0][-1])
    Plot_Timeseries(df_formant_statistic_agesexmatchASDSevere_sorted,df_person_segment_feature_dict_ASD,Inspect_columns,df_formant_statistic_all,\
                    feat_type=feat_type)
    
    df_formant_statistic_TD_normal_sorted=df_formant_statistic_TD_normal.sort_values(Inspect_columns[0][-1])
    Plot_Timeseries(df_formant_statistic_TD_normal_sorted,df_person_segment_feature_dict_TD,Inspect_columns,df_formant_statistic_all,\
                    feat_type=feat_type)


# Debug 
# =============

df_Session_formant_statistic_ASD=Eval_med.Calculate_correlation(label_choose_lst,features_filt,N,columns,feature_type=feat_type)

# =============

# =============================================================================    

'''

    Regression area

'''



''' Single Variable correlation ''' 
# inspect_cols=['BW_sam_wilks(A:,i:,u:)', 'BW_pillai(A:,i:,u:)',
#        'BW_hotelling(A:,i:,u:)', 'BW_roys_root(A:,i:,u:)',
#        'linear_discriminant_covariance(A:,i:,u:)']
inspect_cols=Parameters
Mapping_dict={'girl':2,'boy':1,'M3':3,'M4':4,'None':-1, 'ASD':1,'TD':2}
# =============================================================================
Regess_result_dict=Dict()

gender_set=['None','boy','girl']
module_set=['None','M3','M4']
ASDTD_set=['None','ASD','TD']
Effect_comb=[gender_set,module_set,ASDTD_set]
comcinations=list(itertools.product(*Effect_comb))

for correlation_type in ['spearmanr','pearsonr']:
    df_collector_top=pd.DataFrame()
    for sex_str, module_str, ASDTD_str in comcinations:
        sex=Mapping_dict[sex_str]
        module=Mapping_dict[module_str]
        ASDTD=Mapping_dict[ASDTD_str]
        df_correlation=Eval_med.Calculate_correlation(label_choose_lst,df_formant_statistic_all,N,Parameters,\
                                                      constrain_sex=sex, constrain_module=module, constrain_ASDTD=ASDTD,\
                                                      feature_type='Syncrony_formant')
        if len(df_correlation)>0:
            df_correlation=df_correlation.loc[inspect_cols].round(3)
            # if (df_correlation['spearman_p']<0.05).any() :
            #     print(sex_str, module_str, ASDTD_str )
            
            
            df_collector=pd.DataFrame([],index=df_correlation.index)
            Namer=Dict()
            for col in df_correlation:
                Namer[col]=df_correlation[col].astype(str).values
            if correlation_type == 'pearsonr':
                r_value_str=Namer[list(Namer.keys())[0]].astype(str)
                p_value_str=Namer[list(Namer.keys())[1]].astype(str)
            elif correlation_type == 'spearmanr':
                r_value_str=Namer[list(Namer.keys())[0+2]].astype(str)
                p_value_str=Namer[list(Namer.keys())[1+2]].astype(str)
            p_value_str=np.core.defchararray.add(["("]*len(r_value_str), p_value_str)
            p_value_str=np.core.defchararray.add(p_value_str,[")"]*len(r_value_str))
            corr_result_str=np.core.defchararray.add(r_value_str,p_value_str)
            df_collector['{sex}_{module}_{ASDTD}'.format(sex=sex_str,module=module_str,ASDTD=ASDTD_str).replace('None','')]\
                                                    =corr_result_str
            df_collector_top=pd.concat([df_collector_top,df_collector],axis=1)
    Regess_result_dict[correlation_type]=df_collector_top

Covariances_columns=[
'total_covariance_norm(A:,i:,u:)',
'sam_wilks_lin_norm(A:,i:,u:)',
'within_covariance_norm(A:,i:,u:)',
'between_covariance_norm(A:,i:,u:)',
'total_covariance(A:,i:,u:)',
'sam_wilks_lin(A:,i:,u:)',
'within_covariance(A:,i:,u:)',
'between_covariance(A:,i:,u:)',
'total_variance(A:,i:,u:)',
'hotelling_lin(A:,i:,u:)',
'within_variance(A:,i:,u:)',
'between_variance(A:,i:,u:)',
'VSA1',
'FCR',
'u_num+i_num+a_num',
]
BasicInfoColumns=['totalword']
# Calculate correlations for each sectional features
Aaadf_pearsonr_table_ASD_dict=Eval_med.Calculate_correlation(label_choose_lst,df_formant_statistic_77,\
                                                        N,BasicInfoColumns,constrain_sex=-1, constrain_module=-1,\
                                                        feature_type='')

Aaadf_pearsonr_table_ASD_dict=Eval_med.Calculate_correlation(label_choose_lst,df_formant_statistic_77,\
                                                        N,Covariances_columns,constrain_sex=-1, constrain_module=-1,\
                                                        feature_type='')
# =============================================================================



''' OLS report '''
'''
    Operation steps:
        Unfortunately stepwise remove regression requires heavily on manual operation
        we then set the standard operation rules 
        
        1. put all your variables, and run regression
        2. bookeep the previous formula and eliminate the most unsignificant variables
        3. back to 1. and repeat untill all variables are significant
        
    
'''

def Regression_Preprocess_setp(DV_str, IV_lst ,df,punc=":,()", categorical_cols=['sex']):
    df_remaned=df.rename(columns=lambda s: re.sub(u"[{}]+".format(punc),"",s))
    DV=re.sub(u"[{}]+".format(punc),"",DV_str)
    IV_lst_renamed=[]
    for i,IV_str in enumerate(IV_lst):
        if IV_str in categorical_cols:
            IV_lst_renamed.append('C({})'.format(IV_str))
        else:
            IV_lst_renamed.append(re.sub(u"[{}]+".format(punc),"",IV_str))
    IV_lst_renamed_str = ' + '.join(IV_lst_renamed)
    formula = '{DV} ~ {IV}'.format(DV=DV,IV=IV_lst_renamed_str)
    
    return df_remaned, formula

Formula_bookeep_dict=Dict()


# IV_lst=['between_covariance(A:,i:,u:)']
# DV_str='ADOS'

# df_remaned, formula = Regression_Preprocess_setp(DV_str, IV_lst, df_formant_statistic_all)

# significant_lvl=0.05
# max_p=1
# count=1
# # for count in range(10):
# while max_p > significant_lvl:
#     res = smf.ols(formula=formula, data=df_remaned).fit()    
    
#     max_p=max(res.pvalues  )
#     if max_p < significant_lvl:
#         print(res.summary())
#         break
#     remove_indexs=res.pvalues[res.pvalues==max_p].index
#     print('remove Indexes ',remove_indexs,' with pvalue', max_p)
    
#     Formula_bookeep_dict['step_'+str(count)].formula=formula
#     Formula_bookeep_dict['step_'+str(count)].removed_indexes=remove_indexs.values.astype(str).tolist()
#     Formula_bookeep_dict['step_'+str(count)].removedP=max_p
#     Formula_bookeep_dict['step_'+str(count)].rsquared_adj=res.rsquared_adj
    
#     header_str=formula[:re.search('~ ',formula).end()]
#     formula_lst=formula[re.search('~ ',formula).end():].split(" + ")
#     for rm_str in Formula_bookeep_dict['step_'+str(count)].removed_indexes:
#         rm_str=re.sub("[\[].*?[\]]", "", rm_str)
#         formula_lst.remove(rm_str)
#     formula = header_str + ' + '.join(formula_lst)
#     count+=1

# =============================================================================
''' foward selection '''
# =============================================================================

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

punc=":,()"
df_formant_statistic_all_remaned=df_formant_statistic_77.rename(columns=lambda s: re.sub(u"[{}]+".format(punc),"",s))
df_formant_statistic_all_remaned['Module']=df_formant_statistic_all_remaned['Module'].astype(str)
df_formant_statistic_all_remaned.loc[df_formant_statistic_all_remaned['Module']==3,'Module']=1
df_formant_statistic_all_remaned.loc[df_formant_statistic_all_remaned['Module']==4,'Module']=2


formula='ADOS_C ~ '
formula+= ' between_variance_norm(A:,i:,u:) '
# formula+= ' + BV(A:,i:,u:)_l2'

formula=re.sub(u"[{}]+".format(punc),"",formula)

# formula+=' + totalword '

# formula+=' + C(sex) '
# formula+=' + age '
# formula+=' + C(ASDTD) '
# formula+=' + C(Module) * C(sex)'
# formula+=' + C(Module) * C(ASDTD)'
# formula+=' + C(sex) * C(ASDTD)'
# formula+=' + C(sex) + C(Module) * C(ASDTD)'
res = smf.ols(formula=formula, data=df_formant_statistic_all_remaned).fit()
print(res.summary())




formula='between_variance(A:,i:,u:) ~'
formula=re.sub(u"[{}]+".format(punc),"",formula)
# formula+=' + ADOS '
# formula+=' + C(sex) '
# formula+=' + C(Module) '
formula+=' + age '
# formula+=' ADOS * age '
# formula+=' + C(ASDTD) '
# formula+=' + C(Module) * C(sex)'
# formula+=' + C(Module) * C(ASDTD)'
# formula+=' + C(sex) * C(ASDTD)'
# formula+=' + C(sex) + C(Module) * C(ASDTD)'
res_rev = smf.ols(formula=formula, data=df_formant_statistic_all_remaned).fit()
print(res_rev.summary())



''' small, stepwise forward regression '''

comb_lsts=[' + C(sex) ', ' + age ', ]
df_result=pd.DataFrame([],columns=['R2_adj'])
for i in range(0,len(comb_lsts)+1):
    combs = combinations(comb_lsts, i)
    for additional_str in combs:
        formula='between_variance(A:,i:,u:) ~ ADOS_C'
        formula=re.sub(u"[{}]+".format(punc),"",formula)
        
        formula+= ''.join(additional_str)
        res = smf.ols(formula=formula, data=df_formant_statistic_all_remaned).fit()
        variables = formula[re.search('~ ',formula).end():]
        print(variables, res.rsquared_adj)
        df_result.loc[variables]=res.rsquared_adj


# =============================================================================
''' ANOVA area '''
# =============================================================================
comb=[['df_formant_statistic_TD','df_formant_statistic_77'],]
# IV_list=['sex','Module','ASDTD']
IV_list=['total_variance(A:,i:,u:)',
'total_covariance(A:,i:,u:)',
'between_variance(A:,i:,u:)',
'between_variance_f2(A:,i:,u:)',
'within_variance(A:,i:,u:)',
'within_variance_f2(A:,i:,u:)',
'baseline_numeffect'
]
# comcinations=list(itertools.product(*Effect_comb))
df_formant_statistic=df_formant_statistic_77
ways=1
combination = combinations(IV_list, ways)
for comb in combination:
    IV_lst = list(comb)
    # DV_str='between_variance(A:,i:,u:)'
    DV_str='ADOS_cate'
    df_remaned, formula = Regression_Preprocess_setp(DV_str, IV_lst, df_formant_statistic)
    df_remaned['baseline_numeffect']=df_remaned['u_num+i_num+a_num']
    punc=":,()"
    model = ols(formula, data=df_remaned).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    print(anova)

















# ''' Code backup '''
df_formant_statistic_77_mean=df_formant_statistic_77.mean()
# inspect_cols=['u_num', 'a_num', 'i_num','MSB_f1(A:,i:,u:)', 'MSB_f2(A:,i:,u:)','ADOS']

df_formant_statistic_77_inspect=df_formant_statistic_77[inspect_cols]
df_formant_statistic_TD_inspect=df_formant_statistic_TD[inspect_cols]

df_formant_statistic_77_Notautism_mean=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=0).mean()
df_formant_statistic_77_ASD_mean=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=1).mean()
df_formant_statistic_77_autism_mean=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=2).mean()
df_formant_statistic_TD_mean=df_formant_statistic_TD.mean()

filter_boy=df_formant_statistic_77['sex']==1
filter_girl=df_formant_statistic_77['sex']==2
filter_M3=df_formant_statistic_77['Module']==3
filter_M4=df_formant_statistic_77['Module']==4
# print(df_formant_statistic_77_inspect[filter_boy].mean())
# print(df_formant_statistic_77_inspect[filter_girl].mean())
print(df_formant_statistic_77['VSA1'][filter_M3 & filter_boy].mean())
print(df_formant_statistic_77['VSA1'][filter_M4 & filter_boy].mean())

print(df_formant_statistic_77['VSA1'][filter_M3 & filter_girl].mean())
print(df_formant_statistic_77['VSA1'][filter_M4 & filter_girl].mean())

# Top_data_lst=['df_formant_statistic_77','df_formant_statistic_TD']
df_formant_statistic_77_M3boy=df_formant_statistic_77[filter_M3 & filter_boy]
df_formant_statistic_77_M4boy=df_formant_statistic_77[filter_M4 & filter_boy]
df_formant_statistic_77_M3girl=df_formant_statistic_77[filter_M3 & filter_girl]
df_formant_statistic_77_M4girl=df_formant_statistic_77[filter_M4 & filter_girl]
df_formant_statistic_77_M3=df_formant_statistic_77[filter_M3 ]
df_formant_statistic_77_M4=df_formant_statistic_77[filter_M4 ]

df_formant_statistic_77_Notautism=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=0)
df_formant_statistic_77_ASD=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=1)
df_formant_statistic_77_autism=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=2)
# Top_data_lst=["df_formant_statistic_77_M3boy",\
#               "df_formant_statistic_77_M4boy"]
# Top_data_lst=["df_formant_statistic_77_M3girl",\
#               "df_formant_statistic_77_M4girl"]
# Top_data_lst=["df_formant_statistic_77_Notautism",\
#               "df_formant_statistic_77_ASD"]
Top_data_lst=["df_formant_statistic_77_autism",\
              "df_formant_statistic_77_ASD"]
# Top_data_lst=["df_formant_statistic_77_autism",\
#               "df_formant_statistic_77_Notautism"]
    

    

filter_boy=df_formant_statistic_TD['sex']==1
filter_girl=df_formant_statistic_TD['sex']==2
filter_M3=df_formant_statistic_TD['Module']==3
filter_M4=df_formant_statistic_TD['Module']==4
print(df_formant_statistic_TD_inspect[filter_boy].mean())
print(df_formant_statistic_TD_inspect[filter_girl].mean())

print(df_formant_statistic_TD['VSA1'][filter_M3 & filter_boy].mean())
print(df_formant_statistic_TD['VSA1'][filter_M4 & filter_boy].mean())

print(df_formant_statistic_TD['VSA1'][filter_M3 & filter_girl].mean())
print(df_formant_statistic_TD['VSA1'][filter_M4 & filter_girl].mean())

df_formant_statistic_TD_M3boy=df_formant_statistic_TD[filter_M3 & filter_boy]
df_formant_statistic_TD_M4boy=df_formant_statistic_TD[filter_M4 & filter_boy]
df_formant_statistic_TD_M3girl=df_formant_statistic_TD[filter_M3 & filter_girl]
df_formant_statistic_TD_M4girl=df_formant_statistic_TD[filter_M4 & filter_girl]
df_formant_statistic_TD_M3=df_formant_statistic_TD[filter_M3 ]
df_formant_statistic_TD_M4=df_formant_statistic_TD[filter_M4 ]
# Top_data_lst=["df_formant_statistic_TD_M3boy",\
#               "df_formant_statistic_TD_M4boy"]
# Top_data_lst=["df_formant_statistic_TD_M3girl",\
#               "df_formant_statistic_TD_M4girl"]
Top_data_lst=["df_formant_statistic_TD_M3",\
              "df_formant_statistic_TD_M4"]
# Top_data_lst=['M3','M4']
    
df_ttest_result=pd.DataFrame([],columns=['doc-kid','p-val'])
for role_1,role_2  in comb:
    for parameter in Parameters:
        # test=stats.ttest_ind(vars()[role_1][parameter], vars()[role_2][parameter])
        test=stats.mannwhitneyu(vars()[role_1][parameter], vars()[role_2][parameter])
        # print(parameter, '{0} vs {1}'.format(role_1,role_2),test)
        # print(role_1+':',vars()[role_1][parameter].mean(),role_2+':',vars()[role_2][parameter].mean())
        df_ttest_result.loc[parameter,'doc-kid'] = vars()[role_1][parameter].mean() - vars()[role_2][parameter].mean()
        df_ttest_result.loc[parameter,'p-val'] = test[1]
        
        if test[1] < 0.05:
            print(parameter, "p < 0.05")



inspect_col=['between_variance(A:,i:,u:)', 'between_covariance(A:,i:,u:)',\
             'between_variance_norm(A:,i:,u:)', 'between_covariance_norm(A:,i:,u:)',\
             'within_variance(A:,i:,u:)', 'within_covariance(A:,i:,u:)',\
             'linear_discriminant_covariance(A:,i:,u:)', 'roys_root_lin']

df_Session_formant_statistic_ASD['u_num+i_num+a_num']=df_Session_formant_statistic_ASD['u_num'] +\
                                                df_Session_formant_statistic_ASD['i_num'] +\
                                                df_Session_formant_statistic_ASD['a_num']
df_Session_formant_statistic_TD['u_num+i_num+a_num']=df_Session_formant_statistic_TD['u_num'] +\
                                                df_Session_formant_statistic_TD['i_num'] +\
                                                df_Session_formant_statistic_TD['a_num']
# df_Session_formant_statistic_TD=df_Session_formant_statistic_TD.drop(columns=['dia_num'])