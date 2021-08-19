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
        self.SevereASD_age_sex_match=['2015_12_06_01_045_1',
            '2015_12_27_01_058fu',
            '2016_03_05_01_079fu',
            '2016_10_21_01_202_1',
            '2016_10_22_01_161_fu',
            '2017_01_21_01_242_1',
            '2017_05_21_01_292_1',
            '2017_07_25_01_172',
            '2017_08_07_01_227',
            '2017_08_15_01_413']
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
    def read_dfPersonSegmentFeatureDict(self,feat='df_person_segment_feature_dict'):
        df_person_segment_feature_dict_ASD77_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/{name}_{role}.pkl'.format(name=feat,role='ASD_DOCKID')
        df_person_segment_feature_dict_TD_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/{name}_{role}.pkl'.format(name=feat,role='TD_DOCKID')
        self.df_person_segment_feature_dict_ASD=pickle.load(open(df_person_segment_feature_dict_ASD77_path,'rb'))
        self.df_person_segment_feature_dict_TD=pickle.load(open(df_person_segment_feature_dict_TD_path,'rb'))
        return self.df_person_segment_feature_dict_ASD, self.df_person_segment_feature_dict_TD

# =============================================================================
'''  T-Test ASD vs TD''' 
df_formant_statistic_77=pd.DataFrame()
df_formant_statistic_TD=pd.DataFrame()
feat_getter=Feature_Getter(dfFormantStatisticpath,\
                           Stat_med_str_VSA=Stat_med_str_VSA,\
                           Inspect_features=Inspect_features)
df_Session_formant_statistic_ASD, df_Session_formant_statistic_TD=feat_getter.read_Sessionfeature('Formant_AUI_tVSAFCRFvals')
df_Session_phonation_statistic_ASD, df_Session_phonation_statistic_TD=feat_getter.read_Sessionfeature('Phonation_meanvars')
df_Session_formant_statistic_TD['dia_num']=100


df_Syncrony_formant_statistic_ASD, df_Syncrony_formant_statistic_TD=feat_getter.read_Syncronyfeature('Syncrony_measure_of_variance')
df_Syncrony_phonation_statistic_ASD, df_Syncrony_phonation_statistic_TD=feat_getter.read_Syncronyfeature('Syncrony_measure_of_variance_phonation')
df_dur_strlen_speed_ASD, df_dur_strlen_speed_TD=feat_getter.read_durStrlenSpeedfeature('df_dur_strlen_speed')
df_person_segment_feature_dict_ASD, df_person_segment_feature_dict_TD=feat_getter.read_dfPersonSegmentFeatureDict('df_person_segment_feature_dict')

df_formant_statistic_77=pd.concat([df_formant_statistic_77,df_Session_formant_statistic_ASD],axis=1)
df_formant_statistic_TD=pd.concat([df_formant_statistic_TD,df_Session_formant_statistic_TD],axis=1)

df_formant_statistic_77=pd.concat([df_formant_statistic_77,df_Session_phonation_statistic_ASD],axis=1)
df_formant_statistic_TD=pd.concat([df_formant_statistic_TD,df_Session_phonation_statistic_TD],axis=1)

df_formant_statistic_77=pd.concat([df_formant_statistic_77,df_Syncrony_formant_statistic_ASD],axis=1)
df_formant_statistic_TD=pd.concat([df_formant_statistic_TD,df_Syncrony_formant_statistic_TD],axis=1)

df_formant_statistic_77=pd.concat([df_formant_statistic_77,df_Syncrony_phonation_statistic_ASD],axis=1) #Contains nan
df_formant_statistic_TD=pd.concat([df_formant_statistic_TD,df_Syncrony_phonation_statistic_TD],axis=1)

df_formant_statistic_77=df_formant_statistic_77.loc[:,~df_formant_statistic_77.columns.duplicated()]
df_formant_statistic_TD=df_formant_statistic_TD.loc[:,~df_formant_statistic_TD.columns.duplicated()]

SevereASD_age_sex_match=feat_getter.SevereASD_age_sex_match


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
comb=[['df_formant_statistic_TD','df_formant_statistic_77'],]
base_variance_features=['sam_wilks_lin(A:,i:,u:)',
'between_covariance_norm(A:,i:,u:)',
'within_covariance_norm(A:,i:,u:)',
'total_covariance_norm(A:,i:,u:)',
'between_covariance(A:,i:,u:)',
'within_covariance(A:,i:,u:)',
'total_covariance(A:,i:,u:)',
'localJitter_mean(A:,i:,u:)',
'localJitter_mean(u:)',
'localJitter_mean(i:)',
'localJitter_mean(A:)'
]
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
'localShimmer_mean(A:)'
    ]
base_features=base_variance_features + base_phonation_features
Syncrony_measres=['Divergence[{0}]','Divergence[{0}]_split','Divergence[{0}]_var_p1','Divergence[{0}]_var_p2']
def CombineMeasurefeat(Syncrony_measres,base_features):
    lst=[]
    for basefeat in base_features:
        for sync_M in Syncrony_measres:
            lst.append(sync_M.format(basefeat))
    return lst
Measurement_basefeature=CombineMeasurefeat(Syncrony_measres,base_features)
Parameters=Measurement_basefeature + base_features




print(" Simple between group test (t-test, u-test)")
# Parameters=list(df_Syncrony_formant_statistic_ASD.columns) + list(df_Session_formant_statistic_ASD.columns)
# Parameters=list(df_Syncrony_formant_statistic_ASD.columns)
# Parameters=df_Session_formant_statistic_ASD.columns
# =============================================================================
'''

    Plot area

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



TopTop_data_lst=[]
# TopTop_data_lst.append(['df_formant_statistic_77','df_formant_statistic_TD_normal'])
# TopTop_data_lst.append(['df_formant_statistic_agematchASDSevere','df_formant_statistic_TD_normal'])
# TopTop_data_lst.append(['df_formant_statistic_agematchASDmild','df_formant_statistic_TD_normal'])
# TopTop_data_lst.append(['df_formant_statistic_agematchASDSeverenMild','df_formant_statistic_TD_normal'])
# TopTop_data_lst.append(['df_formant_statistic_ASDagematch','df_formant_statistic_TD_normal'])
TopTop_data_lst.append(['df_formant_statistic_agesexmatch_ASDSevere','df_formant_statistic_TD_normal'])
# TopTop_data_lst.append(['df_formant_statistic_77_Notautism','df_formant_statistic_77_Autism'])
# TopTop_data_lst.append(['df_formant_statistic_77_ASD','df_formant_statistic_77_Autism'])
# TopTop_data_lst.append(['df_formant_statistic_77_Notautism','df_formant_statistic_77_ASD'])
# TopTop_data_lst.append(['df_formant_statistic_77_AD','df_formant_statistic_77_AS'])
# TopTop_data_lst.append(['df_formant_statistic_77_AS','df_formant_statistic_77_HFA'])
# TopTop_data_lst.append(['df_formant_statistic_77_AD','df_formant_statistic_77_HFA'])


# TopTop_data_lst.append(['df_formant_statistic_77_M3','df_formant_statistic_TD_M3'])
# TopTop_data_lst.append(['df_formant_statistic_77_M4','df_formant_statistic_TD_M4'])
# TopTop_data_lst.append(['df_formant_statistic_77_boy','df_formant_statistic_TD_boy'])
# TopTop_data_lst.append(['df_formant_statistic_77_girl','df_formant_statistic_TD_girl'])
# TopTop_data_lst.append(['df_formant_statistic_77_M3boy','df_formant_statistic_TD_M3boy'])
# TopTop_data_lst.append(['df_formant_statistic_77_M3girl','df_formant_statistic_TD_M3girl'])
# TopTop_data_lst.append(['df_formant_statistic_77_M4boy','df_formant_statistic_TD_M4boy'])
# TopTop_data_lst.append(['df_formant_statistic_77_M4girl','df_formant_statistic_TD_M4girl'])

# self_specify_cols=['between_covariance(A:,i:,u:)', 'between_variance(A:,i:,u:)']
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

# self_specify_cols=['Divergence[between_covariance_norm(A:,i:,u:)]',\
#                    'Syncrony[between_variance(A:,i:,u:)]',\
#                    'Syncrony[FCR]']
# self_specify_cols=['Divergence[localShimmer_mean(u:)]_var_p2','Divergence[localabsoluteJitter_mean(A:)]_var_p2']
self_specify_cols=[]
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
        # fig, ax = plt.subplots()
        # =============================================================================
        data=[]
        dataname=[]
        for dstr in Top_data_lst:
            dataname.append(dstr)
            data.append(vars()[dstr])
        # =============================================================================
        # for i,d in enumerate(data):
        #     # ax = sns.distplot(d[columns], ax=ax, kde=False)
        #     ax = sns.distplot(d[columns], ax=ax, label=Top_data_lst)
        #     title='{0}'.format('Inspecting feature ' + columns)
        #     plt.title( title )
        # fig.legend(labels=dataname)  
        
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
        # addtext='{0}/({1})'.format(np.round(mean_difference,3),np.round(p_val,3))
        # text(0.9, 0.9, addtext, ha='center', va='center', transform=ax.transAxes)
        # addtextvariable='{0} vs {1}'.format(Top_data_lst[0],Top_data_lst[1])
        # text(0.9, 0.6, addtextvariable, ha='center', va='center', transform=ax.transAxes)
        # =============================================================================
    warnings.simplefilter('always')
    
    
# =============================================================================
''' Manual check '''
pattern='Divergence[localabsoluteJitter_mean\(.*\)]_var_p2'
selected_rows=[]
for insp_c in inspect_cols:
    x=re.search(pattern, insp_c)
    if x != None:
        print(x)

Base_phones=['A:','u:','i:']
Base_phones+=
selected_rows=
All_cmp_dict['df_formant_statistic_agesexmatch_ASDSevere vs df_formant_statistic_TD_normal'].loc[]
aaa=ccc
# =============================================================================
'''

    Manual area

'''
# =============================================================================


ASDSevere_data_lst=['df_formant_statistic_agematchASDSevere','df_formant_statistic_TD_normal']
ASDMild_data_lst=['df_formant_statistic_agematchASDmild','df_formant_statistic_TD_normal']
ASDSevereMild_data_lst=['df_formant_statistic_agematchASDSeverenMild','df_formant_statistic_TD_normal']
ASDagematch_data_lst=['df_formant_statistic_ASDagematch','df_formant_statistic_TD_normal']
ASDagesexmatch_data_lst=['df_formant_statistic_agesexmatch_ASDSevere','df_formant_statistic_TD_normal']
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

df_Feature_check_severe = Digin_Record_dict(Record_dict, Target_data_lst = ASDSevere_data_lst, Cmp_data_lst = ASDMild_data_lst).sort_index()
df_Feature_check_mild = Digin_Record_dict(Record_dict, Target_data_lst = ASDMild_data_lst, Cmp_data_lst = ASDSevere_data_lst).sort_index()
df_Feature_check_severemild = Digin_Record_dict(Record_dict, Target_data_lst = ASDSevereMild_data_lst, Cmp_data_lst = ASDSevere_data_lst).sort_index()
df_Feature_check_agematch = Digin_Record_dict(Record_dict, Target_data_lst = ASDagematch_data_lst, Cmp_data_lst = ASDSevere_data_lst).sort_index()
df_Feature_check_agesexmatch = Digin_Record_dict(Record_dict, Target_data_lst = ASDagesexmatch_data_lst, Cmp_data_lst = ASDSevere_data_lst).sort_index()

with open(args.ResultsOutpath + 'severe{}.txt'.format(args.epoch), 'w') as f_severe,\
    open(args.ResultsOutpath + 'mild{}.txt'.format(args.epoch), 'w') as f_mild,\
    open(args.ResultsOutpath + 'severemild{}.txt'.format(args.epoch), 'w') as f_severemild,\
    open(args.ResultsOutpath + 'agematch{}.txt'.format(args.epoch), 'w') as f_agematch,\
    open(args.ResultsOutpath + 'agesexmatch{}.txt'.format(args.epoch), 'w') as f_agesexmatch:
    print('df_Feature_check_severe:', df_Feature_check_severe, file=f_severe)
    print('df_Feature_check_mild:', df_Feature_check_mild, file=f_mild)
    print('df_Feature_check_severemild:', df_Feature_check_severemild, file=f_severemild)
    print('df_Feature_check_agematch:', df_Feature_check_agematch, file=f_agematch)
    print('df_Feature_check_agesexmatch:', df_Feature_check_agesexmatch, file=f_agesexmatch)

print(df_Feature_check_agesexmatch)

basic_columns=['u_num', 'a_num', 'i_num', 'ADOS_C', 'dia_num', 'sex', 'age', 'Module','timeSeries_len']
insp_column=basic_columns + ['Divergence[within_covariance_norm(A:,i:,u:)]_var_p1','Divergence[within_covariance_norm(A:,i:,u:)]_var_p2']


Aa_ASD=df_formant_statistic_agesexmatch_ASDSevere[insp_column]
Aa_TD=df_formant_statistic_TD_normal[insp_column]

Inspect_columns=[('localShimmer_mean(u:)','Divergence[localShimmer_mean(u:)]_var_p2')]
score_df_columns=[]
def Plot_Timeseries(df_formant_statistic, df_person_segment_feature,Inspect_columns, score_df):
    fig=plt.figure()
    for people in df_formant_statistic.index:
        # df_person_segment_feature_dict_TD
        Dict_df_ASD=df_person_segment_feature[people]
        for cols, score_cols in Inspect_columns: 
            df_ASD_d=Dict_df_ASD['D'][cols]
            df_ASD_k=Dict_df_ASD['K'][cols]
            df_ASD_d.name="doc"
            df_ASD_k.name="kid"
            df_ASD_dk=pd.concat([df_ASD_d,df_ASD_k],axis=1)
            sns.lineplot(data=df_ASD_dk)
            title='{0}'.format('ASD ' + people, 'Col: ',cols)
            
            plt.title( title )
            
            score=score_df.loc[people,score_cols]
            addtext='score: {0}'.format(np.round(score,3))
            text(0.1, 0.1, addtext, ha='center', va='center')
            plt.show()
            plt.clf()

df_formant_statistic_agesexmatchASDSevere_sorted=df_formant_statistic_agesexmatch_ASDSevere.sort_values(Inspect_columns[0][-1])
Plot_Timeseries(df_formant_statistic_agesexmatchASDSevere_sorted,df_person_segment_feature_dict_ASD,Inspect_columns,df_formant_statistic_all)

df_formant_statistic_TD_normal_sorted=df_formant_statistic_TD_normal.sort_values(Inspect_columns[0][-1])
Plot_Timeseries(df_formant_statistic_TD_normal_sorted,df_person_segment_feature_dict_TD,Inspect_columns,df_formant_statistic_all)

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


Feature_collect['df_Session_formant_statistic_ASD'].columns=intersection(df_formant_statistic_77.columns,df_Session_formant_statistic_ASD.columns)
Feature_collect['df_Session_phonation_statistic_ASD'].columns=intersection(df_formant_statistic_77.columns,df_Session_phonation_statistic_ASD.columns)
Feature_collect['df_Syncrony_formant_statistic_ASD'].columns=intersection(df_formant_statistic_77.columns,df_Syncrony_formant_statistic_ASD.columns)
Feature_collect['df_Syncrony_phonation_statistic_ASD'].columns=intersection(df_formant_statistic_77.columns,df_Syncrony_phonation_statistic_ASD.columns)


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
    else:
        feat_type='Syncrony_formant'
        
    features_filt=criterion_filter(features,N=N,evictNamelst=[],feature_type=feat_type)
    Filtered_feat[feature_strs]=features_filt
    
    
    correlations_ASD=Eval_med.Calculate_correlation(label_choose_lst,features_filt,N,columns,feature_type=feat_type)
    
    Correlation_dict[feature_strs]=correlations_ASD
    



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


# Calculate correlations for each sectional features

Aaadf_pearsonr_table_ASD=Eval_med.Calculate_correlation(label_choose_lst,df_formant_statistic_77,\
                                                        N,Parameters,constrain_sex=-1, constrain_module=-1,\
                                                        feature_type='Syncrony_formant')


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
df_formant_statistic_all_remaned=df_formant_statistic_TD.rename(columns=lambda s: re.sub(u"[{}]+".format(punc),"",s))
df_formant_statistic_all_remaned['Module']=df_formant_statistic_all_remaned['Module'].astype(str)
df_formant_statistic_all_remaned.loc[df_formant_statistic_all_remaned['Module']==3,'Module']=1
df_formant_statistic_all_remaned.loc[df_formant_statistic_all_remaned['Module']==4,'Module']=2


formula='ADOS_C ~ '
formula+= ' between_variance(A:,i:,u:) '
# formula+= ' + BV(A:,i:,u:)_l2'
formula=re.sub(u"[{}]+".format(punc),"",formula)

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
IV_list=['sex','Module','ASDTD']
# IV_lst=['ASDTD']
# comcinations=list(itertools.product(*Effect_comb))
ways=2
combination = combinations(IV_list, ways)
for comb in combination:
    IV_lst = list(comb)
    DV_str='between_variance(A:,i:,u:)'
    df_remaned, formula = Regression_Preprocess_setp(DV_str, IV_lst, df_formant_statistic_all)
    punc=":,()"
    model = ols(formula, data=df_remaned).fit()
    anova = sm.stats.anova_lm(model, typ=ways)
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