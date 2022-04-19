#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:14:41 2020

@author: jackchen

This script does the main expeeriments in Table 1 (Correlation between Level of clustering and ADOS_A) 
1. Data prepare area: 
    Gather raw data of the three critical monophthongs (F1 & F2) and save in: df_formant_statistic.
    
    1-1 Filtering area:
        Filter out the outliers by IQR method (defined in muti.FilterUttDictsByCriterion_map)
    
2. Feature calculating area
    a. We use articulation.calculate_features() method to calculate LOC features 
    
3. Evaluation area


Input:
    Formants_utt_symb

Output:
    df_formant_statistic


"""

import pickle
import argparse
from addict import Dict
import numpy as np
import pandas as pd
from articulation.HYPERPARAM import phonewoprosody, Label
import matplotlib.pyplot as plt
from itertools import combinations

from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from scipy import stats
from scipy.stats import spearmanr,pearsonr 
import statistics 
import os, glob, sys
import statsmodels.api as sm
from varname import nameof
from tqdm import tqdm
import re
from multiprocessing import Pool, current_process
from articulation.articulation import Articulation
import articulation.Multiprocess as Multiprocess
from datetime import datetime as dt
import pathlib

from utils_jack  import  Formant_utt2people_reshape, Gather_info_certainphones, \
                         FilterUttDictsByCriterion, GetValuelimit_IQR, \
                         Get_aligned_sequences, WER, Get_Vowels_AUI
from metric import Evaluation_method 

# from sklearn.utils import (as_float_array, check_array, check_X_y, safe_sqr,
#                      safe_mask)
from scipy import special, stats
import warnings


def criterion_filter(df_formant_statistic,N=10,\
                     constrain_sex=-1, constrain_module=-1,constrain_agemax=-1,constrain_ADOScate=-1,constrain_agemin=-1,\
                     evictNamelst=[]):
    # filter by number of phones
    
    
    
    filter_bool=np.logical_and(df_formant_statistic['u_num']>N,df_formant_statistic['a_num']>N)
    filter_bool=np.logical_and(filter_bool,df_formant_statistic['i_num']>N)
    
    # filer by other biological information
    if constrain_sex != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['sex']==constrain_sex)
    if constrain_module != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['Module']==constrain_module)
    if constrain_agemax != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['age']<=constrain_agemax)
    if constrain_agemin != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['age']>=constrain_agemin)
    if constrain_ADOScate != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['ADOS_cate_C']==constrain_ADOScate)
    
    # filter the names given the name list
    if len(evictNamelst)>0:
        for name in evictNamelst:
            filter_bool.loc[name]=False
    
    # print("filter bool")
    # print(filter_bool)
    # print("df_formant_statistic")
    # print(~df_formant_statistic.isna().T.any())
    # get rid of nan values
    filter_bool=np.logical_and(filter_bool,~df_formant_statistic.isna().T.any())
    return df_formant_statistic[filter_bool]



def NameMatchAssertion(Formants_people_symb,name):
    ''' check the name in  Formants_people_symb matches the names in label'''
    for name in Formants_people_symb.keys():
        assert name in name




def Process_IQRFiltering_Multi(Formants_utt_symb, limit_people_rule,\
                               outpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',\
                               prefix='Formants_utt_symb',\
                               suffix='KID_FromASD_DOCKID'):
    pool = Pool(int(os.cpu_count()))
    keys=[]
    interval=20
    for i in range(0,len(Formants_utt_symb.keys()),interval):
        # print(list(combs_tup.keys())[i:i+interval])
        keys.append(list(Formants_utt_symb.keys())[i:i+interval])
    flat_keys=[item for sublist in keys for item in sublist]
    assert len(flat_keys) == len(Formants_utt_symb.keys())
    muti=Multiprocess.Multi()
    final_results=pool.starmap(muti.FilterUttDictsByCriterion_map, [([Formants_utt_symb,Formants_utt_symb,file_block,limit_people_rule]) for file_block in tqdm(keys)])
    
    Formants_utt_symb_limited=Dict()
    for load_file_tmp,_ in final_results:        
        for utt, df_utt in load_file_tmp.items():
            Formants_utt_symb_limited[utt]=df_utt
    
    pickle.dump(Formants_utt_symb_limited,open(outpath+"/[Analyzing]{0}_limited_{1}.pkl".format(prefix,suffix),"wb"))
    print('Formants_utt_symb saved to ',outpath+"/[Analyzing]{0}_limited_{1}.pkl".format(prefix,suffix))
    
'''
Calculating FCR
FCR=(F2u+F2a+F1i+F1u)/(F2i+F1a)
VSA1=ABS((F1i*(F2a –F2u)+F1a *(F2u–F2i)+F1u*(F2i–F2a))/2)
VSA2=sqrt(S*(S-EDiu)(S-EDia)(S-EDau))
LnVSA=sqrt(LnS*(LnS-LnEDiu)(LnS-LnEDia)(LnS-LnEDau))

where,
u=F12_val_dict['w']
a=F12_val_dict['A']
i=F12_val_dict['j']

EDiu=sqrt((F2u–F2i)^2+(F1u–F1i)^2)
EDia=sqrt((F2a–F2i)^2+(F1a–F1i)^2)
EDau=sqrt((F2u–F2a)^2+(F1u–F1a)^2)
S=(EDiu+EDia+EDau)/2

The above formulas are implemented inside articulation.calculate_features()
'''

# =============================================================================
def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice/articulation',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--inpklpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--outpklpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--Inspect', default=False,
                            help='path of the base directory')
    parser.add_argument('--reFilter', default=False,
                            help='')
    parser.add_argument('--correldf_formant_statistication_type', default='spearmanr',
                            help='spearmanr|pearsonr')
    parser.add_argument('--label_choose_lst', default=['ADOS_C'],
                            help='path of the base directory')
    parser.add_argument('--Stat_med_str_VSA', default='mean',
                            help='path of the base directory')
    parser.add_argument('--poolMed', default='middle',
                            help='path of the base directory')
    parser.add_argument('--poolWindowSize', default=3,
                            help='path of the base directory')
    parser.add_argument('--dataset_role', default='KID_FromASD_DOCKID',
                            help='kid_TD| kid88| DOC_FromASD_DOCKID | KID_FromASD_DOCKID')
    parser.add_argument('--Inspect_features', default=['F1','F2'],
                            help='')

    args = parser.parse_args()
    return args


args = get_args()
base_path=args.base_path

# =============================================================================
'''
    
    1. Data prepare area

'''
# =============================================================================
''' parse namespace '''
args = get_args()
base_path=args.base_path
pklpath=args.inpklpath
INSPECT=args.Inspect
windowsize=args.poolWindowSize
label_choose_lst=args.label_choose_lst # labels are too biased
role=args.dataset_role
Stat_med_str=args.Stat_med_str_VSA
outpklpath=args.inpklpath+"/Session_formants_people_vowel_feat/"
if not os.path.exists(outpklpath):
    os.makedirs(outpklpath)


Formants_utt_symb=pickle.load(open(pklpath+"/Formants_utt_symb_by{0}_window{1}_{2}.pkl".format(args.poolMed,windowsize,role),'rb'))
print("Loading Formants_utt_symb from ", pklpath+"/Formants_utt_symb_by{0}_window{1}_{2}.pkl".format(args.poolMed,windowsize,role))



# =============================================================================
'''

    1-1. Filtering area
    
    Filter out data using by 1.5*IQR

'''
PhoneMapp_dict=phonewoprosody.PhoneMapp_dict
PhoneOfInterest=list(PhoneMapp_dict.keys())
# =============================================================================


''' Vowel AUI rule is using phonewoprosody '''
Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
AUI_info=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)
limit_people_rule=GetValuelimit_IQR(AUI_info,PhoneMapp_dict,args.Inspect_features)



''' multi processing start '''
prefix,suffix = 'Formants_utt_symb', role
# date_now='{0}-{1}-{2} {3}'.format(dt.now().year,dt.now().month,dt.now().day,dt.now().hour)
date_now='{0}-{1}-{2}'.format(dt.now().year,dt.now().month,dt.now().day)
outpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles'
filepath=outpath+"/[Analyzing]{0}_limited_{1}.pkl".format(prefix,suffix)
if os.path.exists(filepath) and args.reFilter==False:
    fname = pathlib.Path(filepath)
    mtime = dt.fromtimestamp(fname.stat().st_mtime)
    # filemtime='{0}-{1}-{2} {3}'.format(mtime.year,mtime.month,mtime.day,mtime.hour)
    filemtime='{0}-{1}-{2}'.format(mtime.year,mtime.month,mtime.day)
    
    # If file last modify time is not now (precisions to the hours) than we create new one
    if filemtime != date_now:
        Process_IQRFiltering_Multi(Formants_utt_symb,limit_people_rule,\
                               outpath=outpath,\
                               prefix=prefix,\
                               suffix=suffix) # the results will be output as pkl file at outpath+"/[Analyzing]Formants_utt_symb_limited.pkl"
else:
    Process_IQRFiltering_Multi(Formants_utt_symb,limit_people_rule,\
                               outpath=outpath,\
                               prefix=prefix,\
                               suffix=suffix)
Formants_utt_symb_limited=pickle.load(open(filepath,"rb"))
''' multi processing end '''
if len(limit_people_rule) >0:
    Formants_utt_symb=Formants_utt_symb_limited



Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
AUI_info=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)

    
# =============================================================================        
''' 

    2. Feature calculating area


'''
#Continuous
'Multi1','Multi2','Multi3','Multi4','T_ADOS_C','T_ADOS_S','T_ADOS_SC'
'VIQ','VCI','ADOS_CSS'
'C1','D1','D2','D3','D4','D5','E1','E2','E3'
'BB1','BB2','BB3','BB4','BB5','BB6','BB7','BB8','BB9','BB10',
'AA1','AA2','AA3','AA4','AA5','AA6','AA7','AA8','AA9'
'ADOS_S','ADOS_C','C','D','C+S',
'fso_com','fst_beh','fso_awr','fsoc_emo','fsrs_t','ssoc','sstere','scom','scq'
'adircta','adirctb','adirctc'
'BRI','MI','GEC'

#Category
'ADOS_cate_C'
'dia_num'
# =============================================================================
Vowels_AUI=Get_Vowels_AUI(AUI_info, args.Inspect_features,VUIsource="From__Formant_people_information")



additional_columns=['ADOS_cate_C','T_ADOS_C','T_ADOS_S','T_ADOS_SC','ADOS_S','ADOS_C','C+S',\
                    'ADOS_CSS','AA1','AA2','AA3','AA4','AA5','AA6','AA7','AA8','AA9','fso_com','fst_beh','fso_awr','fsoc_emo','fsrs_t','ssoc','sstere','scom','scq',\
                    'adircta','adirctb','adirctc']
# additional_columns=['Multi1','Multi2','Multi3','Multi4','T_ADOS_C','T_ADOS_S','T_ADOS_SC',\
#                     'VIQ','VCI','ADOS_CSS',\
#                     'C1','D1','D2','D3','D4','D5','E1','E2','E3',\
#                     'BB1','BB2','BB3','BB4','BB5','BB6','BB7','BB8','BB9','BB10',\
#                     'AA1','AA2','AA3','AA4','AA5','AA6','AA7','AA8','AA9',\
#                     'ADOS_S','ADOS_C','C','D','C+S',\
#                     'ADOS_cate_C']


# label_generate_choose_lst=['ADOS_C','dia_num'] + additional_columns
label_generate_choose_lst=['ADOS_C']



articulation=Articulation(Stat_med_str_VSA='mean')
# df_formant_statistic=articulation.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_generate_choose_lst)
# df_formant_statistic=articulation.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_generate_choose_lst,FILTER_overlap_thrld=0)
# df_formant_statistic=articulation.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_generate_choose_lst, FILTERING_method='Silhouette',FILTER_overlap_thrld=0)
# df_formant_statistic=articulation.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_generate_choose_lst, FILTERING_method='KDE', KDE_THRESHOLD=40)
df_formant_statistic, SCATTER_matrixBookeep_dict=articulation.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_generate_choose_lst, FILTERING_method='KDE', KDE_THRESHOLD=40,RETURN_scatter_matrix=True)



# For pseudo acoustic features generation
df_formant_statistic['u_num+i_num+a_num']=df_formant_statistic['u_num'] +\
                                            df_formant_statistic['i_num'] +\
                                            df_formant_statistic['a_num']

for i in range(len(df_formant_statistic)):
    name=df_formant_statistic.iloc[i].name
    df_formant_statistic.loc[name,'ADOS_cate_C']=Label.label_raw[Label.label_raw['name']==name]['ADOS_cate_C'].values
    ''' ADOS_cate_C, cate stands for category '''
    
# =============================================================================        
''' 

    2. Evaluation area

    We still keep this area to get a peek of the correlation result.
    The evaluation function should be the same as the one in Statistical_tests.py
    
    The evaluation module is defined in Evaluation_method()
    
'''
# =============================================================================

Eval_med=Evaluation_method()


sex=-1
module=-1
agemax=-1
agemin=-1
ADOScate=-1
N=2
df_formant_statistic_77=criterion_filter(df_formant_statistic,\
                                        constrain_sex=sex,constrain_module=module,N=N,constrain_agemax=agemax,constrain_agemin=agemin,constrain_ADOScate=ADOScate,\
                                        evictNamelst=[])

pickle.dump(df_formant_statistic_77,open(outpklpath+"Formant_AUI_tVSAFCRFvals_{}.pkl".format(role),"wb"))
# pickle.dump(df_formant_statistic,open(outpklpath+"Formant_AUI_tVSAFCRFvals_{}.pkl".format(role),"wb"))





''' Calculate correlations for Formant fetures'''
columns=list(set(df_formant_statistic.columns) - set(additional_columns)) # Exclude added labels
columns=list(set(columns) - set([co for co in columns if "_norm" not in co]))
columns= columns + [co for co in columns if "Between_Within" in co]
columns= columns + ['ConvexHull', 'MeanVFD','VSA2','FCR2']
# columns= columns + ['absAng_a','absAng_u','absAng_i', 'ang_ai','ang_iu','ang_ua','Angles']
columns= columns + ['dcov_12','dcorr_12','dvar_1', 'dvar_2','pear_12','spear_12','kendall_12']
columns= columns + ['pointDistsTotal','repulsive_force']
# columns= columns + ['FCR2_uF2','FCR2_aF2','FCR2_iF1','FCR2_uF1','aF2']



# def Known_featuresAddition(df_formant_statistic ,columns, comb_features_collections=['between_covariance_norm(A:,i:,u:)','ang_ua','ang_ai','absAng_a','hotelling_lin_norm(A:,i:,u:)']):
#     comb2=combinations(comb_features_collections,2)
#     comb3=combinations(comb_features_collections,3)
#     comb4=combinations(comb_features_collections,4)
#     comb5=combinations(comb_features_collections,5)
#     combs=list(comb2)+list(comb3)+list(comb4)+list(comb5)
#     for comb in combs:
#         new_variable='+'.join(comb)
#         df_formant_statistic[new_variable]=0
#         for element in comb:
#             df_formant_statistic[new_variable] += df_formant_statistic[element]
#         columns+=[new_variable]
#     return df_formant_statistic ,columns
# df_formant_statistic ,columns= Known_featuresAddition(df_formant_statistic ,columns , df_formant_statistic.columns)


ManualCondition=Dict()
suffix='.xlsx'
condfiles=glob.glob('Inspect/condition/*'+suffix)
for file in condfiles:
    df_cond=pd.read_excel(file)
    name=os.path.basename(file).replace(suffix,"")
    ManualCondition[name]=df_cond['Unnamed: 0'][df_cond['50%']==True]

label_correlation_choose_lst=label_generate_choose_lst
# label_correlation_choose_lst=['ADOS_C','T_ADOS_C','AA1','AA2','AA3','AA4','AA5','AA6','AA7','AA8','AA9']

# df_formant_statistic=df_formant_statistic.drop(index='2015_12_06_01_097')
N=2
Eval_med=Evaluation_method()
Aaadf_spearmanr_table_NoLimit=Eval_med.Calculate_correlation(label_correlation_choose_lst,df_formant_statistic,N,columns,constrain_sex=-1, constrain_module=-1,feature_type='Session_formant')


# Just a little check for equal indexes
lab_chos_lst=['ADOS_C']

indexes_dict={}
for label_choose in lab_chos_lst:
    df_formant_statistic=Eval_med._Postprocess_InCalCorr( label_choose,df_formant_statistic,N) #Filter unwanted samples
    # print(len(df_formant_statistic))
    indexes_dict[label_choose]=df_formant_statistic.index
df_RESULT_list=pd.DataFrame.from_dict(indexes_dict)

# =============================================================================
'''

Parse through evaluation dictionary

'''





# =============================================================================
def Survey_nice_variable(df_result_table):
    def Criteria(df_Corr_val):
        pear_str='pearson_p'
        spear_str='spearman_p'
        
        criteria_bool=(df_Corr_val[pear_str]<=0.05) & (df_Corr_val[spear_str]<=0.05)
        return df_Corr_val[criteria_bool]
    
    
    for key in df_result_table.keys():
        df_Corr_val=df_result_table[key]
        df_Corr_val_criteria=Criteria(df_Corr_val)
        if len(df_Corr_val_criteria)>0:
            print('Predicting label', key)
            print("=========================")
            print(df_Corr_val_criteria)
            print("                         ")


Survey_nice_variable(Aaadf_spearmanr_table_NoLimit)
aaa=ccc
# =============================================================================
'''
    feature Classification 
    

'''
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,recall_score,roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from pylab import text
dfFormantStatisticpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles'
feat='Formant_AUI_tVSAFCRFvals'

df_formant_statistic77_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/{name}_{role}.pkl'.format(name=feat,role='KID_FromASD_DOCKID')
df_feature_ASD=pickle.load(open(df_formant_statistic77_path,'rb'))
df_formant_statistic_ASDTD_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/{name}_{role}.pkl'.format(name=feat,role='kid_TD')
if not os.path.exists(df_formant_statistic_ASDTD_path) or not os.path.exists(df_formant_statistic77_path):
    raise FileExistsError
df_feature_TD=pickle.load(open(df_formant_statistic_ASDTD_path,'rb'))


def Add_label(df_formant_statistic,Label,label_choose='ADOS_S'):
    for people in df_formant_statistic.index:
        bool_ind=Label.label_raw['name']==people
        df_formant_statistic.loc[people,label_choose]=Label.label_raw.loc[bool_ind,label_choose].values
    return df_formant_statistic
# =============================================================================
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

TopTop_data_lst=[]
TopTop_data_lst.append(['df_feature_ASD','df_feature_TD'])
TopTop_data_lst.append(['df_feature_low_CSS','df_feature_TD'])
TopTop_data_lst.append(['df_feature_moderate_CSS','df_feature_TD'])
TopTop_data_lst.append(['df_feature_high_CSS','df_feature_TD'])
TopTop_data_lst.append(['df_feature_lowMinimal_CSS','df_feature_TD'])
TopTop_data_lst.append(['df_feature_moderatehigh_CSS','df_feature_TD'])

TopTop_data_lst.append(['df_feature_low_CSS','df_feature_moderate_CSS'])
TopTop_data_lst.append(['df_feature_moderate_CSS','df_feature_high_CSS'])
TopTop_data_lst.append(['df_feature_low_CSS','df_feature_high_CSS'])
TopTop_data_lst.append(['df_feature_lowMinimal_CSS','df_feature_moderate_CSS'])
TopTop_data_lst.append(['df_feature_lowMinimal_CSS','df_feature_high_CSS'])

TopTop_data_lst.append(['df_feature_Notautism_TC','df_feature_TD'])
TopTop_data_lst.append(['df_feature_ASD_TC','df_feature_TD'])
TopTop_data_lst.append(['df_feature_NotautismandASD_TC','df_feature_TD'])
TopTop_data_lst.append(['df_feature_Autism_TC','df_feature_TD'])

TopTop_data_lst.append(['df_feature_Notautism_TC','df_feature_ASD_TC'])
TopTop_data_lst.append(['df_feature_ASD_TC','df_feature_Autism_TC'])
TopTop_data_lst.append(['df_feature_Notautism_TC','df_feature_Autism_TC'])
TopTop_data_lst.append(['df_feature_Notautism_TC','df_feature_ASD_TC','df_feature_Autism_TC'])


TopTop_data_lst.append(['df_feature_Notautism_TS','df_feature_TD'])
TopTop_data_lst.append(['df_feature_ASD_TS','df_feature_TD'])
TopTop_data_lst.append(['df_feature_NotautismandASD_TS','df_feature_TD'])
TopTop_data_lst.append(['df_feature_Autism_TS','df_feature_TD'])

TopTop_data_lst.append(['df_feature_Notautism_TS','df_feature_ASD_TS'])
TopTop_data_lst.append(['df_feature_ASD_TS','df_feature_Autism_TS'])
TopTop_data_lst.append(['df_feature_Notautism_TS','df_feature_Autism_TS'])
TopTop_data_lst.append(['df_feature_Notautism_TS','df_feature_ASD_TS','df_feature_Autism_TS'])

self_specify_cols=[
    'FCR2',
    'VSA2',
    'between_covariance_norm(A:,i:,u:)', 
    'between_variance_norm(A:,i:,u:)',
    'within_covariance_norm(A:,i:,u:)', 
    'within_variance_norm(A:,i:,u:)',
    'total_covariance_norm(A:,i:,u:)', 
    'total_variance_norm(A:,i:,u:)',
    'sam_wilks_lin_norm(A:,i:,u:)', 
    'pillai_lin_norm(A:,i:,u:)',
    'hotelling_lin_norm(A:,i:,u:)', 
    'roys_root_lin_norm(A:,i:,u:)',
    'Between_Within_Det_ratio_norm(A:,i:,u:)',
    'Between_Within_Tr_ratio_norm(A:,i:,u:)',
    'pear_12',
    'spear_12',
    'kendall_12',
    'dcorr_12'
    ]

if len(self_specify_cols) > 0:
    inspect_cols=self_specify_cols
else:
    inspect_cols=columns

print('Start doing U-tests and T-tests ')

plot=False
Record_dict=Dict()
All_cmp_dict=Dict()
for Top_data_lst in TopTop_data_lst:
    Record_dict[' vs '.join(Top_data_lst)]=pd.DataFrame(index=inspect_cols)
    All_cmp_dict[' vs '.join(Top_data_lst)]=pd.DataFrame(index=inspect_cols)
    import warnings
    warnings.filterwarnings("ignore")
    for columns in inspect_cols:
        # =============================================================================
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
                    fig, ax = plt.subplots()
                    data=[]
                    dataname=[]
                    for dstr in Top_data_lst:
                        dataname.append(dstr)
                        data.append(vars()[dstr])
                    for i,d in enumerate(data):
                        # ax = sns.distplot(d[columns], ax=ax, kde=False)
                        ax = sns.distplot(d[columns], ax=ax, label=Top_data_lst)
                        title='{0}'.format('Inspecting feature ' + columns)
                        plt.title( title )
                    fig.legend(labels=dataname)  
                    
                    addtext='{0}/({1})'.format(np.round(mean_difference,3),np.round(p_val,3))
                    text(0.9, 0.9, addtext, ha='center', va='center', transform=ax.transAxes)
                    addtextvariable='{0} vs {1}'.format(Top_data_lst[0],Top_data_lst[1])
                    text(0.9, 0.6, addtextvariable, ha='center', va='center', transform=ax.transAxes)
    warnings.simplefilter('always')

# Record_certainCol_dict={}
df_CertainCol_U=pd.DataFrame()
df_CertainCol_T=pd.DataFrame()
for test_name, values in Record_dict.items():
    data_T=values.loc[:,values.columns.str.startswith("TTest")]
    data_U=values.loc[:,values.columns.str.startswith("UTest")]
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

aaa=ccc
# =============================================================================
# 
# =============================================================================
lab_chos_lst=['ASDclassify']
# feature_chos_lst_top=['between_covariance_norm(A:,i:,u:)','dcorr_12']


# C_variable=np.array(np.arange(0.1,1.1,0.2))
# epsilon=np.array(np.arange(0.1,1.5,0.1) )
# epsilon=np.array(np.arange(0.01,0.15,0.02))
# C_variable=np.array([0.001,0.01,10.0,50,100] + list(np.arange(0.1,1.5,0.1)))
C_variable=np.array([0.001,0.01,10.0,50,100])
Classifier={}
loo=LeaveOneOut()
# CV_settings=loo
CV_settings=10
pca = PCA(n_components=1)

# C_variable=np.array(np.arange(0.1,1.5,0.1))
# C_variable=np.array([0.001,0.01,10.0,50,100] + list(np.arange(0.1,1.5,0.2))  )
# C_variable=np.array([0.01, 0.1,0.5,1.0, 5.0])
n_estimator=[ 32, 50, 64, 100 ,128, 256]
# =============================================================================
Classifier={}
Classifier['SVC']={'model':sklearn.svm.SVC(),\
                  'parameters':{'model__random_state':[1],\
                    'model__C':C_variable,\
                    'model__kernel': ['rbf'],\
                      # 'model__gamma':['auto'],\
                    'model__probability':[True],\
                                }}
Classifier['DT']={'model':DecisionTreeClassifier(),\
                  'parameters':{'model__random_state':[1],\
                                'model__criterion':['gini','entropy'],
                                'model__splitter':['splitter','random'],\
                                }}

    
clf=Classifier['SVC']



columns=[
    'FCR2',
    'VSA2',
    'between_covariance_norm(A:,i:,u:)', 
    'between_variance_norm(A:,i:,u:)',
    'within_covariance_norm(A:,i:,u:)', 
    'within_variance_norm(A:,i:,u:)',
    'total_covariance_norm(A:,i:,u:)', 
    'total_variance_norm(A:,i:,u:)',
    'sam_wilks_lin_norm(A:,i:,u:)', 
    'pillai_lin_norm(A:,i:,u:)',
    'hotelling_lin_norm(A:,i:,u:)', 
    'roys_root_lin_norm(A:,i:,u:)',
    'Between_Within_Det_ratio_norm(A:,i:,u:)',
    'Between_Within_Tr_ratio_norm(A:,i:,u:)',
    'pear_12',
    'spear_12',
    'kendall_12',
    'dcorr_12'
    ]

featuresOfInterest=[ [col] for col in columns]



combinations_lsts=[ k for k in featuresOfInterest]
combinations_keylsts=[ k[0] for k in featuresOfInterest]


Top_RESULT_dict=Dict()
for Top_data_lst in TopTop_data_lst:
    print(Top_data_lst[0], ' vs ', Top_data_lst[1])
    if len(Top_data_lst) == 2:
        df_asdTmp, df_tdTmp=vars()[Top_data_lst[0]].copy(), vars()[Top_data_lst[1]].copy()
        df_asdTmp["ASDclassify"]=1
        df_tdTmp["ASDclassify"]=2
        df_ASDcmpVombineTD=pd.concat([df_asdTmp,df_tdTmp],axis=0)
    elif len(Top_data_lst) == 3:
        df_adTmp, df_asdTmp, df_nonasdTmp=vars()[Top_data_lst[0]].copy(), vars()[Top_data_lst[1]].copy(), vars()[Top_data_lst[2]].copy()
        df_adTmp["ASDclassify"]=1
        df_asdTmp["ASDclassify"]=2
        df_nonasdTmp["ASDclassify"]=3
        df_ASDcmpVombineTD=pd.concat([df_adTmp,df_asdTmp,df_nonasdTmp],axis=0)

    RESULT_dict=Dict()
    for key,feature_chos_tup in zip(combinations_keylsts,combinations_lsts):
        feature_chos_lst=list(feature_chos_tup)
        for feature_chooses in [feature_chos_lst]:
            # pipe = Pipeline(steps=[("model", clf['model'])])
            pipe = Pipeline(steps=[('scalar',StandardScaler()),("model", clf['model'])])
            # pipe = Pipeline(steps=[ ("pca", pca), ("model", clf['model'])])
            p_grid=clf['parameters']
    
            Gclf = GridSearchCV(pipe, param_grid=p_grid, scoring='recall_macro', cv=CV_settings, refit=True, n_jobs=-1)
            
            features=Dict()
            # 1. 要多一個columns 是ASDTD
            features.X=df_ASDcmpVombineTD[feature_chooses]
            features.y=df_ASDcmpVombineTD[lab_chos_lst]
            # StandardScaler().fit_transform(features.X)
            
            # 2. 改成算UAR, AUC
            # CVscore=cross_val_score(Gclf, features.X, features.y.values.ravel(), cv=CV_settings,scoring='recall_macro')
            CVpredict=cross_val_predict(Gclf, features.X, features.y.values.ravel(), cv=CV_settings)  
            
            n,p=features.X.shape
            UAR=recall_score(features.y, CVpredict, average='macro')
            AUC=roc_auc_score(features.y, CVpredict)
            f1Score=f1_score(features.y, CVpredict, average='macro')
            
            # feature_keys='+'.join(feature_chooses)
            feature_keys=key
            print('Feature {0}, UAR {1}, AUC {2} ,f1Score {3}'.format(feature_keys, UAR, AUC,f1Score))
            RESULT_dict[feature_keys]=[UAR,AUC,f1Score]
    
    
    df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict,orient='index')
    df_RESULT_list.columns=['UAR','AUC','f1Score']
    print(df_RESULT_list)
    
    Expiment_str=' vs '.join(Top_data_lst)
    Top_RESULT_dict[Expiment_str]=df_RESULT_list

Result_UAR_summary={}
Inspect_metric='UAR'
for Expiment_str, values in Top_RESULT_dict.items():
    Result_UAR_summary[Expiment_str]=values[Inspect_metric]
    


df_Result_UAR_summary_list=pd.DataFrame.from_dict(Result_UAR_summary,orient='index')
del Gclf




# =============================================================================
''' 
    Multi feature prediction 

'''
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def TBMEB1Preparation_LoadForFromOtherData(dfFormantStatisticpath):
    '''
        
        We generate data for nested cross-valated analysis in Table.5 in TBME2021
        
        The data will be stored at Pickles/Session_formants_people_vowel_feat
    
    '''
    dfFormantStatisticFractionpath='Pickles/Session_formants_people_vowel_feat'
    if not os.path.exists(dfFormantStatisticFractionpath):
        raise FileExistsError('Directory not exist')
    df_phonation_statistic_77=pickle.load(open(dfFormantStatisticFractionpath+'/df_phonation_statistic_77.pkl','rb'))
    return df_phonation_statistic_77
df_phonation_statistic_77=TBMEB1Preparation_LoadForFromOtherData(pklpath)
df_formant_statistic_added=pd.concat([df_phonation_statistic_77,df_formant_statistic],axis=1)
df_formant_statistic_added=df_formant_statistic_added.loc[:,~df_formant_statistic_added.columns.duplicated()]

sex=-1
module=-1
agemax=-1
agemin=-1
ADOScate=-1
N=2
df_formant_statistic_added=criterion_filter(df_formant_statistic_added,\
                                        constrain_sex=sex,constrain_module=module,N=N,constrain_agemax=agemax,constrain_agemin=agemin,constrain_ADOScate=ADOScate,\
                                        evictNamelst=[])
''' cross validation prediction '''
# feature_chos_lst=['between_covariance_norm(A:,i:,u:)',
# 'sam_wilks_lin_norm(A:,i:,u:)',
# 'hotelling_lin_norm(A:,i:,u:)',
# 'pillai_lin_norm(A:,i:,u:)']

# feature_chos_lst_top=['between_covariance_norm(A:,i:,u:)']
# feature_chos_lst_top=['between_variance_norm(A:,i:,u:)']

# feature_chos_lst_top=['roys_root_lin_norm(A:,i:,u:)', 'Angles']
# feature_chos_lst_top=['Between_Within_Det_ratio_norm(A:,i:,u:)']
# feature_chos_lst=['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ai']
# feature_chos_lst=['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ua']
# feature_chos_lst=['pillai_lin_norm(A:,i:,u:)']
# feature_chos_lst=['ang_ai']
# feature_chos_lst=['ang_ua']
# feature_chos_lst=['FCR2']
# feature_chos_lst=['FCR2','ang_ai']
# feature_chos_lst=['FCR2','ang_ua']
# feature_chos_lst=['FCR2','ang_ai','ang_ua']
# feature_chos_lst_top=['between_covariance_norm(A:,i:,u:)','localabsoluteJitter_mean(A:,i:,u:)','dcorr_12']
# feature_chos_lst_top=['between_covariance_norm(A:,i:,u:)','localabsoluteJitter_mean(A:,i:,u:)']
feature_chos_lst_top=['between_covariance_norm(A:,i:,u:)','dcorr_12']
baseline_lst=['FCR2']


# C_variable=np.array([0.001,0.01, 0.1,0.5,1.0,10.0,50,100])
C_variable=np.array([0.001,0.01, 0.1,0.5,1.0,10.0,50,100])
Classifier={}
loo=LeaveOneOut()
# CV_settings=loo
CV_settings=10
pca = PCA(n_components=1)

# =============================================================================
Classifier['SVR']={'model':sklearn.svm.SVR(),\
                  'parameters':{'C':C_variable,\
                    # 'kernel': ['rbf','sigmoid'],\
                    'kernel': ['rbf'],\
                    'gamma': ['scale'],\
                    # 'gamma': ['auto'],\
                    # 'gamma': ['scale','auto'],\
                                }}
Classifier['EN']={'model':ElasticNet(random_state=0),\
                  'parameters':{'alpha':np.arange(0,1,0.25),\
                                'l1_ratio': np.arange(0,1,0.25),\
                                'max_iter':[2000]}} #Just a initial value will be changed by parameter tuning
                                                    # l1_ratio = 1 is the lasso penalty

Classifier['LinR']={'model':sklearn.linear_model.LinearRegression(),\
                  'parameters':{'fit_intercept':[True],\
                                }}

    
clf=Classifier['SVR']
# comb2 = combinations(feature_chos_lst_top, 2)
# comb3 = combinations(feature_chos_lst_top, 3)
# comb4 = combinations(feature_chos_lst_top, 4)
# combinations_lsts=list(comb2) + list(comb3)+ list(comb4)
combinations_lsts=[feature_chos_lst_top]
lab_chos_lst=['ADOS_C']

RESULT_dict=Dict()
for feature_chos_tup in combinations_lsts:
    feature_chos_lst=list(feature_chos_tup)
    for feature_chooses in [feature_chos_lst,baseline_lst]:
        # pipe = Pipeline(steps=[("model", clf['model'])])
        pipe = Pipeline(steps=[('scalar',StandardScaler()),("model", clf['model'])])
        # pipe = Pipeline(steps=[ ("pca", pca), ("model", clf['model'])])
        param_grid = {
        # "pca__n_components": [3],
        "model__C": C_variable,
        # "model__l1_ratio": np.arange(0,1,0.25),
        # "model__alpha": np.arange(0,1,0.25),
        # "model__max_iter": [2000],
        }
        Gclf = GridSearchCV(pipe, param_grid=param_grid, scoring='neg_mean_squared_error', cv=CV_settings, refit=True, n_jobs=-1)
        
        features=Dict()
        # features.X=df_formant_statistic[feature_chooses]
        # features.y=df_formant_statistic[lab_chos_lst]
        
        # features.X=df_formant_statistic_added[feature_chooses]
        # features.y=df_formant_statistic_added[lab_chos_lst]
        
        features.X=df_feature_ASD[feature_chooses]
        features.y=df_feature_ASD[lab_chos_lst]
        
        
        
        # Score=cross_val_score(Gclf, features.X, features.y, cv=10)
        CVpredict=cross_val_predict(Gclf, features.X, features.y.values.ravel(), cv=CV_settings)  
        r2=r2_score(features.y,CVpredict )
        n,p=features.X.shape
        r2_adj=1-(1-r2)*(n-1)/(n-p-1)
        pearson_result, pearson_p=pearsonr(features.y.values.ravel(),CVpredict )
        spear_result, spearman_p=spearmanr(features.y.values.ravel(),CVpredict )
        
        
        feature_keys='+'.join(feature_chooses)
        print('Feature {0}, r2_result {1}, pearson_result {2} ,spear_result {3}'.format(feature_keys, r2_adj, pearson_result,spear_result))
        RESULT_dict[feature_keys]=[r2_adj,pearson_result,spear_result]


df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict,orient='index')
del Gclf
''' multiple regression model '''
# =============================================================================
# 
# =============================================================================

import statsmodels.api as sm
from itertools import combinations
feature_chos_lst=['between_covariance_norm(A:,i:,u:)',
'sam_wilks_lin_norm(A:,i:,u:)',
'hotelling_lin_norm(A:,i:,u:)',
'pillai_lin_norm(A:,i:,u:)']
baseline_lst=['FCR2']
comb2 = combinations(feature_chos_lst, 2)
print(list(comb2))

[('between_covariance_norm(A:,i:,u:)', 'sam_wilks_lin_norm(A:,i:,u:)'),
 ('between_covariance_norm(A:,i:,u:)', 'hotelling_lin_norm(A:,i:,u:)'),
 ('between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)'),
 ('sam_wilks_lin_norm(A:,i:,u:)', 'hotelling_lin_norm(A:,i:,u:)'),
 ('sam_wilks_lin_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)'),
 ('hotelling_lin_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)')]

['absAng_a','absAng_u','absAng_i', 'ang_ai','ang_iu','ang_ua']

# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)','Between_Within_Det_ratio_norm(A:,i:,u:)']]
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','dcorr_12']]
# X = df_formant_statistic[[ 'pillai_lin_norm(A:,i:,u:)','dcorr_12']]
# X = df_formant_statistic[['dcorr_12','dcov_12']]
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ai']]
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ua']] 
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ai','ang_ua']] 
# X = df_formant_statistic[['FCR2']] 
X = df_formant_statistic_added[['between_covariance_norm(A:,i:,u:)','localabsoluteJitter_mean(A:,i:,u:)','dcorr_12']]
y = df_formant_statistic_added[lab_chos_lst]
## fit a OLS model with intercept on TV and Radio
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
est.summary()

ypred = est.predict(X)
spear_result, spearman_p=spearmanr(y,ypred )
print('Manual test , spear_result {1}'.format(feature_keys,spear_result))



# =============================================================================
# 'between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)'
# r2_adj: 0.144   spear: 0.43





from scipy.stats import f_oneway
import itertools
''' Classification area '''
category_col=['ADOS_cate_C']
for cate in category_col:
    for col in columns:
        groups_compare=df_formant_statistic.groupby(cate)[col].apply(list)
        groups_compare_keys=df_formant_statistic.groupby(cate)[col].groups.keys()
        # f_oneway(groups_compare[0],groups_compare[1],groups_compare[2])
        result=f_oneway(*groups_compare)
        if result[-1] <= 0.05:
            print("anova result")
            print("Autisttic trait: ",cate, " Feature: ",col)
            print(result)
        
        
        
        # Two by two comparison
        # iteration_sel=list(itertools.combinations(groups_compare_keys, 2))
        # for key in iteration_sel:
        #     pair=[groups_compare[key[0]],groups_compare[key[1]]]
        #     # result=stats.ttest_ind(pair[0],pair[1])
        #     # if result[-1] <= 0.05:
        #     #     print("ttest result")
        #     #     print("Autisttic trait: ",cate, " Feature: ",col, "Between keys: ", key)
        #     #     print(result)
        #     result=stats.mannwhitneyu(pair[0],pair[1])
        #     if result[-1] <= 0.05:
        #         print("utest result")
        #         print("Autisttic trait: ",cate, " Feature: ",col, "Between keys: ", key)
        #         print(result)
            

# Plot
# import seaborn as sns
# sns.boxplot(x="dia_num", y="between_covariance_norm(A:,i:,u:)", data=df_formant_statistic)

# sns.boxplot(x="dia_num", y="sam_wilks_lin_norm(A:,i:,u:)", data=df_formant_statistic)

# sns.boxplot(x="dia_num", y="pillai_lin_norm(A:,i:,u:)", data=df_formant_statistic)





# =============================================================================
#  Pearson
#  between_covariance(A:,i:,u:)       -0.346864  ...         86.0
#  between_variance(A:,i:,u:)         -0.465486  ...         86.0




# =============================================================================



# =============================================================================
'''

    WorkSpace

'''
# =============================================================================
df_formant_statistic_toy=pd.DataFrame()
for people in SCATTER_matrixBookeep_dict.keys():
    Result_dict={}
    # Add basic information
    for label_choose in label_generate_choose_lst:
        Result_dict[label_choose]=Label.label_raw[label_choose][Label.label_raw['name']==people].values    
    Result_dict['u_num'], Result_dict['a_num'], Result_dict['i_num']=\
                len(Vowels_AUI[people]['u:']),len(Vowels_AUI[people]['A:']),len(Vowels_AUI[people]['i:'])
    # Matrix information
    Sctter_matrix_person=SCATTER_matrixBookeep_dict[people]
    for SCtr_max in Sctter_matrix_person.keys():
        prefix=SCtr_max+'_'
        Sctter_matrix=Sctter_matrix_person[SCtr_max]
        F1_var=Sctter_matrix[0][0]
        linear_coeff=Sctter_matrix[0][1]
        F2_var=Sctter_matrix[1][1]
        linear_coeff2=Sctter_matrix[1][0]
        
        Result_dict[prefix+'F1_var']=F1_var
        Result_dict[prefix+'linear_coeff']=linear_coeff
        Result_dict[prefix+'F2_var']=F2_var
        Result_dict[prefix+'linear_coeff2']=linear_coeff2
    # df_RESULT_list=pd.DataFrame.from_dict(Result_dict,orient='index').T
    df_RESULT_list=pd.DataFrame.from_dict(Result_dict)
    df_RESULT_list.index=[people]

    df_formant_statistic_toy=df_formant_statistic_toy.append(df_RESULT_list)

''' Calculate correlations for Formant fetures'''
columns=list(set(df_formant_statistic_toy.columns)  - set(additional_columns)) # Exclude added labels
columns=list(set(columns)  - set(['a_num','u_num','i_num']))

Aaad_Correlation_toy=Eval_med.Calculate_correlation(label_correlation_choose_lst,df_formant_statistic_toy,N,columns,constrain_sex=-1, constrain_module=-1,feature_type='Session_formant')
Survey_nice_variable(Aaad_Correlation_toy)



# =============================================================================

# Play code for convex hull area
# from scipy.spatial import ConvexHull, convex_hull_plot_2d
# import matplotlib.pyplot as plt

# for people in Vowels_AUI.keys(): #update 2021/05/27 fixed 
#     points=np.empty((0,2), float)
#     F12_raw_dict=Vowels_AUI[people]
#     for k, phone in F12_raw_dict.items():
#         data=phone[args.Inspect_features].copy()
#         # points=np.vstack((points,data.values))
#         points = np.append(points, data.values, axis=0)
#     hull = ConvexHull(points)
#     plt.plot(points[:,0], points[:,1], 'o')
#     for simplex in hull.simplices:

#         plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
# =====================
# Play code for Vowel vector dispersion
# import math
# Info_dict_total=Dict()
# for people in list(Vowels_AUI.keys()): #update 2021/05/27 fixed 
#     Info_dict=Dict()
    
#     points=np.empty((0,2), float)
#     F12_raw_dict=Vowels_AUI[people]
#     for k, phone in F12_raw_dict.items():
#         data=phone[args.Inspect_features].copy()
#         # points=np.vstack((points,data.values))
#         points = np.append(points, data.values, axis=0)
#     # Determine the f1 mean
#     F1m=np.mean(points[:,0])
#     # Determine the f2 mean
#     f2_criteria_bag=np.empty((0,2), float)
#     for V in points:
#         if V[0]<F1m:
#             f2_criteria_bag=np.append(f2_criteria_bag, V.reshape((1,-1)), axis=0)
#     F2m=np.mean(f2_criteria_bag[:,1])
    
#     center_coordinate=np.array([F1m,F2m])
#     Info_dict['center']=center_coordinate
#     Info_dict['vector'], Info_dict['scalar'], Info_dict['angle']=[], [], []
#     # Calculate the Vowel vector, angle and scale 
#     for i, V in enumerate(points):
#         VFDi=V-center_coordinate
#         Info_dict['vector'].append(VFDi)
#         Info_dict['scalar'].append(np.linalg.norm(VFDi))
        
#         # The math.atan2() method returns the arc tangent of y/x, and has take care the special conditions
#         omega=math.atan2(V[0]-center_coordinate[0], V[1]-center_coordinate[1])
    
#         Info_dict['angle'].append(omega)
#     Info_dict_total[people]=Info_dict

# count=0
# for people, Info_dict in Info_dict_total.items():
#     plt.figure(count)
#     plt.plot(Info_dict['center'][0], Info_dict['center'][1], 'o')
#     for vector in Info_dict['vector']:
#         # matplotlib.pyplot.arrow(x, y, dx, dy, **kwargs)
#         plt.arrow(Info_dict['center'][0],Info_dict['center'][1],vector[0],vector[1])
#     count+=1

# Play code for scatter plot with silhouette_samples score calculated
# import seaborn as sns
# import sklearn
# count=0
# for people in Vowels_AUI.keys():
#     plt.figure(count)
#     F12_raw_dict=Vowels_AUI[people]
#     df_vowel = pd.DataFrame()
#     for keys in F12_raw_dict.keys():
#         if len(df_vowel) == 0:
#             df_vowel=F12_raw_dict[keys]
#             df_vowel['vowel']=keys
#         else:
#             df_=F12_raw_dict[keys]
#             df_['vowel']=keys
#             df_vowel=df_vowel.append(df_)
    
#     X=df_vowel[args.Inspect_features]
#     labels=df_vowel['vowel']
#     sklearn.metrics.silhouette_samples(X, labels,  metric='euclidean')
#     sklearn.metrics.calinski_harabasz_score(X, labels)
#     sns.scatterplot(data=df_vowel, x="F1", y="F2", hue="vowel")
#     count+=1
# =====================
    



# Play code for relative angles
# import seaborn as sns
# import dcor
# import math
# import matplotlib.patheffects as pe
# from matplotlib.offsetbox import AnchoredText

# def cosAngle(a, b, c):
#     # angles between line segments (Python) from https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
#     cosine_angle = np.dot((b-a), (b-c)) / (np.linalg.norm((b-a)) * np.linalg.norm((b-c)))
#     angle = np.arccos(cosine_angle)
#     return np.degrees(angle)

# count=0
# for people in list(Vowels_AUI.keys())[:10]:
#     plt.figure(count)
#     F12_raw_dict=Vowels_AUI[people]
#     df_vowel = pd.DataFrame()
#     for keys in F12_raw_dict.keys():
#         if len(df_vowel) == 0:
#             df_vowel=F12_raw_dict[keys]
#             df_vowel['vowel']=keys
#         else:
#             df_=F12_raw_dict[keys]
#             df_['vowel']=keys
#             df_vowel=df_vowel.append(df_)
    
    
#     def Calculate_raelative_angles(df_vowel, additional_infos=False):
        
#         a=df_vowel[df_vowel['vowel']=='A:'][args.Inspect_features]
#         u=df_vowel[df_vowel['vowel']=='u:'][args.Inspect_features]
#         i=df_vowel[df_vowel['vowel']=='i:'][args.Inspect_features]
        
#         a_center=a.mean()
#         u_center=u.mean()
#         i_center=i.mean()
#         total_center=df_vowel.mean()
#         # gravity_center=(a_center*len(a) + u_center*len(u) + i_center*len(i)) / len(df_vowel)
        
        
        
#         omega_a=np.degrees(math.atan2((a_center - total_center)[1], (a_center - total_center)[0]))
#         omega_u=np.degrees(math.atan2((u_center - total_center)[1], (u_center - total_center)[0]))
#         omega_i=np.degrees(math.atan2((i_center - total_center)[1], (i_center - total_center)[0]))
    
#         ang_ai = cosAngle(a_center,total_center,i_center)
#         ang_iu = cosAngle(i_center,total_center,u_center)
#         ang_ua = cosAngle(u_center,total_center,a_center)
        
#         absolute_ang=[omega_a, omega_u, omega_i]
#         relative_ang=[ang_ai, ang_iu, ang_ua]
#         addition_info=[total_center, a_center, u_center, i_center]
        
#         if additional_infos != True:
#             return absolute_ang, relative_ang
#         else:
#             return absolute_ang, relative_ang, addition_info
    
    
#     absolute_ang, relative_ang, addition_info = Calculate_raelative_angles(df_vowel, additional_infos=True)
#     [omega_a, omega_u, omega_i]=absolute_ang
#     [ang_ai, ang_iu, ang_ua]=relative_ang
#     [total_center, a_center, u_center, i_center]=addition_info
#     print(sum([ang_ai,ang_iu,ang_ua]))
    
#     # omega_a_deg=omega_a*(180/np.pi)
    
#     center2plot=total_center
#     plt.plot(center2plot[0],center2plot[1],'*',markersize=30)
#     # dx,dy=(a_center - center2plot)
#     # origin=center2plot
#     # dest=center2plot+[dx,dy]
    
    
    
#     plt.plot([center2plot[0],center2plot[0] + (a_center - center2plot)[0]],[center2plot[1],center2plot[1] + (a_center - center2plot)[1]],'ro-', color="green", linewidth=3,\
#               path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])
#     plt.plot([center2plot[0],center2plot[0] + (u_center - center2plot)[0]],[center2plot[1],center2plot[1] + (u_center - center2plot)[1]],'ro-', color="blue", linewidth=3,\
#               path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])
#     plt.plot([center2plot[0],center2plot[0] + (i_center - center2plot)[0]],[center2plot[1],center2plot[1] + (i_center - center2plot)[1]],'ro-', color="orange", linewidth=3,\
#               path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])
#     sns.scatterplot(data=df_vowel, x="F1", y="F2", hue="vowel")
    
#     at = AnchoredText(
#     "angles\na:{0}\nu:{1}\ni:{2}".format(np.round(omega_a,2),np.round(omega_u,2),np.round(omega_i,2)), prop=dict(size=15), frameon=True, loc='lower right')
#     # at = AnchoredText(
#     # "angles\nai{0}\niu{1}\nua{2}".format(np.round(ang_ai,2),np.round(ang_iu,2),np.round(ang_ua,2)), prop=dict(size=15), frameon=True, loc='lower right')
#     plt.setp(at.patch, facecolor='white', alpha=0.5)
#     plt.gca().add_artist(at)
    
#     plt.show()
#     count+=1



# Play code for distance covariance
# import dcor
# import math
# from scipy.stats import spearmanr,pearsonr 
# import seaborn as sns
# from matplotlib.offsetbox import AnchoredText
# count=0
# for people in list(Vowels_AUI.keys())[:]:
#     plt.figure(count)
#     F12_raw_dict=Vowels_AUI[people]
#     df_vowel = pd.DataFrame()
#     for keys in F12_raw_dict.keys():
#         if len(df_vowel) == 0:
#             df_vowel=F12_raw_dict[keys]
#             df_vowel['vowel']=keys
#         else:
#             df_=F12_raw_dict[keys]
#             df_['vowel']=keys
#             df_vowel=df_vowel.append(df_)
    
#     def Calculate_distanceCorr(df_vowel):
#         a=df_vowel[df_vowel['vowel']=='A:'][args.Inspect_features]
#         u=df_vowel[df_vowel['vowel']=='u:'][args.Inspect_features]
#         i=df_vowel[df_vowel['vowel']=='i:'][args.Inspect_features]
        
        
#         a_1, a_2=a['F1'], a['F2']
#         u_1, u_2=u['F1'], u['F2']
#         i_1, i_2=i['F1'], i['F2']
        
#         d_stats_a=dcor.distance_stats(a_1, a_2)
#         d_stats_u=dcor.distance_stats(u_1, u_2)
#         d_stats_i=dcor.distance_stats(i_1, i_2)
        
#         pear_a=pearsonr(a_1, a_2)[0]
#         pear_u=pearsonr(u_1, u_2)[0]
#         pear_i=pearsonr(i_1, i_2)[0]
        
#         def get_values(X_stats):
#             cov_xy, corr_xy, var_x, var_y=X_stats
#             return cov_xy, corr_xy, var_x, var_y
        
        
#         data_a = get_values(d_stats_a)
#         data_u = get_values(d_stats_u)
#         data_i = get_values(d_stats_i)
        
#         Cov_sum=[sum(x) for x in zip(data_a,data_u,data_i)]
#         Corr_aui=[data_a[1],data_u[1],data_i[1]]

#         pear_sum=sum([pear_a,pear_u,pear_i])
#         return Cov_sum, Corr_aui, pear_sum
#     ADOS_lab=df_formant_statistic[df_formant_statistic.index==people]['ADOS_C'].values[0]
#     [a,b,c,d], [dcor_a,dcor_u,dcor_i] ,pearsum=Calculate_distanceCorr(df_vowel)
#     sns.scatterplot(data=df_vowel, x="F1", y="F2", hue="vowel")
#     at = AnchoredText(
#     "dcorr:{0}\nADOS:{1}".format(np.round(b,2),np.round(ADOS_lab,2)), prop=dict(size=15), frameon=True, loc='lower right')
#     plt.setp(at.patch, facecolor='white', alpha=0.5)
#     plt.gca().add_artist(at)
    
#     plt.show()
#     count+=1


# Play code for scipy c distance
# import scipy
# count=0
# for people in list(Vowels_AUI.keys())[:10]:
#     # plt.figure(count)
#     F12_raw_dict=Vowels_AUI[people]
#     df_vowel = pd.DataFrame()
#     for keys in F12_raw_dict.keys():
#         if len(df_vowel) == 0:
#             df_vowel=F12_raw_dict[keys]
#             df_vowel['vowel']=keys
#         else:
#             df_=F12_raw_dict[keys]
#             df_['vowel']=keys
#             df_vowel=df_vowel.append(df_)
    
#     def Calculate_pointDistsTotal(df_vowel):
        # a=df_vowel[df_vowel['vowel']=='A:'][args.Inspect_features]
        # u=df_vowel[df_vowel['vowel']=='u:'][args.Inspect_features]
        # i=df_vowel[df_vowel['vowel']=='i:'][args.Inspect_features]
        
#         dist_au=scipy.spatial.distance.cdist(a,u)
#         dist_ai=scipy.spatial.distance.cdist(a,i)
#         dist_iu=scipy.spatial.distance.cdist(i,u)
#         mean_dist_au=np.mean(dist_au)
#         mean_dist_ai=np.mean(dist_ai)
#         mean_dist_iu=np.mean(dist_iu)
#         dist_total=mean_dist_au+mean_dist_ai+mean_dist_iu
#         return dist_total
#     dist_tot=Calculate_pointDistsTotal(df_vowel)


# play code for betweenClusters_distrib_dist from Li-Min Wang
# import scipy
# count=0
# for people in list(Vowels_AUI.keys())[:10]:
#     # plt.figure(count)
#     F12_raw_dict=Vowels_AUI[people]
#     df_vowel = pd.DataFrame()
#     for keys in F12_raw_dict.keys():
#         if len(df_vowel) == 0:
#             df_vowel=F12_raw_dict[keys]
#             df_vowel['vowel']=keys
#         else:
#             df_=F12_raw_dict[keys]
#             df_['vowel']=keys
#             df_vowel=df_vowel.append(df_)
            
    # a=df_vowel[df_vowel['vowel']=='A:'][args.Inspect_features]
    # u=df_vowel[df_vowel['vowel']=='u:'][args.Inspect_features]
    # i=df_vowel[df_vowel['vowel']=='i:'][args.Inspect_features]

#     def calculate_betweenClusters_distrib_dist(df_vowel):
#         def Within_matrices(df_vowel):
#             ''' Not used'''
#             class_feature_means = pd.DataFrame(columns=list(set(df_vowel['vowel'])))
#             n_samples = len(df_vowel)

#             for c, rows in df_vowel.groupby('vowel'):
#                 class_feature_means[c] = rows[args.Inspect_features].mean()
            
#             groups_num=len(class_feature_means.index)
#             # Within class scatter matrix 
#             within_class_scatter_matrix = np.zeros((groups_num,groups_num))
#             for c, rows in df_vowel.groupby('vowel'):
#                 rows = rows[args.Inspect_features]
#                 s = np.zeros((groups_num,groups_num))
                                
#                 for index, row in rows.iterrows():
#                     x, mc = row.values.reshape(groups_num,1), class_feature_means[c].values.reshape(groups_num,1)
#                     class_variance=(x - mc).dot((x - mc).T).astype(float)
#                     s += class_variance
                    
#                 within_class_scatter_matrix += s
#             within_class_scatter_matrix_norm = within_class_scatter_matrix / n_samples
        
#             return within_class_scatter_matrix_norm
#         def calculate_stdev(df_vowel_phone):
#             ''' Not used'''
#             u_scatter_mtrx_norm=Within_matrices(df_vowel_phone)
#             eigen_values, _ = np.linalg.eig(u_scatter_mtrx_norm)
#             stdev=np.sqrt(sum(eigen_values))
#             return stdev
        
#         def calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel', vowel1='A:',\
#                                                vowel2='u:', Inspect_features=['F1', 'F2']):
#             v1=df_vowel[df_vowel[vowelCol_name]==vowel1][args.Inspect_features]
#             v2=df_vowel[df_vowel[vowelCol_name]==vowel2][args.Inspect_features]
            
#             mean_dist_v1v2=np.sum(scipy.spatial.distance.cdist(v1,v2.mean().values.reshape(1,-1)))
#             mean_dist_v2=np.sum(scipy.spatial.distance.cdist(v2,v2.mean().values.reshape(1,-1)))
#             dristrib_dist_v1v2= mean_dist_v2/mean_dist_v1v2
            
#             # stdev_v2=calculate_stdev(df_vowel[df_vowel[vowelCol_name]==vowel1])
#             # dristrib_dist_v1v2= stdev_v2/mean_dist_v1v2
            
#             mean_dist_v2v1=np.mean(scipy.spatial.distance.cdist(v2,v1.mean().values.reshape(1,-1)))
#             mean_dist_v1=np.mean(scipy.spatial.distance.cdist(v1,v1.mean().values.reshape(1,-1)))
#             dristrib_dist_v2v1= mean_dist_v1/mean_dist_v2v1
#             # stdev_v1=calculate_stdev(df_vowel[df_vowel[vowelCol_name]==vowel2])
#             # dristrib_dist_v2v1= stdev_v1/mean_dist_v2v1
            
#             dristrib_dist=dristrib_dist_v1v2 * dristrib_dist_v2v1
#             return dristrib_dist
        
#         dristrib_dist_au= calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel', vowel1='A:',\
#                                            vowel2='u:', Inspect_features=args.Inspect_features)
#         dristrib_dist_ui= calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel', vowel1='u:',\
#                                            vowel2='i:', Inspect_features=args.Inspect_features)
#         dristrib_dist_ai= calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel', vowel1='A:',\
#                                            vowel2='i:', Inspect_features=args.Inspect_features)
#         return sum(dristrib_dist_au,dristrib_dist_ui,dristrib_dist_ai)
    
# play code for repulsive force
# count=0
# import scipy
# for people in list(Vowels_AUI.keys())[:10]:
#     # plt.figure(count)
#     F12_raw_dict=Vowels_AUI[people]
#     df_vowel = pd.DataFrame()
#     for keys in F12_raw_dict.keys():
#         if len(df_vowel) == 0:
#             df_vowel=F12_raw_dict[keys]
#             df_vowel['vowel']=keys
#         else:
#             df_=F12_raw_dict[keys]
#             df_['vowel']=keys
#             df_vowel=df_vowel.append(df_)
    
    
#     def calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel'):
#         repuls_forc_inst_bag=[]
#         for index, row in df_vowel.iterrows():
#             phone=row[vowelCol_name]
#             formant_values=row[args.Inspect_features]
#             other_phones=df_vowel[df_vowel[vowelCol_name]!=phone]
#             other_phones_values=other_phones[args.Inspect_features]
            
#             repuls_forc_inst=np.mean(1/scipy.spatial.distance.cdist(other_phones_values,formant_values.values.reshape(1,-1)))
#             repuls_forc_inst_bag.append(repuls_forc_inst)
#         assert len(repuls_forc_inst_bag) == len(df_vowel)
#         return np.mean(repuls_forc_inst_bag)
#     repulsive_force_norm=calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel')
#     print(repulsive_force_norm)



# Play code for inner and outer 
# from scipy.spatial import distance
# Vowels_AUI_inner=Dict()
# Vowels_AUI_outer=Dict()
# for people in tqdm(list(Vowels_AUI.keys())):
#     # plt.figure(count)
#     F12_raw_dict=Vowels_AUI[people]
#     df_vowel = pd.DataFrame()
#     for keys in F12_raw_dict.keys():
#         if len(df_vowel) == 0:
#             df_vowel=F12_raw_dict[keys]
#             df_vowel['vowel']=keys
#         else:
#             df_=F12_raw_dict[keys]
#             df_['vowel']=keys
#             df_vowel=df_vowel.append(df_)
    
#     gravity_mean = df_vowel.mean() #Total mean
#     #Class mean
#     class_feature_means = pd.DataFrame(columns=list(set(df_vowel['vowel'])))
#     for c, rows in df_vowel.groupby('vowel'):
#         class_feature_means[c] = rows[args.Inspect_features].mean()
#     groups_num=len(class_feature_means.index)
    
#     class_feature_means_reshape=class_feature_means.T
#     class_feature_means_reshape.loc['gravity_mean']=gravity_mean

#     a=df_vowel[df_vowel['vowel']=='A:'][args.Inspect_features]
#     u=df_vowel[df_vowel['vowel']=='u:'][args.Inspect_features]
#     i=df_vowel[df_vowel['vowel']=='i:'][args.Inspect_features]
    
#     # a['distToClassMean']=np.linalg.norm(a[args.Inspect_features].values-class_feature_means_reshape[args.Inspect_features].values, axis=1)
#     distance_matrix=distance.cdist(df_vowel[args.Inspect_features],class_feature_means_reshape[args.Inspect_features])
    
#     df_distance_matrix=pd.DataFrame(distance_matrix,columns=class_feature_means_reshape.index, index=df_vowel['vowel'])
#     df_distance_matrix_forCheck=pd.DataFrame(distance_matrix,columns=class_feature_means_reshape.index, index=[df_vowel.index])
    
#     # unit test (check by eye)
#     df_unittest=pd.DataFrame()
#     for index, row in df_vowel.iterrows():
#         needed_value=np.linalg.norm(row[args.Inspect_features].values-class_feature_means_reshape.loc[row['vowel']][args.Inspect_features].values)
#         df_needed_value=pd.DataFrame(needed_value,columns=[row['vowel']],index=[index])
#         df_unittest=df_unittest.append(df_needed_value)
#     df_unittest.index=df_vowel.index
    
#     check_phone=df_distance_matrix.index[0]
#     findphone=''
#     phone_found_bool=False
#     for phone, values in PhoneMapp_dict.items():
#         if check_phone in values:
#             findphone=check_phone
#             phone_found_bool
    
#     df_vowel_inner=pd.DataFrame()
#     df_vowel_outer=pd.DataFrame()
#     for P in PhoneOfInterest:
#         if P not in df_distance_matrix.index:
#             Vowels_AUI_inner[people][P]=pd.DataFrame([],columns=args.Inspect_features)
#             Vowels_AUI_outer[people][P]=pd.DataFrame([],columns=args.Inspect_features)
#             continue
#         VowelInner_bool=np.where(df_distance_matrix[P] >= df_distance_matrix["gravity_mean"], True, False)
#         VowelOuter_bool=np.where(df_distance_matrix[P] > df_distance_matrix["gravity_mean"], True, False)
        
#         VowelBool=df_distance_matrix.index.values==P
        
#         Bool_innerAndvowel=np.logical_and(VowelInner_bool,VowelBool)
#         Bool_outerAndvowel=np.logical_and(VowelOuter_bool,VowelBool)
        

#         df_vowel_inner=df_vowel_inner.append(df_vowel[Bool_innerAndvowel])
#         df_vowel_outer=df_vowel_outer.append(df_vowel[Bool_outerAndvowel])
        
#         Vowels_AUI_inner[people][P]=df_vowel[Bool_innerAndvowel]
#         Vowels_AUI_outer[people][P]=df_vowel[Bool_outerAndvowel]

# df_formant_statistic_inner=articulation.calculate_features(Vowels_AUI_inner,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_generate_choose_lst)
# df_formant_statistic_outer=articulation.calculate_features(Vowels_AUI_outer,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_generate_choose_lst)


# columns=['between_variance_norm(A:,i:,u:)',
#  'Minor_vector_W_norm(A:,i:,u:)',
#  'total_covariance_norm(A:,i:,u:)',
#  'within_variance_norm(A:,i:,u:)',
#  'between_covariance_norm(A:,i:,u:)',
#  'within_covariance_norm(A:,i:,u:)',
#  'hotelling_lin_norm(A:,i:,u:)',
#  'Minor_vector_lin_norm(A:,i:,u:)',
#  'Major_vector_B_norm(A:,i:,u:)',
#  'Major_vector_W_norm(A:,i:,u:)',
#  'roys_root_lin_norm(A:,i:,u:)',
#  'total_variance_norm(A:,i:,u:)',
#  'Major_vector_lin_norm(A:,i:,u:)',
#  'Minor_vector_B_norm(A:,i:,u:)',
#  'sam_wilks_lin_norm(A:,i:,u:)',
#  'pillai_lin_norm(A:,i:,u:)',
#  'VSA1',
#  'FCR',
#  'ConvexHull',
#  'MeanVFD',
#  'VSA2',
#  'FCR2',
#  'absAng_a',
#  'absAng_u',
#  'absAng_i',
#   'ang_ai',
#  # 'ang_iu',
#  # 'ang_ua',
#  # 'Angles',
#  'pointDistsTotal',
#  'repulsive_force',
#  ]

# N=2
# Eval_med=Evaluation_method()
# Aaadf_spearmanr_table_outer=Eval_med.Calculate_correlation(label_correlation_choose_lst,df_formant_statistic_outer,N,columns,constrain_sex=-1, constrain_module=-1,feature_type='Session_formant')
# Aaadf_spearmanr_table_inner=Eval_med.Calculate_correlation(label_correlation_choose_lst,df_formant_statistic_inner,N,columns,constrain_sex=-1, constrain_module=-1,feature_type='Session_formant')

# def LDA_LevelOfClustering_feats(df_vowel):
#     '''    Calculate class variance by LDA. Vowel space features are in this function 
    
#             Suffix "_norm"" represents normalized matrix or scalar
#     '''  
#     within_class_scatter_matrix, between_class_scatter_matrix,\
#             within_class_scatter_matrix_norm, between_class_scatter_matrix_norm, linear_discriminant_norm, Total_scatter_matrix_norm = LDA_scatter_matrices(df_vowel)
    
#     # eigen_values_lin, eigen_vectors_lin = np.linalg.eig(linear_discriminant)
#     eigen_values_lin_norm, eigen_vectors_lin_norm = np.linalg.eig(linear_discriminant_norm)
#     eigen_values_B, eigen_vectors_B = np.linalg.eig(between_class_scatter_matrix)
#     eigen_values_B_norm, eigen_vectors_B_norm = np.linalg.eig(between_class_scatter_matrix_norm)
#     eigen_values_W, eigen_vectors_W = np.linalg.eig(within_class_scatter_matrix)
#     eigen_values_W_norm, eigen_vectors_W_norm = np.linalg.eig(within_class_scatter_matrix_norm)
#     # eigen_values_T, eigen_vectors_T = np.linalg.eig(Total_scatter_matrix)
#     eigen_values_T_norm, eigen_vectors_T_norm = np.linalg.eig(Total_scatter_matrix_norm)

    
#     def Covariance_representations(eigen_values):
#         sam_wilks=1
#         pillai=0
#         hotelling=0
#         for eigen_v in eigen_values:
#             wild_element=1.0/np.float(1+eigen_v)
#             sam_wilks*=wild_element
#             pillai+=wild_element * eigen_v
#             hotelling+=eigen_v
#         roys_root=np.max(eigen_values)
#         return sam_wilks, pillai, hotelling, roys_root
#     Covariances={}
#     # Covariances['sam_wilks_lin'], Covariances['pillai_lin'], Covariances['hotelling_lin'], Covariances['roys_root_lin'] = Covariance_representations(eigen_values_lin)
#     Covariances['sam_wilks_lin_norm'], Covariances['pillai_lin_norm'], Covariances['hotelling_lin_norm'], Covariances['roys_root_lin_norm'] = Covariance_representations(eigen_values_lin_norm)
#     # Covariances['sam_wilks_B'], Covariances['pillai_B'], Covariances['hotelling_B'], Covariances['roys_root_B'] = Covariance_representations(eigen_values_B)
#     # Covariances['sam_wilks_Bnorm'], Covariances['pillai_Bnorm'], Covariances['hotelling_Bnorm'], Covariances['roys_root_Bnorm'] = Covariance_representations(eigen_values_B_norm)
#     # Covariances['sam_wilks_W'], Covariances['pillai_W'], Covariances['hotelling_W'], Covariances['roys_root_W'] = Covariance_representations(eigen_values_W)

    
#     Multi_Variances={}
#     Multi_Variances['between_covariance_norm'] = np.prod(eigen_values_B_norm)# product of every element
#     Multi_Variances['between_variance_norm'] = np.sum(eigen_values_B_norm)
#     # Multi_Variances['between_covariance'] = np.prod(eigen_values_B)# product of every element
#     # Multi_Variances['between_variance'] = np.sum(eigen_values_B)
#     Multi_Variances['within_covariance_norm'] = np.prod(eigen_values_W_norm)
#     Multi_Variances['within_variance_norm'] = np.sum(eigen_values_W_norm)
#     # Multi_Variances['within_covariance'] = np.prod(eigen_values_W)
#     # Multi_Variances['within_variance'] = np.sum(eigen_values_W)
#     Multi_Variances['total_covariance_norm'] = np.prod(eigen_values_T_norm)
#     Multi_Variances['total_variance_norm'] = np.sum(eigen_values_T_norm)
#     # Multi_Variances['total_covariance'] = np.prod(eigen_values_T)
#     # Multi_Variances['total_variance'] = np.sum(eigen_values_T)
#     Covariances['Between_Within_Det_ratio_norm'] = Multi_Variances['between_covariance_norm'] / Multi_Variances['within_covariance_norm']
#     Covariances['Between_Within_Tr_ratio_norm'] = Multi_Variances['between_variance_norm'] / Multi_Variances['within_variance_norm']
#     return Covariances, Multi_Variances

# def Store_FeatVals(RESULT_dict,df_vowel,Inspect_features=['F1','F2'], cluster_str='u:,i:,A:'):
#     Covariances, Multi_Variances\
#         =LDA_LevelOfClustering_feats(df_vowel[Inspect_features+['vowel']])

#     # RESULT_dict['between_covariance({0})'.format(cluster_str)]=between_covariance
#     # RESULT_dict['between_variance({0})'.format(cluster_str)]=between_variance
#     # RESULT_dict['between_covariance_norm({0})'.format(cluster_str)]=between_covariance_norm
#     # RESULT_dict['between_variance_norm({0})'.format(cluster_str)]=between_variance_norm
#     # RESULT_dict['within_covariance({0})'.format(cluster_str)]=within_covariance
#     # RESULT_dict['within_variance({0})'.format(cluster_str)]=within_variance
#     # RESULT_dict['linear_discriminant_covariance({0})'.format(cluster_str)]=linear_discriminant_covariance
    
#     # for keys, values in Single_Variances.items():
#     #     RESULT_dict[keys+'({0})'.format(cluster_str)]=values
#     for keys, values in Multi_Variances.items():
#         RESULT_dict[keys+'({0})'.format(cluster_str)]=values
#     for keys, values in Covariances.items():
#         RESULT_dict[keys+'({0})'.format(cluster_str)]=values
#     return RESULT_dict


# Play code for KDE filtering
# count=0
# from sklearn.neighbors import KernelDensity
# from sklearn import preprocessing
# from matplotlib.offsetbox import AnchoredText
# THRESHOLD=40
# for THRESHOLD in [40]:
#     scale_factor=100
#     N=2
#     RESULT_DICTIONARY=Dict()
#     df_simulate=pd.DataFrame()
#     # for people in list(Vowels_AUI.keys())[:3]:
#     for people in Vowels_AUI.keys():
#         # plt.figure(count)
#         F12_raw_dict=Vowels_AUI[people]
#         df_vowel = pd.DataFrame()
#         for keys in F12_raw_dict.keys():
#             if len(df_vowel) == 0:
#                 df_vowel=F12_raw_dict[keys]
#                 df_vowel['vowel']=keys
#             else:
#                 df_=F12_raw_dict[keys]
#                 df_['vowel']=keys
#                 df_vowel=df_vowel.append(df_)
        
#         len_a=len(np.where(df_vowel['vowel']=='A:')[0])
#         len_u=len(np.where(df_vowel['vowel']=='u:')[0])
#         len_i=len(np.where(df_vowel['vowel']=='i:')[0])
        
        
#         if len_a<=N or len_u<=N or len_i<=N:
#             continue
        
#         def KDE_Filtering(df_vowel,THRESHOLD=10,scale_factor=100):
#             X=df_vowel[args.Inspect_features].values
#             labels=df_vowel['vowel']
            
#             df_vowel_calibrated=pd.DataFrame([])
#             for phone in set(labels):
                
#                 df=df_vowel[df_vowel['vowel']==phone][args.Inspect_features]
#                 data_array=df_vowel[df_vowel['vowel']==phone][args.Inspect_features].values
    
#                 x=data_array[:,0]
#                 y=data_array[:,1]
#                 xmin = x.min()
#                 xmax = x.max()        
#                 ymin = y.min()
#                 ymax = y.max()
                
#                 image_num=1j
#                 X, Y = np.mgrid[xmin:xmax:image_num*scale_factor, ymin:ymax:image_num*scale_factor]
                
#                 positions = np.vstack([X.ravel(), Y.ravel()])
                
#                 values = np.vstack([x, y])
                
#                 kernel = stats.gaussian_kde(values)
                        
#                 Z = np.reshape(kernel(positions).T, X.shape)
#                 normalized_z = preprocessing.normalize(Z)
                
#                 df['x_to_scale'] = (100*(x - np.min(x))/np.ptp(x)).astype(int) 
#                 df['y_to_scale'] = (100*(y - np.min(y))/np.ptp(y)).astype(int) 
                
#                 normalized_z=(100*(Z - np.min(Z.ravel()))/np.ptp(Z.ravel())).astype(int)
#                 to_delete = zip(*np.where((normalized_z<THRESHOLD) == True))
                
#                 # The indexes that are smaller than threshold
#                 deletepoints_bool=df.apply(lambda x: (x['x_to_scale'], x['y_to_scale']), axis=1).isin(to_delete)
#                 df_calibrated=df.loc[(deletepoints_bool==False).values]
#                 df_deleted_after_calibrated=df.loc[(deletepoints_bool==True).values]
                
#                 df_vowel_calibrated_tmp=df_calibrated.drop(columns=['x_to_scale','y_to_scale'])
#                 df_vowel_calibrated_tmp['vowel']=phone
#                 df_vowel_output=df_vowel_calibrated_tmp.copy()
#                 df_vowel_calibrated=df_vowel_calibrated.append(df_vowel_output)
                
                
#                 # Data prepare for plotting 
#                 # import seaborn as sns
#                 # df_calibrated_tocombine=df_calibrated.copy()
#                 # df_calibrated_tocombine['cal']='calibrated'
#                 # df_deleted_after_calibrated['cal']='deleted'
#                 # df_calibratedcombined=df_calibrated_tocombine.append(df_deleted_after_calibrated)
                
#                 # #Plotting code
#                 # fig = plt.figure(figsize=(8,8))
#                 # ax = fig.gca()
#                 # ax.set_xlim(xmin, xmax)
#                 # ax.set_ylim(ymin, ymax)
#                 # # cfset = ax.contourf(X, Y, Z, cmap='coolwarm')
#                 # # ax.imshow(Z, cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
#                 # # cset = ax.contour(X, Y, Z, colors='k')
#                 # cfset = ax.contourf(X, Y, normalized_z, cmap='coolwarm')
#                 # ax.imshow(normalized_z, cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
#                 # cset = ax.contour(X, Y, normalized_z, colors='k')
#                 # ax.clabel(cset, inline=1, fontsize=10)
#                 # ax.set_xlabel('X')
#                 # ax.set_ylabel('Y')
#                 # plt.title('2D Gaussian Kernel density estimation')
                
#                 # sns.scatterplot(data=df_vowel[df_vowel['vowel']==phone], x="F1", y="F2")
#                 # sns.scatterplot(data=df_calibratedcombined, x="F1", y="F2",hue='cal')
#             return df_vowel_calibrated
        
        
#         def Calculate_pointDistsTotal(df_vowel,dist_type='euclidean'):
#             import scipy
#             a=df_vowel[df_vowel['vowel']=='A:'][args.Inspect_features]
#             u=df_vowel[df_vowel['vowel']=='u:'][args.Inspect_features]
#             i=df_vowel[df_vowel['vowel']=='i:'][args.Inspect_features]
            
            
            
#             dist_au=scipy.spatial.distance.cdist(a,u,dist_type)
#             dist_ai=scipy.spatial.distance.cdist(a,i,dist_type)
#             dist_iu=scipy.spatial.distance.cdist(i,u,dist_type)
#             mean_dist_au=np.mean(dist_au)
#             mean_dist_ai=np.mean(dist_ai)
#             mean_dist_iu=np.mean(dist_iu)
#             dist_total=mean_dist_au*mean_dist_ai*mean_dist_iu
#             # return dist_total, [mean_dist_au,mean_dist_ai,mean_dist_iu]
#             return dist_total
        
        
#         df_vowel_calibrated=KDE_Filtering(df_vowel,THRESHOLD=THRESHOLD,scale_factor=100)
    
#         a=df_vowel_calibrated[df_vowel_calibrated['vowel']=='A:'][args.Inspect_features].mean()
#         u=df_vowel_calibrated[df_vowel_calibrated['vowel']=='u:'][args.Inspect_features].mean()
#         i=df_vowel_calibrated[df_vowel_calibrated['vowel']=='i:'][args.Inspect_features].mean()
    
#         numerator=u[1] + a[1] + i[0] + u[0]
#         demominator=i[1] + a[0]
#         RESULT_dict={}
#         RESULT_dict['FCR2']=np.float(numerator/demominator)
#         # RESULT_dict['FCR2_uF2']=np.float(u[1]/demominator)
#         # RESULT_dict['FCR2_aF2']=np.float(a[1]/demominator)
#         # RESULT_dict['FCR2_iF1']=np.float(i[0]/demominator)
#         # RESULT_dict['FCR2_uF1']=np.float(u[0]/demominator)
        
#         #Get total between cluster distances
#         # distance_types=['euclidean','minkowski','cityblock','seuclidean','sqeuclidean','cosine',\
#         #      'correlation','jaccard','jensenshannon','chebyshev','canberra','braycurtis',\
#         #      'mahalanobis','sokalsneath']
            
#         # for dst_tpe in distance_types:
#         #     # RESULT_dict[dst_tpe+" "+'pointDistsTotal'],\
#         #     #     [RESULT_dict[dst_tpe+" "+'dist_au'],RESULT_dict[dst_tpe+" "+'dist_ai'],RESULT_dict[dst_tpe+" "+'dist_iu']]\
#         #     #         =Calculate_pointDistsTotal(df_vowel_calibrated,dist_type=dst_tpe)
#         #     RESULT_dict[dst_tpe+" "+'pointDistsTotal']=Calculate_pointDistsTotal(df_vowel_calibrated,dist_type=dst_tpe)
#         #     RESULT_dict[dst_tpe+" "+'pointDistsTotal_norm']=Calculate_pointDistsTotal(df_vowel_calibrated,dist_type=dst_tpe) / numerator
        
        
#         for label_choose in label_choose_lst:
#             RESULT_dict[label_choose]=Label.label_raw[label_choose][Label.label_raw['name']==people].values    
#         RESULT_dict['u_num'], RESULT_dict['a_num'], RESULT_dict['i_num']=len_u,len_a,len_i
        
#         cluster_str=','.join(sorted(F12_raw_dict.keys()))
#         RESULT_dict=Store_FeatVals(RESULT_dict,df_vowel_calibrated,args.Inspect_features, cluster_str=cluster_str)    
        
#         ''' End of feature calculation '''
#         # =============================================================================
#         df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict)
#         df_RESULT_list.index=[people]
#         df_simulate=df_simulate.append(df_RESULT_list)
#         # sns.scatterplot(data=df_vowel_calibrated, x="F1", y="F2", hue="vowel")
        
#         # at = AnchoredText(
#         # "FCR2:{0}\nuF2{1}\naF2{2}\niF1{3}\nuF1{4}".format(\
#         #     np.round(RESULT_dict['FCR2'],2),np.round(RESULT_dict['FCR2_uF2'],2),np.round(RESULT_dict['FCR2_aF2'],2),\
#         #     np.round(RESULT_dict['FCR2_iF1'],2),np.round(RESULT_dict['FCR2_uF1'],2)),prop=dict(size=15), frameon=True, loc='lower right')
#         # # at = AnchoredText(
#         # # "angles\nai{0}\niu{1}\nua{2}".format(np.round(ang_ai,2),np.round(ang_iu,2),np.round(ang_ua,2)), prop=dict(size=15), frameon=True, loc='lower right')
#         # plt.setp(at.patch, facecolor='white', alpha=0.5)
#         # plt.gca().add_artist(at)
#         # plt.show()
#         count+=1
        
#     Eval_med=Evaluation_method()
    
#     columns_sel=df_simulate.columns
#     # columns_sel=list(set(columns_sel) - set([co for co in columns_sel if " pointDistsTotal" not in co]))
#     columns_sel=list(set(columns_sel) - set(['u_num','a_num','i_num','ADOS_C']))
#     # columns_sel= columns_sel + ['FCR2']
    
    
#     for N in [2]:
#     # for N in [7,8,9,10]:
#         Aaadf_results=Eval_med.Calculate_correlation(['ADOS_C'],df_simulate,N,columns_sel,constrain_sex=-1, constrain_module=-1,feature_type='Session_formant')
#         Survey_nice_variable(Aaadf_results)

    


# def LDA_scatter_matrices(df_vowel):
#     a=df_vowel[df_vowel['vowel']=='A:'][args.Inspect_features].mean()
#     u=df_vowel[df_vowel['vowel']=='u:'][args.Inspect_features].mean()
#     i=df_vowel[df_vowel['vowel']=='i:'][args.Inspect_features].mean()
    
#     def Normalize_operation(df_vowel, normalized_method='TotalMin'):
#         a_raw=df_vowel[df_vowel['vowel']=='A:'][args.Inspect_features]
#         u_raw=df_vowel[df_vowel['vowel']=='u:'][args.Inspect_features]
#         i_raw=df_vowel[df_vowel['vowel']=='i:'][args.Inspect_features]
    
#         a_min=df_vowel[df_vowel['vowel']=='A:'][args.Inspect_features].min()
#         u_min=df_vowel[df_vowel['vowel']=='u:'][args.Inspect_features].min()
#         i_min=df_vowel[df_vowel['vowel']=='i:'][args.Inspect_features].min()
    
#         a_norm=a_raw/a_min
#         u_norm=u_raw/u_min
#         i_norm=i_raw/i_min
#         a_norm['vowel']='A:'
#         u_norm['vowel']='u:'
#         i_norm['vowel']='i:'
        
#         # df_vowel_min=pd.DataFrame([],columns=args.Inspect_features)
#         df_vowel_min=df_vowel[args.Inspect_features].min()
    
        
#         # normalize data before Sb Sw calculation
#         if normalized_method == 'TotalMin':
#             df_vowel_normalized=df_vowel.copy()[args.Inspect_features]/df_vowel_min
#             df_vowel_normalized['vowel']=df_vowel['vowel']
#         elif normalized_method == 'WithinVowelMin':
#             df_vowel_normalized=pd.concat([u_norm,i_norm,a_norm],axis=0)
#         else:
#             raise KeyError()
#         return df_vowel_normalized
#     # df_vowel=Normalize_operation(df_vowel, normalized_method='TotalMin') 
    
    
    
    
#     numerator=u[1] + a[1] + i[0] + u[0]
#     demominator=i[1] + a[0]

    
#     class_feature_means = pd.DataFrame(columns=list(set(df_vowel['vowel'])))
#     n_samples = len(df_vowel)
#     for c, rows in df_vowel.groupby('vowel'):
#         class_feature_means[c] = rows[args.Inspect_features].mean()
    
#     groups_num=len(class_feature_means.index)
#     # Within class scatter matrix 
#     within_class_scatter_matrix = np.zeros((groups_num,groups_num))
#     for c, rows in df_vowel.groupby('vowel'):
#         rows = rows[args.Inspect_features]
#         s = np.zeros((groups_num,groups_num))
        
#         for index, row in rows.iterrows():
#             x, mc = row.values.reshape(groups_num,1), class_feature_means[c].values.reshape(groups_num,1)
#             # class_variance=((x - mc) / mc).dot(((x - mc) / mc).T).astype(float)
#             # class_variance=((x - mc) / numerator).dot(((x - mc) / numerator).T).astype(float) #BCC norm2 + WCC norm2
#             # class_variance=((x - mc) / numerator).dot(((x - mc) ).T).astype(float) #BCC norm2 + WCC norm1
#             class_variance=((x - mc) ).dot(((x - mc) ).T).astype(float) #BCC norm2
#             # class_variance=( x  / mc).dot(( x / mc).T).astype(float) 
#             s += class_variance
            
#         within_class_scatter_matrix += s
#     within_class_scatter_matrix_norm = within_class_scatter_matrix / n_samples
#     # within_class_scatter_matrix_norm = within_class_scatter_matrix / n_samples / numerator**2
#     within_class_scatter_matrix_norm = within_class_scatter_matrix / numerator**2
        
#     # Between class scatter matrix 
#     feature_means = df_vowel.mean()
#     between_class_scatter_matrix = np.zeros((groups_num,groups_num))
#     for c in class_feature_means:    
#         n = len(df_vowel.loc[df_vowel['vowel'] == c].index)
        
#         mc, m = class_feature_means[c].values.reshape(groups_num,1), feature_means[args.Inspect_features].values.reshape(groups_num,1)
        
#         # between_class_variance = n * ( (mc - m) / m).dot(((mc - m) / m).T)
#         # between_class_variance = n * ( (mc - m) / numerator ).dot(((mc - m) / numerator ).T) #BCC norm2 + WCC norm2
#         # between_class_variance = n * ( (mc - m) / numerator ).dot(((mc - m)  ).T) #BCC norm1 + WCC norm2
#         between_class_variance = n * ( (mc - m)  ).dot(((mc - m)  ).T) #BCC norm1 + WCC norm2
#         # between_class_variance = n * ( mc/ m).dot((mc / m).T)
        
#         between_class_scatter_matrix += between_class_variance
#     between_class_scatter_matrix_norm = between_class_scatter_matrix / n_samples
#     # between_class_scatter_matrix_norm = between_class_scatter_matrix / n_samples / numerator**2
#     between_class_scatter_matrix_norm = between_class_scatter_matrix / numerator**2
    
#     Total_scatter_matrix_norm=within_class_scatter_matrix_norm + between_class_scatter_matrix_norm
    
#     # Calculate eigen values
#     linear_discriminant_norm=np.linalg.inv(within_class_scatter_matrix_norm ).dot(between_class_scatter_matrix_norm )
#     # linear_discriminant=np.linalg.inv(within_class_scatter_matrix_norm ).dot(between_class_scatter_matrix )
    
#     return within_class_scatter_matrix, between_class_scatter_matrix,\
#             within_class_scatter_matrix_norm, between_class_scatter_matrix_norm, linear_discriminant_norm, Total_scatter_matrix_norm

