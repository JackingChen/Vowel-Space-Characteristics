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
from articulation.HYPERPARAM.PeopleSelect import SellectP_define
import matplotlib.pyplot as plt
from itertools import combinations

from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from scipy import stats
from scipy.stats import spearmanr,pearsonr 
import statistics 
import os, glob, sys
import statsmodels.api as sm
# from varname import nameof
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
from utils_jack  import Info_name_sex
import warnings
import math

def Add_label(df_formant_statistic,Label,label_choose='ADOS_cate_C'):
    for people in df_formant_statistic.index:
        bool_ind=Label.label_raw['name']==people
        df_formant_statistic.loc[people,label_choose]=Label.label_raw.loc[bool_ind,label_choose].values
    return df_formant_statistic

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
            
    # get rid of nan values
    filter_bool=np.logical_and(filter_bool,~df_formant_statistic.isna().T.any())
    return df_formant_statistic[filter_bool]



def NameMatchAssertion(Formants_people_symb,name):
    ''' check the name in  Formants_people_symb matches the names in label'''
    for name in Formants_people_symb.keys():
        assert name in name




def Process_IQRFiltering_Multi(Formants_utt_symb, limit_people_rule,\
                               outpath='/media/jack/workspace/DisVoice/articulation/Pickles',\
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
    parser.add_argument('--base_path', default='/media/jack/workspace/DisVoice/articulation',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--inpklpath', default='/media/jack/workspace/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--outpklpath', default='/media/jack/workspace/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--dfFormantStatisticpath', default='/media/jack/workspace/DisVoice/articulation/Pickles',
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
    parser.add_argument('--dataset_role', default='KID_FromTD_DOCKID',
                            help='DOC_FromTD_DOCKID |KID_FromTD_DOCKID | DOC_FromASD_DOCKID | KID_FromASD_DOCKID')
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
dfFormantStatisticpath=args.dfFormantStatisticpath
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
limit_people_rule=GetValuelimit_IQR(AUI_info,PhoneMapp_dict,args.Inspect_features)  # 每個人都有自己Formant的IQR值的統計分佈， limit_people_rule就是存每個人的boundary



''' multi processing start '''
prefix,suffix = 'Formants_utt_symb', role
# date_now='{0}-{1}-{2} {3}'.format(dt.now().year,dt.now().month,dt.now().day,dt.now().hour)
date_now='{0}-{1}-{2}'.format(dt.now().year,dt.now().month,dt.now().day)
outpath='/media/jack/workspace/DisVoice/articulation/Pickles'
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
'ADOS_cate_C'
#Category
# =============================================================================
Vowels_AUI=Get_Vowels_AUI(AUI_info, args.Inspect_features,VUIsource="From__Formant_people_information")



additional_columns=['ADOS_cate_C','dia_num']

label_generate_choose_lst=['ADOS_C'] + additional_columns

class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def _func1(self, x):
        """
            x should be dataFrame
        """
        norm_x = 1127 * x.applymap(lambda val: math.log(float(val) / 700 + 1))
        return norm_x
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

# 年齡性別的data存在變數:Info_name_sex
# _func1: 
normalizer=Normalizer()
Vowels_AUI_norm=Dict()
for people in Vowels_AUI.keys():
    for phone in Vowels_AUI[people].keys():
        df_Vwls=Vowels_AUI[people][phone]
        df_Vwls=df_Vwls[args.Inspect_features]
        df_Vwls_norm=normalizer._func1(df_Vwls)
        Vowels_AUI_norm[people][phone]=df_Vwls_norm

#               F1           F2
# text                         
# A:    626.063374  2211.949605
# A:    897.777810  1526.367364
# A:    741.677195  1871.745522
# A:1   326.671436  1068.048559
# A:3   880.188256  1674.493077
# A:3   857.329737  1737.416894
# A:3   721.434689  1665.090699
# A:3   444.509805  1476.154364
# A:4   396.851885  1754.904178
# A:4   788.004335  1781.307778
# A:4   722.525178  1797.408315
# A:5   825.557645  1418.863323
# A:5   682.699067  1790.946459
aaa=ccc
# =============================================================================
import copy

Vowels_AUI = copy.deepcopy(Vowels_AUI_norm)

=copy.

articulation=Articulation(Stat_med_str_VSA='mean')
# df_formant_statistic=articulation.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_generate_choose_lst, FILTERING_method='KDE', KDE_THRESHOLD=40)
df_formant_statistic, SCATTER_matrixBookeep_dict=articulation.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_generate_choose_lst, FILTERING_method='KDE', KDE_THRESHOLD=40,RETURN_scatter_matrix=True)
pickle.dump(SCATTER_matrixBookeep_dict,open(outpklpath+"/SCATTER_matrixBookeep_dict_{}.pkl".format(role),"wb"))

# For pseudo acoustic features generation
df_formant_statistic['u_num+i_num+a_num']=df_formant_statistic['u_num'] +\
                                            df_formant_statistic['i_num'] +\
                                            df_formant_statistic['a_num']




df_formant_statistic=Add_label(df_formant_statistic,Label,label_choose='ADOS_cate_C')
# =============================================================================        
''' 

    2. Evaluation area

    We still keep this area to get a peek of the correlation result.
    The evaluation function should be the same as the one in Statistical_tests.py
    
    The evaluation module is defined in Evaluation_method()
    
'''
# =============================================================================

Eval_med=Evaluation_method()
df_formant_statistic=Eval_med._Postprocess_dfformantstatistic(df_formant_statistic) #Filter unwanted samples


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
if role == 'KID_FromTD_DOCKID' or role ==  'DOC_FromTD_DOCKID':
    pickle.dump(df_formant_statistic,open(outpklpath+"Formant_AUI_tVSAFCRFvals_{}.pkl".format(role),"wb"))



''' Calculate correlations for Formant fetures'''
columns=list(set(df_formant_statistic.columns) - set(additional_columns)) # Exclude added labels
columns=list(set(columns) - set([co for co in columns if "_norm" not in co]))
columns = columns + ['VSA2','FCR2']



# ManualCondition=Dict()
# suffix='.xlsx'
# condfiles=glob.glob('Inspect/condition/*'+suffix)
# for file in condfiles:
#     df_cond=pd.read_excel(file)
#     name=os.path.basename(file).replace(suffix,"")
#     ManualCondition[name]=df_cond['Unnamed: 0'][df_cond['50%']==True]

label_correlation_choose_lst=label_generate_choose_lst
# label_correlation_choose_lst=['ADOS_C']

# feature生出來後先做個簡單的evaluation
N=2
Eval_med=Evaluation_method()
Aaadf_spearmanr_table_NoLimit=Eval_med.Calculate_correlation(label_correlation_choose_lst,df_formant_statistic,N,columns,constrain_sex=-1, constrain_module=-1,feature_type='Session_formant')

# =============================================================================
#  Pearson
#  between_covariance(A:,i:,u:)       -0.346864  ...         86.0
#  between_variance(A:,i:,u:)         -0.465486  ...         86.0




# =============================================================================
# 以下的部份都是實驗或測試用的， 不在主要的流程裡面，
# maintain程式的話以下的部份都跳過。
# =============================================================================
'''

Parse through evaluation dictionary

'''



def Criteria(df_Corr_val):
    pear_str='pearson_p'
    spear_str='spearman_p'
    
    criteria_bool=(df_Corr_val[pear_str]<=0.05) & (df_Corr_val[spear_str]<=0.05)
    return df_Corr_val[criteria_bool]

# =============================================================================
for key in Aaadf_spearmanr_table_NoLimit.keys():
    df_Corr_val=Aaadf_spearmanr_table_NoLimit[key]
    df_Corr_val_criteria=Criteria(df_Corr_val)
    if len(df_Corr_val_criteria)>0:
        print('Predicting label', key)
        print("=========================")
        print(df_Corr_val_criteria)
        print("                         ")


# =============================================================================
# Generate LOC indexes for fraction people for ASD/non-ASD classification
# =============================================================================
dfFormantStatisticFractionpath=dfFormantStatisticpath+'/Fraction'
if not os.path.exists(dfFormantStatisticFractionpath):
    os.makedirs(dfFormantStatisticFractionpath)
sellect_people_define=SellectP_define()
if role == 'KID_FromASD_DOCKID':
    df_formant_statistic_agesexmatch_ASDSevere=df_formant_statistic_77.loc[sellect_people_define.SevereASD_age_sex_match_ver2]
    df_formant_statistic_agesexmatch_ASDMild=df_formant_statistic_77.loc[sellect_people_define.MildASD_age_sex_match_ver2]
    
    label_add='ADOS_cate_C'
    if label_add  not in df_formant_statistic_agesexmatch_ASDSevere.columns:
        df_formant_statistic_agesexmatch_ASDSevere=Add_label(df_formant_statistic_agesexmatch_ASDSevere,Label,label_choose=label_add)
    if label_add  not in df_formant_statistic_agesexmatch_ASDMild.columns:
        df_formant_statistic_agesexmatch_ASDMild=Add_label(df_formant_statistic_agesexmatch_ASDMild,Label,label_choose=label_add)
    
    # 1 represents ASD, 2 represents TD
    label_add='ASDTD' 
    if label_add not in df_formant_statistic_agesexmatch_ASDSevere.columns:
        df_formant_statistic_agesexmatch_ASDSevere[label_add]=sellect_people_define.ASDTD_label['ASD']
    if label_add not in df_formant_statistic_agesexmatch_ASDMild.columns:
        df_formant_statistic_agesexmatch_ASDMild[label_add]=sellect_people_define.ASDTD_label['ASD']
        
    pickle.dump(df_formant_statistic_agesexmatch_ASDSevere,open(dfFormantStatisticFractionpath+'/df_formant_statistic_agesexmatch_ASDSevereGrp_kid.pkl','wb'))
    pickle.dump(df_formant_statistic_agesexmatch_ASDMild,open(dfFormantStatisticFractionpath+'/df_formant_statistic_agesexmatch_ASDMildGrp_kid.pkl','wb'))
    
    
elif role == 'DOC_FromASD_DOCKID':
    df_formant_statistic_agesexmatch_ASDSevere=df_formant_statistic_77.loc[sellect_people_define.SevereASD_age_sex_match_ver2]
    df_formant_statistic_agesexmatch_ASDMild=df_formant_statistic_77.loc[sellect_people_define.MildASD_age_sex_match_ver2]
    
    label_add='ADOS_cate_C'
    if label_add  not in df_formant_statistic_agesexmatch_ASDSevere.columns:
        df_formant_statistic_agesexmatch_ASDSevere=Add_label(df_formant_statistic_agesexmatch_ASDSevere,Label,label_choose=label_add)
    if label_add  not in df_formant_statistic_agesexmatch_ASDMild.columns:
        df_formant_statistic_agesexmatch_ASDMild=Add_label(df_formant_statistic_agesexmatch_ASDMild,Label,label_choose=label_add)
    
    # 1 represents ASD, 2 represents TD
    label_add='ASDTD' 
    if label_add not in df_formant_statistic_agesexmatch_ASDSevere.columns:
        df_formant_statistic_agesexmatch_ASDSevere[label_add]=sellect_people_define.ASDTD_label['ASD']
    if label_add not in df_formant_statistic_agesexmatch_ASDMild.columns:
        df_formant_statistic_agesexmatch_ASDMild[label_add]=sellect_people_define.ASDTD_label['ASD']
        
    pickle.dump(df_formant_statistic_agesexmatch_ASDSevere,open(dfFormantStatisticFractionpath+'/df_formant_statistic_agesexmatch_ASDSevereGrp_doc.pkl','wb'))
    pickle.dump(df_formant_statistic_agesexmatch_ASDMild,open(dfFormantStatisticFractionpath+'/df_formant_statistic_agesexmatch_ASDMildGrp_doc.pkl','wb'))
    
elif role == 'DOC_FromTD_DOCKID':
    df_formant_TD_normal=df_formant_statistic.loc[sellect_people_define.TD_normal_ver2]
    
    # 1 represents ASD, 2 represents TD
    label_add='ASDTD' 
    if label_add not in df_formant_TD_normal.columns:
        df_formant_TD_normal[label_add]=sellect_people_define.ASDTD_label['TD']
        
    pickle.dump(df_formant_TD_normal,open(dfFormantStatisticFractionpath+'/df_formant_statistic_TD_normal_doc.pkl','wb'))
elif role == 'KID_FromTD_DOCKID':
    df_formant_TD_normal=df_formant_statistic.loc[sellect_people_define.TD_normal_ver2]
    
    label_add='ASDTD' 
    if label_add not in df_formant_TD_normal.columns:
        df_formant_TD_normal[label_add]=sellect_people_define.ASDTD_label['TD']
        
    pickle.dump(df_formant_TD_normal,open(dfFormantStatisticFractionpath+'/df_formant_statistic_TD_normal_kid.pkl','wb'))
else:
    raise KeyError("The key has not been registered")


# =============================================================================
'''

    Merge Doc and kid matrixes


    This area is independent from the code above, if the "SCATTER_matrixBookeep_dict"s 
    have been generated before, then we can merge them
'''

# MERGE_INDEXES_ROLE='TD'
MERGE_INDEXES_ROLE=''
scatter_matrix_path=outpklpath

Scatter_mrtx_lst=['Norm(WC)', 'Norm(BC)', 'Norm(TotalVar)']
# =============================================================================

if MERGE_INDEXES_ROLE != '':
    if MERGE_INDEXES_ROLE == 'ASD':
        try:
            SCATTER_matrix_KID_dict=pickle.load(open(scatter_matrix_path+"/SCATTER_matrixBookeep_dict_{}.pkl".format('KID_FromASD_DOCKID'),"rb"))
            SCATTER_matrix_DOC_dict=pickle.load(open(scatter_matrix_path+"/SCATTER_matrixBookeep_dict_{}.pkl".format('DOC_FromASD_DOCKID'),"rb"))
        except FileNotFoundError:
            raise FileNotFoundError("Scatter matrices from Doc and Kid should be prepared in advance")
    elif MERGE_INDEXES_ROLE == 'TD':
        try:
            SCATTER_matrix_KID_dict=pickle.load(open(scatter_matrix_path+"/SCATTER_matrixBookeep_dict_{}.pkl".format('KID_FromTD_DOCKID'),"rb"))
            SCATTER_matrix_DOC_dict=pickle.load(open(scatter_matrix_path+"/SCATTER_matrixBookeep_dict_{}.pkl".format('DOC_FromTD_DOCKID'),"rb"))
        except FileNotFoundError:
            raise FileNotFoundError("Scatter matrices from Doc and Kid should be prepared in advance")
    else:
        raise KeyError("MERGE_INDEXES_ROLE whould be [ASD, TD, '']")
    
    # 1. First check the people are the same, you should take union
    kid_people_lst=list(SCATTER_matrix_KID_dict.keys())
    doc_people_lst=list(SCATTER_matrix_DOC_dict.keys())
    
    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3
    
    def Covariance_representations(eigen_values):
        sam_wilks=1
        pillai=0
        hotelling=0
        for eigen_v in eigen_values:
            wild_element=1.0/np.float(1+eigen_v)
            sam_wilks*=wild_element
            pillai+=wild_element * eigen_v
            hotelling+=eigen_v
        roys_root=np.max(eigen_values)
        return sam_wilks, pillai, hotelling, roys_root
    
    people_intersection=intersection(kid_people_lst, doc_people_lst)
    df_FormantRatios_statistic=pd.DataFrame()
    for people in people_intersection:
        
        Tmp_dict={}
        Covariances={}
        for e in Scatter_mrtx_lst:
        
            Tmp_dict[e+'_kid']=SCATTER_matrix_KID_dict[people][e] 
            Tmp_dict[e+'_doc']=SCATTER_matrix_DOC_dict[people][e] 
        
            Tmp_dict['RatioMatrix_{}_D_K'.format(e)]=np.linalg.inv(Tmp_dict[e+'_kid']).dot(Tmp_dict[e+'_doc'])
            
            eigen_values, _ = np.linalg.eig(Tmp_dict['RatioMatrix_{}_D_K'.format(e)])
            
            Covariances[e+'_sam_wilks_{}'.format('DKRaito')], Covariances[e+'_pillai_{}'.format('DKRaito')],\
                Covariances[e+'_hotelling_{}'.format('DKRaito')], Covariances[e+'_roys_root_{}'.format('DKRaito')]=Covariance_representations(eigen_values)
            
            # make it a list is because nothing but make the code no errors
            Covariances[e+'_Det_DKRaito']=[np.linalg.det(Tmp_dict[e+'_doc'])/np.linalg.det(Tmp_dict[e+'_kid'])]
            Covariances[e+'_Tr_DKRaito']=[np.trace(Tmp_dict[e+'_doc'])/np.trace(Tmp_dict[e+'_kid'])]
        
        df_RESULT_list=pd.DataFrame.from_dict(Covariances)
        df_RESULT_list.index=[people]
        df_FormantRatios_statistic=df_FormantRatios_statistic.append(df_RESULT_list)


    pickle.dump(df_FormantRatios_statistic,open(outpklpath+"Formant_AUI_tVSAFCRFvals_{0}{1}.pkl".format(MERGE_INDEXES_ROLE,'DKRaito'),"wb"))    
    
    if MERGE_INDEXES_ROLE == 'ASD':
        df_formant_statistic_agesexmatch_ASDSevere=df_FormantRatios_statistic.loc[sellect_people_define.SevereASD_age_sex_match_ver2]
        df_formant_statistic_agesexmatch_ASDMild=df_FormantRatios_statistic.loc[sellect_people_define.MildASD_age_sex_match_ver2]
        
        label_add='ADOS_cate_C'
        if label_add  not in df_formant_statistic_agesexmatch_ASDSevere.columns:
            df_formant_statistic_agesexmatch_ASDSevere=Add_label(df_formant_statistic_agesexmatch_ASDSevere,Label,label_choose=label_add)
        if label_add  not in df_formant_statistic_agesexmatch_ASDMild.columns:
            df_formant_statistic_agesexmatch_ASDMild=Add_label(df_formant_statistic_agesexmatch_ASDMild,Label,label_choose=label_add)
        
        # 1 represents ASD, 2 represents TD
        label_add='ASDTD' 
        if label_add not in df_formant_statistic_agesexmatch_ASDSevere.columns:
            df_formant_statistic_agesexmatch_ASDSevere[label_add]=sellect_people_define.ASDTD_label['ASD']
        if label_add not in df_formant_statistic_agesexmatch_ASDMild.columns:
            df_formant_statistic_agesexmatch_ASDMild[label_add]=sellect_people_define.ASDTD_label['ASD']
            
        pickle.dump(df_formant_statistic_agesexmatch_ASDSevere,open(dfFormantStatisticFractionpath+'/df_formant_statistic_agesexmatch_ASDSevereGrp_DKRatio.pkl','wb'))
        pickle.dump(df_formant_statistic_agesexmatch_ASDMild,open(dfFormantStatisticFractionpath+'/df_formant_statistic_agesexmatch_ASDMildGrp_DKRatio.pkl','wb'))
    elif MERGE_INDEXES_ROLE == 'TD':
        df_formant_TD_normal=df_FormantRatios_statistic.loc[sellect_people_define.TD_normal_ver2]
        
        # 1 represents ASD, 2 represents TD
        label_add='ASDTD' 
        if label_add not in df_formant_TD_normal.columns:
            df_formant_TD_normal[label_add]=sellect_people_define.ASDTD_label['TD']
            
        pickle.dump(df_formant_TD_normal,open(dfFormantStatisticFractionpath+'/df_formant_statistic_TD_normalGrp_DKRatio.pkl','wb'))
    