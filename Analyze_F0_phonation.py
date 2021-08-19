#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:14:41 2020

@author: jackchen

This script does the main expeeriments in Table 1 (Correlation between Level of clustering and ADOS_A) 
1. Data prepare area: Gather raw data of the three critical monophthongs (F1 & F2) and save in: df_formant_statistic.
    * Note we hack into scipy f-classif function to decompose ssbn and sswn. We found that ssbn is the main factor of correlation 
2. Correlation area: Write a standard correlation function to calculate the correlations between all features ADOS_C 
3. t-test area: t-test between each groups 



4. Manual Ttest area: Not really important, just ignore it

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
# from articulation.articulation import Articulation
import articulation.Multiprocess as Multiprocess
from datetime import datetime as dt
import pathlib
from phonation.phonation import  Phonation

def criterion_filter(df_formant_statistic,N=10,\
                     constrain_sex=-1, constrain_module=-1,constrain_agemax=-1,constrain_ADOScate=-1,constrain_agemin=-1,\
                     evictNamelst=[]):
    filter_bool=np.logical_and(df_formant_statistic['u_num']>N,df_formant_statistic['a_num']>N)
    # filter_bool=np.logical_and(df_formant_statistic['a_num']>N)
    filter_bool=np.logical_and(filter_bool,df_formant_statistic['i_num']>N)
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



def NameMatchAssertion(Formants_people_symb,name):
    ''' check the name in  Formants_people_symb matches the names in label'''
    for name in Formants_people_symb.keys():
        assert name in name

def find_group(ADOS_label,group):
    flag=-1
    for i,g in enumerate(group):
        if ADOS_label in g:
            flag=0
            return i
    if flag ==-1: # -1 if for conditions that groups are hisglow groups (which means we don't have middle class group) 
        return flag


def to_matrix(l, n): #Create a 2D list out of 1D list
    return [l[i:i+n] for i in range(0, len(l), n)]


from scipy import special, stats
import warnings

    
def Process_IQRFiltering_Phonation_Multi(Formants_utt_symb, limit_people_rule, outpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles'):
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
    # final_results=pool.starmap(FilterUttDictsByCriterion_map, [([Formants_utt_symb,Formants_utt_symb,file_block,limit_people_rule]) for file_block in tqdm(keys)])
    
    Formants_utt_symb_limited=Dict()
    for load_file_tmp,_ in final_results:        
        for utt, df_utt in load_file_tmp.items():
            Formants_utt_symb_limited[utt]=df_utt
    
    pickle.dump(Formants_utt_symb_limited,open(outpath+"/[Analyzing]Phonation_utt_symb_limited.pkl","wb"))
    print('Formants_utt_symb saved to ',outpath+"/[Analyzing]Phonation_utt_symb_limited.pkl")
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
    parser.add_argument('--reFilter', default=True,
                            help='')
    parser.add_argument('--correlation_type', default='spearmanr',
                            help='spearmanr|pearsonr')
    parser.add_argument('--label_choose_lst', default=['ADOS_C'],
                            help='path of the base directory')
    parser.add_argument('--Stat_med_str_VSA', default='mean',
                            help='path of the base directory')
    parser.add_argument('--dataset_role', default='kid88',
                            help='kid_TD| kid88')
    # parser.add_argument('--Inspect_features', default=['F1','F2'],
    #                         help='')
    parser.add_argument('--Inspect_features_phonations', default=['stdevF0','localJitter', 'localabsoluteJitter','localShimmer'],
                            help='')
    
    args = parser.parse_args()
    return args


args = get_args()
base_path=args.base_path

# path_app = base_path+'/../'
# sys.path.append(path_app)
from utils_jack  import  Formant_utt2people_reshape, Gather_info_certainphones, \
                         FilterUttDictsByCriterion, GetValuelimit_IQR, \
                         Get_aligned_sequences, WER, Get_Vowels_AUI
from metric import Evaluation_method     


# =============================================================================
'''
    
    1. Data prepare area

'''
# =============================================================================
''' parse namespace '''
args = get_args()
base_path=args.base_path
pklpath=args.inpklpath
label_choose_lst=args.label_choose_lst # labels are too biased
role=args.dataset_role
outpklpath=args.inpklpath+"/Session_formants_people_vowel_feat/"
if not os.path.exists(outpklpath):
    os.makedirs(outpklpath)


Phonation_utt_symb=pickle.load(open(pklpath+"/Phonation_utt_symb_{role}.pkl".format(role=role),'rb'))
label_set=['ADOS_C','ADOS_S','ADOS_SC']


# =============================================================================
'''

    Filter out data using by 1.5*IQR

'''
PhoneMapp_dict=phonewoprosody.PhoneMapp_dict


PhoneOfInterest=list(PhoneMapp_dict.keys())


# =============================================================================


''' Filter unqualified Phonation vowels '''
Phonation_people_information=Formant_utt2people_reshape(Phonation_utt_symb,Phonation_utt_symb,Align_OrinCmp=False)
AUI_info_phonation=Gather_info_certainphones(Phonation_people_information,PhoneMapp_dict,PhoneOfInterest)
limit_people_rule_Phonation=GetValuelimit_IQR(AUI_info_phonation,PhoneMapp_dict,args.Inspect_features_phonations)

''' multi processing start '''
date_now='{0}-{1}-{2} {3}'.format(dt.now().year,dt.now().month,dt.now().day,dt.now().hour)
outpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles'
filepath=outpath+"/[Analyzing]Phonation_utt_symb_limited.pkl"
if os.path.exists(filepath) and args.reFilter==False:
    fname = pathlib.Path(filepath)
    mtime = dt.fromtimestamp(fname.stat().st_mtime)
    filemtime='{0}-{1}-{2} {3}'.format(mtime.year,mtime.month,mtime.day,mtime.hour)
    
    # If file last modify time is not now (precisions to the hours) than we create new one
    if filemtime != date_now:
        Process_IQRFiltering_Phonation_Multi(Phonation_utt_symb,limit_people_rule_Phonation) # the results will be output as pkl file at outpath+"/[Analyzing]Phonation_utt_symb_limited.pkl"
else:
    Process_IQRFiltering_Phonation_Multi(Phonation_utt_symb,limit_people_rule_Phonation)

Phonation_utt_symb_limited=pickle.load(open(filepath,"rb"))
''' multi processing end '''
if len(limit_people_rule_Phonation) >0:
    Phonation_utt_symb=Phonation_utt_symb_limited

Phonation_people_information=Formant_utt2people_reshape(Phonation_utt_symb,Phonation_utt_symb,Align_OrinCmp=False)
AUI_info_phonation=Gather_info_certainphones(Phonation_people_information,PhoneMapp_dict,PhoneOfInterest)
# =============================================================================
'''

    Calculate each vowel formant duration


      
    averaged duration
    u
    0.0860
    i
    0.0705
    a
    0.0932

'''
# =============================================================================
# def Calculate_each_vowel_formant_duration(AUI_info):
#     Dict_phoneDuration=Dict()
#     Dict_phoneDuration_mean=pd.DataFrame([])
#     for phone in PhoneOfInterest:
#         Dict_phoneDuration[phone]=pd.DataFrame([],columns=['dur'])
#         for people in AUI_info.keys():
#             df_data=AUI_info[people][phone]            
#             Dict_phoneDuration[phone].loc[people,'dur']=(df_data['end']-df_data['start']).mean()
#         Dict_phoneDuration_mean.loc[phone,'mean']=Dict_phoneDuration[phone].mean().values
#     return Dict_phoneDuration, Dict_phoneDuration_mean
# Dict_phoneDuration, Dict_phoneDuration_mean = Calculate_each_vowel_formant_duration(AUI_info)
# =============================================================================




''' Calculate temporal variance features '''
Vowels_AUI_phonation=Get_Vowels_AUI(AUI_info_phonation, args.Inspect_features_phonations,VUIsource="From__Formant_people_information")



# Calculate phonation features
phonation=Phonation(Inspect_features=args.Inspect_features_phonations)
df_phonation_statistic=phonation.calculate_features(Vowels_AUI_phonation,Label,PhoneOfInterest,label_choose_lst=label_choose_lst)

for i in range(len(df_phonation_statistic)):
    name=df_phonation_statistic.iloc[i].name
    df_phonation_statistic.loc[name,'ADOS_cate']=Label.label_raw[Label.label_raw['name']==name]['ADOS_cate'].values
pickle.dump(df_phonation_statistic,open(outpklpath+"Phonation_meanvars_{}.pkl".format(role),"wb"))

sex=-1
module=-1
agemax=-1
agemin=-1
ADOScate=-1
N=0
df_phonation_statistic_77=criterion_filter(df_phonation_statistic,\
                                        constrain_sex=sex,constrain_module=module,N=N,constrain_agemax=agemax,constrain_agemin=agemin,constrain_ADOScate=ADOScate,\
                                        evictNamelst=[])

# =============================================================================
'''

    2. Correlation area

'''
# =============================================================================
''' Calculate correlations for Phonation fetures'''
columns=[
    'stdevF0_mean(u:)', 'localJitter_mean(u:)',
    'localabsoluteJitter_mean(u:)', 'localShimmer_mean(u:)',
    'stdevF0_var(u:)', 'localJitter_var(u:)', 'localabsoluteJitter_var(u:)',
    'localShimmer_var(u:)', 'stdevF0_mean(i:)', 'localJitter_mean(i:)',
    'localabsoluteJitter_mean(i:)', 'localShimmer_mean(i:)',
    'stdevF0_var(i:)', 'localJitter_var(i:)', 'localabsoluteJitter_var(i:)',
    'localShimmer_var(i:)', 'stdevF0_mean(A:)', 'localJitter_mean(A:)',
    'localabsoluteJitter_mean(A:)', 'localShimmer_mean(A:)',
    'stdevF0_var(A:)', 'localJitter_var(A:)', 'localabsoluteJitter_var(A:)',
    'localShimmer_var(A:)', 'stdevF0_mean(A:,i:,u:)',
    'localJitter_mean(A:,i:,u:)', 'localabsoluteJitter_mean(A:,i:,u:)',
    'localShimmer_mean(A:,i:,u:)', 'stdevF0_var(A:,i:,u:)',
    'localJitter_var(A:,i:,u:)', 'localabsoluteJitter_var(A:,i:,u:)',
    'localShimmer_var(A:,i:,u:)'
    ]

df_phonation_statistic_77['u_num+i_num+a_num']=df_phonation_statistic_77['u_num'] +\
                                            df_phonation_statistic_77['i_num'] +\
                                            df_phonation_statistic_77['a_num']


# ManualCondition=Dict()
# suffix='.xlsx'
# condfiles=glob.glob('Inspect/condition/*'+suffix)
# for file in condfiles:
#     df_cond=pd.read_excel(file)
#     name=os.path.basename(file).replace(suffix,"")
#     ManualCondition[name]=df_cond['Unnamed: 0'][df_cond['50%']==True]

N=0
Eval_med=Evaluation_method()
Aaadf_spearmanr_table_NoLimit=Eval_med.Calculate_correlation(label_choose_lst,df_phonation_statistic_77,N,columns,constrain_sex=-1, constrain_module=-1)
