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
from HYPERPARAM import phonewoprosody, Label
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
from articulation import Articulation
import Multiprocess
from datetime import datetime as dt
import pathlib


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


from sklearn.utils import (as_float_array, check_array, check_X_y, safe_sqr,
                     safe_mask)
from scipy import special, stats
import warnings

def Process_IQRFiltering_Multi(Formants_utt_symb, limit_people_rule, outpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles'):
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
    
    pickle.dump(Formants_utt_symb_limited,open(outpath+"/[Analyzing]Formants_utt_symb_limited.pkl","wb"))
    print('Formants_utt_symb saved to ',outpath+"/[Analyzing]Formants_utt_symb_limited.pkl")
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
    parser.add_argument('--Inspect', default=False,
                            help='path of the base directory')
    parser.add_argument('--reFilter', default=False,
                            help='')
    parser.add_argument('--correlation_type', default='spearmanr',
                            help='spearmanr|pearsonr')
    parser.add_argument('--label_choose_lst', default=['ADOS_C'],
                            help='path of the base directory')
    parser.add_argument('--Stat_med_str_VSA', default='mean',
                            help='path of the base directory')
    parser.add_argument('--poolMed', default='middle',
                            help='path of the base directory')
    parser.add_argument('--poolWindowSize', default=3,
                            help='path of the base directory')
    parser.add_argument('--role', default='ASDkid',
                            help='path of the base directory')
    parser.add_argument('--Inspect_features', default=['F1','F2'],
                            help='')
    
    args = parser.parse_args()
    return args


args = get_args()
base_path=args.base_path

# path_app = base_path+'/../'
# sys.path.append(path_app)
from utils_jack  import  Formant_utt2people_reshape, Gather_info_certainphones, \
                         FilterUttDictsByCriterion, GetValuelimit_IQR, \
                         Get_aligned_sequences, WER 
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
INSPECT=args.Inspect
windowsize=args.poolWindowSize
label_choose_lst=args.label_choose_lst # labels are too biased
role=args.role
Stat_med_str=args.Stat_med_str_VSA
outpklpath=args.inpklpath+"/Session_formants_people_vowel_feat/"
if not os.path.exists(outpklpath):
    os.makedirs(outpklpath)


Formants_utt_symb=pickle.load(open(pklpath+"/Formants_utt_symb_by{0}_window{1}_{2}.pkl".format(args.poolMed,windowsize,role),'rb'))
# Formants_people_symb=pickle.load(open(pklpath+"/Formants_people_symb_bymiddle.pkl".format(role),"rb"))
label_set=['ADOS_C','ADOS_S','ADOS_SC']


# =============================================================================
''' grouping severity '''
ADOS_label=Label.label_raw['ADOS_C']
array=list(set(ADOS_label))
phoneme="A:"
groups=[np.split(array, idx)  for n_splits in range(3)  for idx in combinations(range(1, len(array)), n_splits) if len(np.split(array, idx))>1]
# =============================================================================




######################################
''' iteration area '''

label_choose='ADOS_C'

#######################################
# =============================================================================
'''

    Filter out data using by 1.5*IQR

'''
from HYPERPARAM import phonewoprosody, Label
PhoneMapp_dict=phonewoprosody.PhoneMapp_dict

# PhoneMapp_dict={'u:':phonewoprosody.Phoneme_sets['u_'],\
#                 'i:':phonewoprosody.Phoneme_sets['i_']+['j'],\
#                 'A:':phonewoprosody.Phoneme_sets['A_']}
    


PhoneOfInterest=list(PhoneMapp_dict.keys())
# =============================================================================


''' Vowel AUI rule is using phonewoprosody '''


Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
AUI_info=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)
limit_people_rule=GetValuelimit_IQR(AUI_info,PhoneMapp_dict,args.Inspect_features)



''' multi processing start '''
date_now='{0}-{1}-{2} {3}'.format(dt.now().year,dt.now().month,dt.now().day,dt.now().hour)
outpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles'
filepath=outpath+"/[Analyzing]Formants_utt_symb_limited.pkl"
if os.path.exists(filepath) and args.reFilter==False:
    fname = pathlib.Path(filepath)
    mtime = dt.fromtimestamp(fname.stat().st_mtime)
    filemtime='{0}-{1}-{2} {3}'.format(mtime.year,mtime.month,mtime.day,mtime.hour)
    
    # If file last modify time is not now (precisions to the hours) than we create new one
    if filemtime != date_now:
        Process_IQRFiltering_Multi(Formants_utt_symb,limit_people_rule) # the results will be output as pkl file at outpath+"/[Analyzing]Formants_utt_symb_limited.pkl"
else:
    Process_IQRFiltering_Multi(Formants_utt_symb,limit_people_rule)
Formants_utt_symb_limited=pickle.load(open(filepath,"rb"))
''' multi processing end '''
if len(limit_people_rule) >0:
    Formants_utt_symb=Formants_utt_symb_limited



Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
AUI_info=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)

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
def Calculate_each_vowel_formant_duration(AUI_info):
    Dict_phoneDuration=Dict()
    Dict_phoneDuration_mean=pd.DataFrame([])
    for phone in PhoneOfInterest:
        Dict_phoneDuration[phone]=pd.DataFrame([],columns=['dur'])
        for people in AUI_info.keys():
            df_data=AUI_info[people][phone]            
            Dict_phoneDuration[phone].loc[people,'dur']=(df_data['end']-df_data['start']).mean()
        Dict_phoneDuration_mean.loc[phone,'mean']=Dict_phoneDuration[phone].mean().values
    return Dict_phoneDuration, Dict_phoneDuration_mean
Dict_phoneDuration, Dict_phoneDuration_mean = Calculate_each_vowel_formant_duration(AUI_info)
# =============================================================================
    
    

def Get_Vowels_AUI(AUI_info,VUIsource="From__Formant_people_information"):
    if VUIsource=="From__Formant_people_information": # Mainly use this
        Vowels_AUI=Dict()
        for people in AUI_info.keys():
            for phone, values in AUI_info[people].items():
                Vowels_AUI[people][phone]=AUI_info[people][phone][AUI_info[people][phone]['cmps']=='ori'][args.Inspect_features]
    
    
    elif VUIsource=="From__Formants_people_symb":
        Formants_people_symb=pickle.load(open(pklpath+"/Formants_people_symb_by{0}_window{1}_{2}.pkl".format(args.poolMed,windowsize,role),"rb"))
        ''' assert if the names don't matches' '''
        NameMatchAssertion(Formants_people_symb,Label.label_raw['name'].values)
        Vowels_AUI=Dict()
        # for people in Label.label_raw.sort_values(by='ADOS_C')['name']:
        for people in Formants_people_symb.keys():    
            for phone, values in Formants_people_symb[people].items():
                if phone not in [e for _, phoneme in PhoneMapp_dict.items() for e in phoneme]: # update fixed 2021/05/27
                    continue
                else:
                    for p_key, p_val in PhoneMapp_dict.items():
                        if phone in p_val:
                            Phone_represent=p_key
                    if people not in Vowels_AUI.keys():
                        if Phone_represent not in Vowels_AUI[people].keys():
                            Vowels_AUI[people][Phone_represent]=values
                        else:
                            Vowels_AUI[people][Phone_represent].extend(values)
                    else:
                        if Phone_represent not in Vowels_AUI[people].keys():
                            Vowels_AUI[people][Phone_represent]=values
                        else:
                            Vowels_AUI[people][Phone_represent].extend(values)
    return Vowels_AUI
# =============================================================================
    

Vowels_AUI=Get_Vowels_AUI(AUI_info)
pickle.dump(Vowels_AUI,open(outpklpath+"Vowels_AUI_{}.pkl".format(role),"wb"))

articulation=Articulation()
df_formant_statistic=articulation.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest)


for i in range(len(df_formant_statistic)):
    name=df_formant_statistic.iloc[i].name
    df_formant_statistic.loc[name,'ADOS_cate']=Label.label_raw[Label.label_raw['name']==name]['ADOS_cate'].values


Eval_med=Evaluation_method()
df_formant_statistic=Eval_med._Postprocess_dfformantstatistic(df_formant_statistic)
pickle.dump(df_formant_statistic,open(outpklpath+"Formant_AUI_tVSAFCRFvals_{}.pkl".format(role),"wb"))


# kidM3=df_formant_statistic[df_formant_statistic['Module']==3]
# kidM4=df_formant_statistic[df_formant_statistic['Module']==4]
# pickle.dump(kidM4,open(outpklpath+"Formant_AUI_tVSAFCRFvals_{}.pkl".format('ASDkidM4'),"wb"))
# pickle.dump(kidM3,open(outpklpath+"Formant_AUI_tVSAFCRFvals_{}.pkl".format('ASDkidM3'),"wb"))



# =============================================================================
'''

    2. Correlation area

'''
# =============================================================================

''' Calculate correlations '''

# columns=['FCR','VSA1','F_vals_f1(A:,i:,u:)', 'F_vals_f2(A:,i:,u:)',
#        'F_val_mix(A:,i:,u:)', 'MSB_f1(A:,i:,u:)', 'MSB_f2(A:,i:,u:)',
#        'MSB_mix', 'F_vals_f1(A:,u:)', 'F_vals_f2(A:,u:)', 'F_val_mix(A:,u:)',
#        'MSB_f1(A:,u:)', 'MSB_f2(A:,u:)', 'F_vals_f1(A:,i:)',
#        'F_vals_f2(A:,i:)', 'F_val_mix(A:,i:)', 'MSB_f1(A:,i:)',
#        'MSB_f2(A:,i:)', 'F_vals_f1(i:,u:)', 'F_vals_f2(i:,u:)',
#        'F_val_mix(i:,u:)', 'MSB_f1(i:,u:)', 'MSB_f2(i:,u:)']
columns=['VSA1','FCR','u_num+i_num+a_num',
       'BW_sam_wilks(A:,i:,u:)', 'BW_pillai(A:,i:,u:)',
       'BW_hotelling(A:,i:,u:)', 'BW_roys_root(A:,i:,u:)',
       'between_covariance(A:,i:,u:)', 'between_variance(A:,i:,u:)',
       'within_covariance(A:,i:,u:)', 'within_variance(A:,i:,u:)']
df_formant_statistic['u_num+i_num+a_num']=df_formant_statistic['u_num'] +\
                                            df_formant_statistic['i_num'] +\
                                            df_formant_statistic['a_num']
# columns=['FCR','VSA1','F_vals_f1', 'F_vals_f2', 'F_val_mix','MSB_f1','MSB_f2',\
#          'dau1','dai1','diu1','daudai1','daudiu1','daidiu1','daidiudau1',\
#          'dau2','dai2','diu2','daudai2','daudiu2','daidiu2','daidiudau2',\
#          'F2i_u','F1a_u']
# def Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns, corr_label='ADOS', constrain_sex=-1, constrain_module=-1, constrain_assessment=-1,evictNamelst=[]):
#     '''
#         constrain_sex: 1 for boy, 2 for girl
#         constrain_module: 3 for M3, 4 for M4
#     '''
#     df_pearsonr_table=pd.DataFrame([],columns=[args.correlation_type,'{}_pvalue'.format(args.correlation_type[:5]),'de-zero_num'])
#     for lab_choose in label_choose_lst:
#         filter_bool=np.logical_and(df_formant_statistic['u_num']>N,df_formant_statistic['a_num']>N)
#         filter_bool=np.logical_and(filter_bool,df_formant_statistic['i_num']>N)
#         filter_bool=np.logical_and(filter_bool,df_formant_statistic['ADOS'].isna()!=True)
#         if constrain_sex != -1:
#             filter_bool=np.logical_and(filter_bool,df_formant_statistic['sex']==constrain_sex)
#         if constrain_module != -1:
#             filter_bool=np.logical_and(filter_bool,df_formant_statistic['Module']==constrain_module)
#         if constrain_assessment != -1:
#             filter_normal=df_formant_statistic['ADOS']<2
#             filter_ASD=(df_formant_statistic['ADOS']<3) & (df_formant_statistic['ADOS']>=2)
#             filter_autism=df_formant_statistic['ADOS']>=3
            
#             if constrain_assessment == 0:
#                 filter_bool=np.logical_and(filter_bool,filter_normal)
#             elif constrain_assessment == 1:
#                 filter_bool=np.logical_and(filter_bool,filter_ASD)
#             elif constrain_assessment == 2:
#                 filter_bool=np.logical_and(filter_bool,filter_autism)
#         if len(evictNamelst)>0:
#             for name in evictNamelst:
#                 filter_bool.loc[name]=False
            
#         df_formant_qualified=df_formant_statistic[filter_bool]
#         for col in columns:
#             spear,spear_p=spearmanr(df_formant_qualified[col],df_formant_qualified[corr_label])
#             pear,pear_p=pearsonr(df_formant_qualified[col],df_formant_qualified[corr_label])

#             if args.correlation_type == 'pearsonr':
#                 df_pearsonr_table.loc[col]=[pear,pear_p,len(df_formant_qualified[col])]
#                 # pear,pear_p=pearsonr(df_denan["{}_LPP_{}".format(ps,ps)],df_formant_qualified['ADOS'])
#                 # df_pearsonr_table_GOP.loc[ps]=[pear,pear_p,len(df_denan)]
#             elif args.correlation_type == 'spearmanr':
#                 df_pearsonr_table.loc[col]=[spear,spear_p,len(df_formant_qualified[col])]
#         print("Setting N={0}, the correlation metric is: ".format(N))
#         print("Using evaluation metric: {}".format(args.correlation_type))
#         print(df_pearsonr_table)
#     return df_pearsonr_table
# for N in range(10):
#     df_pearsonr_table=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=-1, constrain_module=-1)


ManualCondition=Dict()
suffix='.xlsx'
condfiles=glob.glob('Inspect/condition/*'+suffix)
for file in condfiles:
    df_cond=pd.read_excel(file)
    name=os.path.basename(file).replace(suffix,"")
    ManualCondition[name]=df_cond['Unnamed: 0'][df_cond['50%']==True]

N=2

# tmp_dct={}
# for N in range(1,20,1):
Aaadf_spearmanr_table_NoLimit=Eval_med.Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=-1, constrain_module=-1,evictNamelst=ManualCondition[ 'unreasonable_all'])
Aaadf_spearmanr_table_NoLimit=Eval_med.Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=-1, constrain_module=-1)

    # target=Aaadf_spearmanr_table_NoLimit
    # diff_r=(np.abs(target.loc['BWratio(A:,i:,u:)']) - np.abs(target.loc['u_num+i_num+a_num'])).iloc[0]
    # tmp_dct[N]=diff_r

# Aaadf_pearsonr_table_NoLimitWithADOScat=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,corr_label='ADOS_cate',constrain_sex=-1, constrain_module=-1)
# Aaadf_pearsonr_table_normal=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_assessment=0)
# Aaadf_pearsonr_table_ASD=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_assessment=1)
# Aaadf_pearsonr_table_autism=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_assessment=2)

# Aaadf_pearsonr_table_boy_M3=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=1, constrain_module=3)
# Aaadf_pearsonr_table_girl_M3=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=2, constrain_module=3)
# Aaadf_pearsonr_table_boy_M4=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=1, constrain_module=4)
# Aaadf_pearsonr_table_girl_M4=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=2, constrain_module=4)
# Aaadf_pearsonr_table_M3=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=-1, constrain_module=3)
# Aaadf_pearsonr_table_M4=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=-1, constrain_module=4)
# Aaadf_pearsonr_table_boy=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=1, constrain_module=-1)
# Aaadf_pearsonr_table_girl=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=2, constrain_module=-1)
# Aaadf_pearsonr_table_N7=Calculate_correlation(df_formant_statistic,7,columns)



# =============================================================================
'''

    3. t-test area

'''
# =============================================================================
'''  T-Test between group Within group''' 
filter_boy=df_formant_statistic['sex']==1
filter_girl=df_formant_statistic['sex']==2
filter_M3=df_formant_statistic['Module']==3
filter_M4=df_formant_statistic['Module']==4

filter_boy_M3 = filter_boy & filter_M3
filter_boy_M4 = filter_boy & filter_M4
filter_girl_M3 = filter_girl & filter_M3
filter_girl_M4 = filter_girl & filter_M4



from itertools import combinations
Filter_list=['filter_boy','filter_girl','filter_M3','filter_M4','filter_boy_M3',\
             'filter_boy_M4','filter_boy_M4','filter_girl_M3','filter_girl_M4']
feature_list=['ADOS','F_vals_f1','F_vals_f2','F_val_mix','VSA1','FCR','MSB_f1','MSB_f2']
comb=combinations(Filter_list,2)
Filter_pair_list=[[x1,x2] for x1, x2 in comb]


df_TTest_results=pd.DataFrame([],columns=feature_list)
for filt1, filt2 in Filter_pair_list:
    for feature in feature_list:
        sign=np.sign((df_formant_statistic[vars()[filt1]][feature].mean() - df_formant_statistic[vars()[filt2]][feature].mean()))
        df_TTest_results.loc[','.join([filt1,filt2]),feature] = np.round(sign* float(stats.ttest_ind(df_formant_statistic[vars()[filt1]][feature], df_formant_statistic[vars()[filt2]][feature])[1]),3)
        # print(','.join([filt1,filt2]),feature,stats.ttest_ind(df_formant_statistic[vars()[filt1]][feature], df_formant_statistic[vars()[filt2]][feature])[1])

df_TTest_results=df_TTest_results.astype(float)

'''  T-Test ASD vs TD''' 
df_formant_statistic_doc=pickle.load(open(outpklpath+"Formant_AUI_tVSAFCRFvals_ASDdoc.pkl","rb"))
df_formant_statistic_kid=pickle.load(open(outpklpath+"Formant_AUI_tVSAFCRFvals_ASDkid.pkl","rb"))
df_formant_statistic77_path='Pickles/Session_formants_people_vowel_feat/Formant_AUI_tVSAFCRFvals_ASDkid.pkl'
df_formant_statistic_77=pickle.load(open(df_formant_statistic77_path,'rb'))
df_formant_statistic_ASDTD_path='Pickles/Session_formants_people_vowel_feat/Formant_AUI_tVSAFCRFvals_ASDTD.pkl'
df_formant_statistic_TD=pickle.load(open(df_formant_statistic_ASDTD_path,'rb'))


def criterion_filter(df_formant_statistic,N=10,constrain_sex=-1, constrain_module=-1):
    filter_bool=np.logical_and(df_formant_statistic['u_num']>N,df_formant_statistic['a_num']>N)
    filter_bool=np.logical_and(filter_bool,df_formant_statistic['i_num']>N)
    if constrain_sex != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['sex']==constrain_sex)
    if constrain_module != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['Module']==constrain_module)
    return df_formant_statistic[filter_bool]
sex=-1
df_formant_statistic_77=criterion_filter(df_formant_statistic_77,constrain_sex=sex)
df_formant_statistic_TD=criterion_filter(df_formant_statistic_TD,constrain_sex=sex)


comb=[['df_formant_statistic_TD','df_formant_statistic_77'],]
Parameters=['F_vals_f1','F_vals_f2','F_val_mix','MSB_f1','MSB_f2','MSB_mix','MSW_f1','MSW_f2','MSW_mix']

df_ttest_result=pd.DataFrame([],columns=['doc-kid','p-val'])
for role_1,role_2  in comb:
    for parameter in Parameters:
        test=stats.ttest_ind(vars()[role_1][parameter], vars()[role_2][parameter])
        print(parameter, '{0} vs {1}'.format(role_1,role_2),test)
        print(role_1+':',vars()[role_1][parameter].mean(),role_2+':',vars()[role_2][parameter].mean())
        df_ttest_result.loc[parameter,'doc-kid'] = vars()[role_1][parameter].mean() - vars()[role_2][parameter].mean()
        df_ttest_result.loc[parameter,'p-val'] = test[1]
        
aaa=ccc


# =============================================================================
'''

Plotting area

'''
phoneme_color_map={'a':'tab:blue','u:':'tab:orange','i:':'tab:green',\
                   'A:':'tab:blue','A:1':'tab:orange','A:2':'tab:green','A:3':'tab:red','A:4':'tab:purple','A:5':'tab:gray'}
# =============================================================================

Plotout_path="Plots/"

if not os.path.exists(Plotout_path):
    os.makedirs(Plotout_path)



def plot(Vowels_AUI,outname=Plotout_path+'{0}_ADOS{1}'):
    for people in Vowels_AUI.keys():
        if people not in df_formant_statistic.index:
            continue
        formant_info=df_formant_statistic.loc[people]
        ASDlab=Label.label_raw[label_choose][Label.label_raw['name']==people].values
        fig, ax = plt.subplots()
        for phone, values in Vowels_AUI[people].items():
            x,y=values.loc[:,'F1'],values.loc[:,'F2']
    
            area=np.repeat(1,len(x))
            cms=np.repeat(phoneme_color_map[phone],len(x))
            plt.scatter(x, y, s=area, c=cms,  label=phone)
            plt.title('{0}_ADOS{1}'.format(pid_dict[people],ASDlab[0]))
            
        additional_info1="FCR={FCR}, VSA1={VSA1}, VSA2={VSA2}, LnVSA={LnVSA}".format(\
            FCR=formant_info['FCR'],VSA1=formant_info['VSA1'],VSA2=formant_info['VSA2'],LnVSA=formant_info['LnVSA'])
        additional_info2="F_vals_f1={F_vals_f1}, F_vals_f2={F_vals_f2}, F_val_mix={F_val_mix}".format(\
            F_vals_f1=formant_info['F_vals_f1'],F_vals_f2=formant_info['F_vals_f2'],F_val_mix=formant_info['F_val_mix'])

        plt.ylim(0, 5000)
        plt.xlim(0, 5000)
        ax.legend()
        plt.figtext(0,0,additional_info1)
        plt.figtext(0,-0.1,additional_info2)
        plt.savefig(outname.format(pid_dict[people],ASDlab[0]),dpi=300, bbox_inches = "tight")
        plt.show()


plot(Vowels_AUI,Plotout_path+'{0}_ADOS{1}')

if INSPECT:
    phone="ax5"
    
    label_ADOSC=Label.label_raw[['name','ADOS_C']].set_index("name")
    Num_spokenphone=Vowels_AUI_sampNum[people][phone]
    
    df_formant_statistic[phone]=pd.concat([df_formant_statistic[phone],Num_spokenphone],axis=1)

    # =============================================================================
    '''
    
    Plot x = desired value, y= ADOS score. We want to sample critical samples for further inspection
    
    '''
    # num_spoken_phone=pd.DataFrame.from_dict(Vowels_AUI_sampNum,orient='index',columns=['num_spoken_phone'])
    N=10
    # =============================================================================
    fig, ax = plt.subplots()
    feature_str='F_val_mix'
    df_formant_statistic_col=df_formant_statistic.astype(float)[feature_str]
    label_ADOSC=Label.label_raw[['name','ADOS_C']].set_index("name")
    df_formant_statistic_qualified=pd.concat([df_formant_statistic_col,label_ADOSC],axis=1)
    # df_formant_statistic_col=pd.concat([df_formant_statistic_col,num_spoken_phone],axis=1)
    # df_formant_statistic_qualified=df_formant_statistic_col[df_formant_statistic_col['num_spoken_phone']>N]
    
    area=np.repeat(10,len(df_formant_statistic_qualified))
    cms=range(len(df_formant_statistic_qualified))
    
    
    x, y=df_formant_statistic_qualified.iloc[:,0], df_formant_statistic_qualified.iloc[:,1]
    plt.scatter(x,y, c=cms, s=area)
    for xi, yi, pidi in zip(x,y,df_formant_statistic_qualified.index):
        ax.annotate(str(pid_dict[pidi]), xy=(xi,yi))
    plt.title(feature_str)
    plt.xlabel("feature")
    plt.ylabel("ADOS")
    ax.legend()
    plt.savefig(Plotout_path+'Acorrelation{0}.png'.format(feature_str),dpi=300, bbox_inches = "tight")
    plt.show()
df_pid=pd.DataFrame.from_dict(pid_dict,orient='index')
df_pid.columns=['name_id']
df_pid[df_pid['name_id']==69].index    



def FilterUttDictsByCriterion_map(Formants_utt_symb,Formants_utt_symb_cmp,keys,limit_people_rule):
    # Masks will be generated by setting criterion on Formants_utt_symb
    # and Formants_utt_symb_cmp will be masked by the same mask as Formants_utt_symb
    # we need to make sure two things:
    #   1. the length of Formants_utt_symb_cmp and Formants_utt_symb are the same
    #   2. the phone sequences are aligned correctly
    Formants_utt_symb_limited=Dict()
    Formants_utt_symb_cmp_limited=Dict()
    for utt in tqdm(keys):
        people=utt[:utt.find(re.findall("_[K|D]",utt)[0])]
        df_ori=Formants_utt_symb[utt].sort_values(by="start")
        df_cmp=Formants_utt_symb_cmp[utt].sort_values(by="start")
        df_ori['text']=df_ori.index
        df_cmp['text']=df_cmp.index
        
        r=df_cmp.index.astype(str)
        h=df_ori.index.astype(str)
        
        error_info, WER_value=WER(r,h)
        utt_human_ali, utt_hype_ali=Get_aligned_sequences(ref=df_cmp, hype=df_ori ,error_info=error_info) # This step cannot gaurentee hype and human be exact the same string
                                                                                                          # because substitude error also counts when selecting the optimal 
                                                                                                          # matched string         
        utt_human_ali.index=utt_human_ali['text']
        utt_human_ali=utt_human_ali.drop(columns=["text"])
        utt_hype_ali.index=utt_hype_ali['text']
        utt_hype_ali=utt_hype_ali.drop(columns=["text"])
        
        assert len(utt_human_ali) == len(utt_hype_ali)
        limit_rule=limit_people_rule[people]
        SymbRuleChecked_bookkeep={}
        for symb_P in limit_rule.keys():
            values_limit=limit_rule[symb_P]
    
            filter_bool=utt_hype_ali.index.str.contains(symb_P)  #  1. select the phone with criterion
            filter_bool_inv=np.invert(filter_bool)           #  2. create a mask for unchoosed phones
                                                             #  we want to make sure that only 
                                                             #  the phones not match the criterion will be False
            for feat in values_limit.keys():
                feat_max_value=values_limit[feat]['max']
                filter_bool=np.logical_and(filter_bool , (utt_hype_ali[feat]<=feat_max_value))
                feat_min_value=values_limit[feat]['min']
                filter_bool=np.logical_and(filter_bool , (utt_hype_ali[feat]>=feat_min_value))
                
            filter_bool=np.logical_or(filter_bool_inv,filter_bool)
            
            # check & debug
            if not filter_bool.all():
                print(utt,filter_bool[filter_bool==False])
            
            SymbRuleChecked_bookkeep[symb_P]=filter_bool.to_frame()
        
        df_True=pd.DataFrame(np.array([True]*len(utt_hype_ali)))
        for keys, values in SymbRuleChecked_bookkeep.items():
            df_True=np.logical_and(values,df_True)
        
        Formants_utt_symb_limited[utt]=utt_hype_ali[df_True[0].values]
        Formants_utt_symb_cmp_limited[utt]=utt_human_ali[df_True[0].values]
    return Formants_utt_symb_limited,Formants_utt_symb_cmp_limited

# df_formant_statistic=pd.DataFrame()
# for people in Vowels_AUI.keys(): #update 2021/05/27 fixed 
#     RESULT_dict={}
#     F12_raw_dict=Vowels_AUI[people]
#     F12_val_dict={k:[] for k in ['u','a','i']}
#     for k,v in F12_raw_dict.items():
#         if Stat_med_str == 'mode':
#             F12_val_dict[k]=Statistic_method[Stat_med_str](v,axis=0)[0].ravel()
#         else:
#             F12_val_dict[k]=Statistic_method[Stat_med_str](v,axis=0)
    
#     RESULT_dict['u_num'], RESULT_dict['a_num'], RESULT_dict['i_num']=len(Vowels_AUI[people]['u']),len(Vowels_AUI[people]['a']),len(Vowels_AUI[people]['i'])
    
#     RESULT_dict['ADOS']=Label.label_raw[label_choose][Label.label_raw['name']==people].values    
#     RESULT_dict['sex']=Label.label_raw['sex'][Label.label_raw['name']==people].values[0]
#     RESULT_dict['age']=Label.label_raw['age_year'][Label.label_raw['name']==people].values[0]
#     RESULT_dict['Module']=Label.label_raw['Module'][Label.label_raw['name']==people].values[0]
    
#     u=F12_val_dict['u']
#     a=F12_val_dict['a']
#     i=F12_val_dict['i']

    
#     if len(u)==0 or len(a)==0 or len(i)==0:
#         u_num= RESULT_dict['u_num'] if type(RESULT_dict['u_num'])==int else 0
#         i_num= RESULT_dict['i_num'] if type(RESULT_dict['i_num'])==int else 0
#         a_num= RESULT_dict['a_num'] if type(RESULT_dict['a_num'])==int else 0
#         df_RESULT_list=pd.DataFrame(np.zeros([1,len(df_formant_statistic.columns)]),columns=df_formant_statistic.columns)
#         df_RESULT_list.index=[people]
#         df_RESULT_list['FCR']=10
#         df_RESULT_list['ADOS']=RESULT_dict['ADOS'][0]
#         df_RESULT_list[['u_num','a_num','i_num']]=[u_num, i_num, a_num]
#         df_formant_statistic=df_formant_statistic.append(df_RESULT_list)
#         continue
    
#     numerator=u[1] + a[1] + i[0] + u[0]
#     demominator=i[1] + a[0]
#     RESULT_dict['FCR']=np.float(numerator/demominator)
#     RESULT_dict['F2i_u']= u[1]/i[1]
#     RESULT_dict['F1a_u']= u[0]/a[0]
#     # assert FCR <=2
    
#     RESULT_dict['VSA1']=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
    
#     RESULT_dict['LnVSA']=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
    
#     EDiu=np.sqrt((u[1]-i[1])**2+(u[0]-i[0])**2)
#     EDia=np.sqrt((a[1]-i[1])**2+(a[0]-i[0])**2)
#     EDau=np.sqrt((u[1]-a[1])**2+(u[0]-a[0])**2)
#     S=(EDiu+EDia+EDau)/2
#     RESULT_dict['VSA2']=np.sqrt(S*(S-EDiu)*(S-EDia)*(S-EDau))
    
#     RESULT_dict['LnVSA']=np.sqrt(np.log(S)*(np.log(S)-np.log(EDiu))*(np.log(S)-np.log(EDia))*(np.log(S)-np.log(EDau)))
    
#     ''' a u i distance '''
#     RESULT_dict['dau1'] = np.abs(a[0] - u[0])
#     RESULT_dict['dai1'] = np.abs(a[0] - i[0])
#     RESULT_dict['diu1'] = np.abs(i[0] - u[0])
#     RESULT_dict['daudai1'] = RESULT_dict['dau1'] + RESULT_dict['dai1']
#     RESULT_dict['daudiu1'] = RESULT_dict['dau1'] + RESULT_dict['diu1']
#     RESULT_dict['daidiu1'] = RESULT_dict['dai1'] + RESULT_dict['diu1']
#     RESULT_dict['daidiudau1'] = RESULT_dict['dai1'] + RESULT_dict['diu1']+ RESULT_dict['dau1']
    
#     RESULT_dict['dau2'] = np.abs(a[1] - u[1])
#     RESULT_dict['dai2'] = np.abs(a[1] - i[1])
#     RESULT_dict['diu2'] = np.abs(i[1] - u[1])
#     RESULT_dict['daudai2'] = RESULT_dict['dau2'] + RESULT_dict['dai2']
#     RESULT_dict['daudiu2'] = RESULT_dict['dau2'] + RESULT_dict['diu2']
#     RESULT_dict['daidiu2'] = RESULT_dict['dai2'] + RESULT_dict['diu2']
#     RESULT_dict['daidiudau2'] = RESULT_dict['dai2'] + RESULT_dict['diu2']+ RESULT_dict['dau2']
    
#     # =============================================================================
#     ''' F-value, Valid Formant measure '''
    
#     # =============================================================================
#     # Get data
#     F12_raw_dict=Vowels_AUI[people]
#     u=F12_raw_dict['u']
#     a=F12_raw_dict['a']
#     i=F12_raw_dict['i']
#     df_vowel = pd.DataFrame(np.vstack([u,a,i]),columns=['F1','F2'])
#     df_vowel['vowel'] = np.hstack([np.repeat('u',len(u)),np.repeat('a',len(a)),np.repeat('i',len(i))])
#     df_vowel['target']=pd.Categorical(df_vowel['vowel'])
#     df_vowel['target']=df_vowel['target'].cat.codes
#     # F-test
#     print("utt number of group u = {0}, utt number of group i = {1}, utt number of group A = {2}".format(\
#         len(u),len(a),len(i)))
#     F_vals=f_classif(df_vowel[['F1','F2']].values,df_vowel['target'].values)[0]
#     RESULT_dict['F_vals_f1']=F_vals[0]
#     RESULT_dict['F_vals_f2']=F_vals[1]
#     RESULT_dict['F_val_mix']=RESULT_dict['F_vals_f1'] + RESULT_dict['F_vals_f2']
    
#     msb=f_classif(df_vowel[['F1','F2']].values,df_vowel['target'].values)[2]
#     msw=f_classif(df_vowel[['F1','F2']].values,df_vowel['target'].values)[3]
#     ssbn=f_classif(df_vowel[['F1','F2']].values,df_vowel['target'].values)[4]
    
    
    
#     RESULT_dict['MSB_f1']=msb[0]
#     RESULT_dict['MSB_f2']=msb[1]
#     MSB_f1 , MSB_f2 = RESULT_dict['MSB_f1'], RESULT_dict['MSB_f2']
#     RESULT_dict['MSB_mix']=MSB_f1 + MSB_f2
#     RESULT_dict['MSW_f1']=msw[0]
#     RESULT_dict['MSW_f2']=msw[1]
#     MSW_f1 , MSW_f2 = RESULT_dict['MSW_f1'], RESULT_dict['MSW_f2']
#     RESULT_dict['MSW_mix']=MSW_f1 + MSW_f2
#     RESULT_dict['SSBN_f1']=ssbn[0]
#     RESULT_dict['SSBN_f2']=ssbn[1]
    
#     # =============================================================================
#     # criterion
#     # F1u < F1a
#     # F2u < F2a
#     # F2u < F2i
#     # F1i < F1a
#     # F2a < F2i
#     # =============================================================================
#     u_mean=F12_val_dict['u']
#     a_mean=F12_val_dict['a']
#     i_mean=F12_val_dict['i']
    
#     F1u, F2u=u_mean[0], u_mean[1]
#     F1a, F2a=a_mean[0], a_mean[1]
#     F1i, F2i=i_mean[0], i_mean[1]
    
#     filt1 = [1 if F1u < F1a else 0]
#     filt2 = [1 if F2u < F2a else 0]
#     filt3 = [1 if F2u < F2i else 0]
#     filt4 = [1 if F1i < F1a else 0]
#     filt5 = [1 if F2a < F2i else 0]
#     RESULT_dict['criterion_score']=np.sum([filt1,filt2,filt3,filt4,filt5])

#     df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict)
#     df_RESULT_list.index=[people]
#     df_formant_statistic=df_formant_statistic.append(df_RESULT_list)