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
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['ADOS_cate']==constrain_ADOScate)
    
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
    parser.add_argument('--reFilter', default=True,
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
    parser.add_argument('--dataset_role', default='kid88',
                            help='kid_TD| kid88')
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

    2. Feature calculating area


'''
# =============================================================================
Vowels_AUI=Get_Vowels_AUI(AUI_info, args.Inspect_features,VUIsource="From__Formant_people_information")


label_generate_choose_lst=['ADOS_C','dia_num']
articulation=Articulation(Stat_med_str_VSA='mean')
df_formant_statistic=articulation.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_generate_choose_lst)

# For pseudo acoustic features generation
df_formant_statistic['u_num+i_num+a_num']=df_formant_statistic['u_num'] +\
                                            df_formant_statistic['i_num'] +\
                                            df_formant_statistic['a_num']

for i in range(len(df_formant_statistic)):
    name=df_formant_statistic.iloc[i].name
    df_formant_statistic.loc[name,'ADOS_cate']=Label.label_raw[Label.label_raw['name']==name]['ADOS_cate'].values
    ''' ADOS_cate, cate stands for category '''
    
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

pickle.dump(df_formant_statistic,open(outpklpath+"Formant_AUI_tVSAFCRFvals_{}.pkl".format(role),"wb"))


''' Calculate correlations for Formant fetures'''
columns=df_formant_statistic.columns



ManualCondition=Dict()
suffix='.xlsx'
condfiles=glob.glob('Inspect/condition/*'+suffix)
for file in condfiles:
    df_cond=pd.read_excel(file)
    name=os.path.basename(file).replace(suffix,"")
    ManualCondition[name]=df_cond['Unnamed: 0'][df_cond['50%']==True]

label_correlation_choose_lst=['ADOS_C',]


N=2
Eval_med=Evaluation_method()
Aaadf_spearmanr_table_NoLimit=Eval_med.Calculate_correlation(label_correlation_choose_lst,df_formant_statistic,N,columns,constrain_sex=-1, constrain_module=-1,feature_type='Session_formant')

# =============================================================================
#  Pearson
#  between_covariance(A:,i:,u:)       -0.346864  ...         86.0
#  between_variance(A:,i:,u:)         -0.465486  ...         86.0
# =============================================================================

















        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# =============================================================================
'''

    Calculate each vowel formant duration
    (This part is for debugging, and is not part of the paper TBME2021)

      
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



def plot(Vowels_AUI,outname=Plotout_path+'{0}_ADOS{1}',label_choose='ADOS_C'):
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




# =============================================================================
'''Function that is used but placed at the bottom to make the code neat''' 
# =============================================================================

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
