#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 12:21:17 2021

@author: jackchen
"""

'''
        1. Data prepare area
            a. Filter out data using by 1.5*IQR
        2-1. Personal timeseries generation (Details in TBME2021)
        2-2. Calculate LOC timeseries features within each defined timesteps (Details in TBME2021)
        3. Calculate syncrony features based on feature timeseries

        Input:  Formants_utt_symb
        Output: df_syncrony_measurement.loc[people,feature]
    
'''


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
from varname import nameof
from tqdm import tqdm
import re
from multiprocessing import Pool, current_process
from articulation.articulation import Articulation
import articulation.Multiprocess as Multiprocess
from datetime import datetime as dt
import pathlib

from scipy import special, stats
import warnings
from Syncrony import Syncrony
from SlidingWindow import slidingwindow as SW

from articulation.HYPERPARAM.PlotFigureVars import *

# =============================================================================
def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--inpklpath', default='articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--outpklpath', default='articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--dfFormantStatisticpath', default='articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--reFilter', default=False, type=bool,
                            help='')
    parser.add_argument('--check', default=True, type=bool,
                            help='')
    parser.add_argument('--label_choose_lst', default=['ADOS_C'],
                            help=['ADOS_C','dia_num'])
    parser.add_argument('--poolMed', default='middle',
                            help='path of the base directory')
    parser.add_argument('--poolWindowSize', default=3,
                            help='path of the base directory')
    parser.add_argument('--MinPhoneNum', default=3,
                            help='path of the base directory')
    # parser.add_argument('--Randseed', default=5998,
    #                         help='path of the base directory')
    parser.add_argument('--dataset_role', default='ASD_DOCKID',
                            help='[TD_DOCKID_emotion | ASD_DOCKID_emotion | kid_TD | kid88]')
    parser.add_argument('--Inspect_features', default=['F1','F2'],
                            help='')
    parser.add_argument('--Inspect_roles', default=['D','K'],
                            help='')
    parser.add_argument('--basic_columns', default=['u_num', 'a_num', 'i_num', 'ADOS_C', 'dia_num', 'sex', 'age', 'Module','ADOS_cate_C', 'u_num+i_num+a_num'],
                            help='')
    args = parser.parse_args()
    return args

args = get_args()

from utils_jack  import  Formant_utt2people_reshape, Gather_info_certainphones, \
                         FilterUttDictsByCriterion, GetValuelimit_IQR, \
                         Get_aligned_sequences, WER, Get_Vowels_AUI
from metric import Evaluation_method     
import random

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
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['ADOS_cate_C']==constrain_ADOScate)
        
    if len(evictNamelst)>0:
        for name in evictNamelst:
            filter_bool.loc[name]=False
    # get rid of nan values
    filter_bool=np.logical_and(filter_bool,~df_formant_statistic.isna().T.any())
    return df_formant_statistic[filter_bool]

def Add_label(df_formant_statistic,Label,label_choose='ADOS_cate_C'):
    for people in df_formant_statistic.index:
        bool_ind=Label.label_raw['name']==people
        df_formant_statistic.loc[people,label_choose]=Label.label_raw.loc[bool_ind,label_choose].values
    return df_formant_statistic

def GetPersonalSegmentFeature_map(keys_people, Formants_people_segment_role_utt_dict, People_data_distrib,\
                              PhoneMapp_dict, PhoneOfInterest ,\
                              Inspect_roles ,Inspect_features, In_Segments_order,\
                              Feature_calculator, vowel_min_num=3):
    Eval_med=Evaluation_method()
    
    Segments_order=In_Segments_order
    Vowels_AUI_info_dict=Dict()
    df_person_segment_feature_dict=Dict()
    MissingSegment_bag=[]
    for people in keys_people:
        Formants_segment_role_utt_dict=Formants_people_segment_role_utt_dict[people]
        if len(In_Segments_order) ==0 :
            Segments_order=sorted(list(Formants_segment_role_utt_dict.keys()))
            
        for role in Inspect_roles:
            df_person_segment_feature=pd.DataFrame([])
            for segment in Segments_order:
                Formants_utt_symb_SegmentRole=Formants_segment_role_utt_dict[segment][role]
                if len(Formants_utt_symb_SegmentRole)==0:
                    MissingSegment_bag.append([people,role,segment])
                
                # 解決數值問題，如果遇到空的或不足的就填補那個人acoustic value的統計值
                AUI_info_filled = Fill_n_Create_AUIInfo(Formants_utt_symb_SegmentRole, People_data_distrib, Inspect_features ,PhoneMapp_dict, PhoneOfInterest ,people, vowel_min_num)
    
                Vowels_AUI=Get_Vowels_AUI(AUI_info_filled, Inspect_features,VUIsource="From__Formant_people_information")
                Vowels_AUI_info_dict[people][segment][role]=Vowels_AUI #bookeeping
                
                
                df_formant_statistic=Feature_calculator.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_choose_lst)
                # add ADOS_cate_C to df_formant_statistic
                for i in range(len(df_formant_statistic)):
                    name=df_formant_statistic.iloc[i].name
                    df_formant_statistic.loc[name,'ADOS_cate_C']=Label.label_raw[Label.label_raw['name']==name]['ADOS_cate_C'].values
                df_formant_statistic['u_num+i_num+a_num']=df_formant_statistic['u_num'] +\
                                                df_formant_statistic['i_num'] +\
                                                df_formant_statistic['a_num']
                if len(PhoneOfInterest) >= 3:
                    df_formant_statistic=Eval_med._Postprocess_dfformantstatistic(df_formant_statistic)
                assert len(df_formant_statistic.columns) > 10 #check if df_formant_statistic is empty DF
                if len(df_person_segment_feature) == 0:
                    df_person_segment_feature=pd.DataFrame([],columns=df_formant_statistic.columns)
                df_person_segment_feature.loc[segment]=df_formant_statistic.loc[people]
            df_person_segment_feature_dict[people][role]=df_person_segment_feature

    return df_person_segment_feature_dict, Vowels_AUI_info_dict, MissingSegment_bag

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
    
    
def GetValuemeanstd(AUI_info_total,PhoneMapp_dict,Inspect_features):
    People_data_distrib=Dict()
    for people in AUI_info_total.keys():
        for phoneRepresent in PhoneMapp_dict.keys():
            df_values = AUI_info_total[people][phoneRepresent][AUI_info_total[people][phoneRepresent]['cmps'] == 'ori']
            People_data_distrib[people][phoneRepresent].means=df_values[Inspect_features].mean()
            People_data_distrib[people][phoneRepresent].stds=df_values[Inspect_features].std()
    return People_data_distrib

def FillData_NormDistrib(AUI_info,People_data_distrib,Inspect_features,PhoneOfInterest, vowel_min_num=3,\
                         verbose=True):
    # Fill data with random samples sampled from normal distribution with ones mean and std
    AUI_info_filled=Dict()
    for people in AUI_info.keys():  # there will be only one people
        for phonesymb, phonevalues in AUI_info[people].items():
            if phonesymb in PhoneOfInterest: # Only fill missing phones in PhoneOfInterest
                AUI_info_filled[people][phonesymb]=AUI_info[people][phonesymb]            
                if len(phonevalues) == 0:
                    phoneNum=0
                else:
                    phonevalues_ori=phonevalues[phonevalues['cmps']=='ori']
                    phoneNum=len(phonevalues_ori)
                if phoneNum < vowel_min_num:
                    Num2sample=vowel_min_num -phoneNum
                    df_filled_samples=pd.DataFrame()
                    for feat in Inspect_features:
                        mu=People_data_distrib[people][phonesymb].means.loc[feat]
                        std=People_data_distrib[people][phonesymb].stds.loc[feat]
                        
                        samples=np.random.normal(mu, std, Num2sample)
                        df_filled_samples[feat]=samples
                    df_filled_samples_cmp=df_filled_samples.copy()
                    df_filled_samples_cmp['cmps']='cmp'
                    df_filled_samples_ori=df_filled_samples.copy()
                    df_filled_samples_ori['cmps']='ori'
                    df_filled_samples=pd.concat([df_filled_samples_ori,df_filled_samples_cmp],axis=0)
                    AUI_info_filled[people][phonesymb]=AUI_info_filled[people][phonesymb].append(df_filled_samples)        

    return AUI_info_filled

def Fill_n_Create_AUIInfo(Formants_utt_symb_SegmentRole, People_data_distrib, Inspect_features ,PhoneMapp_dict, PhoneOfInterest ,people, vowel_min_num=3):
    Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb_SegmentRole,Formants_utt_symb_SegmentRole,Align_OrinCmp=False)
    AUI_info=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)
    
    if len(AUI_info) == 0: # If someone does not have the whole segment
                           # create a fake one for him  
        AUI_info=Dict()
        for phone_symb in PhoneMapp_dict.keys():
            AUI_info[people][phone_symb]=pd.DataFrame()
        
    AUI_info_filled = FillData_NormDistrib(AUI_info,People_data_distrib,Inspect_features,PhoneOfInterest, vowel_min_num)
    return AUI_info_filled




# =============================================================================
'''
    
    1. Data prepare area
    
        a. Filter out data using by 1.5*IQR

'''
# =============================================================================
''' parse namespace '''
args = get_args()
pklpath=args.inpklpath
windowsize=args.poolWindowSize
dfFormantStatisticpath=args.dfFormantStatisticpath
label_choose_lst=args.label_choose_lst # labels are too biased
dataset_role=args.dataset_role
# Randseed=args.Randseed
outpklpath=args.inpklpath+"/Session_formants_people_vowel_feat/"
if not os.path.exists(outpklpath):
    os.makedirs(outpklpath)


Formants_utt_symb=pickle.load(open(pklpath+"/Formants_utt_symb_by{0}_window{1}_{2}.pkl".format(args.poolMed,windowsize,dataset_role),'rb'))
print("Loading Formants_utt_symb from ", pklpath+"/Formants_utt_symb_by{0}_window{1}_{2}.pkl".format(args.poolMed,windowsize,dataset_role))

PhoneMapp_dict=phonewoprosody.PhoneMapp_dict
PhoneOfInterest=sorted(list(PhoneMapp_dict.keys()))

''' Vowel AUI rule is using phonewoprosody '''
Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
AUI_info_total=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)
limit_people_rule=GetValuelimit_IQR(AUI_info_total,PhoneMapp_dict,args.Inspect_features)


''' multi processing start '''
prefix,suffix = 'Formants_utt_symb', dataset_role
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



Formants_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
AUI_info_total=Gather_info_certainphones(Formants_people_information,PhoneMapp_dict,PhoneOfInterest)

# 這個部份要統計TASLP database部份要秀的phone數量
Phone_num_dict={}
for people in AUI_info_total.keys():
    corner_phone_num=0
    for phone in AUI_info_total[people].keys():
        corner_phone_num+=len(AUI_info_total[people][phone][AUI_info_total[people][phone]['cmps'] == 'ori'])
    Phone_num_dict[people]=corner_phone_num

df_Phone_num_dict=pd.DataFrame.from_dict(Phone_num_dict, orient='index')
df_Phone_num_dict[0].mean()

# =============================================================================
''' TD mean phone num: 591.55  '''
''' ASD mean phone num: 665.579  '''
''' (591.55*20+665.579*88)/(20+88)  '''

# =============================================================================


# =============================================================================
'''

    2-1. Personal timeseries generation (Details in TBME2021)
    
    We have several methods to determine each timestamp. 
        1. define timestamp by four emotion questions
        2. define timestamp by a minimum about of collected phones (TBME2021 uses this setting)
        3. define timestamp by first and last half
    
    Input: Formants_utt_symb
    Output: Formants_people_segment_role_utt_dict[people][timestep][role] -> raw data, 
            Formants_people_half_role_utt_dict[people][{first_half,last_half}[role] -> raw data, 
    

'''
HalfDesider={'first_half':["happy","afraid"],
                 'last_half':["angry","sad"]}
emotion2half={}
for key, values in HalfDesider.items():
    for v in values:        
        emotion2half[v]=key

# =============================================================================
def Reorder2Emotion_PER_utt(Formants_utt_symb, HalfDesider, PhonesOfInterest=['u:', 'i:', 'A:']):
    # 這個函數可以把原本Session level的Formants_utt_symb切成emotion segments或是對半切
    Formants_people_segment_role_utt_dict=Dict()
    Formants_people_half_role_utt_dict=Dict()
    FormantsUttKeys_numberOrder=sorted(list(Formants_utt_symb.keys()),key=lambda x: (x[:re.search("_[K|D]_",x).start()], int(x.split("_")[-1])))
    for i, keys in enumerate(FormantsUttKeys_numberOrder):
        values= Formants_utt_symb[keys]
        name=keys[:re.search("_[K|D]_",keys).start()]

        
        res_key_str=keys[re.search("_[K|D]_",keys).start()+1:]
        res_key = res_key_str.split("_")

        if len(res_key) != 3:
            raise ValueError("using emotion data, but Perhaps using the worng Alignments")
        role, turn_number, segment=res_key
        Formants_people_segment_role_utt_dict[name][segment][role][keys]=values
        
        for k, v in  HalfDesider.items():
            if segment in v:
                Formants_people_half_role_utt_dict[name][k][role][keys]=values
    
    return Formants_people_segment_role_utt_dict, Formants_people_half_role_utt_dict


def POI_List2str(list):
    return ','.join(list)

def POI_Str2list(str):
    return str.split(',')


if 'emotion' in  dataset_role:
    Formants_people_segment_role_utt_dict, Formants_people_half_role_utt_dict\
        =Reorder2Emotion_PER_utt(Formants_utt_symb, HalfDesider,PhoneOfInterest)

else:
    sliding_window=SW()
    # Formants_people_segment_role_utt_dict=Reorder2_PER_utt(Formants_utt_symb,PhoneOfInterest,MinNum=args.MinPhoneNum)
    Formants_people_segment_role_utt_dict=sliding_window.Reorder2_PER_utt_formants(Formants_utt_symb,PhoneMapp_dict,\
                                                           PhoneOfInterest,args.Inspect_roles,\
                                                           MinNum=args.MinPhoneNum)            
    Formants_people_half_role_utt_dict=Dict()
    Formants_people_whole_role_utt_dict=Dict()
    for people in Formants_people_segment_role_utt_dict.keys():
        split_num=len(Formants_people_segment_role_utt_dict[people])//2
        for segment in Formants_people_segment_role_utt_dict[people].keys():
            for role in Formants_people_segment_role_utt_dict[people][segment].keys():
                if segment <= split_num:
                    Formants_people_half_role_utt_dict[people]['first_half'][role].update(Formants_people_segment_role_utt_dict[people][segment][role])
                else:
                    Formants_people_half_role_utt_dict[people]['last_half'][role].update(Formants_people_segment_role_utt_dict[people][segment][role])
                Formants_people_whole_role_utt_dict[people]['whole'][role].update(Formants_people_segment_role_utt_dict[people][segment][role])

# Data Statistics for the use of filling empty segments
People_data_distrib=GetValuemeanstd(AUI_info_total,PhoneMapp_dict,args.Inspect_features)



# =============================================================================
'''
    2-2. Calculate LOC timeseries features within each defined timesteps (Details in TBME2021)
    
Prepare df_formant_statistics for each segment

Input:  Formants_people_segment_role_utt_dict
Output: df_person_segment_feature_dict[people][role]=df_person_segment_feature

df_person_segment_feature=
        u_num  a_num  ...  linear_discriminant_covariance(A:,i:)  ADOS_cate_C
happy     5.0    6.0  ...                          -4.877535e-15        1.0
afraid    3.0    3.0  ...                           8.829873e-18        1.0
angry     6.0   13.0  ...                           1.658604e-17        1.0
sad       7.0   10.0  ...                           0.000000e+00        1.0

'''
emotion_timeorder=['happy', 'afraid', 'angry', 'sad']
articulation=Articulation()
# =============================================================================
keys=[]
interval=5
for i in range(0,len(People_data_distrib.keys()),interval):
    keys.append(list(People_data_distrib.keys())[i:i+interval])
flat_keys=[item for sublist in keys for item in sublist]
assert len(flat_keys) == len(People_data_distrib)

pool = Pool(os.cpu_count())
if 'emotion' in  dataset_role:
    Segment_lst=emotion_timeorder
else:
    Segment_lst=[]

final_result = pool.starmap(GetPersonalSegmentFeature_map, [(key,Formants_people_segment_role_utt_dict,People_data_distrib, \
                                  PhoneMapp_dict, PhoneOfInterest ,\
                                  args.Inspect_roles, args.Inspect_features,\
                                  Segment_lst, articulation) for key in tqdm(keys)])
print('GetPersonalSegmentFeature_map segment done !!!')
df_person_segment_feature_dict=Dict()
Vowels_AUI_info_segments_dict=Dict()
MissSeg=[]
for d, vowelinfoseg, missSeg in tqdm(final_result):
    MissSeg.extend(missSeg) #Bookeep the people that the timestamps are missing 
    for spk in d.keys():
        df_person_segment_feature_dict[spk]=d[spk]
        Vowels_AUI_info_segments_dict[spk]=d[spk]

# Some people having too little timeseries will have zero last-half, so Fill_n_Create_AUIInfo
# will fill values to each phone, each role. The total filled message will be len(Missbag) * 3
final_result = pool.starmap(GetPersonalSegmentFeature_map, [(key,Formants_people_half_role_utt_dict,People_data_distrib, \
                              PhoneMapp_dict, PhoneOfInterest ,\
                              args.Inspect_roles, args.Inspect_features,\
                              list(HalfDesider.keys()), articulation) for key in keys])
print('GetPersonalSegmentFeature_map done')
df_person_half_feature_dict=Dict()
Vowels_AUI_half_dict=Dict()
MissHalf=[]
for d, vowelinfoseg, missHal in tqdm(final_result):
    MissHalf.extend(missHal) #Bookeep the people that the first/last half is missing 
    for spk in d.keys():
        df_person_half_feature_dict[spk]=d[spk]
        Vowels_AUI_half_dict[spk]=d[spk]


final_result = pool.starmap(GetPersonalSegmentFeature_map, [(key,Formants_people_whole_role_utt_dict,People_data_distrib, \
                              PhoneMapp_dict, PhoneOfInterest ,\
                              args.Inspect_roles, args.Inspect_features,\
                              ['whole'], articulation) for key in keys])

# This section is to inspect features if we collect vowel information for the whole session    
# The purpose of doing this is because the whole session feature should equal or be similar to LOC features (The total session as a timestep)
print('GetPersonalSegmentFeature_map whole done')
df_person_whole_feature_dict=Dict()
Vowels_AUI_whold_dict=Dict()
for d, vowelinfoseg, missHal in tqdm(final_result):
    for spk in d.keys():
        df_person_whole_feature_dict[spk]=d[spk]

# This will be used in Statistical tests
pickle.dump(df_person_segment_feature_dict,open(outpklpath+"df_person_segment_feature_dict_{0}_{1}.pkl".format(dataset_role, 'formant'),"wb"))


# =============================================================================
'''

    3. Calculate syncrony features based on feature timeseries


    Input: df_person_segment_feature_dict
    Output: 

'''



features=['VSA2',
       'FCR2', 'between_covariance_norm(A:,i:,u:)',
       'between_variance_norm(A:,i:,u:)', 'between_covariance(A:,i:,u:)',
       'between_variance(A:,i:,u:)', 'within_covariance_norm(A:,i:,u:)',
       'within_variance_norm(A:,i:,u:)', 'within_covariance(A:,i:,u:)',
       'within_variance(A:,i:,u:)', 'total_covariance_norm(A:,i:,u:)',
       'total_variance_norm(A:,i:,u:)', 'sam_wilks_lin_norm(A:,i:,u:)',
       'pillai_lin_norm(A:,i:,u:)', 'hotelling_lin_norm(A:,i:,u:)',
       'roys_root_lin_norm(A:,i:,u:)',
       'Between_Within_Det_ratio_norm(A:,i:,u:)',
       'Between_Within_Tr_ratio_norm(A:,i:,u:)', 'ConvexHull', 'MeanVFD',
       'SumVFD',  'dcov_12', 'dcorr_12', 'dvar_1', 'dvar_2',
       'pear_12', 'pointDistsTotal', 'repulsive_force',
       'ADOS_cate_C', 'u_num+i_num+a_num']
exclude_cols=['ADOS_cate_C']   # To avoid labels that will cause errors 
                             # Covariance of only two classes are easily to be zero
FilteredFeatures = [c for c in features if c not in exclude_cols]
# =============================================================================

syncrony=Syncrony()
PhoneOfInterest_str=''
df_syncrony_measurement=syncrony.calculate_features(df_person_segment_feature_dict, df_person_half_feature_dict,\
                               FilteredFeatures,PhoneOfInterest_str,\
                               args.Inspect_roles, Label,\
                               MinNumTimeSeries=2, label_choose_lst=['ADOS_C'])

timeSeries_len_columns=[col  for col in df_syncrony_measurement.columns if 'timeSeries_len' in col]
df_syncrony_measurement['timeSeries_len']=df_syncrony_measurement[timeSeries_len_columns].min(axis=1)
    

feat_type='Syncrony_formant'
N=2
if dataset_role == 'ASD_DOCKID':
    df_syncrony_measurement=criterion_filter(df_syncrony_measurement,N=N,evictNamelst=[],feature_type=feat_type)
    

pickle.dump(df_syncrony_measurement,open(outpklpath+"Syncrony_measure_of_variance_{}.pkl".format(dataset_role),"wb"))



# =============================================================================
# Generate LOC indexes for fraction people for ASD/non-ASD classification
# 以下不需要了因為我們最後沒有手動選擇mild/severe組別的個案
# =============================================================================
dfFormantStatisticFractionpath=dfFormantStatisticpath+'/Fraction'
if not os.path.exists(dfFormantStatisticFractionpath):
    os.makedirs(dfFormantStatisticFractionpath)
sellect_people_define=SellectP_define()
if dataset_role == 'ASD_DOCKID':
    df_syncrony_statistic_agesexmatch_ASDSevere=df_syncrony_measurement.loc[sellect_people_define.SevereASD_age_sex_match_ver2]
    df_syncrony_statistic_agesexmatch_ASDMild=df_syncrony_measurement.loc[sellect_people_define.MildASD_age_sex_match_ver2]
    
    label_add='ADOS_cate_C'
    if label_add  not in df_syncrony_statistic_agesexmatch_ASDSevere.columns:
        df_syncrony_statistic_agesexmatch_ASDSevere=Add_label(df_syncrony_statistic_agesexmatch_ASDSevere,Label,label_choose=label_add)
    if label_add  not in df_syncrony_statistic_agesexmatch_ASDMild.columns:
        df_syncrony_statistic_agesexmatch_ASDMild=Add_label(df_syncrony_statistic_agesexmatch_ASDMild,Label,label_choose=label_add)
    
    # 1 represents ASD, 2 represents TD
    label_add='ASDTD' 
    if label_add not in df_syncrony_statistic_agesexmatch_ASDSevere.columns:
        df_syncrony_statistic_agesexmatch_ASDSevere[label_add]=sellect_people_define.ASDTD_label['ASD']
    if label_add not in df_syncrony_statistic_agesexmatch_ASDMild.columns:
        df_syncrony_statistic_agesexmatch_ASDMild[label_add]=sellect_people_define.ASDTD_label['ASD']
        
    pickle.dump(df_syncrony_statistic_agesexmatch_ASDSevere,open(dfFormantStatisticFractionpath+'/df_syncrony_statistic_agesexmatch_ASDSevereGrp.pkl','wb'))
    pickle.dump(df_syncrony_statistic_agesexmatch_ASDMild,open(dfFormantStatisticFractionpath+'/df_syncrony_statistic_agesexmatch_ASDMildGrp.pkl','wb'))
    
elif dataset_role == 'TD_DOCKID':
    df_syncrony_TD_normal=df_syncrony_measurement.loc[sellect_people_define.TD_normal_ver2]
    
    # 1 represents ASD, 2 represents TD
    label_add='ASDTD' 
    if label_add not in df_syncrony_TD_normal.columns:
        df_syncrony_TD_normal[label_add]=sellect_people_define.ASDTD_label['TD']
        
    pickle.dump(df_syncrony_TD_normal,open(dfFormantStatisticFractionpath+'/df_syncrony_statistic_TD_normalGrp.pkl','wb'))    
else:
    raise KeyError("The key has not been registered")

