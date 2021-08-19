#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 12:21:17 2021

@author: jackchen
"""

'''
    
        1. Keep in mind that last frame may have unenough  vowels, so sometimes the 
           latter code will fill arbitrary data
        
        2. Some people having too little timeseries will have zero last-half, so Fill_n_Create_AUIInfo
           will fill values to each phone, each role. The total filled message will be len(Missbag) * 3
    
'''


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
import articulation.Multiprocess as Multiprocess
from datetime import datetime as dt
import pathlib

from scipy import special, stats
import warnings
from Syncrony import Syncrony

# =============================================================================
def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--inpklpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--outpklpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--reFilter', default=True, type=bool,
                            help='')
    parser.add_argument('--check', default=True, type=bool,
                            help='')
    parser.add_argument('--label_choose_lst', default=['ADOS_C'],
                            help=['ADOS_C','dia_num'])
    parser.add_argument('--MinNum', default=3,
                            help='path of the base directory')
    # parser.add_argument('--Randseed', default=5998,
    #                         help='path of the base directory')
    parser.add_argument('--dataset_role', default='ASD_DOCKID',
                            help='[TD_DOCKID_emotion | ASD_DOCKID_emotion | kid_TD | kid88]')
    parser.add_argument('--Inspect_roles', default=['D','K'],
                            help='')
    parser.add_argument('--Inspect_features_phonations', default=['stdevF0','localJitter', 'localabsoluteJitter','localShimmer'],
                            help='')
    parser.add_argument('--basic_columns', default=['u_num', 'a_num', 'i_num', 'ADOS_C', 'dia_num', 'sex', 'age', 'Module','ADOS_cate', 'u_num+i_num+a_num'],
                            help='')
    args = parser.parse_args()
    return args

args = get_args()

from utils_jack  import  Formant_utt2people_reshape, Gather_info_certainphones, \
                         FilterUttDictsByCriterion, GetValuelimit_IQR, \
                         Get_aligned_sequences, WER, Get_Vowels_AUI
from metric import Evaluation_method     
import random
from phonation.phonation import  Phonation

def GetPersonalSegmentFeature_map(keys_people, Formants_people_segment_role_utt_dict, People_data_distrib,\
                              PhoneMapp_dict, PhoneOfInterest ,\
                              Inspect_roles ,Inspect_features, In_Segments_order,\
                              Feature_calculator, vowel_min_num=3):
    
    # keys_people, Formants_people_segment_role_utt_dict, People_data_distrib,\
    #                           PhoneMapp_dict, PhoneOfInterest ,\
    #                           Inspect_roles ,Inspect_features, In_Segments_order,\
    #                           Feature_calculator, vowel_min_num=key,Phonation_people_segment_role_utt_dict,People_data_distrib_phonation, \
    #                                   PhoneMapp_dict, PhoneOfInterest ,\
    #                                   args.Inspect_roles, args.Inspect_features_phonations,\
    #                                   Segment_lst, phonation, phonation.vowel_min_num

    # Inspect_roles = args.Inspect_roles
    # Segments_order = emotion_timeorder
    Eval_med=Evaluation_method()
    
    Segments_order=In_Segments_order
    Vowels_AUI_info_dict=Dict()
    df_person_segment_feature_dict=Dict()
    MissingSegment_bag=[]
    for people in tqdm(keys_people):
        Formants_segment_role_utt_dict=Formants_people_segment_role_utt_dict[people]
        if len(In_Segments_order) ==0 :
            Segments_order=sorted(list(Formants_segment_role_utt_dict.keys()))
            
        for role in Inspect_roles:
            df_person_segment_feature=pd.DataFrame([])
            for segment in Segments_order:
                Formants_utt_symb_SegmentRole=Formants_segment_role_utt_dict[segment][role]
                if len(Formants_utt_symb_SegmentRole)==0:
                    MissingSegment_bag.append([people,role,segment])
                
                AUI_info_filled = Fill_n_Create_AUIInfo(Formants_utt_symb_SegmentRole, People_data_distrib, Inspect_features ,PhoneMapp_dict, PhoneOfInterest ,people, vowel_min_num)
    
                Vowels_AUI=Get_Vowels_AUI(AUI_info_filled, Inspect_features,VUIsource="From__Formant_people_information")
                Vowels_AUI_info_dict[people][segment][role]=Vowels_AUI #bookeeping
                
                
                df_formant_statistic=Feature_calculator.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_choose_lst)
                # add ADOS_cate to df_formant_statistic
                for i in range(len(df_formant_statistic)):
                    name=df_formant_statistic.iloc[i].name
                    df_formant_statistic.loc[name,'ADOS_cate']=Label.label_raw[Label.label_raw['name']==name]['ADOS_cate'].values
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
            
            # df_person_segment_feature_check=df_person_segment_feature.drop(columns=args.basic_columns)
            # if df_person_segment_feature_check.isnull().values.any():
            #     print('people ',people,' contains nan')
            #     assert not df_person_segment_feature_check.isnull().values.any()
    return df_person_segment_feature_dict, Vowels_AUI_info_dict, MissingSegment_bag

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
                    if verbose:
                        print("Filled ", Num2sample ,'EmptIES ')
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

'''
# =============================================================================
''' parse namespace '''
args = get_args()
pklpath=args.inpklpath
label_choose_lst=args.label_choose_lst # labels are too biased
dataset_role=args.dataset_role
# Randseed=args.Randseed
outpklpath=args.inpklpath+"/Session_formants_people_vowel_feat/"
if not os.path.exists(outpklpath):
    os.makedirs(outpklpath)

# np.random.seed(Randseed)
# os.environ['PYTHONHASHSEED'] = str(Randseed)
# random.seed(Randseed)

Phonation_utt_symb=pickle.load(open(pklpath+"/Phonation_utt_symb_{role}.pkl".format(role=dataset_role),'rb'))

# =============================================================================
'''

    Filter out data using by 1.5*IQR

'''
PhoneMapp_dict=phonewoprosody.PhoneMapp_dict
PhoneOfInterest=list(PhoneMapp_dict.keys())

# =============================================================================
'''

    Filter F1 F2 by IQR method

'''
# =============================================================================
''' Filter unqualified Phonation vowels '''
Phonation_people_information=Formant_utt2people_reshape(Phonation_utt_symb,Phonation_utt_symb,Align_OrinCmp=False)
AUI_info_phonation_total=Gather_info_certainphones(Phonation_people_information,PhoneMapp_dict,PhoneOfInterest)
limit_people_rule_Phonation=GetValuelimit_IQR(AUI_info_phonation_total,PhoneMapp_dict,args.Inspect_features_phonations)

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

    Reformat data to segmental dicts

'''
# =============================================================================


HalfDesider={'first_half':["happy","afraid"],
                 'last_half':["angry","sad"]}
emotion2half={}
for key, values in HalfDesider.items():
    for v in values:        
        emotion2half[v]=key



    

# =============================================================================
def Reorder2Emotion_PER_utt(Formants_utt_symb, HalfDesider, PhonesOfInterest=['u:', 'i:', 'A:']):
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

def Reorder2_PER_utt(Formants_utt_symb, PhonesOfInterest=['u:', 'i:', 'A:'],MinNum=3):
    # segment_timeorder=['happy', 'afraid', 'angry', 'sad'],\
    # inspect_roles=['D','K']
    # =============================================================================
    '''
    
        Keep in mind that last frame may have unenough  vowels, so sometimes the 
        latter code will fill arbitrary data
    
    '''
    def Initialize_VowelStack(PhoneMapp_dict, PhonesOfInterest):
        Vowel_stack={k:pd.DataFrame([]) for k in PhonesOfInterest }
        return Vowel_stack
    
    
    def Initialize_VowelStackPair(PhoneMapp_dict, Inspect_roles, PhonesOfInterest):
        Vowel_stack_DK=Dict()
        for R in Inspect_roles:
            Vowel_stack=Initialize_VowelStack(PhoneMapp_dict, PhonesOfInterest)
            Vowel_stack_DK[R]=Vowel_stack
        return Vowel_stack_DK
    
    def PhoneIn_check(phone, PhoneMapp_dict, PhonesOfInterest):
        for k in PhonesOfInterest:
            for p in PhoneMapp_dict[k]:
                if phone in p:
                    return k
        return -1
    
    def Fillin_VowelStack(Vowel_stack, values, PhoneMapp_dict, PhonesOfInterest):
        # put target vowels in stack
        for i,ind in enumerate(list(values.index)):
            if PhoneIn_check(ind, PhoneMapp_dict, PhonesOfInterest) != -1:
                Vowel_stack[PhoneIn_check(ind, PhoneMapp_dict, PhonesOfInterest)]=\
                    Vowel_stack[PhoneIn_check(ind, PhoneMapp_dict, PhonesOfInterest)].append(values.iloc[i])
        return Vowel_stack
    def Check_VowelStackFull(Vowel_stack, minNum=3):
        cond= True
        for p_symb in Vowel_stack.keys():
            cond = cond and len(Vowel_stack[p_symb]) >= minNum
        return cond
    
    def Refill_VowelStack_condition(Vowel_stack_DK,minNum=3):
        cond_top=True
        for R in Vowel_stack_DK.keys():
            cond=Check_VowelStackFull(Vowel_stack_DK[R], minNum=minNum)
            cond_top = cond_top and cond
        return cond_top
    def Check_nextperson(nextname, name):
        if nextname != name:
            return True
        else:
            return False
    # =============================================================================
    
    
    
    Vowel_stack_DK=Initialize_VowelStackPair(PhoneMapp_dict, args.Inspect_roles, PhonesOfInterest)
    Formants_people_segment_role_utt_dict=Dict()
    
    FormantsUttKeys_numberOrder=sorted(list(Formants_utt_symb.keys()),key=lambda x: (x[:re.search("_[K|D]_",x).start()], int(x.split("_")[-1])))
    NumberOrder_checkdict=Dict()
    segment_Num=0
    for i, keys in enumerate(FormantsUttKeys_numberOrder):
        values= Formants_utt_symb[keys]
        name=keys[:re.search("_[K|D]_",keys).start()]
        if i < len(FormantsUttKeys_numberOrder)-1:
            nextname=FormantsUttKeys_numberOrder[i+1][:re.search("_[K|D]_",FormantsUttKeys_numberOrder[i+1]).start()]
        
        
        res_key_str=keys[re.search("_[K|D]_",keys).start()+1:]
        res_key = res_key_str.split("_")
        if len(res_key) != 2:
            raise ValueError("not using emotion data, and Perhaps using the worng Alignments")
        role, turn_number=res_key
        if args.check:
            if name not in NumberOrder_checkdict.keys():
                NumberOrder_checkdict[name]=[]
            NumberOrder_checkdict[name].append(int(turn_number))

        Vowel_stack = Vowel_stack_DK[role].copy()  # We use Vowel_stack to bookeep a u i vowels
        Vowel_stack = Fillin_VowelStack(Vowel_stack, values, PhoneMapp_dict, PhonesOfInterest)# function: Fillin_VowelStack, fills PhoneOfInterests from value to the Vowel stack
        Vowel_stack_DK[role] = Vowel_stack # Vowel_stack for both Doctor and kid
        cond = Refill_VowelStack_condition(Vowel_stack_DK, minNum=MinNum) # The condition to renew the Vowel_stack is that both Doc and Kid should have enough PhoneOfInterests
        # cond=Check_VowelStackFull(Vowel_stack_DK[role], minNum=2)
        Formants_people_segment_role_utt_dict[name][segment_Num][role][keys]=values
        if Check_nextperson(nextname, name) or i == len(FormantsUttKeys_numberOrder)-1: 
                                              # If next utterance is from another person (or last person)
                                              # Reinitialize Vowel_stack, set segment_Num to 0 
                                              # and delete the incomplete segment
            if not cond: # The number of vowels does not satisfy to renew Vowel_stack_DK
                del Formants_people_segment_role_utt_dict[name][segment_Num]
            segment_Num = 0
            Vowel_stack_DK=Initialize_VowelStackPair(PhoneMapp_dict, args.Inspect_roles, PhonesOfInterest)
            continue
        
        if cond:
            Vowel_stack_DK=Initialize_VowelStackPair(PhoneMapp_dict, args.Inspect_roles, PhonesOfInterest)
            segment_Num+=1 #next segment_Num will be different

    return Formants_people_segment_role_utt_dict


def POI_List2str(list):
    return ','.join(list)

def POI_Str2list(str):
    return str.split(',')


Phonation_POI_people_segment_role_utt_dict=Dict()
for PhoneOfInterest in [['A:'],['u:'],['i:'],sorted(list(PhoneMapp_dict.keys()))]:
    if 'emotion' in  dataset_role:
        Phonation_people_segment_role_utt_dict, Phonation_people_half_role_utt_dict\
            =Reorder2Emotion_PER_utt(Phonation_utt_symb, HalfDesider,PhoneOfInterest)
    else:
        Phonation_people_segment_role_utt_dict=Reorder2_PER_utt(Phonation_utt_symb,PhoneOfInterest,MinNum=args.MinNum)
                                                                            
        Phonation_people_half_role_utt_dict=Dict()
        for people in Phonation_people_segment_role_utt_dict.keys():
            split_num=len(Phonation_people_segment_role_utt_dict[people])//2
            for segment in Phonation_people_segment_role_utt_dict[people].keys():
                for role in Phonation_people_segment_role_utt_dict[people][segment].keys():
                    if segment <= split_num:
                        Phonation_people_half_role_utt_dict[people]['first_half'][role].update(Phonation_people_segment_role_utt_dict[people][segment][role])
                    else:
                        Phonation_people_half_role_utt_dict[people]['last_half'][role].update(Phonation_people_segment_role_utt_dict[people][segment][role])
    
    Phonation_POI_people_segment_role_utt_dict[POI_List2str(PhoneOfInterest)].segment=Phonation_people_segment_role_utt_dict
    Phonation_POI_people_segment_role_utt_dict[POI_List2str(PhoneOfInterest)].half=Phonation_people_half_role_utt_dict

# Data Statistics for the use of filling empty segments
# People_data_distrib=GetValuemeanstd(AUI_info_total,PhoneMapp_dict,args.Inspect_features)

People_data_distrib_phonation=GetValuemeanstd(AUI_info_phonation_total,PhoneMapp_dict,args.Inspect_features_phonations)


# =============================================================================
'''
Prepare df_formant_statistics for each segment

df_person_segment_feature_dict[people][role]=df_person_segment_feature

df_person_segment_feature=

        u_num  a_num  ...  linear_discriminant_covariance(A:,i:)  ADOS_cate
happy     5.0    6.0  ...                          -4.877535e-15        1.0
afraid    3.0    3.0  ...                           8.829873e-18        1.0
angry     6.0   13.0  ...                           1.658604e-17        1.0
sad       7.0   10.0  ...                           0.000000e+00        1.0

'''
emotion_timeorder=['happy', 'afraid', 'angry', 'sad']
phonation=Phonation(Inspect_features=args.Inspect_features_phonations)
phonation._updateISSegmentFeature(True)
# =============================================================================
df_POI_person_segment_feature_dict=Dict()

# cols=['A:']
for PhoneOfInterest_str in Phonation_POI_people_segment_role_utt_dict.keys():
    Phonation_people_segment_role_utt_dict=Phonation_POI_people_segment_role_utt_dict[PhoneOfInterest_str].segment
    Phonation_people_half_role_utt_dict=Phonation_POI_people_segment_role_utt_dict[PhoneOfInterest_str].half
    
    
    PhoneOfInterest=POI_Str2list(PhoneOfInterest_str)
    ''' This part for phonation '''            
    keys=[]
    interval=5
    for i in range(0,len(People_data_distrib_phonation.keys()),interval):
        # print(list(Utt_ctxdepP_dict.keys())[i:i+interval])
        keys.append(list(People_data_distrib_phonation.keys())[i:i+interval])
    flat_keys=[item for sublist in keys for item in sublist]
    assert len(flat_keys) == len(People_data_distrib_phonation)
    
    pool = Pool(os.cpu_count())
    # pool = Pool(2)
    if 'emotion' in  dataset_role:
        Segment_lst=emotion_timeorder
    else:
        Segment_lst=[]
    
    final_result = pool.starmap(GetPersonalSegmentFeature_map, [(key,Phonation_people_segment_role_utt_dict,People_data_distrib_phonation, \
                                      PhoneMapp_dict, PhoneOfInterest ,\
                                      args.Inspect_roles, args.Inspect_features_phonations,\
                                      Segment_lst, phonation, phonation.vowel_min_num) for key in tqdm(keys)])
    print('GetPersonalSegmentFeature_map done')
    df_person_segment_feature_dict=Dict()
    Vowels_AUI_info_segments_dict=Dict()
    MissSeg=[]
    for d, vowelinfoseg, missSeg in tqdm(final_result):
        MissSeg.extend(missSeg)
        for spk in d.keys():
            df_person_segment_feature_dict[spk]=d[spk]
            Vowels_AUI_info_segments_dict[spk]=d[spk]
    
    # Some people having too little timeseries will have zero last-half, so Fill_n_Create_AUIInfo
    # will fill values to each phone, each role. The total filled message will be len(Missbag) * 3
    final_result = pool.starmap(GetPersonalSegmentFeature_map, [(key,Phonation_people_half_role_utt_dict,People_data_distrib_phonation, \
                                  PhoneMapp_dict, PhoneOfInterest ,\
                                  args.Inspect_roles, args.Inspect_features_phonations,\
                                  list(HalfDesider.keys()), phonation, phonation.vowel_min_num) for key in tqdm(keys)])
    print('GetPersonalSegmentFeature_map done')
    df_person_half_feature_dict=Dict()
    Vowels_AUI_half_dict=Dict()
    MissHalf=[]
    for d, vowelinfoseg, missHal in tqdm(final_result):
        MissHalf.extend(missHal)
        for spk in d.keys():
            df_person_half_feature_dict[spk]=d[spk]
            Vowels_AUI_half_dict[spk]=d[spk]
    
    df_POI_person_segment_feature_dict[PhoneOfInterest_str].segment=df_person_segment_feature_dict
    df_POI_person_segment_feature_dict[PhoneOfInterest_str].half=df_person_half_feature_dict



# # Then merge All df_POI_person_segment_feature_dict sub dicts!!
# df_POI_person_segment_feature_merged_dict=Dict()
# for PhoneOfInterest_str in df_POI_person_segment_feature_dict.keys():
#     for segmenthalf in df_POI_person_segment_feature_dict[PhoneOfInterest_str].keys():
#         for people in df_POI_person_segment_feature_dict[PhoneOfInterest_str][segmenthalf].keys():
#             for role in df_POI_person_segment_feature_dict[PhoneOfInterest_str][segmenthalf][people].keys():
#                 if role not in df_POI_person_segment_feature_merged_dict[segmenthalf][people].keys():
#                     df_POI_person_segment_feature_merged_dict[segmenthalf][people][role]=pd.DataFrame()
#                 df_POI_person_segment_feature_merged_dict[segmenthalf][people][role]=\
#                     pd.concat([df_POI_person_segment_feature_merged_dict[segmenthalf][people][role],\
#                                df_POI_person_segment_feature_dict[PhoneOfInterest_str][segmenthalf][people][role]],axis=1)
#                 df_POI_person_segment_feature_merged_dict[segmenthalf][people][role] = \
#                     df_POI_person_segment_feature_merged_dict[segmenthalf][people][role].drop(columns=basic_columns)
#                 # df_POI_person_segment_feature_merged_dict[segmenthalf][people][role] = \
#                 #     df_POI_person_segment_feature_merged_dict[segmenthalf][people][role].loc[:,~df_POI_person_segment_feature_merged_dict[segmenthalf][people][role].columns.duplicated()]



# =============================================================================
'''

    Calculate syncrony features

'''
features=[
    'stdevF0_mean(A:,i:,u:)', 'localJitter_mean(A:,i:,u:)',
       'localabsoluteJitter_mean(A:,i:,u:)', 'localShimmer_mean(A:,i:,u:)',
       'stdevF0_var(A:,i:,u:)', 'localJitter_var(A:,i:,u:)',
       'localabsoluteJitter_var(A:,i:,u:)', 'localShimmer_var(A:,i:,u:)',
       ]
exclude_cols=['ADOS_cate']   # covariance of only two classes are easily to be zero
FilteredFeatures = [c for c in features if c not in exclude_cols]
# =============================================================================
            
syncrony=Syncrony()

Phonation_temporal_poolMed=['mean','var']
df_syncrony_measurement_phonation_all=pd.DataFrame()

for PhoneOfInterest_str in df_POI_person_segment_feature_dict.keys():
    df_POI_person_segment_feature_PhoneOfInterest_str_dict=df_POI_person_segment_feature_dict[PhoneOfInterest_str]
    df_person_segment_feature_dict=df_POI_person_segment_feature_PhoneOfInterest_str_dict.segment
    df_person_half_feature_dict=df_POI_person_segment_feature_PhoneOfInterest_str_dict.half
    
    features=['{0}_{1}({2})'.format(feat , pool, PhoneOfInterest_str) for pool in Phonation_temporal_poolMed for feat in args.Inspect_features_phonations]
    
    df_syncrony_measurement_phonation=syncrony.calculate_features(df_person_segment_feature_dict, df_person_half_feature_dict,\
                               features,PhoneOfInterest_str,\
                               args.Inspect_roles, Label,\
                               MinNumTimeSeries=2, label_choose_lst=['ADOS_C'])
    
        
    df_syncrony_measurement_phonation_all=pd.concat([df_syncrony_measurement_phonation_all,df_syncrony_measurement_phonation], axis=1)    
        
timeSeries_len_columns=[col  for col in df_syncrony_measurement_phonation_all.columns if 'timeSeries_len' in col]
df_syncrony_measurement_phonation_all['timeSeries_len']=df_syncrony_measurement_phonation_all[timeSeries_len_columns].min(axis=1)


df_syncrony_measurement_phonation_all=df_syncrony_measurement_phonation_all.loc[:,~df_syncrony_measurement_phonation_all.columns.duplicated()]

pickle.dump(df_syncrony_measurement_phonation_all,open(outpklpath+"Syncrony_measure_of_variance_phonation_{}.pkl".format(dataset_role),"wb"))

    
df_syncrony_measurement=df_syncrony_measurement_phonation_all
lst=[]
for col in df_syncrony_measurement.columns:
    if df_syncrony_measurement[col].isnull().values.any():
        lst.append(col)
# =============================================================================
''' Correaltion area ''' 
correlationColumns=df_syncrony_measurement.columns    

N=2
Eval_med=Evaluation_method()
Aaadf_spearmanr_table_NoLimit=Eval_med.Calculate_correlation(label_choose_lst,df_syncrony_measurement,\
                                                             N,correlationColumns,\
                                                             constrain_sex=-1, constrain_module=-1,\
                                                             feature_type='Syncrony_formant')


