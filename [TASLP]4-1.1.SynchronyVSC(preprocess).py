#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 12:21:17 2021

@author: jackchen
"""

'''
這個腳本是源自舊版本中的[Debug]4.Calculate_coordinationindexes.py

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
from tqdm import tqdm
import re
from multiprocessing import Pool, current_process
from articulation.articulation import Articulation
# from articulation.articulation_cmp import Articulation
import articulation.Multiprocess as Multiprocess
from datetime import datetime as dt
import pathlib

from scipy import special, stats
import warnings
from Syncrony import Syncrony
from SlidingWindow import slidingwindow as SW

import matplotlib.pyplot as plt
import matplotlib
from sklearn import neighbors
import seaborn as sns
from pylab import text
# =============================================================================
def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--inpklpath', default='/media/jack/workspace/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--outpklpath', default='/media/jack/workspace/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--dfFormantStatisticpath', default='/media/jack/workspace/DisVoice/articulation/Pickles',
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
                            help='[TD_DOCKID | ASD_DOCKID_emotion | kid_TD | kid88]')
    parser.add_argument('--Inspect_features', default=['F1','F2'],
                            help='')
    parser.add_argument('--Inspect_roles', default=['D','K'],
                            help='')
    parser.add_argument('--basic_columns', default=['u_num', 'a_num', 'i_num', 'ADOS_C', 'dia_num', 'sex', 'age', 'Module','ADOS_cate_C', 'u_num+i_num+a_num'],
                            help='')
    parser.add_argument('--Normalize_way', default='proposed',
                            help='')
    args = parser.parse_args()
    return args

args = get_args()

from utils_jack  import  Formant_utt2people_reshape, Gather_info_certainphones, \
                         FilterUttDictsByCriterion, GetValuelimit_IQR, \
                         Get_aligned_sequences, WER, Get_Vowels_AUI
from metric import Evaluation_method     
import random
import copy

import os

class VariableSaver:
    def __init__(self, save_dir='./'):
        self.save_dir = save_dir
    def updateVar(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            print(f"更新變數 {key}：{value}")



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

# def GetPersonalSegmentFeature_map(keys_people, Formants_people_segment_role_utt_dict, People_data_distrib,\
#                               PhoneMapp_dict, PhoneOfInterest ,\
#                               Inspect_roles ,Inspect_features, In_Segments_order,\
#                               Feature_calculator, vowel_min_num=3):
#     Eval_med=Evaluation_method()
    
#     Segments_order=In_Segments_order
#     Vowels_AUI_info_dict=Dict()
#     df_person_segment_feature_dict=Dict()
#     MissingSegment_bag=[]
#     for people in keys_people:
#         Formants_segment_role_utt_dict=Formants_people_segment_role_utt_dict[people]
#         if len(In_Segments_order) ==0 :
#             Segments_order=sorted(list(Formants_segment_role_utt_dict.keys()))
            
#         for role in Inspect_roles:
#             df_person_segment_feature=pd.DataFrame([])
#             for segment in Segments_order:
#                 Formants_utt_symb_SegmentRole=Formants_segment_role_utt_dict[segment][role]
#                 if len(Formants_utt_symb_SegmentRole)==0:
#                     MissingSegment_bag.append([people,role,segment])
                
#                 AUI_info_filled = Fill_n_Create_AUIInfo(Formants_utt_symb_SegmentRole, People_data_distrib[role], Inspect_features ,PhoneMapp_dict, PhoneOfInterest ,people, vowel_min_num)
    
#                 Vowels_AUI=Get_Vowels_AUI(AUI_info_filled, Inspect_features,VUIsource="From__Formant_people_information")
#                 Vowels_AUI_info_dict[people][segment][role]=Vowels_AUI #bookeeping
                
                
#                 df_formant_statistic=Feature_calculator.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_choose_lst)
#                 # add ADOS_cate_C to df_formant_statistic
#                 for i in range(len(df_formant_statistic)):
#                     name=df_formant_statistic.iloc[i].name
#                     df_formant_statistic.loc[name,'ADOS_cate_C']=Label.label_raw[Label.label_raw['name']==name]['ADOS_cate_C'].values
#                 df_formant_statistic['u_num+i_num+a_num']=df_formant_statistic['u_num'] +\
#                                                 df_formant_statistic['i_num'] +\
#                                                 df_formant_statistic['a_num']
#                 if len(PhoneOfInterest) >= 3:
#                     df_formant_statistic=Eval_med._Postprocess_dfformantstatistic(df_formant_statistic)
#                 assert len(df_formant_statistic.columns) > 10 #check if df_formant_statistic is empty DF
#                 if len(df_person_segment_feature) == 0:
#                     df_person_segment_feature=pd.DataFrame([],columns=df_formant_statistic.columns)
#                 df_person_segment_feature.loc[segment]=df_formant_statistic.loc[people]
#             df_person_segment_feature_dict[people][role]=df_person_segment_feature

#     return df_person_segment_feature_dict, Vowels_AUI_info_dict, MissingSegment_bag
def GetPersonalSegmentFeature_map(keys_people, Formants_people_segment_role_utt_dict, People_data_distrib,\
                              PhoneMapp_dict, PhoneOfInterest ,df_general_info,\
                              Inspect_roles ,Inspect_features, In_Segments_order,\
                              Feature_calculator, vowel_min_num):
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
                
                AUI_info_filled = Fill_n_Create_AUIInfo(Formants_utt_symb_SegmentRole, People_data_distrib[role], Inspect_features ,PhoneMapp_dict, PhoneOfInterest ,people, vowel_min_num)
    
                Vowels_AUI=Get_Vowels_AUI(AUI_info_filled, Inspect_features,VUIsource="From__Formant_people_information")
                Vowels_AUI_info_dict[people][segment][role]=Vowels_AUI #bookeeping
                
                # Calculate articulation related features
                # Will contain only one person
                try:
                    df_formant_statistic=Feature_calculator.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_choose_lst)
                except:
                    print("Error people: ",people," Role = ",role)
                    raise ValueError()
                    
                if len(df_formant_statistic.columns) < 10:
                    print("error people= ",people)
                    print("Tryping to set N =2 for calculate_features",)
                    Feature_calculator._updateN(2)
                    df_formant_statistic=Feature_calculator.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_choose_lst)
                    if len(df_formant_statistic.columns) < 10:
                        print("people: ",people,"Not successfully generated")
                        
                        
                # Add informations to columns that are needed to df_formant_statistic
                for i in range(len(df_formant_statistic)):
                    name=df_formant_statistic.iloc[i].name
                    df_formant_statistic.loc[name,'ADOS_cate_C']=Label.label_raw[Label.label_raw['name']==name]['ADOS_cate_C'].values
                df_formant_statistic['u_num+i_num+a_num']=df_formant_statistic['u_num'] +\
                                                df_formant_statistic['i_num'] +\
                                                df_formant_statistic['a_num']
                query_lst=list(Formants_utt_symb_SegmentRole.keys())
                df_general_info_query=df_general_info.query("utt == @query_lst")
                
                # Padd a default IPU value for AUI_info_filled data
                if len(df_general_info_query['st'])==0:
                    IPU_start_time=0
                else:
                    IPU_start_time=df_general_info_query['st'].min()
                if len(df_general_info_query['ed'])==0:
                    IPU_end_time=25
                else:
                    IPU_end_time=df_general_info_query['ed'].max()
                df_formant_statistic['IPU_st']=IPU_start_time
                df_formant_statistic['IPU_ed']=IPU_end_time
                # if len(PhoneOfInterest) >= 3:
                #     df_formant_statistic=Eval_med._Postprocess_dfformantstatistic(df_formant_statistic)
                
                if len(df_person_segment_feature) == 0:
                    df_person_segment_feature=pd.DataFrame([],columns=df_formant_statistic.columns)
                df_person_segment_feature.loc[segment]=df_formant_statistic.loc[people]
                # try:
                #     df_person_segment_feature.loc[segment]=df_formant_statistic.loc[people]
                # except KeyError: 
                #     print(df_formant_statistic)
                #     raise KeyError
            df_person_segment_feature_dict[people][role]=df_person_segment_feature
    return df_person_segment_feature_dict, Vowels_AUI_info_dict, MissingSegment_bag


# def Process_IQRFiltering_Multi(Formants_utt_symb, limit_people_rule,\
#                                 outpath='/media/jack/workspace/DisVoice/articulation/Pickles',\
#                                 prefix='Formants_utt_symb',\
#                                 suffix='KID_FromASD_DOCKID'):
#     pool = Pool(int(os.cpu_count()))
#     keys=[]
#     interval=20
#     for i in range(0,len(Formants_utt_symb.keys()),interval):
#         # print(list(combs_tup.keys())[i:i+interval])
#         keys.append(list(Formants_utt_symb.keys())[i:i+interval])
#     flat_keys=[item for sublist in keys for item in sublist]
#     assert len(flat_keys) == len(Formants_utt_symb.keys())
#     muti=Multiprocess.Multi()
#     final_results=pool.starmap(muti.FilterUttDictsByCriterion_map, [([Formants_utt_symb,Formants_utt_symb,file_block,limit_people_rule]) for file_block in tqdm(keys)])
    
#     Formants_utt_symb_limited=Dict()
#     for load_file_tmp,_ in final_results:        
#         for utt, df_utt in load_file_tmp.items():
#             Formants_utt_symb_limited[utt]=df_utt
    
#     pickle.dump(Formants_utt_symb_limited,open(outpath+"/[Analyzing]{0}_limited_{1}.pkl".format(prefix,suffix),"wb"))
#     print('Formants_utt_symb saved to ',outpath+"/[Analyzing]{0}_limited_{1}.pkl".format(prefix,suffix))
    

# This is from other script
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


def GetValuemeanstd(AUI_info_total,PhoneMapp_dict,Inspect_features):
    People_data_distrib=Dict()
    for people in AUI_info_total.keys():
        for phoneRepresent in PhoneMapp_dict.keys():
            df_values = AUI_info_total[people][phoneRepresent][AUI_info_total[people][phoneRepresent]['cmps'] == 'ori']
            df_values_K=df_values[df_values['utt'].str.contains("_K_")]
            df_values_D=df_values[df_values['utt'].str.contains("_D_")]
            if len(df_values_D) == 0:
                People_data_distrib['D'][people][phoneRepresent].means=pd.Series([0]*len(Inspect_features),index=Inspect_features)
                People_data_distrib['D'][people][phoneRepresent].stds=pd.Series([0]*len(Inspect_features),index=Inspect_features)
            else:
                People_data_distrib['D'][people][phoneRepresent].means=df_values_D[Inspect_features].mean()
                if len(df_values_D) == 1:
                    People_data_distrib['D'][people][phoneRepresent].stds=pd.Series([0]*len(Inspect_features),index=Inspect_features)
                else:
                    People_data_distrib['D'][people][phoneRepresent].stds=df_values_D[Inspect_features].std()
            
            if len(df_values_K) == 0:
                People_data_distrib['K'][people][phoneRepresent].means=pd.Series([0]*len(Inspect_features),index=Inspect_features)
                People_data_distrib['K'][people][phoneRepresent].stds=pd.Series([0]*len(Inspect_features),index=Inspect_features)
            else:
                People_data_distrib['K'][people][phoneRepresent].means=df_values_K[Inspect_features].mean()
                if len(df_values_K) == 1:
                    People_data_distrib['K'][people][phoneRepresent].stds=pd.Series([0]*len(Inspect_features),index=Inspect_features)
                else:
                    People_data_distrib['K'][people][phoneRepresent].stds=df_values_K[Inspect_features].std()
            
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
Formants_utt_symb_onlyKid=pickle.load(open(pklpath+"/Formants_utt_symb_by{0}_window{1}_{2}.pkl".format(args.poolMed,windowsize,'KID_FromASD_DOCKID'),'rb'))



''' Get general timestamps '''
ASD_info_path='/media/jack/workspace/DisVoice/data/Segments_info_ADOS_ASD.txt'
TD_info_path='/media/jack/workspace/DisVoice/data/Segments_info_ADOS_TD.txt'
# Backup segments info path = /homes/ssd1/jackchen/gop_prediction
# ASD_info_path='/homes/ssd1/jackchen/gop_prediction/Segments_info_ADOS_ASD.txt'
# TD_info_path='/homes/ssd1/jackchen/gop_prediction/Segments_info_ADOS_TD.txt'


General_info_col=['utt','st','ed','txt','spk']
if dataset_role == 'TD_DOCKID':
    df_general_info=pd.read_csv(TD_info_path, sep='\t',header=None)
    df_general_info.columns=General_info_col
else:
    df_general_info=pd.read_csv(ASD_info_path, sep='\t',header=None)
    df_general_info.columns=General_info_col



'''End of get general timestamps '''


PhoneMapp_dict=phonewoprosody.PhoneMapp_dict
PhoneOfInterest=sorted(list(PhoneMapp_dict.keys()))

Formants_utt_symb_Top=Dict()
Formants_utt_symb_Top['K']={k:v for k,v in Formants_utt_symb.items() if "_K_" in k}
Formants_utt_symb_Top['D']={k:v for k,v in Formants_utt_symb.items() if "_D_" in k}


def Check_IfKID_FromASD_DOCKID_eq_ASD_DOCKID_K():    
    Formants_people_information=Formant_utt2people_reshape(Formants_utt_symb_Top['K'],Formants_utt_symb_Top['K'],Align_OrinCmp=False)
    AUI_info_total_fromDK=Gather_info_certainphones(Formants_people_information,PhoneMapp_dict,PhoneOfInterest)
    
    Formants_people_information=Formant_utt2people_reshape(Formants_utt_symb_onlyKid,Formants_utt_symb_onlyKid,Align_OrinCmp=False)
    AUI_info_total=Gather_info_certainphones(Formants_people_information,PhoneMapp_dict,PhoneOfInterest)
    
    
    #Compare AUI_info_total_fromDK & AUI_info_total
    for people in AUI_info_total.keys():
        for POI in PhoneOfInterest:
            assert (AUI_info_total[people][POI] == AUI_info_total_fromDK[people][POI] ).all().all()
    # The result is comfirmed

def Check_Iflimit_people_rule_fromDK_eq_limit_people_rule():
    Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb_Top['K'],Formants_utt_symb_Top['K'],Align_OrinCmp=False)
    AUI_info_total_fromDK=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)
    limit_people_rule_fromDK=GetValuelimit_IQR(AUI_info_total_fromDK,PhoneMapp_dict,args.Inspect_features)
    
    Formants_people_information=Formant_utt2people_reshape(Formants_utt_symb_onlyKid,Formants_utt_symb_onlyKid,Align_OrinCmp=False)
    AUI_info_total=Gather_info_certainphones(Formants_people_information,PhoneMapp_dict,PhoneOfInterest)
    limit_people_rule=GetValuelimit_IQR(AUI_info_total,PhoneMapp_dict,args.Inspect_features)
    
    for people in limit_people_rule.keys():
        for Phone in limit_people_rule[people]:
            for feat in limit_people_rule[people][Phone]:
                for max_min in limit_people_rule[people][Phone][feat]:
                    assert (limit_people_rule[people][Phone][feat][max_min] == limit_people_rule_fromDK[people][Phone][feat][max_min] )
    # The result is comfirmed
def Check_IflenFormants_utt_symb_Top_eq_Formants_utt_symb_onlyKid():
    for people in Formants_utt_symb_onlyKid.keys():
        Aac=Formants_utt_symb_Top[role][people]
        Aad=Formants_utt_symb_onlyKid[people]
    
        assert len(Aac) == len(Aad)

for role in Formants_utt_symb_Top.keys():
# for role in ['K']:
    ''' Vowel AUI rule is using phonewoprosody '''
    Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb_Top[role],Formants_utt_symb_Top[role],Align_OrinCmp=False)
    AUI_info_total=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)
    limit_people_rule=GetValuelimit_IQR(AUI_info_total,PhoneMapp_dict,args.Inspect_features)
    
    # Check_Iflimit_people_rule_fromDK_eq_limit_people_rule()
    # print("Check_Iflimit_people_rule_fromDK_eq_limit_people_rule passed")
    # Check_IfKID_FromASD_DOCKID_eq_ASD_DOCKID_K()
    # print("Check_IfKID_FromASD_DOCKID_eq_ASD_DOCKID_K passed")
    # Check_IflenFormants_utt_symb_Top_eq_Formants_utt_symb_onlyKid()
    # print("Check_IflenFormants_utt_symb_Top_eq_Formants_utt_symb_onlyKid passed")
    
    
    ''' multi processing start '''
    prefix,suffix = 'Formants_utt_symb_Top', '{dataset}_{role}'.format(dataset=dataset_role,role=role)
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
            Process_IQRFiltering_Multi(Formants_utt_symb_Top[role],limit_people_rule,\
                                   outpath=outpath,\
                                   prefix=prefix,\
                                   suffix=suffix) # the results will be output as pkl file at outpath+"/[Analyzing]Formants_utt_symb_Top[role]_limited.pkl"
    else:
        Process_IQRFiltering_Multi(Formants_utt_symb_Top[role],limit_people_rule,\
                                   outpath=outpath,\
                                   prefix=prefix,\
                                   suffix=suffix)    
    Formants_utt_symb_Top_limited=pickle.load(open(filepath,"rb"))
    ''' multi processing end '''
    if len(limit_people_rule) >0:
        Formants_utt_symb_Top[role]=Formants_utt_symb_Top_limited

def Check_IfFormants_utt_symb_Top_limited_onlyKid_eq_Formants_utt_symb_Top_limited():
    Formants_utt_symb_Top_limited_onlyKid=pickle.load(open('articulation/Pickles/[Analyzing]Formants_utt_symb_limited_KID_FromASD_DOCKID.pkl',"rb"))
    length_bag=[]
    for people in Formants_utt_symb_Top_limited.keys():
        length_bag.append(len(Formants_utt_symb_Top_limited_onlyKid[people]) - len(Formants_utt_symb_Top_limited[people] ))
        if len(Formants_utt_symb_Top_limited_onlyKid[people]) != len(Formants_utt_symb_Top_limited[people]):
            Aaa=Formants_utt_symb_Top_limited_onlyKid[people]
            Aab=Formants_utt_symb_Top_limited[people]
            Aac=Formants_utt_symb_Top[role][people]
            Aad=Formants_utt_symb_onlyKid[people]
            assert  len(Aaa) == len(Aab)
            assert len(Aac) == len(Aad)
        



Formants_utt_symb={**Formants_utt_symb_Top['D'], **Formants_utt_symb_Top['K']}

Formants_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
AUI_info_total=Gather_info_certainphones(Formants_people_information,PhoneMapp_dict,PhoneOfInterest)




#%%
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
    def SplitRole_Formants_utt(Formants_utt_symb,Inspect_roles):
        Formants_utt_symb_Individual=Dict()
        for query_role in Inspect_roles:
            for k, values in Formants_utt_symb.items():
                prefix, suffix='_', '_'
                if prefix+query_role+suffix in k:
                    Formants_utt_symb_Individual[query_role][k]=values
        return Formants_utt_symb_Individual
    Formants_utt_symb_Individual=SplitRole_Formants_utt(Formants_utt_symb,args.Inspect_roles)
    
    Formants_people_segment_DKIndividual_utt_dict=Dict()
    Formants_people_segment_DKIndividual_utt_dict['D']=sliding_window.Reorder2_PER_utt_formants(Formants_utt_symb_Individual['D'],PhoneMapp_dict,\
                                                            PhoneOfInterest,['D'],\
                                                            MinNum=args.MinPhoneNum)
    
    
    Formants_people_segment_DKIndividual_utt_dict['K']=sliding_window.Reorder2_PER_utt_formants(Formants_utt_symb_Individual['K'],PhoneMapp_dict,\
                                                            PhoneOfInterest,['K'],\
                                                            MinNum=args.MinPhoneNum)
    
    
    
    # Formants_people_segmentutt_dict,Formants_utt_symb=copy.deepcopy(Formants_people_segment_DKIndividual_utt_dict),copy.deepcopy(Formants_utt_symb_Individual)
    # minNum=3                                                                               
    # role='K'
    # backoff_minNum_lst=[ 1 ] 
    # length_thrld=4
    # # This function will repair the segments with short timeseries with at least more timeseries
    # # by setting smaller MinNum (loosen the restriction of phone numbers )
    # # 這個補償函數需要確保不會有人被捨棄掉，採以下兩步驟
    # # 1. 如果小於length_thrld就放寬phone定義條件
    # # 2. 如果time series長度為1那就補長到跟MinPhoneNum一樣
    # def Check_short_TS(Formants_people_segmentutt_dict,role='K',length_thrld=length_thrld):
    #     empty_lst_pp=[]
    #     # for role in Formants_people_segmentutt_dict.keys():
    #     for people,IPU_collection in Formants_people_segmentutt_dict[role].items():
    #         TS_len=len(Formants_people_segmentutt_dict[role][people])
    #         if TS_len < length_thrld: #For some people who talks less, the maximum timeseries they can get is 1, so set this as default value
    #             empty_lst_pp.append(people)
    #     return empty_lst_pp
    
    
    # # for backoff_minNum in [args.MinPhoneNum - i  for i in range(1,10) if args.MinPhoneNum - i >0]:
    # for backoff_minNum in backoff_minNum_lst:
    #     empty_lst_pp=Check_short_TS(Formants_people_segmentutt_dict,role)
    #     if len(empty_lst_pp) > 0: # there's people got empty timeseries list
    #         # print(empty_lst_pp)
            
    #         ''' Change to Reorder2_PER_utt_phonation in phonation script'''
    #         tmp_sldWind=sliding_window.Reorder2_PER_utt_formants(Formants_utt_symb[role],PhoneMapp_dict,\
    #                                                         PhoneOfInterest,[role],\
    #                                                         MinNum=backoff_minNum)
    #         for people,segment_lst in Formants_people_segmentutt_dict[role].items():
    #             # 1. 如果小於length_thrld就放寬phone定義條件
    #             if len(segment_lst) < length_thrld:
                    
    #                 Formants_people_segmentutt_dict[role][people]=tmp_sldWind[people]
    #                 print("TS of people ", people, " swapped with smaller MinNum: ",backoff_minNum)
    #             # 2. 如果time series長度為1那就補長到跟MinPhoneNum一樣
    #             if len(Formants_people_segmentutt_dict[role][people]) < minNum:
    #                 padd_num=minNum-len(Formants_people_segmentutt_dict[role][people])
    #                 st_idx=len(Formants_people_segmentutt_dict[role][people])
    #                 for n in range(st_idx,st_idx+padd_num,1):
    #                     Formants_people_segmentutt_dict[role][people][n]=Formants_people_segmentutt_dict[role][people][n-1]
    # aaa=ccc
        
    
    
    def Compensate_EmptySegments_DKIndividual(Formants_people_segmentutt_dict,Formants_utt_symb,role,\
                                     backoff_minNum_lst=[ 2 ], length_thrld=1, minNum=3):
        # This function will repair the segments with short timeseries with at least more timeseries
        # by setting smaller MinNum (loosen the restriction of phone numbers )
        # 這個補償函數需要確保不會有人被捨棄掉，採以下兩步驟
        # 1. 如果小於length_thrld就放寬phone定義條件
        # 2. 如果time series長度為1那就補長到跟MinPhoneNum一樣
        # backoff_minNum_lst 裡面至少要是2 不然算DEP features的spear時候會至少需要2個phone
        def Check_short_TS(Formants_people_segmentutt_dict,role='K',length_thrld=length_thrld):
            empty_lst_pp=[]
            # for role in Formants_people_segmentutt_dict.keys():
            for people,IPU_collection in Formants_people_segmentutt_dict[role].items():
                TS_len=len(Formants_people_segmentutt_dict[role][people])
                if TS_len < length_thrld: #For some people who talks less, the maximum timeseries they can get is 1, so set this as default value
                    empty_lst_pp.append(people)
            return empty_lst_pp
        
        
        # for backoff_minNum in [args.MinPhoneNum - i  for i in range(1,10) if args.MinPhoneNum - i >0]:
        for backoff_minNum in backoff_minNum_lst:
            empty_lst_pp=Check_short_TS(Formants_people_segmentutt_dict,role)
            if len(empty_lst_pp) > 0: # there's people got empty timeseries list
                # print(empty_lst_pp)
                
                ''' Change to Reorder2_PER_utt_phonation in phonation script'''
                tmp_sldWind=sliding_window.Reorder2_PER_utt_formants(Formants_utt_symb[role],PhoneMapp_dict,\
                                                                PhoneOfInterest,[role],\
                                                                MinNum=backoff_minNum)
                for people,segment_lst in Formants_people_segmentutt_dict[role].items():
                    # 1. 如果小於length_thrld就放寬phone定義條件
                    if len(segment_lst) < length_thrld:
                        Formants_people_segmentutt_dict[role][people]=tmp_sldWind[people]
                        print("TS of people ", people, " swapped with smaller MinNum: ",backoff_minNum)
                    # 2. 如果time series長度為1那就補長到跟MinPhoneNum一樣
                    # 2-1. 如果是空的就不補
                    if len(Formants_people_segmentutt_dict[role][people]) < minNum:
                        padd_num=minNum-len(Formants_people_segmentutt_dict[role][people])
                        st_idx=len(Formants_people_segmentutt_dict[role][people])
                        for n in range(st_idx,st_idx+padd_num,1):
                            if len(Formants_people_segmentutt_dict[role][people][n-1])!=0:
                                Formants_people_segmentutt_dict[role][people][n]=Formants_people_segmentutt_dict[role][people][n-1]
                    if len(Formants_people_segmentutt_dict[role][people]) == 0:
                            Formants_people_segmentutt_dict[role][people][0]=Dict()
                            Formants_people_segmentutt_dict[role][people][0][role]=pd.DataFrame([]) # If the segment is empty, add something so that the latter functions will fill some values for this
        return Formants_people_segmentutt_dict
    
    
    
    
    
    
    for role in Formants_people_segment_DKIndividual_utt_dict.keys():
        Formants_people_segment_DKIndividual_utt_dict_new=Compensate_EmptySegments_DKIndividual(copy.deepcopy(Formants_people_segment_DKIndividual_utt_dict),\
                                                                                   copy.deepcopy(Formants_utt_symb_Individual),role,\
                                                                                   backoff_minNum_lst=[ 2 ], length_thrld=1, minNum=args.MinPhoneNum)
    Formants_people_segment_DKIndividual_utt_dict = Formants_people_segment_DKIndividual_utt_dict_new 
    
    
    def Compensate_EmptySegments_unity_check(Formants_people_segment_DKIndividual_utt_dict,Formants_people_segment_DKIndividual_utt_dict_new):
        # Verification code
        diff_dict=Dict()
        for role in Formants_people_segment_DKIndividual_utt_dict.keys():
            for people in Formants_people_segment_DKIndividual_utt_dict[role].keys():
                for d_key in Formants_people_segment_DKIndividual_utt_dict[role][people].keys():
                    
                    # cond=[]
                    # for k in Formants_people_segment_DKIndividual_utt_dict_new[role][people][d_key][role].keys():
                    #     cond.append((Formants_people_segment_DKIndividual_utt_dict[role][people][d_key][role][k] == Formants_people_segment_DKIndividual_utt_dict_new[role][people][d_key][role][k]).all().all())
                            
                    cond= len(Formants_people_segment_DKIndividual_utt_dict[role][people][d_key][role]) == len(Formants_people_segment_DKIndividual_utt_dict_new[role][people][d_key][role])
                    if not cond:
                        diff_dict['{0}-{1}-{2}'.format(role,people,d_key)]['before']=Formants_people_segment_DKIndividual_utt_dict[role][people][d_key][role]
                        diff_dict['{0}-{1}-{2}'.format(role,people,d_key)]['after']=Formants_people_segment_DKIndividual_utt_dict_new[role][people][d_key][role]
    
    
    
    def Compensate_EmptySegments_DKCriteria(Formants_utt_people_segmentutt_dict,Formants_utt_utt_symb,\
                                     backoff_minNum_lst=[ 1 ], length_thrld=1, minNum=3):
            # This function will repair the segments with short timeseries with at least more timeseries
            # by setting smaller MinNum (loosen the restriction of phone numbers )
            # 這個補償函數需要確保不會有人被捨棄掉，採以下兩步驟
            # 1. 如果小於length_thrld就放寬phone定義條件
            # 2. 如果time series長度為1那就補長到跟MinPhoneNum一樣
            # backoff_minNum_lst 裡面至少要是2 不然算DEP features的spear時候會至少需要2個phone
            def Check_short_TS(Formants_utt_people_segmentutt_dict,length_thrld=length_thrld):
                empty_lst_pp=[]
                # for role in Formants_utt_people_segmentutt_dict.keys():
                for people,IPU_collection in Formants_utt_people_segmentutt_dict.items():
                    TS_len=len(Formants_utt_people_segmentutt_dict[people])
                    if TS_len < length_thrld: #For some people who talks less, the maximum timeseries they can get is 1, so set this as default value
                        empty_lst_pp.append(people)
                return empty_lst_pp
            
            
            # for backoff_minNum in [args.MinPhoneNum - i  for i in range(1,10) if args.MinPhoneNum - i >0]:
            for backoff_minNum in backoff_minNum_lst:
                empty_lst_pp=Check_short_TS(Formants_utt_people_segmentutt_dict)
                if len(empty_lst_pp) > 0: # there's people got empty timeseries list
                    # print(empty_lst_pp)
                    
                    ''' Change to Reorder2_PER_utt_phonation in phonation script'''
                    tmp_sldWind=sliding_window.Reorder2_PER_utt_formants(Formants_utt_utt_symb,PhoneMapp_dict,\
                                                                    PhoneOfInterest,args.Inspect_roles,\
                                                                    MinNum=backoff_minNum)
                    for people,segment_lst in Formants_utt_people_segmentutt_dict.items():
                        # 1. 如果小於length_thrld就放寬phone定義條件
                        if len(segment_lst) < length_thrld:
                            Formants_utt_people_segmentutt_dict[people]=tmp_sldWind[people]
                            print("TS of people ", people, " swapped with smaller MinNum: ",backoff_minNum)
                        # 2. 如果time series長度為1那就補長到跟MinPhoneNum一樣
                        # 2-1. 如果是空的就不補
                        # 2-2. 如果完全沒有segment就放一個空df讓後面算feature的function補值
                        if len(Formants_utt_people_segmentutt_dict[people]) < minNum:
                            padd_num=minNum-len(Formants_utt_people_segmentutt_dict[people])
                            st_idx=len(Formants_utt_people_segmentutt_dict[people])
                            for n in range(st_idx,st_idx+padd_num,1):
                                if len(Formants_utt_people_segmentutt_dict[people][n-1])!=0:
                                    Formants_utt_people_segmentutt_dict[people][n]=Formants_utt_people_segmentutt_dict[people][n-1]
                        if len(Formants_utt_people_segmentutt_dict[people]) == 0:
                            Formants_utt_people_segmentutt_dict[people][0]=Dict()
                            for role in args.Inspect_roles:
                                Formants_utt_people_segmentutt_dict[people][0][role]=pd.DataFrame([]) # If the segment is empty, add something so that the latter functions will fill some values for this
            return Formants_utt_people_segmentutt_dict
    
    
    Formants_people_segment_DKcriteria_utt_dict=sliding_window.Reorder2_PER_utt_formants(Formants_utt_symb,PhoneMapp_dict,\
                                                            PhoneOfInterest,args.Inspect_roles,\
                                                            MinNum=args.MinPhoneNum)            
    Formants_people_segment_DKcriteria_utt_dict_new=Compensate_EmptySegments_DKCriteria(copy.deepcopy(Formants_people_segment_DKcriteria_utt_dict),\
                                                                                   copy.deepcopy(Formants_utt_symb),\
                                                                                   backoff_minNum_lst=[ 1 ], length_thrld=1)
    Formants_people_segment_DKcriteria_utt_dict = Formants_people_segment_DKcriteria_utt_dict_new 
    # Formants_people_half_role_utt_dict=Dict()
    # Formants_people_whole_role_utt_dict=Dict()
    # for people in Formants_people_segment_DKcriterial_utt_dict.keys():
    #     split_num=len(Formants_people_segment_DKcriterial_utt_dict[people])//2
    #     for segment in Formants_people_segment_DKcriterial_utt_dict[people].keys():
    #         for role in Formants_people_segment_DKcriterial_utt_dict[people][segment].keys():
    #             if segment <= split_num:
    #                 Formants_people_half_role_utt_dict[people]['first_half'][role].update(Formants_people_segment_DKcriterial_utt_dict[people][segment][role])
    #             else:
    #                 Formants_people_half_role_utt_dict[people]['last_half'][role].update(Formants_people_segment_DKcriterial_utt_dict[people][segment][role])
    #             Formants_people_whole_role_utt_dict[people]['whole'][role].update(Formants_people_segment_DKcriterial_utt_dict[people][segment][role])

    
    
    

pickle.dump(Formants_people_segment_DKIndividual_utt_dict,open(outpklpath+"Formants_people_segment_DKIndividual_utt_dict_{0}.pkl".format(dataset_role),"wb"))
pickle.dump(Formants_people_segment_DKcriteria_utt_dict,open(outpklpath+"Formants_people_segment_DKcriterial_utt_dict{0}.pkl".format(dataset_role),"wb"))
def Check_AUI_info_from_people_segment_dict(role='D'):
        AUI_info_total_check=Dict()
        # for role in Formants_people_segment_DKIndividual_utt_dict.keys():
        for people in Formants_people_segment_DKIndividual_utt_dict[role].keys():
            for i,n in enumerate(Formants_people_segment_DKIndividual_utt_dict[role][people]):
                Formants_utt_symb_fake=Formants_people_segment_DKIndividual_utt_dict[role][people][n][role]
                
                
                Formants_people_information=Formant_utt2people_reshape(Formants_utt_symb_fake,Formants_utt_symb_fake,Align_OrinCmp=False)
                AUI_info=Gather_info_certainphones(Formants_people_information,PhoneMapp_dict,PhoneOfInterest)
                if people not in AUI_info_total_check.keys():
                    AUI_info_total_check[people]=AUI_info[people]
                for P in PhoneOfInterest:
                    AUI_info_total_check[people][P] = AUI_info_total_check[people][P].append(AUI_info[people][P])

#%%
Formants_people_segment_DKIndividual_utt_dict=pickle.load(open(outpklpath+"Formants_people_segment_DKIndividual_utt_dict_{0}.pkl".format(dataset_role),"rb"))
Formants_people_segment_DKcriterial_utt_dict=pickle.load(open(outpklpath+"Formants_people_segment_DKcriterial_utt_dict{0}.pkl".format(dataset_role),"rb"))

# Data Statistics for the use of filling empty segments
People_data_distrib=GetValuemeanstd(AUI_info_total,PhoneMapp_dict,args.Inspect_features)


# =============================================================================
'''
    2-2. Calculate LOC timeseries features within each defined timesteps (Details in TBME2021)
    
Prepare df_formant_statistics for each segment

Input:  Formants_people_segment_DKcriterial_utt_dict
Output: df_person_segment_feature_dict[people][role]=df_person_segment_feature

df_person_segment_feature=
        u_num  a_num  ...  linear_discriminant_covariance(A:,i:)  ADOS_cate_C
happy     5.0    6.0  ...                          -4.877535e-15        1.0
afraid    3.0    3.0  ...                           8.829873e-18        1.0
angry     6.0   13.0  ...                           1.658604e-17        1.0
sad       7.0   10.0  ...                           0.000000e+00        1.0

'''

emotion_timeorder=['happy', 'afraid', 'angry', 'sad']
articulation=Articulation(Normalize_way=args.Normalize_way)


''' '''

Used_IPU_format=Formants_people_segment_DKcriterial_utt_dict


''' '''
# =============================================================================
keys=[]
interval=5
for i in range(0,len(Used_IPU_format.keys()),interval):
    keys.append(list(Used_IPU_format.keys())[i:i+interval])
flat_keys=[item for sublist in keys for item in sublist]
assert len(flat_keys) == len(Used_IPU_format)

pool = Pool(os.cpu_count())
if 'emotion' in  dataset_role:
    Segment_lst=emotion_timeorder
else:
    Segment_lst=[]

# def GetPersonalSegmentFeature_map(keys_people, Formants_people_segment_role_utt_dict, People_data_distrib,\
#                               PhoneMapp_dict, PhoneOfInterest ,df_general_info,\
#                               Inspect_roles ,Inspect_features, In_Segments_order,\
#                               Feature_calculator, vowel_min_num):


# If IPU format = DKcriterial
final_result = pool.starmap(GetPersonalSegmentFeature_map, [(key,Formants_people_segment_DKcriterial_utt_dict,People_data_distrib, \
                                  PhoneMapp_dict, PhoneOfInterest ,df_general_info,\
                                  args.Inspect_roles, args.Inspect_features,\
                                  Segment_lst, articulation,args.MinPhoneNum) for key in tqdm(keys)])
print('GetPersonalSegmentFeature_map segment done !!!')
df_person_segment_feature_DKcriteria_dict=Dict()
Vowels_AUI_info_segments_dict=Dict()
MissSeg=[]
for d, vowelinfoseg, missSeg in tqdm(final_result):
    MissSeg.extend(missSeg) #Bookeep the people that the timestamps are missing 
    for spk in d.keys():
        df_person_segment_feature_DKcriteria_dict[spk]=d[spk]
        Vowels_AUI_info_segments_dict[spk]=d[spk]

# If IPU format = DKIndividual
df_person_segment_feature_DKIndividual_dict=Dict()
Vowels_AUI_info_segments_DKIndividual_dict=Dict()
for role in Formants_people_segment_DKIndividual_utt_dict.keys():
    final_result = pool.starmap(GetPersonalSegmentFeature_map, [(key,Formants_people_segment_DKIndividual_utt_dict[role],People_data_distrib, \
                                      PhoneMapp_dict, PhoneOfInterest ,df_general_info,\
                                      [role], args.Inspect_features,\
                                      Segment_lst, articulation,args.MinPhoneNum) for key in tqdm(keys)])
    print('GetPersonalSegmentFeature_map segment done !!!')
    
    MissSeg=[]
    for d, vowelinfoseg, missSeg in tqdm(final_result):
        MissSeg.extend(missSeg) #Bookeep the people that the timestamps are missing 
        for spk in d.keys():
            df_person_segment_feature_DKIndividual_dict[spk][role]=d[spk][role]
            Vowels_AUI_info_segments_DKIndividual_dict[spk][role]=d[spk][role]




# Some people having too little timeseries will have zero last-half, so Fill_n_Create_AUIInfo
# will fill values to each phone, each role. The total filled message will be len(Missbag) * 3
# final_result = pool.starmap(GetPersonalSegmentFeature_map, [(key,Formants_people_half_role_utt_dict,People_data_distrib, \
#                               PhoneMapp_dict, PhoneOfInterest ,df_general_info,\
#                               args.Inspect_roles, args.Inspect_features,\
#                               list(HalfDesider.keys()), articulation,args.MinPhoneNum) for key in keys])
# print('GetPersonalSegmentFeature_map done')


# df_person_half_feature_dict=Dict()
# Vowels_AUI_half_dict=Dict()
# MissHalf=[]
# for d, vowelinfoseg, missHal in tqdm(final_result):
#     MissHalf.extend(missHal) #Bookeep the people that the first/last half is missing 
#     for spk in d.keys():
#         df_person_half_feature_dict[spk]=d[spk]
#         Vowels_AUI_half_dict[spk]=d[spk]







# final_result = pool.starmap(GetPersonalSegmentFeature_map, [(key,Formants_people_whole_role_utt_dict,People_data_distrib, \
#                               PhoneMapp_dict, PhoneOfInterest ,df_general_info,\
#                               args.Inspect_roles, args.Inspect_features,\
#                               ['whole'], articulation,args.MinPhoneNum) for key in keys])

# # This section is to inspect features if we collect vowel information for the whole session    
# # The purpose of doing this is because the whole session feature should equal or be similar to LOC features (The total session as a timestep)
# print('GetPersonalSegmentFeature_map whole done')
# df_person_whole_feature_dict=Dict()
# Vowels_AUI_whold_dict=Dict()
# for d, vowelinfoseg, missHal in tqdm(final_result):
#     for spk in d.keys():
#         df_person_whole_feature_dict[spk]=d[spk]


# This will be used in Statistical tests
pickle.dump(df_person_segment_feature_DKIndividual_dict,open(outpklpath+"df_person_segment_feature_DKIndividual_dict_{0}_{1}.pkl".format(dataset_role, 'formant'),"wb"))
pickle.dump(df_person_segment_feature_DKcriteria_dict,open(outpklpath+"df_person_segment_feature_DKcriteria_dict_{0}_{1}.pkl".format(dataset_role, 'formant'),"wb"))

#%%
# =============================================================================
'''

    3. Calculate syncrony features based on feature timeseries


    Input: df_person_segment_feature_dict
    Output: 

'''
dataset_role=args.dataset_role
Reorder_type='DKIndividual'   #[DKIndividual, DKcriteria]
if Reorder_type == 'DKIndividual':
    df_person_segment_feature_DKIndividual_dict=pickle.load(open(outpklpath+"df_person_segment_feature_{Reorder_type}_dict_{0}_{1}.pkl".format(dataset_role, 'formant',Reorder_type=Reorder_type),"rb"))
elif Reorder_type == 'DKcriteria':
    df_person_segment_feature_DKcriteria_dict=pickle.load(open(outpklpath+"df_person_segment_feature_{Reorder_type}_dict_{0}_{1}.pkl".format(dataset_role, 'formant',Reorder_type=Reorder_type),"rb"))


features=[
 'VSA2',
 'FCR2',
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
 'dcorr_12',
 ]




exclude_cols=['ADOS_cate_C']   # To avoid labels that will cause errors 
                             # Covariance of only two classes are easily to be zero
FilteredFeatures = [c for c in features if c not in exclude_cols]
# =============================================================================
label_generate_choose_lst=['ADOS_C','ADOS_S']


syncrony=Syncrony()
PhoneOfInterest_str=''
knn_neighbors=2
MinNumTimeSeries=knn_neighbors+1
knn_weights="distance"

if Reorder_type == 'DKIndividual':
    df_syncrony_measurement=syncrony.calculate_features_continuous_modulized(df_person_segment_feature_DKIndividual_dict,features,PhoneOfInterest_str,\
                                                                    args.Inspect_roles, Label,\
                                                                    knn_weights=knn_weights,knn_neighbors=knn_neighbors,\
                                                                    MinNumTimeSeries=MinNumTimeSeries, label_choose_lst=label_generate_choose_lst,Knn_aggressive_mode=True)

    
    # Inspect_roles=args.Inspect_roles
    # df_basic_additional_info=syncrony._Add_additional_info(df_person_segment_feature_DKIndividual_dict,Label,label_choose_lst,\
    #                                               Inspect_roles, MinNumTimeSeries=MinNumTimeSeries,PhoneOfInterest_str=PhoneOfInterest_str)
    # df_syncrony_measurement_merge=pd.DataFrame()
    # def KNNFitting(self,df_person_segment_feature_DKIndividual_dict,\
    #             col_choose,Inspect_roles,\
    #             knn_weights='uniform',knn_neighbors=2,MinNumTimeSeries=3,\
    #             st_col_str='IPU_st', ed_col_str='IPU_ed', aggressive_mode=False):
    #     p_1=Inspect_roles[0]
    #     p_2=Inspect_roles[1]
        
    #     functionDK_people=Dict()
    #     for people in df_person_segment_feature_DKIndividual_dict.keys():
    #         if not aggressive_mode:
    #             #Aggressive mode means that we don't want to skip any participant, if there is a value error, then make sure previous procedure doesn't generate unavailable data
    #             if len(df_person_segment_feature_DKIndividual_dict[people][p_1])<MinNumTimeSeries or len(df_person_segment_feature_DKIndividual_dict[people][p_2])<MinNumTimeSeries:
    #                 continue
    #         df_person_segment_feature_role_dict=df_person_segment_feature_DKIndividual_dict[people]  
    #         try:
    #             Totalendtime=min([df_person_segment_feature_role_dict[role][ed_col_str].values[-1]  for role in Inspect_roles])
    #         except:
    #             print("The people causing error happens at people: ",people)
    #             print("The problem file is ",df_person_segment_feature_role_dict)
    #             raise KeyError
    #         T = np.linspace(0, Totalendtime, int(Totalendtime))[:, np.newaxis]
            
    #         functionDK={}
    #         for role_choose in Inspect_roles:
    #             df_dynVals=df_person_segment_feature_role_dict[role_choose][col_choose]
    #             if not np.isnan(np.abs(stats.zscore(df_dynVals))).all(): 
    #                 # remove outlier that is falls over 3 times of the std
    #                 df_dynVals_deleteOutlier=df_dynVals[(np.abs(stats.zscore(df_dynVals)) < 3)]
    #             else:
    #                 #Situation for padding when the time series is too short
    #                 df_dynVals_deleteOutlier=df_dynVals
    #             df_stidx=df_person_segment_feature_role_dict[role_choose][st_col_str]
    #             df_edidx=df_person_segment_feature_role_dict[role_choose][ed_col_str]
                
                
    #             Mid_positions=[]
    #             for x_1 , x_2, y in zip(df_stidx.values ,df_edidx.values,df_dynVals_deleteOutlier.values):            
    #                 start_time=x_1
    #                 end_time=x_2
    #                 mid_time=(start_time+end_time)/2
    #                 Mid_positions.append(mid_time)    
                
    #             if aggressive_mode:
    #                 #Aggressive mode means that we don't want to skip any participant
    #                 knn_neighbors = min (knn_neighbors,len(df_dynVals_deleteOutlier))
    #             knn = neighbors.KNeighborsRegressor(knn_neighbors, weights=knn_weights)
    #             X, y=np.array(Mid_positions).reshape(-1,1), df_dynVals_deleteOutlier
    #             try:
    #                 y_ = knn.fit(X, y.values).predict(T)
    #             except ValueError:
    #                 print("Problem people happen at ", people, role_choose)
    #                 print("df_dynVals", df_dynVals)
    #                 print("==================================================")
    #                 print("df_dynVals_deleteOutlier", df_dynVals_deleteOutlier)
    #                 raise ValueError
    #             functionDK[role_choose]=y_
    #         functionDK['T']=T
    #         functionDK_people[people]=functionDK
    #     return functionDK_people
    # Knn_aggressive_mode=True
    # for col in features:
    #     Col_continuous_function_DK=syncrony.KNNFitting(df_person_segment_feature_DKIndividual_dict,\
    #                 col, args.Inspect_roles,\
    #                 knn_weights=knn_weights,knn_neighbors=knn_neighbors,MinNumTimeSeries=MinNumTimeSeries,\
    #                 st_col_str='IPU_st', ed_col_str='IPU_ed', aggressive_mode=Knn_aggressive_mode)
        
    #     df_syncrony_measurement_col=syncrony._calculate_features_col(Col_continuous_function_DK,col)
    #     if df_syncrony_measurement_col.isna().any().any():
    #         print("The columns with Nan is ", col)
            
    # aaa=ccc
        
    #     df_syncrony_measurement_merge=pd.concat([df_syncrony_measurement_merge,df_syncrony_measurement_col],axis=1)
    # df_syncrony_measurement=pd.concat([df_basic_additional_info,df_syncrony_measurement_merge],axis=1)
        
        
        
if Reorder_type == 'DKcriteria':
    df_syncrony_measurement=syncrony.calculate_features_continuous_modulized(df_person_segment_feature_DKcriteria_dict,features,PhoneOfInterest_str,\
                                                                    args.Inspect_roles, Label,\
                                                                    knn_weights=knn_weights,knn_neighbors=knn_neighbors,\
                                                                    MinNumTimeSeries=MinNumTimeSeries, label_choose_lst=label_generate_choose_lst,Knn_aggressive_mode=True)
                                        

timeSeries_len_columns=[col  for col in df_syncrony_measurement.columns if 'timeSeries_len' in col]
df_syncrony_measurement['timeSeries_len']=df_syncrony_measurement[timeSeries_len_columns].min(axis=1)
    



pickle.dump(df_syncrony_measurement,open(outpklpath+"Syncrony_measure_of_variance_{Reorder_type}_{}.pkl".format(dataset_role,Reorder_type=Reorder_type),"wb"))
Aaa=df_syncrony_measurement[df_syncrony_measurement.isna().any(axis=1)]
print("df_syncrony_measurement", df_syncrony_measurement)
print("Aaa=df_syncrony_measurement[df_syncrony_measurement.isna().any(axis=1)]", Aaa)
#%%
# =============================================================================
'''

    Unit Check
    
    以下的程式是在作function修改後驗證使用的，驗證成功後就沒有用處了

'''
# =============================================================================
# variablesaver=VariableSaver()
# # variablesaver.updateVar(df_syncrony_measurement_old=df_syncrony_measurement)
# variablesaver.updateVar(df_syncrony_measurement_new=df_syncrony_measurement)
# def unit_check(VarA,VarB,Inspect_columns,tolerance=1e-5):
#     """
#         當舊的function改成新的時候用unit check來驗證
#     """
#     diff_Vars=VarA - VarB
#     diff_dict={}
#     for col in diff_Vars.columns:
#         diff_dict[col]=diff_Vars[col].mean()
#         if col in Inspect_columns and diff_dict[col] < tolerance:
#             assert True

# diff_dfsyncronymeasurement=variablesaver.df_syncrony_measurement_old-variablesaver.df_syncrony_measurement_new
# import articulation.HYPERPARAM.FeatureSelect as FeatSel
# FeatSel.CategoricalName2cols.keys()
# Syncrony_functions=['Proximity[{}]','Trend[{}]_d','Trend[{}]_k','Convergence[{}]','Syncrony[{}]']

# Include_columns=[]
# for Syncwrapper in FeatSel.Syncrony_functions:
#     Selcted_cate=FeatSel.CategoricalName2cols[Syncwrapper.format('LOCDEP')]
#     Include_columns+=Selcted_cate
# unit_check(variablesaver.df_syncrony_measurement_old,variablesaver.df_syncrony_measurement_new,Include_columns)

# =============================================================================
'''

    Correlation analysis

'''
# =============================================================================
label_correlation_choose_lst=label_generate_choose_lst
additional_columns=label_correlation_choose_lst+['timeSeries_len']

columns=list(set(df_syncrony_measurement.columns) - set(additional_columns)) # Exclude added labels

# columns=list(set(columns) - set([co for co in columns if "Syncrony" in co or "Proximity" in co or "Convergence" in co]))

Eval_med=Evaluation_method()
if Reorder_type == 'DKIndividual':
    df_syncrony_measurement_DKIndividual=pickle.load(open(outpklpath+"Syncrony_measure_of_variance_{Reorder_type}_{}.pkl".format(dataset_role,Reorder_type=Reorder_type),"rb"))
    Aaadf_spearmanr_table_NoLimit_DKIndividual=Eval_med.Calculate_correlation(label_correlation_choose_lst,df_syncrony_measurement_DKIndividual,MinNumTimeSeries,columns,constrain_sex=-1, constrain_module=-1,feature_type='Syncrony_formant')
if Reorder_type == 'DKcriteria':
    df_syncrony_measurement_DKcriteria=pickle.load(open(outpklpath+"Syncrony_measure_of_variance_{Reorder_type}_{}.pkl".format(dataset_role,Reorder_type=Reorder_type),"rb"))
    Aaadf_spearmanr_table_NoLimit_DKIndividual=Eval_med.Calculate_correlation(label_correlation_choose_lst,df_syncrony_measurement_DKcriteria,MinNumTimeSeries,columns,constrain_sex=-1, constrain_module=-1,feature_type='Syncrony_formant')
