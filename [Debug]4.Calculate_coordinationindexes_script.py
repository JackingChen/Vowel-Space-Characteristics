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
import articulation.HYPERPARAM.FeatureSelect as FeatSel

from scipy import special, stats
import warnings
from Syncrony import Syncrony
from SlidingWindow import slidingwindow as SW

import matplotlib.pyplot as plt
import matplotlib
from sklearn import neighbors

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
    parser.add_argument('--dfFormantStatisticpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
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
    parser.add_argument('--knn_weights', default='distance',
                            help='path of the base directory')
    parser.add_argument('--knn_neighbors', default=2,  type=int,
                            help='path of the base directory')
    parser.add_argument('--Reorder_type', default='DKIndividual',
                            help='[DKIndividual, DKcriteria]')
    parser.add_argument('--Result_path', default='./Result_Interaction',
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
                df_formant_statistic=Feature_calculator.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_choose_lst)
                # Add informations to columns that are needed to df_formant_statistic
                for i in range(len(df_formant_statistic)):
                    name=df_formant_statistic.iloc[i].name
                    df_formant_statistic.loc[name,'ADOS_cate_C']=Label.label_raw[Label.label_raw['name']==name]['ADOS_cate_C'].values
                df_formant_statistic['u_num+i_num+a_num']=df_formant_statistic['u_num'] +\
                                                df_formant_statistic['i_num'] +\
                                                df_formant_statistic['a_num']
                query_lst=list(Formants_utt_symb_SegmentRole.keys())
                df_general_info_query=df_general_info.query("utt == @query_lst")
                IPU_start_time=df_general_info_query['st'].min()
                IPU_end_time=df_general_info_query['ed'].max()
                df_formant_statistic['IPU_st']=IPU_start_time
                df_formant_statistic['IPU_ed']=IPU_end_time
                
                
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
            df_values_K=df_values[df_values['utt'].str.contains("_K_")]
            df_values_D=df_values[df_values['utt'].str.contains("_D_")]
            People_data_distrib['D'][people][phoneRepresent].means=df_values_D[Inspect_features].mean()
            People_data_distrib['D'][people][phoneRepresent].stds=df_values_D[Inspect_features].std()
            
            People_data_distrib['K'][people][phoneRepresent].means=df_values_K[Inspect_features].mean()
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
# dataset_role=args.dataset_role
# Randseed=args.Randseed
outpklpath=args.inpklpath+"/Session_formants_people_vowel_feat/"
if not os.path.exists(outpklpath):
    os.makedirs(outpklpath)
    
knn_weights=args.knn_weights
knn_neighbors=args.knn_neighbors
Reorder_type=args.Reorder_type
Result_path=args.Result_path
if not os.path.exists(Result_path):
    os.makedirs(Result_path)
#%%
# =============================================================================
'''

    3. Calculate syncrony features based on feature timeseries


    Input: df_person_segment_feature_dict
    Output: 

'''
# dataset_role='ASD_DOCKID'
for dataset_role in ['ASD_DOCKID','TD_DOCKID']:
    print('Start preparing Syncrony_measure_of_variance ')
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
    MinNumTimeSeries=knn_neighbors+1

    
    # df_syncrony_measurement_cmp=syncrony.calculate_features_continuous(df_person_segment_feature_DKIndividual_dict,features,PhoneOfInterest_str,\
    #                             args.Inspect_roles, Label,\
    #                             knn_weights=knn_weights,knn_neighbors=knn_neighbors,\
    #                             MinNumTimeSeries=MinNumTimeSeries, label_choose_lst=label_generate_choose_lst,plot=False)
    if Reorder_type == 'DKIndividual':
        df_syncrony_measurement=syncrony.calculate_features_continuous_modulized(df_person_segment_feature_DKIndividual_dict,features,PhoneOfInterest_str,\
                                args.Inspect_roles, Label,\
                                knn_weights=knn_weights,knn_neighbors=knn_neighbors,\
                                MinNumTimeSeries=MinNumTimeSeries, label_choose_lst=label_generate_choose_lst)
    elif Reorder_type == 'DKcriteria':
        df_syncrony_measurement=syncrony.calculate_features_continuous_modulized(df_person_segment_feature_DKcriteria_dict,features,PhoneOfInterest_str,\
                                args.Inspect_roles, Label,\
                                knn_weights=knn_weights,knn_neighbors=knn_neighbors,\
                                MinNumTimeSeries=MinNumTimeSeries, label_choose_lst=label_generate_choose_lst)
    
    
    # df_syncrony_measurement=syncrony.calculate_features_continuous(df_person_segment_feature_DKcriteria_dict,features,PhoneOfInterest_str,\
    #                         args.Inspect_roles, Label,\
    #                         knn_weights="uniform",knn_neighbors=knn_neighbors,\
    #                         MinNumTimeSeries=MinNumTimeSeries, label_choose_lst=label_generate_choose_lst,plot=False)
        
    # df_syncrony_measurement=syncrony.calculate_features(df_person_segment_feature_dict, df_person_half_feature_dict,\
    #                                 FilteredFeatures,PhoneOfInterest_str,\
    #                                 args.Inspect_roles, Label,\
    #                                 MinNumTimeSeries=2, label_choose_lst=['ADOS_C'])
    
    # timeSeries_len_columns=[col  for col in df_syncrony_measurement.columns if 'timeSeries_len' in col]
    # df_syncrony_measurement['timeSeries_len']=df_syncrony_measurement[timeSeries_len_columns].min(axis=1)
        
    
    # feat_type='Syncrony_formant'
    # N=2
    # if dataset_role == 'ASD_DOCKID':
    #     df_syncrony_measurement=criterion_filter(df_syncrony_measurement,N=N,evictNamelst=[],feature_type=feat_type)
        
    
    
    pickle.dump(df_syncrony_measurement,open(outpklpath+"Syncrony_measure_of_variance_{Reorder_type}_{}.pkl".format(dataset_role,Reorder_type=Reorder_type),"wb"))
    # pickle.dump(df_syncrony_measurement,open(outpklpath+"Syncrony_measure_of_variance_{Reorder_type}_{}.pkl".format(dataset_role,Reorder_type=Reorder_type),"wb"))

# =============================================================================
'''

    Correlation analysis

'''
# =============================================================================
label_correlation_choose_lst=label_generate_choose_lst
additional_columns=label_correlation_choose_lst+['timeSeries_len']

columns=list(set(df_syncrony_measurement.columns) - set(additional_columns)) # Exclude added labels

# columns=list(set(columns) - set([co for co in columns if "Syncrony" in co or "Proximity" in co or "Convergence" in co]))

# Eval_med=Evaluation_method()
# if Reorder_type == 'DKIndividual':
#     df_syncrony_measurement_DKIndividual=pickle.load(open(outpklpath+"Syncrony_measure_of_variance_{Reorder_type}_{}.pkl".format(dataset_role,Reorder_type=Reorder_type),"rb"))
#     Aaadf_spearmanr_table_NoLimit_DKIndividual=Eval_med.Calculate_correlation(label_correlation_choose_lst,df_syncrony_measurement_DKIndividual,MinNumTimeSeries,columns,constrain_sex=-1, constrain_module=-1,feature_type='Syncrony_formant')
# if Reorder_type == 'DKcriteria':
#     df_syncrony_measurement_DKcriteria=pickle.load(open(outpklpath+"Syncrony_measure_of_variance_{Reorder_type}_{}.pkl".format(dataset_role,Reorder_type=Reorder_type),"rb"))
#     Aaadf_spearmanr_table_NoLimit_DKIndividual=Eval_med.Calculate_correlation(label_correlation_choose_lst,df_syncrony_measurement_DKcriteria,MinNumTimeSeries,columns,constrain_sex=-1, constrain_module=-1,feature_type='Syncrony_formant')



# Aaadf_spearmanr_table_NoLimit_DKcriteria=Eval_med.Calculate_correlation(label_correlation_choose_lst,df_syncrony_measurement_DKcriteria,MinNumTimeSeries,columns,constrain_sex=-1, constrain_module=-1,feature_type='Syncrony_formant')


# =============================================================================
# Generate LOC indexes for fraction people for ASD/non-ASD classification
# =============================================================================
# dfFormantStatisticFractionpath=dfFormantStatisticpath+'/Fraction'
# if not os.path.exists(dfFormantStatisticFractionpath):
#     os.makedirs(dfFormantStatisticFractionpath)
# sellect_people_define=SellectP_define()
# if dataset_role == 'ASD_DOCKID':
#     df_syncrony_statistic_agesexmatch_ASDSevere=df_syncrony_measurement.loc[sellect_people_define.SevereASD_age_sex_match_ver2]
#     df_syncrony_statistic_agesexmatch_ASDMild=df_syncrony_measurement.loc[sellect_people_define.MildASD_age_sex_match_ver2]
    
#     label_add='ADOS_cate_C'
#     if label_add  not in df_syncrony_statistic_agesexmatch_ASDSevere.columns:
#         df_syncrony_statistic_agesexmatch_ASDSevere=Add_label(df_syncrony_statistic_agesexmatch_ASDSevere,Label,label_choose=label_add)
#     if label_add  not in df_syncrony_statistic_agesexmatch_ASDMild.columns:
#         df_syncrony_statistic_agesexmatch_ASDMild=Add_label(df_syncrony_statistic_agesexmatch_ASDMild,Label,label_choose=label_add)
    
#     # 1 represents ASD, 2 represents TD
#     label_add='ASDTD' 
#     if label_add not in df_syncrony_statistic_agesexmatch_ASDSevere.columns:
#         df_syncrony_statistic_agesexmatch_ASDSevere[label_add]=sellect_people_define.ASDTD_label['ASD']
#     if label_add not in df_syncrony_statistic_agesexmatch_ASDMild.columns:
#         df_syncrony_statistic_agesexmatch_ASDMild[label_add]=sellect_people_define.ASDTD_label['ASD']
        
#     pickle.dump(df_syncrony_statistic_agesexmatch_ASDSevere,open(dfFormantStatisticFractionpath+'/df_syncrony_statistic_agesexmatch_ASDSevereGrp.pkl','wb'))
#     pickle.dump(df_syncrony_statistic_agesexmatch_ASDMild,open(dfFormantStatisticFractionpath+'/df_syncrony_statistic_agesexmatch_ASDMildGrp.pkl','wb'))
    
# elif dataset_role == 'TD_DOCKID':
#     df_syncrony_TD_normal=df_syncrony_measurement.loc[sellect_people_define.TD_normal_ver2]
    
#     # 1 represents ASD, 2 represents TD
#     label_add='ASDTD' 
#     if label_add not in df_syncrony_TD_normal.columns:
#         df_syncrony_TD_normal[label_add]=sellect_people_define.ASDTD_label['TD']
        
#     pickle.dump(df_syncrony_TD_normal,open(dfFormantStatisticFractionpath+'/df_syncrony_statistic_TD_normalGrp.pkl','wb'))    
# else:
#     raise KeyError("The key has not been registered")
#%%
# =============================================================================
'''

    Analysis area

'''
import seaborn as sns
from pylab import text
dfFormantStatisticpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles'
feat='Syncrony_measure_of_variance_{Reorder_type}'.format(Reorder_type=Reorder_type)
# =============================================================================
df_formant_statistic77_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/{name}_{role}.pkl'.format(name=feat,role='ASD_DOCKID')
df_feature_ASD=pickle.load(open(df_formant_statistic77_path,'rb'))
df_formant_statistic_ASDTD_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/{name}_{role}.pkl'.format(name=feat,role='TD_DOCKID')
if not os.path.exists(df_formant_statistic_ASDTD_path) or not os.path.exists(df_formant_statistic77_path):
    raise FileExistsError
df_feature_TD=pickle.load(open(df_formant_statistic_ASDTD_path,'rb'))

# ADD label
df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='ADOS_cate_CSS')
df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='ADOS_cate_C')
df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='ADOS_cate_S')
df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='ADOS_cate_SC')
df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='age_year')
df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='sex')
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

filter_Notautism_TSC=df_feature_ASD['ADOS_cate_SC']==0
filter_ASD_TSC=df_feature_ASD['ADOS_cate_SC']==1
filter_Autism_TSC=df_feature_ASD['ADOS_cate_SC']==2


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

df_feature_Notautism_TSC=df_feature_ASD[filter_Notautism_TSC]
df_feature_ASD_TSC=df_feature_ASD[filter_ASD_TSC]
df_feature_NotautismandASD_TSC=df_feature_ASD[filter_Notautism_TSC | filter_ASD_TSC]
df_feature_Autism_TSC=df_feature_ASD[filter_Autism_TSC]

# sellect_people_define=SellectP_define()
# SevereASD_age_sex_match=sellect_people_define.SevereASD_age_sex_match_ver2
# MildASD_age_sex_match=sellect_people_define.MildASD_age_sex_match_ver2
# TD_normal_ver2=sellect_people_define.TD_normal_ver2

# df_formant_statistic_agesexmatch_ASDSevere=df_feature_ASD.copy().loc[SevereASD_age_sex_match]
# df_formant_statistic_agesexmatch_ASDMild=df_feature_ASD.copy().loc[MildASD_age_sex_match]
# df_formant_statistic_TD_normal=df_feature_TD.copy().loc[TD_normal_ver2]

TopTop_data_lst=[]
# TopTop_data_lst.append(['df_formant_statistic_agesexmatch_ASDSevere','df_formant_statistic_TD_normal'])
# TopTop_data_lst.append(['df_formant_statistic_agesexmatch_ASDMild','df_formant_statistic_TD_normal'])
# TopTop_data_lst.append(['df_formant_statistic_agesexmatch_ASDMild','df_formant_statistic_agesexmatch_ASDSevere'])
''' Notice, ASD should be on the left '''

# TopTop_data_lst.append(['df_feature_ASD','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_low_CSS','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_moderate_CSS','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_high_CSS','df_feature_TD'])
TopTop_data_lst.append(['df_feature_lowMinimal_CSS','df_feature_TD'])
TopTop_data_lst.append(['df_feature_moderate_CSS','df_feature_TD'])
TopTop_data_lst.append(['df_feature_high_CSS','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_moderatehigh_CSS','df_feature_TD'])

# TopTop_data_lst.append(['df_feature_low_CSS','df_feature_moderate_CSS'])
# TopTop_data_lst.append(['df_feature_moderate_CSS','df_feature_high_CSS'])
# TopTop_data_lst.append(['df_feature_low_CSS','df_feature_high_CSS'])
# TopTop_data_lst.append(['df_feature_lowMinimal_CSS','df_feature_moderate_CSS'])
# TopTop_data_lst.append(['df_feature_lowMinimal_CSS','df_feature_high_CSS'])

# TopTop_data_lst.append(['df_feature_Notautism_TC','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_ASD_TC','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_NotautismandASD_TC','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_Autism_TC','df_feature_TD'])

# TopTop_data_lst.append(['df_feature_Notautism_TC','df_feature_ASD_TC'])
# TopTop_data_lst.append(['df_feature_ASD_TC','df_feature_Autism_TC'])
# TopTop_data_lst.append(['df_feature_Notautism_TC','df_feature_Autism_TC'])
# TopTop_data_lst.append(['df_feature_Notautism_TC','df_feature_ASD_TC','df_feature_Autism_TC'])


# TopTop_data_lst.append(['df_feature_Notautism_TS','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_ASD_TS','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_NotautismandASD_TS','df_feature_TD'])
# TopTop_data_lst.append(['df_feature_Autism_TS','df_feature_TD'])

# TopTop_data_lst.append(['df_feature_Notautism_TS','df_feature_ASD_TS'])
# TopTop_data_lst.append(['df_feature_ASD_TS','df_feature_Autism_TS'])
# TopTop_data_lst.append(['df_feature_Notautism_TS','df_feature_Autism_TS'])
# TopTop_data_lst.append(['df_feature_Notautism_TS','df_feature_ASD_TS','df_feature_Autism_TS'])


self_specify_cols=FeatSel.LOCDEP_Proximity_cols
# self_specify_cols= 
Parameters=df_syncrony_measurement.columns
if len(self_specify_cols) > 0:
    inspect_cols=self_specify_cols
else:
    inspect_cols=Parameters

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
        if plot:
            fig, ax = plt.subplots()
        # =============================================================================
        data=[]
        dataname=[]
        for dstr in Top_data_lst:
            dataname.append(dstr)
            data.append(vars()[dstr])
        # =============================================================================
        if plot:
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
            addtext='{0}/({1})'.format(np.round(mean_difference,3),np.round(p_val,3))
            text(0.9, 0.9, addtext, ha='center', va='center', transform=ax.transAxes)
            addtextvariable='{0} vs {1}'.format(Top_data_lst[0],Top_data_lst[1])
            text(0.9, 0.6, addtextvariable, ha='center', va='center', transform=ax.transAxes)
        # =============================================================================
    warnings.simplefilter('always')

#Clear Record_dict
Record_cleaned_dict={}
for keys, values in Record_dict.items():
    Record_cleaned_dict[keys]=values.dropna(thresh=2,axis=0)

# =============================================================================
'''

    Classification area

'''
print('Start running classifiers ')
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

# =============================================================================
lab_chos_lst=['ASDTD']
# feature_chos_lst_top=['between_covariance_norm(A:,i:,u:)','dcorr_12']


# C_variable=np.array(np.arange(0.1,1.1,0.2))
epsilon=np.array(np.arange(0.1,1.5,0.1) )
# epsilon=np.array(np.arange(0.01,0.15,0.02))
# C_variable=np.array([0.001,0.01,10.0,50,100] + list(np.arange(0.1,1.5,0.1)))
# C_variable=np.array([0.001,0.01,10.0,50,100])
Classifier={}
loo=LeaveOneOut()
# CV_settings=loo
CV_settings=10
pca = PCA(n_components=1)

# C_variable=np.array(np.arange(0.1,1.5,0.1))
C_variable=np.array([0.001,0.01,0.1,1,5,10.0,25,50,75,100])
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


Comb=Dict()
Comb['Trend_D_cols']=FeatSel.LOCDEP_Trend_D_cols
Comb['Trend_K_cols']=FeatSel.LOCDEP_Trend_K_cols
Comb['Proximity_cols']=FeatSel.LOCDEP_Proximity_cols
Comb['Convergence_cols']=FeatSel.LOCDEP_Convergence_cols
Comb['Syncrony_cols']=FeatSel.LOCDEP_Syncrony_cols

combinations_lsts=[ Comb[k] for k in Comb.keys()]
combinations_keylsts=[ k for k in Comb.keys()]


Top_RESULT_dict=Dict()
for Top_data_lst in TopTop_data_lst:
    print(Top_data_lst[0], ' vs ', Top_data_lst[1])
    if len(Top_data_lst) == 2:
        df_asdTmp, df_tdTmp=vars()[Top_data_lst[0]].copy(), vars()[Top_data_lst[1]].copy()
        df_asdTmp["ASDTD"]=1
        df_tdTmp["ASDTD"]=2
        df_ASDcmpVombineTD=pd.concat([df_asdTmp,df_tdTmp],axis=0)
    elif len(Top_data_lst) == 3:
        df_adTmp, df_asdTmp, df_nonasdTmp=vars()[Top_data_lst[0]].copy(), vars()[Top_data_lst[1]].copy(), vars()[Top_data_lst[2]].copy()
        df_adTmp["ASDTD"]=1
        df_asdTmp["ASDTD"]=2
        df_nonasdTmp["ASDTD"]=3
        df_ASDcmpVombineTD=pd.concat([df_asdTmp,df_tdTmp,df_nonasdTmp],axis=0)

    RESULT_dict=Dict()
    for key,feature_chos_tup in zip(combinations_keylsts,combinations_lsts):
        feature_chos_lst=list(feature_chos_tup)
        for feature_chooses in [feature_chos_lst]:
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
            feature_keys=key
            try:
                CVpredict=cross_val_predict(Gclf, features.X, features.y.values.ravel(), cv=CV_settings)  
            except ValueError:
                print('too few samples')
                RESULT_dict[feature_keys]=[np.nan,np.nan,np.nan]
                continue
            n,p=features.X.shape
            UAR=recall_score(features.y, CVpredict, average='macro')
            AUC=roc_auc_score(features.y, CVpredict)
            f1Score=f1_score(features.y, CVpredict, average='macro')
            
            # feature_keys='+'.join(feature_chooses)
            
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



#%%
out_df_str='df_Result_UAR_summary_list_{knn_weights}_{knn_neighbors}_{Reorder_type}.xlsx'.format(knn_weights=knn_weights,\
                                                                                            knn_neighbors=knn_neighbors,\
                                                                                            Reorder_type=Reorder_type)

df_Result_UAR_summary_list.to_excel(Result_path+"/"+out_df_str)
with open(Result_path+'/result_{knn_weights}_{knn_neighbors}_{Reorder_type}.pkl'.format(knn_weights=knn_weights,\
                                                                                            knn_neighbors=knn_neighbors,\
                                                                                            Reorder_type=Reorder_type), 'wb') as fp:
    pickle.dump(Record_cleaned_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
# =============================================================================
'''

    Plot area

'''
# =============================================================================
# from sklearn import neighbors
# df_person_segment_feature_DKIndividual_dict_TD=pickle.load(open(outpklpath+"df_person_segment_feature_DKIndividual_dict_{0}_{1}.pkl".format('TD_DOCKID', 'formant'),"rb"))
# df_person_segment_feature_DKIndividual_dict_ASD=pickle.load(open(outpklpath+"df_person_segment_feature_DKIndividual_dict_{0}_{1}.pkl".format('ASD_DOCKID', 'formant'),"rb"))

# st_col_str='IPU_st'  #It is related to global info
# ed_col_str='IPU_ed'  #It is related to global info
# Inspect_roles=args.Inspect_roles
# df_person_segment_feature_DKIndividual_dict=df_person_segment_feature_DKIndividual_dict_ASD
# p_1=Inspect_roles[0]
# p_2=Inspect_roles[1]
# df_syncrony_measurement=pd.DataFrame()
# col = 'between_variance_norm(A:,i:,u:)'
# Colormap_role_dict=Dict()
# Colormap_role_dict['D']='orange'
# Colormap_role_dict['K']='blue'
# # knn_weights='distance'
# knn_weights='uniform'
# knn_neighbors=2
# functionDK_people=Dict()
# for people in df_person_segment_feature_DKIndividual_dict.keys():
#     if len(df_person_segment_feature_DKIndividual_dict[people][p_1])<MinNumTimeSeries or len(df_person_segment_feature_DKIndividual_dict[people][p_2])<MinNumTimeSeries:
#         continue
#     df_person_segment_feature_role_dict=df_person_segment_feature_DKIndividual_dict[people]
    
#     RESULT_dict={}
#     # kNN fitting
#     Totalendtime=min([df_person_segment_feature_role_dict[role][ed_col_str].values[-1]  for role in Inspect_roles])
#     Mintimeserieslen=min([len(df_person_segment_feature_role_dict[role])  for role in Inspect_roles])
#     T = np.linspace(0, Totalendtime, int(Totalendtime))[:, np.newaxis]
    
#     RESULT_dict['timeSeries_len[{}]'.format(PhoneOfInterest_str)]=Mintimeserieslen
#     for label_choose in label_choose_lst:
#         RESULT_dict[label_choose]=Label.label_raw[Label.label_raw['name']==people][label_choose].values[0]
#     functionDK={}
#     for role_choose in Inspect_roles:
#         df_dynVals=df_person_segment_feature_role_dict[role_choose][col]
#         # remove outlier that is falls over 3 times of the std
#         df_dynVals_deleteOutlier=df_dynVals[(np.abs(stats.zscore(df_dynVals)) < 3)]
#         df_stidx=df_person_segment_feature_role_dict[role_choose][st_col_str]
#         df_edidx=df_person_segment_feature_role_dict[role_choose][ed_col_str]
        
        
#         Mid_positions=[]
#         for x_1 , x_2, y in zip(df_stidx.values ,df_edidx.values,df_dynVals_deleteOutlier.values):            
#             # ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,0.5,color=Colormap_role_dict[role_choose]))
#             start_time=x_1
#             end_time=x_2
#             mid_time=(start_time+end_time)/2
#             Mid_positions.append(mid_time)    
        
#         recWidth=df_dynVals_deleteOutlier.min()
#         # add an totally overlapped rectangle but it will show the label
#         # ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,recWidth,color=Colormap_role_dict[role_choose],label=role_choose))
#         knn = neighbors.KNeighborsRegressor(knn_neighbors, weights=knn_weights)
#         X, y=np.array(Mid_positions).reshape(-1,1), df_dynVals_deleteOutlier
#         try:
#             y_ = knn.fit(X, y.values).predict(T)
#         except ValueError:
#             print("Problem people happen at ", people, role_choose)
#             print("df_dynVals", df_dynVals)
#             print("==================================================")
#             print("df_dynVals_deleteOutlier", df_dynVals_deleteOutlier)
#             raise ValueError
#         functionDK[role_choose]=y_
#         # plt.plot(y_,color=Colormap_role_dict[role_choose],alpha=0.5)
#     # ax.autoscale()
#     # plt.title(col)
#     # plt.legend()
#     # plt.show()
#     # fig.clf()
#     functionDK_people[people]=functionDK
    
#     proximity=-np.abs(np.mean(functionDK['D'] - functionDK['K']))
#     D_t=-np.abs(functionDK['D']-functionDK['K'])

#     time=T.reshape(-1)
#     Convergence=pearsonr(D_t,time)[0]
#     Trend_D=pearsonr(functionDK['D'],time)[0]
#     Trend_K=pearsonr(functionDK['K'],time)[0]
#     delta=[-15, -10, -5, 0, 5, 10, 15]        
#     syncron_lst=[]
#     for d in delta:
#         if d < 0: #ex_ d=-15
#             f_d_shifted=functionDK['D'][-d:]
#             f_k_shifted=functionDK['K'][:d]
#         elif d > 0: #ex_ d=15
#             f_d_shifted=functionDK['D'][:-d]
#             f_k_shifted=functionDK['K'][d:]
#         else: #d=0
#             f_d_shifted=functionDK['D']
#             f_k_shifted=functionDK['K']
#         syncron_candidate=pearsonr(f_d_shifted,f_k_shifted)[0]
        
#         syncron_lst.append(syncron_candidate)
#     syncrony=syncron_lst[np.argmax(np.abs(syncron_lst))]
    
#     RESULT_dict['Proximity[{}]'.format(col)]=proximity
#     RESULT_dict['Trend[{}]_d'.format(col)]=Trend_D
#     RESULT_dict['Trend[{}]_k'.format(col)]=Trend_K
#     RESULT_dict['Convergence[{}]'.format(col)]=Convergence
#     RESULT_dict['Syncrony[{}]'.format(col)]=syncrony
    
#     df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict,orient='index').T
#     df_RESULT_list.index=[people]
#     df_syncrony_measurement=df_syncrony_measurement.append(df_RESULT_list)


    
# # score_column='Proximity[{}]'.format(col)
# score_column='Trend[{0}]{suffix}'.format(col,suffix='_k')
# score_df=df_syncrony_measurement
# score_cols=[score_column]

# # =============================================================================
# '''
    
#     Plot function

# '''
# # =============================================================================
# for people in list(score_df.sort_values(by=score_column).index):
#     df_person_segment_feature_role_dict=df_person_segment_feature_DKIndividual_dict[people]
#     fig, ax = plt.subplots()
#     for role_choose in args.Inspect_roles:
#         df_dynVals=df_person_segment_feature_role_dict[role_choose][col]
#         # remove outlier that is falls over 3 times of the std
#         df_dynVals_deleteOutlier=df_dynVals[(np.abs(stats.zscore(df_dynVals)) < 3)]
#         df_stidx=df_person_segment_feature_role_dict[role_choose][st_col_str]
#         df_edidx=df_person_segment_feature_role_dict[role_choose][ed_col_str]
#         # recWidth=df_dynVals_deleteOutlier.min()/100
#         recWidth=0.0005
#         for x_1 , x_2, y in zip(df_stidx.values ,df_edidx.values,df_dynVals_deleteOutlier.values):            
#             ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,recWidth,color=Colormap_role_dict[role_choose],alpha=0.5))
        
#         # add an totally overlapped rectangle but it will show the label
#         ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,recWidth,color=Colormap_role_dict[role_choose],label=role_choose,alpha=0.5))
        
        
#         plt.plot(functionDK_people[people][role_choose],color=Colormap_role_dict[role_choose])
#     ax.autoscale()
#     plt.title(col)
#     plt.legend()
    
#     score=score_df.loc[people,score_cols]
#     info_arr=["{}: {}".format(idx,v) for idx, v in zip(score.index.values,np.round(score.values,3))]
#     addtext='\n'.join(info_arr)
#     x0, xmax = plt.xlim()
#     y0, ymax = plt.ylim()
#     data_width = xmax - x0
#     data_height = ymax - y0
#     # text(x0/0.1 + data_width * 0.004, -data_height * 0.002, addtext, ha='center', va='center')
#     text(0, -0.1,addtext, ha='center', va='center', transform=ax.transAxes)
    
#     plt.show()
#     fig.clf()
                
# # =============================================================================
# '''

#     Unused old code

# '''
# # =============================================================================


# ''' Migated to Syncrony '''
# st_col_str='IPU_st'
# ed_col_str='IPU_ed'
# feature_choose='VSA2'
# Colormap_role_dict=Dict()
# Colormap_role_dict['D']='orange'
# Colormap_role_dict['K']='blue'

# def calculate_features_continuous(df_person_segment_feature_DKIndividual_dict,features,PhoneOfInterest_str,\
#                            Inspect_roles, Label,\
#                            MinNumTimeSeries=3, label_choose_lst=['ADOS_C'],plot=False):
#     p_1=Inspect_roles[0]
#     p_2=Inspect_roles[1]

#     df_syncrony_measurement=pd.DataFrame()
#     for people in df_person_segment_feature_DKIndividual_dict.keys():
#         if len(df_person_segment_feature_DKIndividual_dict[people][p_1])<MinNumTimeSeries or len(df_person_segment_feature_DKIndividual_dict[people][p_2])<MinNumTimeSeries:
#             continue
#         df_person_segment_feature_role_dict=df_person_segment_feature_DKIndividual_dict[people]
        
#         RESULT_dict={}
        
#         if plot==True:
#             fig, ax = plt.subplots()
#         # kNN fitting
#         Totalendtime=min([df_person_segment_feature_role_dict[role][ed_col_str].values[-1]  for role in Inspect_roles])
#         Mintimeserieslen=min([len(df_person_segment_feature_role_dict[role])  for role in Inspect_roles])
#         T = np.linspace(0, Totalendtime, int(Totalendtime))[:, np.newaxis]
        
#         RESULT_dict['timeSeries_len[{}]'.format(PhoneOfInterest_str)]=Mintimeserieslen
#         for label_choose in label_choose_lst:
#             RESULT_dict[label_choose]=Label.label_raw[Label.label_raw['name']==people][label_choose].values[0]

#         for col in features:
#             functionDK={}
#             for role_choose in Inspect_roles:
#                 df_dynVals=df_person_segment_feature_role_dict[role_choose][col]
#                 # remove outlier that is falls over 3 times of the std
#                 df_dynVals_deleteOutlier=df_dynVals[(np.abs(stats.zscore(df_dynVals)) < 3)]
#                 df_stidx=df_person_segment_feature_role_dict[role_choose][st_col_str]
#                 df_edidx=df_person_segment_feature_role_dict[role_choose][ed_col_str]
                
                
#                 Mid_positions=[]
#                 for x_1 , x_2, y in zip(df_stidx.values ,df_edidx.values,df_dynVals_deleteOutlier.values):
#                     if plot==True:
#                         ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,0.5,color=Colormap_role_dict[role_choose]))
                    
#                     start_time=x_1
#                     end_time=x_2
#                     mid_time=(start_time+end_time)/2
#                     Mid_positions.append(mid_time)    
#                 if plot==True:
#                     # add an totally overlapped rectangle but it will show the label
#                     ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,0.5,color=Colormap_role_dict[role_choose],label=role_choose))
            
                
#                 knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
#                 X, y=np.array(Mid_positions).reshape(-1,1), df_dynVals_deleteOutlier
#                 y_ = knn.fit(X, y.values).predict(T)
#                 functionDK[role_choose]=y_
                
#                 if plot==True:
#                     plt.plot(y_,color=Colormap_role_dict[role_choose],alpha=0.5)
#             if plot==True:
#                 ax.autoscale()
#                 plt.title(col)
#                 plt.legend()
#                 plt.show()
            
            
#             proximity=-np.abs(np.mean(functionDK['D']) - np.mean(functionDK['K']))
#             D_t=-np.abs(functionDK['D']-functionDK['K'])
    
#             time=T.reshape(-1)
#             Convergence=pearsonr(D_t,time)[0]
#             delta=[-15, -10, -5, 0, 5, 10, 15]        
#             syncron_lst=[]
#             for d in delta:
#                 if d < 0: #ex_ d=-15
#                     f_d_shifted=functionDK['D'][-d:]
#                     f_k_shifted=functionDK['K'][:d]
#                 elif d > 0: #ex_ d=15
#                     f_d_shifted=functionDK['D'][:-d]
#                     f_k_shifted=functionDK['K'][d:]
#                 else: #d=0
#                     f_d_shifted=functionDK['D']
#                     f_k_shifted=functionDK['K']
#                 syncron_candidate=pearsonr(f_d_shifted,f_k_shifted)[0]
                
#                 syncron_lst.append(syncron_candidate)
#             syncrony=syncron_lst[np.argmax(np.abs(syncron_lst))]
            
#             RESULT_dict['Proximity[{}]'.format(col)]=proximity
#             RESULT_dict['Convergence[{}]'.format(col)]=Convergence
#             RESULT_dict['Syncrony[{}]'.format(col)]=syncrony
#         # Turn dict to dataframes
#         df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict,orient='index').T
#         df_RESULT_list.index=[people]
#         df_syncrony_measurement=df_syncrony_measurement.append(df_RESULT_list)
#     return df_syncrony_measurement


# # KNN fitting
# def KNNFitting(df_person_segment_feature_DKIndividual_dict,\
#                col_choose,Inspect_roles,\
#                knn_weights='uniform',knn_neighbors=2,\
#                st_col_str='IPU_st', ed_col_str='IPU_ed'):
#     p_1=Inspect_roles[0]
#     p_2=Inspect_roles[1]
    
#     functionDK_people=Dict()
#     for people in df_person_segment_feature_DKIndividual_dict.keys():
#         if len(df_person_segment_feature_DKIndividual_dict[people][p_1])<MinNumTimeSeries or len(df_person_segment_feature_DKIndividual_dict[people][p_2])<MinNumTimeSeries:
#             continue
#         df_person_segment_feature_role_dict=df_person_segment_feature_DKIndividual_dict[people]  
        
#         Totalendtime=min([df_person_segment_feature_role_dict[role][ed_col_str].values[-1]  for role in Inspect_roles])
#         T = np.linspace(0, Totalendtime, int(Totalendtime))[:, np.newaxis]
        
#         functionDK={}
#         for role_choose in Inspect_roles:
#             df_dynVals=df_person_segment_feature_role_dict[role_choose][col_choose]
#             # remove outlier that is falls over 3 times of the std
#             df_dynVals_deleteOutlier=df_dynVals[(np.abs(stats.zscore(df_dynVals)) < 3)]
#             df_stidx=df_person_segment_feature_role_dict[role_choose][st_col_str]
#             df_edidx=df_person_segment_feature_role_dict[role_choose][ed_col_str]
            
            
#             Mid_positions=[]
#             for x_1 , x_2, y in zip(df_stidx.values ,df_edidx.values,df_dynVals_deleteOutlier.values):            
#                 start_time=x_1
#                 end_time=x_2
#                 mid_time=(start_time+end_time)/2
#                 Mid_positions.append(mid_time)    
            
#             # add an totally overlapped rectangle but it will show the label
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


# def calculate_features_col(functionDK_people,col):
#     df_syncrony_measurement=pd.DataFrame()
#     for people in functionDK_people.keys():
#         functionDK=functionDK_people[people]
#         T=functionDK['T']
#         RESULT_dict={}    
        
#         proximity=-np.abs(np.mean(functionDK['D'] - functionDK['K']))
#         D_t=-np.abs(functionDK['D']-functionDK['K'])
    
#         time=T.reshape(-1)
#         Convergence=pearsonr(D_t,time)[0]
#         Trend_D=pearsonr(functionDK['D'],time)[0]
#         Trend_K=pearsonr(functionDK['K'],time)[0]
#         delta=[-15, -10, -5, 0, 5, 10, 15]        
#         syncron_lst=[]
#         for d in delta:
#             if d < 0: #ex_ d=-15
#                 f_d_shifted=functionDK['D'][-d:]
#                 f_k_shifted=functionDK['K'][:d]
#             elif d > 0: #ex_ d=15
#                 f_d_shifted=functionDK['D'][:-d]
#                 f_k_shifted=functionDK['K'][d:]
#             else: #d=0
#                 f_d_shifted=functionDK['D']
#                 f_k_shifted=functionDK['K']
#             syncron_candidate=pearsonr(f_d_shifted,f_k_shifted)[0]
            
#             syncron_lst.append(syncron_candidate)
#         syncrony=syncron_lst[np.argmax(np.abs(syncron_lst))]
        
#         RESULT_dict['Proximity[{}]'.format(col)]=proximity
#         RESULT_dict['Trend[{}]_d'.format(col)]=Trend_D
#         RESULT_dict['Trend[{}]_k'.format(col)]=Trend_K
#         RESULT_dict['Convergence[{}]'.format(col)]=Convergence
#         RESULT_dict['Syncrony[{}]'.format(col)]=syncrony
        
#         df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict,orient='index').T
#         df_RESULT_list.index=[people]
#         df_syncrony_measurement=df_syncrony_measurement.append(df_RESULT_list)
#     return df_syncrony_measurement

# def Add_additional_info(df_person_segment_feature_DKIndividual_dict,Label,label_choose_lst,\
#                         Inspect_roles,\
#                         MinNumTimeSeries=3,PhoneOfInterest_str=' '):
#     p_1=Inspect_roles[0]
#     p_2=Inspect_roles[1]
#     df_basic_additional_info=pd.DataFrame()
#     for people in df_person_segment_feature_DKIndividual_dict.keys():
#         if len(df_person_segment_feature_DKIndividual_dict[people][p_1])<MinNumTimeSeries or len(df_person_segment_feature_DKIndividual_dict[people][p_2])<MinNumTimeSeries:
#             continue
        
#         df_person_segment_feature_role_dict=df_person_segment_feature_DKIndividual_dict[people]
#         RESULT_dict={}
#         Mintimeserieslen=min([len(df_person_segment_feature_role_dict[role])  for role in Inspect_roles])
#         RESULT_dict['timeSeries_len[{}]'.format(PhoneOfInterest_str)]=Mintimeserieslen
#         for label_choose in label_choose_lst:
#             RESULT_dict[label_choose]=Label.label_raw[Label.label_raw['name']==people][label_choose].values[0]
        
        
#         df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict,orient='index').T
#         df_RESULT_list.index=[people]
#         df_basic_additional_info=df_basic_additional_info.append(df_RESULT_list)
#     return df_basic_additional_info


# def calculate_features_continuous_modulized(df_person_segment_feature_DKIndividual_dict,features,PhoneOfInterest_str,\
#                            Inspect_roles, Label,\
#                            knn_weights="uniform",knn_neighbors=2,\
#                            MinNumTimeSeries=3, label_choose_lst=['ADOS_C']):
#     df_basic_additional_info=Add_additional_info(df_person_segment_feature_DKIndividual_dict,Label,label_choose_lst,\
#                                                  Inspect_roles, MinNumTimeSeries=MinNumTimeSeries,PhoneOfInterest_str=PhoneOfInterest_str)
#     df_syncrony_measurement_merge=pd.DataFrame()
#     for col in features:
#         Col_continuous_function_DK=KNNFitting(df_person_segment_feature_DKIndividual_dict,\
#                    col, Inspect_roles,\
#                    knn_weights=knn_weights,knn_neighbors=knn_neighbors,\
#                    st_col_str='IPU_st', ed_col_str='IPU_ed')
            
#         df_syncrony_measurement_col=calculate_features_col(Col_continuous_function_DK,col)
#         df_syncrony_measurement_merge=pd.concat([df_syncrony_measurement_merge,df_syncrony_measurement_col],axis=1)
#     df_syncrony_measurement=pd.concat([df_basic_additional_info,df_syncrony_measurement_merge],axis=1)
#     return df_syncrony_measurement