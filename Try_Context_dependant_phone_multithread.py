#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 11:57:47 2021

@author: jackchen
"""


from scipy.io.wavfile import read
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.mlab as mlab
import math
import pysptk

import uuid
import pandas as pd
import torch
from tqdm import tqdm
from addict import Dict
import glob
import argparse
import math
from pydub import AudioSegment

from multiprocessing import Pool, current_process
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.feature_selection import f_classif


from articulation.HYPERPARAM import phonewoprosody, Label
import pickle
import itertools
import articulation.articulation
from scipy.stats import spearmanr,pearsonr 
from metric import Evaluation_method
import re
# from utils_jack  import  Postprocess_dfformantstatistic, Calculate_correlation
np.seterr(divide='ignore', invalid='ignore')
class Try_Combination:
    def __init__(self,PhoneOfInterest,\
                 constrain_sex=-1,constrain_module=-1,num_people_min=5,N=2,\
                 label_choose_lst=['ADOS_C'],columns=['MSB_f1(A:,u:,i:)', 'MSB_f2(A:,u:,i:)'],\
                 evictNamelst=[]):
        self.constrain_sex=constrain_sex
        self.constrain_module=constrain_sex
        self.num_people_min=num_people_min
        self.N=N
        self.label_choose_lst=label_choose_lst
        self.columns=columns
        self.PhoneOfInterest=PhoneOfInterest
        self.evictNamelst=evictNamelst
        self.correlation_type='spearmanr'
        self.corr_label='ADOS'
        self.constrain_assessment=-1
    def Try_Combination_map(self,combs_lst,feat,CtxDepVowel_AUI_dict,):
        # print(" process PID", os.getpid(), " running")
        Result_table_dict=Dict()
        for comb in combs_lst:
            # =============================================================================
            '''
            
                ANOVA F value genertor
                
                input: Vowel_AUI
                output df_f_val
            '''
            
            ''' Collect the people that have phones of A: u: i:, and Set_unionPoeple_AUI is a list of people containing the three phones of inspect'''
            Main_dict=Dict()
            Main_dict['A:'], Main_dict['u:'], Main_dict['i:']= CtxDepVowel_AUI_dict[comb[0]], CtxDepVowel_AUI_dict[comb[1]], CtxDepVowel_AUI_dict[comb[2]]
            # Take union of a_dict, u_dict, i_dict
            
            
            Set_unionPoeple_AUI=set(list(Main_dict['A:'].keys())+list(Main_dict['u:'].keys())+list(Main_dict['i:'].keys()))
            Set_unionPoeple_AUI=Set_unionPoeple_AUI.intersection(list(Main_dict['A:'].keys()))
            Set_unionPoeple_AUI=Set_unionPoeple_AUI.intersection(list(Main_dict['u:'].keys()))
            Set_unionPoeple_AUI=Set_unionPoeple_AUI.intersection(list(Main_dict['i:'].keys()))
            Num_people=len(Set_unionPoeple_AUI)
            
            if Num_people > self.num_people_min:
                #The input to articulation.calculate_features has to be Vowels_AUI form
                Vowels_AUI = Dict()
                for people in Set_unionPoeple_AUI:
                    for phone in Main_dict.keys():
                        Vowels_AUI[people][phone]=Main_dict[phone][people]
                
                # try: 
                try:
                    # print(comb)
                    df_formant_statistic=articulation.calculate_features(Vowels_AUI,Label,PhoneOfInterest=self.PhoneOfInterest)
                except np.linalg.LinAlgError:
                    print("LinAlgError: at comb", comb)
                    raise np.linalg.LinAlgError
                df_formant_statistic=Eval_med._Postprocess_dfformantstatistic(df_formant_statistic)
                df_result_table=Eval_med.Calculate_correlation(label_choose_lst=self.label_choose_lst,df_formant_statistic=df_formant_statistic,N=self.N,columns=self.columns,\
                                                               corr_label=self.corr_label,constrain_sex=self.constrain_sex,constrain_module=self.constrain_module,constrain_assessment=self.constrain_assessment,\
                                                               evictNamelst=self.evictNamelst,correlation_type=self.correlation_type)
                
                infocode='N{0}_feat{1}_{2}'.format(N,feat,'__'.join(comb))
                Result_table_dict[infocode]=df_result_table
                # print(df_result_table)
                # except:
                #     print("combination",comb, "has failed")
                #     pass
        return Result_table_dict






# ## Debug
    # df_formant_statistic_here=df_formant_statistic.sort_index()
    # df_formant_statistic_tocmp=pickle.load(open('articulation/Pickles/Session_formants_people_vowel_feat/Formant_AUI_tVSAFCRFvals_ASDkid.pkl',"rb"))
    # df_formant_statistic_tocmp=df_formant_statistic_tocmp.sort_index()
    # Aaa=(df_formant_statistic_here - df_formant_statistic_tocmp)
    # Aab=Aaa[Aaa>0.000001]
    # print(len(np.where(Aaa>0.000001)[0]))

def Vowel_ADOSCode_Combinator(vowel_dict,feat,ui_type='merged',label_choose_lst=['ADOS_C']):
    ''' 
        ui_type= 'jw' or 'iu' or 'merged '
        feat= 'CtxDepVowel_AUI' or 'LeftDepVowel_AUI' or 'RightDepVowel_AUI'
    
    '''
    listA, listu, listi=[], [], []
    for ctx_P in vowel_dict.keys():
        feat_type=feat[:feat.find("Dep")]
                
        if feat_type == 'Ctx':
            critical_P=ctx_P[ctx_P.find('-')+1:ctx_P.find('+')]
        elif feat_type == 'Left':
            critical_P=ctx_P[ctx_P.find('-')+1:]
        elif feat_type == 'Right':
            critical_P=ctx_P[:ctx_P.find('+')]
        
        if ui_type == 'merged' or ui_type == 'iu': # if ui_type is merged, in the previous function will treat w as u and j as i
            if critical_P == 'u':
                listu.append(ctx_P)
            if critical_P == 'A':
                listA.append(ctx_P)
            if critical_P == 'i':
                listi.append(ctx_P)
        elif ui_type == 'jw':
            if critical_P == 'w':
                listu.append(ctx_P)
            if critical_P == 'A':
                listA.append(ctx_P)
            if critical_P == 'j':
                listi.append(ctx_P)
        
    DepAUI_combs=[listA,listu,listi,label_choose_lst]
    return DepAUI_combs

def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--outpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--dfstat_outpath', default='"Features/artuculation_AUI/CtxDepVowels/"',
                        help='path of the base directory')
    parser.add_argument('--constrain_module', default=-1,
                            help='path of the base directory')
    parser.add_argument('--constrain_sex', default=-1,
                            help='path of the base directory')
    parser.add_argument('--label_choose_lst', default=['ADOS_C'],
                            help='')
    parser.add_argument('--check', default=True,
                            help='')
    args = parser.parse_args()
    return args

args = get_args()
base_path=args.base_path
outpath=args.outpath
constrain_sex=args.constrain_sex
constrain_module=args.constrain_module
label_choose_lst=args.label_choose_lst






# =============================================================================
# Initialize some parameters
# =============================================================================
PhoneMapp_dict=phonewoprosody.PhoneMapp_dict
PhoneOfInterest=list(PhoneMapp_dict.keys())
articulation=articulation.articulation.Articulation()
Eval_med=Evaluation_method()
columns=[
       'F_vals_f1(A:,i:,u:)', 'F_vals_f2(A:,i:,u:)',
       'MSB_f1(A:,i:,u:)', 'MSB_f2(A:,i:,u:)',
       'MSB_mix', 'BWratio(A:,i:,u:)', 'BV(A:,i:,u:)_l2', 'WV(A:,i:,u:)_l2',
       'BWratio(i:,u:)', 'BV(i:,u:)_l2',
       'WV(i:,u:)_l2',
       'BWratio(A:,u:)',
       'BV(A:,u:)_l2', 'WV(A:,u:)_l2',
       'BWratio(A:,i:)',
       'BV(A:,i:)_l2', 'WV(A:,i:)_l2']
num_people_min=5
N=2 # The least number of critical phones (A:, u:, i:)
try_combination=Try_Combination(PhoneOfInterest,\
                 constrain_sex,constrain_module,\
                 num_people_min,N,\
                 label_choose_lst=label_choose_lst,columns=columns)

# =============================================================================
'''# Get features'''
# =============================================================================

# Phoneset_sets={'Manner_simp1':phonewoprosody.Manner_sets_simple1, \
#                 'Manner_simp2':phonewoprosody.Manner_sets_simple2, \
#                 'Place_simp1':phonewoprosody.Place_sets_simple1,  \
#                 'Place_simp2':phonewoprosody.Place_sets_simple2}

# =============================================================================
'''# add condition

    We have set some of the people to be unreasonable data (details see articulation/Get_F1F2.py)
    
    
    Output: 
        The whole collection of the correlation results
    
'''

# =============================================================================
Result_table_dict=Dict()
ManualCondition=Dict()
suffix='.xlsx'
condfiles=glob.glob('articulation/Inspect/condition/*'+suffix)
for file in condfiles:
    df_cond=pd.read_excel(file)
    name=os.path.basename(file).replace(suffix,"")
    ManualCondition[name]=df_cond['Unnamed: 0'][df_cond['50%']==True]

for CtxPhone_types in tqdm(['Manner_simp1','Manner_simp2','Place_simp1','Place_simp2','']):
# for CtxPhone_types in tqdm(['']):
    Feature_dicts=pickle.load(open(outpath+"/AUI_ContextDepPhonesMerge_{0}_uwij.pkl".format(CtxPhone_types),"rb"))
    # Feature_dicts=pickle.load(open(outpath+"/AUI_ContextDepPhonesMerge_uwij.pkl","rb"))
    
    # =============================================================================
    '''
        Generate combinations
    
        Create all combinations of ctx dep phones from three sets
        {A+N, ...}, {u+Vowel, ...}, {i+fricative, ...}
    
        Output will be something like : A+N__u+Vowel__i+fricative
    
    '''
    # Pseudo_feat_comb=Dict()
    # Pseudo_feat_comb['[s]-A+[s]']=[]
    # Pseudo_feat_comb['[s]-w+O:3']=[]
    # Pseudo_feat_comb['ts6-j+oU4']=[]
    # =============================================================================
    Combinations_Depvowel_dict=Dict()
    for feat in Feature_dicts.keys():
    # for feat in ['LeftDepVowel_AUI', 'RightDepVowel_AUI']:
        DepVowel_AUI=Feature_dicts[feat]
        # feat_comb=Vowel_Combinator(DepVowel_AUI,feat)
        feat_comb=Vowel_ADOSCode_Combinator(DepVowel_AUI,feat,ui_type='merged')
        Combinations_Depvowel_dict[feat]=list(itertools.product(*feat_comb))
    
    
    
    
    
    ''' multi processing start '''
    import time
    start_time = time.time()
    
    interval=20
    # pool = Pool(int(1))
    pool = Pool(os.cpu_count())
    # pool = Pool(os.cpu_count())
    df_result_FeatComb_table=Dict()
    # for feat in tqdm(['RightDepVowel_AUI', 'LeftDepVowel_AUI', 'CtxDepVowel_AUI']):
    for feat in tqdm(list(Feature_dicts.keys())):
        CtxDepVowel_AUI_dict=Feature_dicts[feat]
        
        if args.check:  #Check if certainphone like 'w' in the CtxPhones    
            for CtxP in CtxDepVowel_AUI_dict.keys():
                for people in CtxDepVowel_AUI_dict[CtxP].keys():
                    for CtxPhone in CtxDepVowel_AUI_dict[CtxP][people].index:
                        left_P=CtxPhone[:CtxPhone.find('-')]
                        right_P=CtxPhone[CtxPhone.find('+')+1:]
                        critical_P=CtxPhone[CtxPhone.find('-')+1:CtxPhone.find('+')]
                        
                        if critical_P == 'w':
                            raise Exception()

            # for people in PeopleLeftDepPhoneFunctional_dict.keys():
            #     for CtxPhone in PeopleLeftDepPhoneFunctional_dict[people].keys():
            #         left_P=CtxPhone[:CtxPhone.find('-')]
            #         critical_P=CtxPhone[CtxPhone.find('-')+1:]
            #         if critical_P == 'w':
            #             aaa=ccc
            # for CtxP in CtxDepVowel_AUI_dict.keys():
            #     for people in CtxDepVowel_AUI_dict[CtxP].keys():
            #         for CtxPhone in CtxDepVowel_AUI_dict[CtxPhone][people].index:
            #             critical_P=CtxPhone[:CtxPhone.find('+')]
            #             right_P=CtxPhone[CtxPhone.find('+')+1:]
            #             if critical_P == 'w':
            #                 aaa=ccc
        
        # print("Stop here and the Ctxphones don't have the w")
        combs_tup=Combinations_Depvowel_dict[feat]
        keys=[]
        for i in range(0,len(combs_tup),interval):
            # print(list(combs_tup.keys())[i:i+interval])
            keys.append(combs_tup[i:i+interval])
        flat_keys=[item for sublist in keys for item in sublist]
        assert len(flat_keys) == len(combs_tup)
        
        final_result = pool.starmap(try_combination.Try_Combination_map, [(combs_lst,feat,CtxDepVowel_AUI_dict, \
                                                          ) \
                                                          for combs_lst in tqdm(keys)])
        df_result_FeatComb_table[feat]=final_result
    
    df_result_FeatComb_table_collect=Dict()
    for feat in df_result_FeatComb_table.keys():
        df_result_FeatComb_table_collect[feat]=[]
        feat_results=df_result_FeatComb_table[feat]
        for feat_result in feat_results:
            if len(feat_result)>0:
                df_result_FeatComb_table_collect[feat].append(feat_result)
    
    pickle.dump(df_result_FeatComb_table_collect,open(outpath+"/df_result_FeatComb_table_collect_{0}.pkl".format(CtxPhone_types),"wb"))
    
    print("--- %s seconds ---" % (time.time() - start_time))
    ''' multi processing end '''

