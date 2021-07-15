#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:23:47 2021

@author: jackchen
"""
import os,glob
from addict import Dict

import articulation.articulation
from articulation.HYPERPARAM import phonewoprosody, Label


from metric import Evaluation_method
# =============================================================================
'''

    Ctx phone selecting 

'''
# =============================================================================
def Collector(df_result_FeatComb_table_collect,correlation_type='spearmanr'):
    df_significant_collection=Dict()
    for feat_type in df_result_FeatComb_table_collect.keys():
        for df_result_dict in df_result_FeatComb_table_collect[feat_type]:
            for combstr, df_result in df_result_dict.items():
                if df_result['de-zero_num'].mean() > 20:
                    df_significant_collection[combstr]=df_result
    return df_significant_collection
# df_result=pd.DataFrame([],columns=['spearmanr', 'spear_pvalue', 'de-zero_num'])
# cond=vars()['df_result']['de-zero_num'].mean() >5


def Selector_Pval(df_significant_collection):
    df_significant_collection_stage2=Dict()
    for combstr, df_result in df_significant_collection.items():
        if df_result['spear_pvalue'].min() < 0.0001:
            df_significant_collection_stage2[combstr]=df_result
            
    return df_significant_collection_stage2

def Selector_Pval_colrow(df_significant_collection,col='spear_pvalue',rows=['MSB_f1(A:,i:,u:)','MSB_f2(A:,i:,u:)']):
    df_significant_collection_stage2=Dict()
    for combstr, df_result in df_significant_collection.items():
        cond=False
        for row in rows:
            cond = cond or (df_result.loc[row,col] < 0.0001)
        if cond:
            df_significant_collection_stage2[combstr]=df_result
            
    return df_significant_collection_stage2

# =============================================================================
'''

    Ctx phone Feature extracting

'''

# =============================================================================

def Calculate_dfStatistics(CtxDepVowel_AUI_dict, comb,PhoneOfInterest):
    #CtxDepVowel_AUI_dict, comb,PhoneOfInterest=Feature_dicts[feattype], phone_combination_list,PhoneOfInterest
    articulatn=articulation.articulation.Articulation()
    Eval_med=Evaluation_method()
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
    # Num_people=len(Set_unionPoeple_AUI)
    
    #The input to articulation.calculate_features has to be Vowels_AUI form
    Vowels_AUI = Dict()
    for people in Set_unionPoeple_AUI:
        for phone in Main_dict.keys():
            Vowels_AUI[people][phone]=Main_dict[phone][people]
        
        
    df_formant_statistic=articulatn.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest)
    df_formant_statistic=Eval_med._Postprocess_dfformantstatistic(df_formant_statistic)
    return df_formant_statistic

def Extract_Vowels_AUI(CtxDepVowel_AUI_dict, comb):
    #CtxDepVowel_AUI_dict, comb,PhoneOfInterest=Feature_dicts[feattype], phone_combination_list,PhoneOfInterest
    # =============================================================================
    '''
    
        ANOVA F value genertor
        
        input: Vowel_AUI
        output df_f_val
    '''
    
    ''' Collect the people that have phones of A: u: i:, and Set_unionPoeple_AUI is a list of people containing the three phones of inspect'''
    cond1=comb[0] in CtxDepVowel_AUI_dict.keys()
    cond2=comb[1] in CtxDepVowel_AUI_dict.keys()
    cond3=comb[2] in CtxDepVowel_AUI_dict.keys()
    # print(comb)
    # print(cond1, cond2, cond3)
    
    
    if cond1 and cond2 and cond3:
        
        Main_dict=Dict()
        Main_dict['A:'], Main_dict['u:'], Main_dict['i:']= CtxDepVowel_AUI_dict[comb[0]], CtxDepVowel_AUI_dict[comb[1]], CtxDepVowel_AUI_dict[comb[2]]
        # Take union of a_dict, u_dict, i_dict
        
        
        Set_unionPoeple_AUI=set(list(Main_dict['A:'].keys())+list(Main_dict['u:'].keys())+list(Main_dict['i:'].keys()))
        Set_unionPoeple_AUI=Set_unionPoeple_AUI.intersection(list(Main_dict['A:'].keys()))
        Set_unionPoeple_AUI=Set_unionPoeple_AUI.intersection(list(Main_dict['u:'].keys()))
        Set_unionPoeple_AUI=Set_unionPoeple_AUI.intersection(list(Main_dict['i:'].keys()))
        # Num_people=len(Set_unionPoeple_AUI)
        
        #The input to articulation.calculate_features has to be Vowels_AUI form
        Vowels_AUI = Dict()
        for people in Set_unionPoeple_AUI:
            for phone in Main_dict.keys():
                Vowels_AUI[people][phone]=Main_dict[phone][people]
    else:
        Vowels_AUI=[]

    return Vowels_AUI