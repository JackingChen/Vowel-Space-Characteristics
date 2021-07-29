#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 11:57:47 2021

@author: jackchen
"""
import os, glob
from addict import Dict
import pandas as pd
from tqdm import tqdm
import pickle
import argparse
import re
from articulation.HYPERPARAM import phonewoprosody, Label
import articulation.articulation
from metric import Evaluation_method

import numpy as np
from matplotlib import pyplot as plt
from metric import independent_corr 

def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--outpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--dfstat_outpath', default='Features/artuculation_AUI/CtxDepVowels/',
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
# =============================================================================
# Initialize some parameters
# =============================================================================
PhoneMapp_dict=phonewoprosody.PhoneMapp_dict
PhoneOfInterest=list(PhoneMapp_dict.keys())
articulation=articulation.articulation.Articulation()
Eval_med=Evaluation_method()
columns=['F_vals_f1(A:,i:,u:)', 'F_vals_f2(A:,i:,u:)',
       'F_val_mix(A:,i:,u:)', 'MSB_f1(A:,i:,u:)', 'MSB_f2(A:,i:,u:)',
       'MSB_mix', 'F_vals_f1(A:,u:)', 'F_vals_f2(A:,u:)', 'F_val_mix(A:,u:)',
       'MSB_f1(A:,u:)', 'MSB_f2(A:,u:)', 'F_vals_f1(A:,i:)',
       'F_vals_f2(A:,i:)', 'F_val_mix(A:,i:)', 'MSB_f1(A:,i:)',
       'MSB_f2(A:,i:)', 'F_vals_f1(i:,u:)', 'F_vals_f2(i:,u:)',
       'F_val_mix(i:,u:)', 'MSB_f1(i:,u:)', 'MSB_f2(i:,u:)']
num_people_min=5
N=2 # The least number of critical phones (A:, u:, i:)

# =============================================================================
# User define conditions (df_result['de-zero_num'].mean() > 5) to get wanted results
# =============================================================================
from Filtering_n_FeatureExtracting import Collector, Selector_Pval, Selector_Pval_colrow, Calculate_dfStatistics, Selector_Rval_colrow,\
                                          Selector_independent_corr  
Result_dict=Dict()
rows=['BWratio(A:,i:,u:)','BV(A:,i:,u:)_l2']
Criteria_dict=Dict()
Criteria_dict['BWratio(A:,i:,u:)'].r = 0.468
Criteria_dict['BWratio(A:,i:,u:)'].N = 82
Criteria_dict['BV(A:,i:,u:)_l2'].r = 0.481
Criteria_dict['BV(A:,i:,u:)_l2'].N = 82
significant_val=0.05
r_val=0.
for CtxPhone_types in tqdm(['Manner_simp1','Manner_simp2','Place_simp1','Place_simp2','']):
# for CtxPhone_types in tqdm(['']):    
    df_result_FeatComb_table_collect=pickle.load(open(outpath+"/df_result_FeatComb_table_collect_{0}.pkl".format(CtxPhone_types),"rb"))
    # df_result_FeatComb_table_collect=pickle.load(open(outpath+"/df_result_FeatComb_table_collect_{0}.pkl".format(CtxPhone_types),"rb"))
    # print(df_result_FeatComb_table_collect['LeftDepVowel_AUI'], df_result_FeatComb_table_collect['RightDepVowel_AUI'])
    
    df_significant_collection=Collector(df_result_FeatComb_table_collect)
    # df_significant_collection_stage2=Selector_Pval(df_significant_collection, significant_val=significant_val)
    # df_significant_collection_stage2_msbf1f2=Selector_Pval_colrow(df_significant_collection,col='spear_pvalue', rows=rows, significant_val=significant_val)
    # df_significant_collection_stage2_msbf1f2=Selector_Rval_colrow(df_significant_collection,col='spearmanr', rows=rows, r_val=significant_val)
    df_significant_collection_stage2_spearrsig=Selector_independent_corr(df_significant_collection,col='spearmanr',rows=rows,\
                              Criteria_dict=Criteria_dict,\
                              significant_val=0.95)  #Try to find the r value larger than non context dependent versions
    # Result_dict['min< thresh'].update(df_significant_collection_stage2)
    Result_dict['R significant'].update(df_significant_collection_stage2_spearrsig)

def arrange_Resultdict(Result_dict,rows):
    Result_feature_dict=Dict()
    for row in rows:
        Result_feature_dict[row]=pd.DataFrame()
        for criterion in Result_dict.keys():
            for keys, values in Result_dict[criterion].items():
                df_result=values.loc[[row],:].copy()
                df_result.index= keys + df_result.index
                Result_feature_dict[row]=Result_feature_dict[row].append(df_result)
    return Result_feature_dict



cmp_r=0.468
N_cmp_r=82
Arranged_resultdict=arrange_Resultdict(Result_dict,rows)
for feature in Arranged_resultdict.keys():
    for idx in Arranged_resultdict[feature].index:
        df_=Arranged_resultdict[feature].loc[idx]
        xy, n= np.abs(cmp_r), N_cmp_r
        ab, n2= np.abs(df_.iloc[0]), df_.iloc[-1]
        z, p = independent_corr(xy, ab, n, n2 = n2, twotailed=False, method='fisher')
        if (p < significant_val and ab > xy):
            # print(idx)
            # print(feature)
            print(df_)
            print("r: ",ab, ' p: ',p)




for CtxPhone_types in tqdm(['Manner_simp1','Manner_simp2','Place_simp1','Place_simp2']):
    Feature_dicts=pickle.load(open(outpath+"/AUI_ContextDepPhonesMerge_{0}_uwij.pkl".format(CtxPhone_types),"rb"))
    for comb in Result_dict['R significant'].keys():
        feattype=comb[re.search('_feat',comb).end():re.search('CtxDepVowel_AUI|LeftDepVowel_AUI|RightDepVowel_AUI',comb).end()]
        phone_str=comb[re.search('CtxDepVowel_AUI|LeftDepVowel_AUI|RightDepVowel_AUI',comb).end()+1:]
        CtxPhone={}
        CtxPhone['A:'], CtxPhone['u:'], CtxPhone['i:'], label=phone_str.split("__") # dtype: strings
        
        # There's a bug above, we temprory avoid the bug to do the later stuffs
        if args.check:
            # assert CtxPhone['A:'] in Feature_dicts[feattype].keys()
            # assert CtxPhone['u:'] in Feature_dicts[feattype].keys()
            # assert CtxPhone['i:'] in Feature_dicts[feattype].keys()
            A_valid=CtxPhone['A:'] in Feature_dicts[feattype].keys()
            u_valid=CtxPhone['u:'] in Feature_dicts[feattype].keys()
            i_valid=CtxPhone['i:'] in Feature_dicts[feattype].keys()
            valid=A_valid and u_valid and i_valid
        if valid:
            phone_combination_list=[CtxPhone['A:'], CtxPhone['u:'], CtxPhone['i:'], label]
            df_formant_statistic=Calculate_dfStatistics(Feature_dicts[feattype], phone_combination_list,PhoneOfInterest)
            
            feature_path=base_path+"/"+args.dfstat_outpath
            pickle.dump(df_formant_statistic,open(feature_path+"/{0}.pkl".format(phone_str),"wb"))
            

# =============================================================================
'''

    Inspect area

'''
from Filtering_n_FeatureExtracting import Extract_Vowels_AUI
# =============================================================================
Feature_dicts_top=Dict()
for CtxPhone_types in tqdm(['Manner_simp1','Manner_simp2','Place_simp1','Place_simp2']):
    Feature_dicts_top[CtxPhone_types]=pickle.load(open(outpath+"/AUI_ContextDepPhonesMerge_{0}_ij.pkl".format(CtxPhone_types),"rb"))
    


''' query by comb string '''
Eval_med=Evaluation_method()
role='ASDkid'
BasicVowelAUI_path='articulation/Pickles/Session_formants_people_vowel_feat'
BasicVowelAUI=pickle.load(open(BasicVowelAUI_path+"/Vowels_AUI_{}.pkl".format(role),"rb"))
df_formant_statistic_basic=articulation.calculate_features(BasicVowelAUI,Label,PhoneOfInterest=PhoneOfInterest)
df_formant_statistic_basic=Eval_med._Postprocess_dfformantstatistic(df_formant_statistic_basic)
df_formant_statistic_basic=Eval_med._Postprocess_dfformantstatistic_N_notnanADOS(df_formant_statistic_basic,N=1)


combstr_lst=['A+alveolar__u+All__i+All__ADOS_C-MSB_f1(A:,i:,u:)-MSB_f2(A:,i:,u:)']
for combstr in combstr_lst:
    phoneLab_str=combstr.split("-")[0]
    CtxPhone={}
    CtxPhone['A:'], CtxPhone['u:'], CtxPhone['i:'], label=phoneLab_str.split("__") # dtype: strings
    
    cond_plus='+' in CtxPhone['A:']
    cond_minus='-' in CtxPhone['A:']
    
    
    phone_str='__'.join([CtxPhone['A:'], CtxPhone['u:'], CtxPhone['i:']])
    for CtxPhone_types in Feature_dicts_top.keys(): #'Manner_simp1' ...
        Feature_dicts=Feature_dicts_top[CtxPhone_types]
        for feattype in Feature_dicts.keys():
            dict_tmp=Extract_Vowels_AUI(Feature_dicts[feattype],comb=[CtxPhone['A:'], CtxPhone['u:'], CtxPhone['i:']])
            if len(dict_tmp) != 0:
                Vowels_AUI=dict_tmp
                df_formant_statistic=Calculate_dfStatistics(Feature_dicts[feattype],[CtxPhone['A:'], CtxPhone['u:'], CtxPhone['i:']],PhoneOfInterest)
                df_formant_statistic=Eval_med._Postprocess_dfformantstatistic(df_formant_statistic)
                df_formant_statistic=Eval_med._Postprocess_dfformantstatistic_N_notnanADOS(df_formant_statistic,N=1)
    
    
    featrue_inspect='MSB_f1(A:,i:,u:)'
    ''' Ploting area '''
    Set_unionPoeple_AUI=set(list(df_formant_statistic.index)+list(df_formant_statistic_basic.index))
    Set_unionPoeple_AUI=Set_unionPoeple_AUI.intersection(list(df_formant_statistic_basic.index))
    Set_unionPoeple_AUI=Set_unionPoeple_AUI.intersection(list(df_formant_statistic.index))
    data=pd.DataFrame()
    data[['X0','Y0']]=df_formant_statistic.loc[Set_unionPoeple_AUI].sort_index()[[featrue_inspect,'ADOS']]
    data[['X1','Y1']]=df_formant_statistic_basic.loc[Set_unionPoeple_AUI].sort_index()[[featrue_inspect,'ADOS']]
    
    
    # data = np.genfromtxt('file1.dat', delimiter=',', skip_header=1, names=['MAG', 'X0', 'Y0','X1','Y1'])
    
    plt.scatter(data['X0'], data['Y0'], color='r', zorder=10)
    plt.scatter(data['X1'], data['Y1'], color='b', zorder=10)
    for i in range(len(data)):
        d=data.iloc[i]
        # plt.arrow(d['X0'],d['Y0'],d['X1']-d['X0'], d['Y1']-d['Y0'], 
        # shape='full', color='b', lw=d['MAG']/2., length_includes_head=True, 
        # zorder=0, head_length=3., head_width=1.5)
        
        plt.arrow(d['X0'],d['Y0'],d['X1']-d['X0'], d['Y1']-d['Y0'], 
        shape='full', color='b', length_includes_head=True, 
        zorder=0, head_length=3., head_width=1.5)
    
    plt.show()