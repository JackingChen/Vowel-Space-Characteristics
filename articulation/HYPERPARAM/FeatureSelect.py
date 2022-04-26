#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:44:54 2022

@author: jackchen
"""
from itertools import combinations
from addict import Dict
# =============================================================================
'''

    Phonation Interaction feature
'''
# =============================================================================


Phonation_Proximity_cols=[
 'Proximity[intensity_mean_mean(A:,i:,u:)]',
 'Proximity[meanF0_mean(A:,i:,u:)]',
 'Proximity[stdevF0_mean(A:,i:,u:)]',
 'Proximity[hnr_mean(A:,i:,u:)]',
 'Proximity[localJitter_mean(A:,i:,u:)]',
 'Proximity[localabsoluteJitter_mean(A:,i:,u:)]',
 # 'Proximity[rapJitter_mean(A:,i:,u:)]',
 # 'Proximity[ddpJitter_mean(A:,i:,u:)]',
 'Proximity[localShimmer_mean(A:,i:,u:)]',
 'Proximity[localdbShimmer_mean(A:,i:,u:)]',
 'Proximity[intensity_mean_var(A:,i:,u:)]',
 # 'Proximity[meanF0_var(A:,i:,u:)]',
 # 'Proximity[stdevF0_var(A:,i:,u:)]',
 # 'Proximity[hnr_var(A:,i:,u:)]',
 # 'Proximity[localJitter_var(A:,i:,u:)]',
 # 'Proximity[localabsoluteJitter_var(A:,i:,u:)]',
 # 'Proximity[rapJitter_var(A:,i:,u:)]',
 # 'Proximity[ddpJitter_var(A:,i:,u:)]',
 # 'Proximity[localShimmer_var(A:,i:,u:)]',
 # 'Proximity[localdbShimmer_var(A:,i:,u:)]',
 'Proximity[intensity_mean_max(A:,i:,u:)]',
 'Proximity[meanF0_max(A:,i:,u:)]',
 'Proximity[stdevF0_max(A:,i:,u:)]',
 'Proximity[hnr_max(A:,i:,u:)]',
 'Proximity[localJitter_max(A:,i:,u:)]',
 'Proximity[localabsoluteJitter_max(A:,i:,u:)]',
 # 'Proximity[rapJitter_max(A:,i:,u:)]',
 # 'Proximity[ddpJitter_max(A:,i:,u:)]',
 'Proximity[localShimmer_max(A:,i:,u:)]',
 'Proximity[localdbShimmer_max(A:,i:,u:)]'
 ]
Phonation_Trend_D_cols=[
 'Trend[intensity_mean_mean(A:,i:,u:)]_d',
 'Trend[meanF0_mean(A:,i:,u:)]_d',
 'Trend[stdevF0_mean(A:,i:,u:)]_d',
 'Trend[hnr_mean(A:,i:,u:)]_d',
 'Trend[localJitter_mean(A:,i:,u:)]_d',
 'Trend[localabsoluteJitter_mean(A:,i:,u:)]_d',
 # 'Trend[rapJitter_mean(A:,i:,u:)]_d',
 # 'Trend[ddpJitter_mean(A:,i:,u:)]_d',
 'Trend[localShimmer_mean(A:,i:,u:)]_d',
 'Trend[localdbShimmer_mean(A:,i:,u:)]_d',
 # 'Trend[intensity_mean_var(A:,i:,u:)]_d',
 # 'Trend[meanF0_var(A:,i:,u:)]_d',
 # 'Trend[stdevF0_var(A:,i:,u:)]_d',
 # 'Trend[hnr_var(A:,i:,u:)]_d',
 # 'Trend[localJitter_var(A:,i:,u:)]_d',
 # 'Trend[localabsoluteJitter_var(A:,i:,u:)]_d',
 # 'Trend[rapJitter_var(A:,i:,u:)]_d',
 # 'Trend[ddpJitter_var(A:,i:,u:)]_d',
 # 'Trend[localShimmer_var(A:,i:,u:)]_d',
 # 'Trend[localdbShimmer_var(A:,i:,u:)]_d',
 'Trend[intensity_mean_max(A:,i:,u:)]_d',
 'Trend[meanF0_max(A:,i:,u:)]_d',
 'Trend[stdevF0_max(A:,i:,u:)]_d',
 'Trend[hnr_max(A:,i:,u:)]_d',
 'Trend[localJitter_max(A:,i:,u:)]_d',
 'Trend[localabsoluteJitter_max(A:,i:,u:)]_d',
 # 'Trend[rapJitter_max(A:,i:,u:)]_d',
 # 'Trend[ddpJitter_max(A:,i:,u:)]_d',
 'Trend[localShimmer_max(A:,i:,u:)]_d',
 'Trend[localdbShimmer_max(A:,i:,u:)]_d'
    ]
Phonation_Trend_K_cols=[
 'Trend[intensity_mean_mean(A:,i:,u:)]_k',
 'Trend[meanF0_mean(A:,i:,u:)]_k',
 'Trend[stdevF0_mean(A:,i:,u:)]_k',
 'Trend[hnr_mean(A:,i:,u:)]_k',
 'Trend[localJitter_mean(A:,i:,u:)]_k',
 'Trend[localabsoluteJitter_mean(A:,i:,u:)]_k',
 # 'Trend[rapJitter_mean(A:,i:,u:)]_k',
 # 'Trend[ddpJitter_mean(A:,i:,u:)]_k',
 'Trend[localShimmer_mean(A:,i:,u:)]_k',
 'Trend[localdbShimmer_mean(A:,i:,u:)]_k',
 # 'Trend[intensity_mean_var(A:,i:,u:)]_k',
 # 'Trend[meanF0_var(A:,i:,u:)]_k',
 # 'Trend[stdevF0_var(A:,i:,u:)]_k',
 # 'Trend[hnr_var(A:,i:,u:)]_k',
 # 'Trend[localJitter_var(A:,i:,u:)]_k',
 # 'Trend[localabsoluteJitter_var(A:,i:,u:)]_k',
 # 'Trend[rapJitter_var(A:,i:,u:)]_k',
 # 'Trend[ddpJitter_var(A:,i:,u:)]_k',
 # 'Trend[localShimmer_var(A:,i:,u:)]_k',
 # 'Trend[localdbShimmer_var(A:,i:,u:)]_k',
 'Trend[intensity_mean_max(A:,i:,u:)]_k',
 'Trend[meanF0_max(A:,i:,u:)]_k',
 'Trend[stdevF0_max(A:,i:,u:)]_k',
 'Trend[hnr_max(A:,i:,u:)]_k',
 'Trend[localJitter_max(A:,i:,u:)]_k',
 'Trend[localabsoluteJitter_max(A:,i:,u:)]_k',
 # 'Trend[rapJitter_max(A:,i:,u:)]_k',
 # 'Trend[ddpJitter_max(A:,i:,u:)]_k',
 'Trend[localShimmer_max(A:,i:,u:)]_k',
 'Trend[localdbShimmer_max(A:,i:,u:)]_k'
    ]
Phonation_Convergence_cols=[
 'Convergence[intensity_mean_mean(A:,i:,u:)]',
 'Convergence[meanF0_mean(A:,i:,u:)]',
 'Convergence[stdevF0_mean(A:,i:,u:)]',
 'Convergence[hnr_mean(A:,i:,u:)]',
 'Convergence[localJitter_mean(A:,i:,u:)]',
 'Convergence[localabsoluteJitter_mean(A:,i:,u:)]',
 # 'Convergence[rapJitter_mean(A:,i:,u:)]',
 # 'Convergence[ddpJitter_mean(A:,i:,u:)]',
 'Convergence[localShimmer_mean(A:,i:,u:)]',
 'Convergence[localdbShimmer_mean(A:,i:,u:)]',
 # 'Convergence[intensity_mean_var(A:,i:,u:)]',
 # 'Convergence[meanF0_var(A:,i:,u:)]',
 # 'Convergence[stdevF0_var(A:,i:,u:)]',
 # 'Convergence[hnr_var(A:,i:,u:)]',
 # 'Convergence[localJitter_var(A:,i:,u:)]',
 # 'Convergence[localabsoluteJitter_var(A:,i:,u:)]',
 # 'Convergence[rapJitter_var(A:,i:,u:)]',
 # 'Convergence[ddpJitter_var(A:,i:,u:)]',
 # 'Convergence[localShimmer_var(A:,i:,u:)]',
 # 'Convergence[localdbShimmer_var(A:,i:,u:)]',
 'Convergence[intensity_mean_max(A:,i:,u:)]',
 'Convergence[meanF0_max(A:,i:,u:)]',
 'Convergence[stdevF0_max(A:,i:,u:)]',
 'Convergence[hnr_max(A:,i:,u:)]',
 'Convergence[localJitter_max(A:,i:,u:)]',
 'Convergence[localabsoluteJitter_max(A:,i:,u:)]',
 # 'Convergence[rapJitter_max(A:,i:,u:)]',
 # 'Convergence[ddpJitter_max(A:,i:,u:)]',
 'Convergence[localShimmer_max(A:,i:,u:)]',
 'Convergence[localdbShimmer_max(A:,i:,u:)]'
 ]
Phonation_Syncrony_cols=[
 'Syncrony[intensity_mean_mean(A:,i:,u:)]',
 'Syncrony[meanF0_mean(A:,i:,u:)]',
 'Syncrony[stdevF0_mean(A:,i:,u:)]',
 'Syncrony[hnr_mean(A:,i:,u:)]',
 'Syncrony[localJitter_mean(A:,i:,u:)]',
 'Syncrony[localabsoluteJitter_mean(A:,i:,u:)]',
 # 'Syncrony[rapJitter_mean(A:,i:,u:)]',
 # 'Syncrony[ddpJitter_mean(A:,i:,u:)]',
 'Syncrony[localShimmer_mean(A:,i:,u:)]',
 'Syncrony[localdbShimmer_mean(A:,i:,u:)]',
 # 'Syncrony[intensity_mean_var(A:,i:,u:)]',
 # 'Syncrony[meanF0_var(A:,i:,u:)]',
 # 'Syncrony[stdevF0_var(A:,i:,u:)]',
 # 'Syncrony[hnr_var(A:,i:,u:)]',
 # 'Syncrony[localJitter_var(A:,i:,u:)]',
 # 'Syncrony[localabsoluteJitter_var(A:,i:,u:)]',
 # 'Syncrony[rapJitter_var(A:,i:,u:)]',
 # 'Syncrony[ddpJitter_var(A:,i:,u:)]',
 # 'Syncrony[localShimmer_var(A:,i:,u:)]',
 # 'Syncrony[localdbShimmer_var(A:,i:,u:)]',
 'Syncrony[intensity_mean_max(A:,i:,u:)]',
 'Syncrony[meanF0_max(A:,i:,u:)]',
 'Syncrony[stdevF0_max(A:,i:,u:)]',
 'Syncrony[hnr_max(A:,i:,u:)]',
 'Syncrony[localJitter_max(A:,i:,u:)]',
 'Syncrony[localabsoluteJitter_max(A:,i:,u:)]',
 # 'Syncrony[rapJitter_max(A:,i:,u:)]',
 # 'Syncrony[ddpJitter_max(A:,i:,u:)]',
 'Syncrony[localShimmer_max(A:,i:,u:)]',
 'Syncrony[localdbShimmer_max(A:,i:,u:)]'
    
    ]

# =============================================================================
'''

    LOC Interaction feature

'''
# =============================================================================

LOCDEP_Proximity_cols=[
 'Proximity[VSA2]',
 'Proximity[FCR2]',
 'Proximity[between_covariance_norm(A:,i:,u:)]',
 'Proximity[between_variance_norm(A:,i:,u:)]',
 'Proximity[within_covariance_norm(A:,i:,u:)]',
 'Proximity[within_variance_norm(A:,i:,u:)]',
 'Proximity[total_covariance_norm(A:,i:,u:)]',
 'Proximity[total_variance_norm(A:,i:,u:)]',
 'Proximity[sam_wilks_lin_norm(A:,i:,u:)]',
 'Proximity[pillai_lin_norm(A:,i:,u:)]',
 'Proximity[hotelling_lin_norm(A:,i:,u:)]',
 'Proximity[roys_root_lin_norm(A:,i:,u:)]',
 'Proximity[Between_Within_Det_ratio_norm(A:,i:,u:)]',
 'Proximity[Between_Within_Tr_ratio_norm(A:,i:,u:)]',
  'Proximity[pear_12]',
  'Proximity[spear_12]',
  'Proximity[kendall_12]',
  'Proximity[dcorr_12]'
 ]
LOCDEP_Trend_D_cols=[
 'Trend[VSA2]_d',
 'Trend[FCR2]_d',
 'Trend[between_covariance_norm(A:,i:,u:)]_d',
 'Trend[between_variance_norm(A:,i:,u:)]_d',
 'Trend[within_covariance_norm(A:,i:,u:)]_d',
 'Trend[within_variance_norm(A:,i:,u:)]_d',
 'Trend[total_covariance_norm(A:,i:,u:)]_d',
 'Trend[total_variance_norm(A:,i:,u:)]_d',
 'Trend[sam_wilks_lin_norm(A:,i:,u:)]_d',
 'Trend[pillai_lin_norm(A:,i:,u:)]_d',
 'Trend[hotelling_lin_norm(A:,i:,u:)]_d',
 'Trend[roys_root_lin_norm(A:,i:,u:)]_d',
 'Trend[Between_Within_Det_ratio_norm(A:,i:,u:)]_d',
 'Trend[Between_Within_Tr_ratio_norm(A:,i:,u:)]_d',
  'Trend[pear_12]_d',
  'Trend[spear_12]_d',
  'Trend[kendall_12]_d',
  'Trend[dcorr_12]_d'
    ]
LOCDEP_Trend_K_cols=[
 'Trend[VSA2]_k',
 'Trend[FCR2]_k',
 'Trend[between_covariance_norm(A:,i:,u:)]_k',
 'Trend[between_variance_norm(A:,i:,u:)]_k',
 'Trend[within_covariance_norm(A:,i:,u:)]_k',
 'Trend[within_variance_norm(A:,i:,u:)]_k',
  'Trend[total_covariance_norm(A:,i:,u:)]_k',
  'Trend[total_variance_norm(A:,i:,u:)]_k',
 'Trend[sam_wilks_lin_norm(A:,i:,u:)]_k',
 'Trend[pillai_lin_norm(A:,i:,u:)]_k',
  'Trend[hotelling_lin_norm(A:,i:,u:)]_k',
  'Trend[roys_root_lin_norm(A:,i:,u:)]_k',
 'Trend[Between_Within_Det_ratio_norm(A:,i:,u:)]_k',
 'Trend[Between_Within_Tr_ratio_norm(A:,i:,u:)]_k',
  'Trend[pear_12]_k',
  'Trend[spear_12]_k',
  'Trend[kendall_12]_k',
  'Trend[dcorr_12]_k'
    ]
LOCDEP_Convergence_cols=[
 'Convergence[VSA2]',
 'Convergence[FCR2]',
 'Convergence[between_covariance_norm(A:,i:,u:)]',
 'Convergence[between_variance_norm(A:,i:,u:)]',
 'Convergence[within_covariance_norm(A:,i:,u:)]',
 'Convergence[within_variance_norm(A:,i:,u:)]',
 'Convergence[total_covariance_norm(A:,i:,u:)]',
 'Convergence[total_variance_norm(A:,i:,u:)]',
 'Convergence[sam_wilks_lin_norm(A:,i:,u:)]',
 'Convergence[pillai_lin_norm(A:,i:,u:)]',
 'Convergence[hotelling_lin_norm(A:,i:,u:)]',
 'Convergence[roys_root_lin_norm(A:,i:,u:)]',
 'Convergence[Between_Within_Det_ratio_norm(A:,i:,u:)]',
 'Convergence[Between_Within_Tr_ratio_norm(A:,i:,u:)]',
  'Convergence[pear_12]',
  'Convergence[spear_12]',
  'Convergence[kendall_12]',
  'Convergence[dcorr_12]'
 ]
LOCDEP_Syncrony_cols=[
 'Syncrony[VSA2]',
 'Syncrony[FCR2]',
 'Syncrony[between_covariance_norm(A:,i:,u:)]',
 'Syncrony[between_variance_norm(A:,i:,u:)]',
 'Syncrony[within_covariance_norm(A:,i:,u:)]',
 'Syncrony[within_variance_norm(A:,i:,u:)]',
 'Syncrony[total_covariance_norm(A:,i:,u:)]',
 'Syncrony[total_variance_norm(A:,i:,u:)]',
 'Syncrony[sam_wilks_lin_norm(A:,i:,u:)]',
 'Syncrony[pillai_lin_norm(A:,i:,u:)]',
 'Syncrony[hotelling_lin_norm(A:,i:,u:)]',
 'Syncrony[roys_root_lin_norm(A:,i:,u:)]',
 'Syncrony[Between_Within_Det_ratio_norm(A:,i:,u:)]',
 'Syncrony[Between_Within_Tr_ratio_norm(A:,i:,u:)]',
  'Syncrony[pear_12]',
  'Syncrony[spear_12]',
  'Syncrony[kendall_12]',
  'Syncrony[dcorr_12]'
    ]

# =============================================================================
'''

    Utterance level columns

'''
# =============================================================================
Utt_prosodyF0=[
    'F0avg',
 'F0std',
 'F0max',
 'F0min',
 'F0skew',
 'F0kurt',
 'F0tiltavg',
 'F0mseavg',
 'F0tiltstd',
 'F0msestd',
 'F0tiltmax',
 'F0msemax',
 'F0tiltmin',
 'F0msemin',
 'F0tiltskw',
 'F0mseskw',
 'F0tiltku',
 'F0mseku',
    ]


Utt_VoiceQuality = ['avg Jitter',
 'avg Shimmer',
 # 'avg apq',
 # 'avg ppq',
 # 'avg logE',
 'std Jitter',
 'std Shimmer',
 # 'std apq',
 # 'std ppq',
 # 'std logE',
 'skewness Jitter',
 'skewness Shimmer',
 # 'skewness apq',
 # 'skewness ppq',
 # 'skewness logE',
 'kurtosis Jitter',
 'kurtosis Shimmer',
 # 'kurtosis apq',
 # 'kurtosis ppq',
 # 'kurtosis logE'
 ]

Utt_energy = [
    'avgEvoiced', 'stdEvoiced', 'skwEvoiced', 'kurtosisEvoiced',
        'avgtiltEvoiced', 'stdtiltEvoiced', 'skwtiltEvoiced',
        'kurtosistiltEvoiced', 'avgmseEvoiced', 'stdmseEvoiced',
        'skwmseEvoiced', 'kurtosismseEvoiced', 
        ]

Utt_prosodyF0_VoiceQuality=Utt_prosodyF0+Utt_VoiceQuality
Utt_VoiceQuality_energy=Utt_VoiceQuality+Utt_energy
Utt_prosodyF0_energy=Utt_prosodyF0+Utt_energy
Utt_prosodyF0_VoiceQuality_energy=Utt_prosodyF0+Utt_VoiceQuality+Utt_energy
# =============================================================================
'''

    LOC and DEP columns

'''
# =============================================================================

LOC_columns=[ 'between_covariance_norm(A:,i:,u:)',
        'between_variance_norm(A:,i:,u:)',
        'total_covariance_norm(A:,i:,u:)',
        'total_variance_norm(A:,i:,u:)', 
        'sam_wilks_lin_norm(A:,i:,u:)',
        'pillai_lin_norm(A:,i:,u:)', 
        'hotelling_lin_norm(A:,i:,u:)',
        'roys_root_lin_norm(A:,i:,u:)',
        'Between_Within_Det_ratio_norm(A:,i:,u:)',
        'Between_Within_Tr_ratio_norm(A:,i:,u:)',
       ]

DEP_columns=[
    'pear_12',
    'spear_12',
    'kendall_12',
    'dcorr_12'
    ]

LOCDEP_columns=LOC_columns+DEP_columns
# =============================================================================
'''
    static_feautre_phonation
'''
# =============================================================================

Phonation_columns=[
'intensity_mean_mean(A:,i:,u:)', 'meanF0_mean(A:,i:,u:)',
       'stdevF0_mean(A:,i:,u:)', 'hnr_mean(A:,i:,u:)',
       'localJitter_mean(A:,i:,u:)', 'localabsoluteJitter_mean(A:,i:,u:)',
       # 'rapJitter_mean(A:,i:,u:)', 'ddpJitter_mean(A:,i:,u:)',
       'localShimmer_mean(A:,i:,u:)', 'localdbShimmer_mean(A:,i:,u:)',
]
# =============================================================================
'''

    Columns combination

'''
# def Get_columnproduct(Comb1,Block_columns_dict):
#     Comb1_result = list(itertools.product(*Comb1))
#     comb1_append=[]
#     for c in Comb1_result:
        
#         comb1_append.append(Block_columns_dict[c[0]]+ Block_columns_dict[c[1]])
#     return comb1_append

def Get_columnproduct(Comb1,Block_columns_dict):
    Comb1_result = list(itertools.product(*Comb1))
    comb1_dict={}
    for c in Comb1_result:
        comb1_dict['+'.join(c)]=Block_columns_dict[c[0]]+ Block_columns_dict[c[1]]
        # comb1_dict.append(Block_columns_dict[c[0]]+ Block_columns_dict[c[1]])
    return comb1_dict
# =============================================================================
import itertools

static_feautre_LOC=['LOC_columns','DEP_columns','LOCDEP_columns']
static_feautre_phonation=['Phonation_columns']
dynamic_feature_LOC=['LOCDEP_Proximity_cols','LOCDEP_Trend_D_cols','LOCDEP_Trend_K_cols','LOCDEP_Convergence_cols','LOCDEP_Syncrony_cols']
dynamic_feature_phonation=['Phonation_Trend_D_cols','Phonation_Trend_K_cols','Phonation_Proximity_cols','Phonation_Convergence_cols','Phonation_Syncrony_cols']
Utt_feature=['Utt_prosodyF0','Utt_VoiceQuality','Utt_energy',\
             'Utt_prosodyF0_VoiceQuality','Utt_VoiceQuality_energy','Utt_prosodyF0_energy','Utt_prosodyF0_VoiceQuality_energy']

Block_columns_dict={}
for Lsts in [static_feautre_LOC,static_feautre_phonation,dynamic_feature_LOC,dynamic_feature_phonation,Utt_feature]:
    for L in Lsts:
        Block_columns_dict[L]=vars()[L]


Columns_comb={}

Top_category_lst_comb1=list(combinations(['static_feautre_LOC', 'static_feautre_phonation', 'dynamic_feature_LOC', 'dynamic_feature_phonation'],1))
Top_category_lst_comb2=list(combinations(['static_feautre_LOC', 'static_feautre_phonation', 'dynamic_feature_LOC', 'dynamic_feature_phonation'],2))
Top_combs_dict={}


for pair in Top_category_lst_comb2:
    pair_key='+'.join(pair)
    Top_combs_dict[pair_key]=[vars()[pair[0]],vars()[pair[1]]]


for keys, values in Top_combs_dict.items():
    Columns_comb[keys]=Get_columnproduct(values,Block_columns_dict)


for c in Top_category_lst_comb1: #e.g. static_feautre_LOC, 
    e1=c[0]
    Small_category_dict={}
    for category_str in vars()[e1]: # e.g. LOC_columns
        Small_category_dict[category_str]=vars()[category_str]

    Columns_comb[e1]= Small_category_dict


''' Columns_comb all combine with Utt feature '''
Columns_comb2=Dict()
for key_utt in Utt_feature:
    for key_layer1 in Columns_comb.keys():
        for key_layer2 in Columns_comb[key_layer1].keys():
            Columns_comb2['Utt_feature+'+key_layer1][key_utt+'+'+key_layer2]=Columns_comb[key_layer1][key_layer2]+vars()[key_utt]

for key_utt in Utt_feature:
    Columns_comb2['Utt_feature'][key_utt]=vars()[key_utt]
# Columns_comb['static_feautre_LOC+static_feautre_phonation']=Get_columnproduct(Comb1,Block_columns_dict)
# Columns_comb['static_feautre_LOC+static_feautre_phonation']+=Get_columnproduct(Comb2,Block_columns_dict)

# Comb1 = [
#     static_feautre_LOC,
#     dynamic_feature_LOC,
# ]
# Comb2 = [
#     static_feautre_LOCDEP,
#     dynamic_feature_LOC,
# ]


# Columns_comb['static_feautre_LOC+dynamic_feature_LOC']=Get_columnproduct(Comb1,Block_columns_dict)
# Columns_comb['static_feautre_LOC+dynamic_feature_LOC']+=Get_columnproduct(Comb2,Block_columns_dict)

