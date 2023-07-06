#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:44:54 2022

@author: jackchen
"""
from itertools import combinations
from addict import Dict
import copy
from scipy.special import comb
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
 'Proximity[localShimmer_mean(A:,i:,u:)]',
 'Proximity[localdbShimmer_mean(A:,i:,u:)]',
 'Proximity[intensity_mean_var(A:,i:,u:)]',
 'Proximity[intensity_mean_max(A:,i:,u:)]',
 'Proximity[meanF0_max(A:,i:,u:)]',
 'Proximity[stdevF0_max(A:,i:,u:)]',
 'Proximity[hnr_max(A:,i:,u:)]',
 'Proximity[localJitter_max(A:,i:,u:)]',
 'Proximity[localabsoluteJitter_max(A:,i:,u:)]',
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
 'Trend[localShimmer_mean(A:,i:,u:)]_d',
 'Trend[localdbShimmer_mean(A:,i:,u:)]_d',
 'Trend[intensity_mean_max(A:,i:,u:)]_d',
 'Trend[meanF0_max(A:,i:,u:)]_d',
 'Trend[stdevF0_max(A:,i:,u:)]_d',
 'Trend[hnr_max(A:,i:,u:)]_d',
 'Trend[localJitter_max(A:,i:,u:)]_d',
 'Trend[localabsoluteJitter_max(A:,i:,u:)]_d',
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
 'Trend[localShimmer_mean(A:,i:,u:)]_k',
 'Trend[localdbShimmer_mean(A:,i:,u:)]_k',
 'Trend[intensity_mean_max(A:,i:,u:)]_k',
 'Trend[meanF0_max(A:,i:,u:)]_k',
 'Trend[stdevF0_max(A:,i:,u:)]_k',
 'Trend[hnr_max(A:,i:,u:)]_k',
 'Trend[localJitter_max(A:,i:,u:)]_k',
 'Trend[localabsoluteJitter_max(A:,i:,u:)]_k',
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
 'Convergence[localShimmer_mean(A:,i:,u:)]',
 'Convergence[localdbShimmer_mean(A:,i:,u:)]',
 'Convergence[intensity_mean_max(A:,i:,u:)]',
 'Convergence[meanF0_max(A:,i:,u:)]',
 'Convergence[stdevF0_max(A:,i:,u:)]',
 'Convergence[hnr_max(A:,i:,u:)]',
 'Convergence[localJitter_max(A:,i:,u:)]',
 'Convergence[localabsoluteJitter_max(A:,i:,u:)]',
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
 'Syncrony[localShimmer_mean(A:,i:,u:)]',
 'Syncrony[localdbShimmer_mean(A:,i:,u:)]',
 'Syncrony[intensity_mean_max(A:,i:,u:)]',
 'Syncrony[meanF0_max(A:,i:,u:)]',
 'Syncrony[stdevF0_max(A:,i:,u:)]',
 'Syncrony[hnr_max(A:,i:,u:)]',
 'Syncrony[localJitter_max(A:,i:,u:)]',
 'Syncrony[localabsoluteJitter_max(A:,i:,u:)]',
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
 'std Jitter',
 'std Shimmer',
 'skewness Jitter',
 'skewness Shimmer',
 'kurtosis Jitter',
 'kurtosis Shimmer',
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

LOC_columns=[ 
        'VSA2',
        'FCR2',
        'between_covariance_norm(A:,i:,u:)',
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


LOC_columns_Intra=[ 
    'within_covariance_norm(A:,i:,u:)',
    'within_variance_norm(A:,i:,u:)',
    ]

DEP_columns=[
    'pear_12',
    'spear_12',
    'kendall_12',
    'dcorr_12'
    ]

LOCDEP_columns=LOC_columns+DEP_columns


# =============================================================================
''' Here are categories where the definitions are in paper '''
Vowel_dispersion_inter=[
        'VSA2',
        'FCR2',
        'between_covariance_norm(A:,i:,u:)',
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
Vowel_dispersion_inter__vowel_centralization=[
        'FCR2',
        'sam_wilks_lin_norm(A:,i:,u:)',
        ]

Vowel_dispersion_inter__vowel_dispersion=[
        'VSA2',
        'between_covariance_norm(A:,i:,u:)',
        'between_variance_norm(A:,i:,u:)',
        'total_covariance_norm(A:,i:,u:)',
        'total_variance_norm(A:,i:,u:)',
        'pillai_lin_norm(A:,i:,u:)',
        'hotelling_lin_norm(A:,i:,u:)',
        'roys_root_lin_norm(A:,i:,u:)',
        'Between_Within_Det_ratio_norm(A:,i:,u:)',
        'Between_Within_Tr_ratio_norm(A:,i:,u:)',
        ]
Vowel_dispersion_intra=[
        'within_covariance_norm(A:,i:,u:)',
        'within_variance_norm(A:,i:,u:)',
                     ]
Vowel_dispersion=Vowel_dispersion_inter+Vowel_dispersion_intra    
formant_dependency=['pear_12',
        'spear_12',
        'kendall_12',
        'dcorr_12'
        ]
Syncrony_functions=['Proximity[{}]','Trend[{}]_d','Trend[{}]_k','Convergence[{}]','Syncrony[{}]']
LOCDEP=Vowel_dispersion+formant_dependency

    

# =============================================================================
PaperVariables=['Vowel_dispersion','Vowel_dispersion_inter','Vowel_dispersion_inter__vowel_centralization','Vowel_dispersion_inter__vowel_dispersion',\
 'Vowel_dispersion_intra','formant_dependency','LOCDEP','LOC_columns','DEP_columns','LOCDEP_Syncrony_cols','LOCDEP_Trend_D_cols']

CategoricalName2cols={}
for variables in PaperVariables:
    CategoricalName2cols[variables]=vars()[variables]

for Syncwrapper in Syncrony_functions:
    for variables in PaperVariables:
        FeatSet_bag=[]
        for singlefeatures in vars()[variables]:
            FeatSet_bag.append(Syncwrapper.format(singlefeatures))
        CategoricalName2cols[Syncwrapper.format(variables)]=FeatSet_bag


# 做一個相反的
cols2CategoricalName={}
for FeatCategory, FeatLsts in CategoricalName2cols.items():
    for feat in FeatLsts:
        cols2CategoricalName[feat]=FeatCategory

# =============================================================================
'''
    static_feautre_phonation
'''
# =============================================================================

Phonation_columns=[
'intensity_mean_mean(A:,i:,u:)', 'meanF0_mean(A:,i:,u:)',
       'stdevF0_mean(A:,i:,u:)', 'hnr_mean(A:,i:,u:)',
       'localJitter_mean(A:,i:,u:)', 'localabsoluteJitter_mean(A:,i:,u:)',
       'localShimmer_mean(A:,i:,u:)', 'localdbShimmer_mean(A:,i:,u:)',
]
# =============================================================================
'''

    Columns combination

'''
def Get_columnproduct(Comb1,Block_columns_dict):
    Comb1_result = list(itertools.product(*Comb1))
    comb1_dict={}
    for c in Comb1_result:
        comb1_dict['+'.join(c)]=Block_columns_dict[c[0]]+ Block_columns_dict[c[1]]
        # comb1_dict.append(Block_columns_dict[c[0]]+ Block_columns_dict[c[1]])
    return comb1_dict

def Get_Combs_Feat(Comb1,Block_columns_dict):
    Comb1_result=[]
    Combcount=0
    for n in range(1,len(Comb1)+1):
        Comb1_result+=list(itertools.combinations(Comb1,n))
        Combcount+=comb(len(Comb1),n)
    assert Combcount == len(Comb1_result)
    
    comb1_dict={}
    for c_layer1 in Comb1_result:
        comb1_dict['+'.join(c_layer1)]=[]
        for c_layer2 in c_layer1:
            comb1_dict['+'.join(c_layer1)]+=Block_columns_dict[c_layer2] 
    return comb1_dict



def Get_LOCCombs_withSelectedFeat(Comb1,Block_columns_dict,SelectedFeat='Phonation_Proximity_cols'):
    Comb1_result=[]
    for n in range(1,len(Comb1)+1):
        Comb1_result+=list(itertools.combinations(Comb1,n))
    
    def Fuse_combinationFeatWtihSelected(Comb1_result,Block_columns_dict,SelectedFeat):
        comb1_dict={}
        comb1_dict[SelectedFeat]=Block_columns_dict[SelectedFeat]
        for c_layer1 in Comb1_result:
            comb1_dict['+'.join(c_layer1)+'+'+SelectedFeat]=[]
            for c_layer2 in c_layer1:
                # print(c_layer2)
                # print(Block_columns_dict[c_layer2] )
                comb1_dict['+'.join(c_layer1)+'+'+SelectedFeat]+=Block_columns_dict[c_layer2] 
            comb1_dict['+'.join(c_layer1)+'+'+SelectedFeat]+=Block_columns_dict[SelectedFeat]
        return comb1_dict
        
    Comb_dict=Fuse_combinationFeatWtihSelected(Comb1_result,Block_columns_dict,SelectedFeat)
        
    return Comb_dict



# =============================================================================
import itertools

static_feautre_LOC=['LOC_columns','DEP_columns','LOCDEP_columns','LOC_columns_Intra']
static_feautre_phonation=['Phonation_columns']
dynamic_feature_LOC=['LOCDEP_Proximity_cols','LOCDEP_Trend_D_cols','LOCDEP_Trend_K_cols','LOCDEP_Convergence_cols','LOCDEP_Syncrony_cols']
dynamic_feature_phonation=['Phonation_Trend_D_cols','Phonation_Trend_K_cols','Phonation_Proximity_cols','Phonation_Convergence_cols','Phonation_Syncrony_cols']
Utt_feature=['Utt_prosodyF0','Utt_VoiceQuality','Utt_energy',\
             'Utt_prosodyF0_VoiceQuality','Utt_VoiceQuality_energy','Utt_prosodyF0_energy','Utt_prosodyF0_VoiceQuality_energy']

Block_columns_dict={}
for Lsts in [static_feautre_LOC,static_feautre_phonation,dynamic_feature_LOC,dynamic_feature_phonation,Utt_feature]:
    for L in Lsts:
        Block_columns_dict[L]=vars()[L]
    ''' 
        Manually extended combinations
        注意：不要放single feature的，會造成再跟其他feature set合的時候選用不對的feature
    
    '''
    Block_columns_dict['Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols']=Phonation_Trend_D_cols+Phonation_Proximity_cols
    Block_columns_dict['Phonation_Trend_K_cols+Phonation_Syncrony_cols']=Phonation_Trend_D_cols+Phonation_Proximity_cols
    Block_columns_dict['Phonation_Trend_D_cols+Phonation_Proximity_cols']=Phonation_Trend_D_cols+Phonation_Proximity_cols
    # Block_columns_dict['Phonation_Proximity_cols']=Phonation_Trend_D_cols+Phonation_Proximity_cols





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

''' Baseline_comb = Only  dynamic_feature_LOC dynamic_feature_phonation static_feautre_LOC'''
Baseline_comb=Dict()
for col in ['static_feautre_LOC','dynamic_feature_phonation','dynamic_feature_LOC']:
    Baseline_comb[col]= copy.deepcopy(Columns_comb[col])

''' Columns_comb2 = Columns_comb all combine with Utt feature '''
Columns_comb2=Dict()
for key_utt in Utt_feature:
    for key_layer1 in Columns_comb.keys():
        for key_layer2 in Columns_comb[key_layer1].keys():
            Columns_comb2['Utt_feature+'+key_layer1][key_utt+'+'+key_layer2]=Columns_comb[key_layer1][key_layer2]+vars()[key_utt]

for key_utt in Utt_feature:
    Columns_comb2['Utt_feature'][key_utt]=vars()[key_utt]


''' Columns_comb3 = All possible feature combination + phonation_proximity_col'''
Comb1=['LOC_columns','DEP_columns','LOC_columns_Intra']+dynamic_feature_LOC
Columns_comb3=Dict()
Columns_comb3['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation']=\
    Get_LOCCombs_withSelectedFeat(Comb1,Block_columns_dict,SelectedFeat='Phonation_Proximity_cols')

Columns_comb4=Dict()
Columns_comb4['Utt_feature+static_feautre_LOC+dynamic_feature_LOC']=\
    Get_LOCCombs_withSelectedFeat(Comb1,Block_columns_dict,SelectedFeat='Utt_prosodyF0_VoiceQuality_energy')

''' Columns_comb5 = All possible feature combination + Phonation_Syncrony_cols'''
Comb1=['LOC_columns','LOC_columns_Intra','DEP_columns']+dynamic_feature_LOC
Columns_comb5=Dict()
Columns_comb5['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation']=\
    Get_LOCCombs_withSelectedFeat(Comb1,Block_columns_dict,SelectedFeat='Phonation_Syncrony_cols')

''' Columns_comb6 = All possible feature combination + Phonation_Trend_D_cols'''
Comb1=['LOC_columns','DEP_columns']+dynamic_feature_LOC
Columns_comb6=Dict()
Columns_comb6['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation']=\
    Get_LOCCombs_withSelectedFeat(Comb1,Block_columns_dict,SelectedFeat='Phonation_Trend_D_cols')
    
''' Columns_comb7 = All possible feature combination + (Phonation_Trend_D_cols+Phonation_Proximity_cols)'''
# Columns A+B的示範
Comb1=['LOC_columns','DEP_columns']+dynamic_feature_LOC
Columns_comb7=Dict()
Columns_comb7['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation']=\
    Get_LOCCombs_withSelectedFeat(Comb1,Block_columns_dict,SelectedFeat='Phonation_Trend_D_cols+Phonation_Proximity_cols')
    
    
''' Columns_comb8 = All possible feature combination + Phonation_Trend_K_cols'''
Comb1=['LOC_columns','LOC_columns_Intra','DEP_columns']+dynamic_feature_LOC
Columns_comb8=Dict()
Columns_comb8['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation']=\
    Get_LOCCombs_withSelectedFeat(Comb1,Block_columns_dict,SelectedFeat='Phonation_Trend_K_cols')


''' Columns_comb_dynPhonation = All possible feature combination within dynamic_feature_phonation'''
Comb2=dynamic_feature_phonation
Comb_dynPhonation=Dict()
Comb_dynPhonation['dynamic_feature_phonation']=Get_Combs_Feat(Comb2,Block_columns_dict)

''' Columns_TotalComb = All possible feature combination of static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation'''
Comb4=['LOC_columns','DEP_columns','LOC_columns_Intra'] + dynamic_feature_LOC 
Comb_staticLOCDEP_dynamicLOCDEP=Dict()
Comb_staticLOCDEP_dynamicLOCDEP['static_feautre_LOC+dynamic_feature_LOC']=Get_Combs_Feat(Comb4,Block_columns_dict)
    
''' Columns_TotalComb = All possible feature combination of static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation'''
Comb3=['LOC_columns','DEP_columns'] + dynamic_feature_LOC + dynamic_feature_phonation
Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation=Dict()
Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation']=Get_Combs_Feat(Comb3,Block_columns_dict)


''' Columns_TotalComb = All possible feature combination of static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation + Utt Features'''
Comb3=['LOC_columns','DEP_columns'] + dynamic_feature_LOC + dynamic_feature_phonation
Comb_Utt_feature_staticLOCDEP_dynamicLOCDEP_dynamicphonation=Dict()
Comb_Utt_feature_staticLOCDEP_dynamicLOCDEP_dynamicphonation['Utt_features+static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation']=\
    Get_LOCCombs_withSelectedFeat(Comb3,Block_columns_dict,SelectedFeat='Utt_prosodyF0_VoiceQuality_energy')


''' Columns_comb9 = All possible feature combination of static_feautre_LOC+dynamic_feature_LOC+ featureCombs below:
'Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols'
'Phonation_Trend_K_cols+Phonation_Syncrony_cols'
'Phonation_Trend_D_cols+Phonation_Proximity_cols'
'Phonation_Proximity_cols'
'''
# Comb9_1=Get_LOCCombs_withSelectedFeat(Comb4,Block_columns_dict,SelectedFeat='Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols')
# Comb9_2=Get_LOCCombs_withSelectedFeat(Comb4,Block_columns_dict,SelectedFeat='Phonation_Trend_K_cols+Phonation_Syncrony_cols')
# Comb9_3=Get_LOCCombs_withSelectedFeat(Comb4,Block_columns_dict,SelectedFeat='Phonation_Trend_D_cols+Phonation_Proximity_cols')
# Comb9_4=Get_LOCCombs_withSelectedFeat(Comb4,Block_columns_dict,SelectedFeat='Phonation_Proximity_cols')

# Columns_comb9=Dict()
# Columns_comb9['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation']={}
# Columns_comb9['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation'].update(Comb9_1)
# Columns_comb9['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation'].update(Comb9_2)
# Columns_comb9['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation'].update(Comb9_3)
# Columns_comb9['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation'].update(Comb9_4)


# =============================================================================
# 想製造VSC所有comb跟phonation的product的組合，但是有bug待查
# =============================================================================
# Block_columns_dict_phonation=Comb_dynPhonation['dynamic_feature_phonation']
# Comb_phonation=list(Block_columns_dict_phonation.keys())
# Block_columns_dict_VSC=Comb_staticLOCDEP_dynamicLOCDEP['static_feautre_LOC+dynamic_feature_LOC']
# Comb_VSC=list(Block_columns_dict_VSC.keys())
# from itertools import product
# tup_lst= list(product(Comb_phonation, Comb_VSC))
# Comb_product_phonationVSC=['+'.join(e) for e in tup_lst]

# Block_columns_dict_phonation.update(Block_columns_dict_VSC)
# assert len(Block_columns_dict_phonation) == len(Comb_product_phonationVSC)
# aaa=Get_Combs_Feat(Comb_product_phonationVSC,Block_columns_dict)
