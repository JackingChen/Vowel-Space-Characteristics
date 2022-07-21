#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:44:54 2022

@author: jackchen
"""


import os, sys


sys.path.append(os.path.dirname(__file__))
# import .FeatureSelect as FeatSel


# Used when I need to debug
# from FeatureSelect import *


from itertools import combinations
from addict import Dict
import copy
import numpy as np


def Swap2PaperName(feature_rawname,PprNmeMp):
    if feature_rawname in PprNmeMp.Paper_name_map.keys():
        featurename_paper=PprNmeMp.Paper_name_map[feature_rawname]
        feature_keys=featurename_paper
    else: 
        feature_keys=feature_rawname
    return feature_keys


Syncrony_functions=['Trend[{}]_d','Trend[{}]_k','Syncrony[{}]','Proximity[{}]','Convergence[{}]']
Syncrony_functions2PprNme_map={
    'Trend[{}]_d':"$GC[%s]_{inv}$",
    'Trend[{}]_k':"$GC[%s]_{part}$",
    'Syncrony[{}]':'Syncrony[%s]',
    'Proximity[{}]':'Proximity[%s]',
    'Convergence[{}]':'Convergence[%s]'
    }


Paper_name_map={}    
Paper_name_map['between_covariance_norm(A:,i:,u:)']='$BCC$'
Paper_name_map['between_variance_norm(A:,i:,u:)']='$BCV$'
Paper_name_map['within_covariance_norm(A:,i:,u:)']='$WCC$'
Paper_name_map['within_variance_norm(A:,i:,u:)']='$WCV$'
Paper_name_map['total_covariance_norm(A:,i:,u:)']='$TC$'
Paper_name_map['total_variance_norm(A:,i:,u:)']='$TV$'
Paper_name_map['sam_wilks_lin_norm(A:,i:,u:)']='$Wilks$'
Paper_name_map['pillai_lin_norm(A:,i:,u:)']='$Pillai$'
Paper_name_map['hotelling_lin_norm(A:,i:,u:)']='$Hotel$'
Paper_name_map['roys_root_lin_norm(A:,i:,u:)']='$Roys$'
Paper_name_map['Between_Within_Det_ratio_norm(A:,i:,u:)']='$Det(W^{-1}B)$'
Paper_name_map['Between_Within_Tr_ratio_norm(A:,i:,u:)']='$Tr(W^{-1}B)$'
Paper_name_map['dcorr_12']='$DCorrF1F2$'
Paper_name_map['pear_12']='$PearF1F2$'
Paper_name_map['spear_12']='$SpearF1F2$'
Paper_name_map['kendall_12']='$KendallF1F2$'
Paper_name_map['FCR2']='FCR'
Paper_name_map['VSA2']='VSA'

Paper_name_map['intensity_mean_mean(A:,i:,u:)']='$Mean(\overline{int})$'
Paper_name_map['meanF0_mean(A:,i:,u:)']='$Mean(\\overline{F0})$'
Paper_name_map['meanF0_mean(A:,i:,u:)']='$Mean(\\rho(F0))$'
Paper_name_map['hnr_mean(A:,i:,u:)']='$Mean(HNR)$'
Paper_name_map['localShimmer_mean(A:,i:,u:)']='$Mean(Shimmer)$'
Paper_name_map['localdbShimmer_mean(A:,i:,u:)']='$Mean(Shimmer)$'
Paper_name_map['localJitter_mean(A:,i:,u:)']='$Mean(Jitter)$'
# Paper_name_map['']='$Std(\overline{F0})$'
Paper_name_map['stdevF0_mean(A:,i:,u:)']='$Std(\\rho(F0))$'
Paper_name_map['intensity_mean_var(A:,i:,u:)']='$Std(\overline{int})$'
Paper_name_map['meanF0_var(A:,i:,u:)']='$Std(\\overline{F0})$'
Paper_name_map['localabsoluteJitter_mean(A:,i:,u:)']='$Mean(Jitter)$'
Paper_name_map['intensity_mean_max(A:,i:,u:)']='$Max(\overline{int})$'
Paper_name_map['meanF0_max(A:,i:,u:)']='$Max(\\rho(F0))$'
Paper_name_map['hnr_max(A:,i:,u:)']='$Max(HNR)$'
Paper_name_map['localShimmer_max(A:,i:,u:)']='$Max(Shimmer)$'
Paper_name_map['localJitter_mean_max(A:,i:,u:)']='$Max(Jitter)$'
# Paper_name_map['']=''


Paper_name_map_ToBeAdded={}
for Syncwrapper in Syncrony_functions:
    Syncwrapper_PprNme=Syncrony_functions2PprNme_map[Syncwrapper]
    for origin_varName in Paper_name_map.keys():
        # Paper_name_map_ToBeAdded[Syncwrapper.format(origin_varName)]=Syncwrapper_PprNme.format(Paper_name_map[origin_varName])
        Paper_name_map_ToBeAdded[Syncwrapper.format(origin_varName)]=Syncwrapper_PprNme % (Paper_name_map[origin_varName],)
Paper_name_map.update(Paper_name_map_ToBeAdded)


# Paper_name_map['Divergence[pillai_lin_norm(A:,i:,u:)]']='$Div(Pillai)$'
# Paper_name_map['Trend[pillai_lin_norm(A:,i:,u:)]_d']='$Mod(Pillai)_{inv}$'
# Paper_name_map['Trend[pillai_lin_norm(A:,i:,u:)]_k']='$Mod(Pillai)_{part}$'
# Paper_name_map['Trend[roys_root_lin_norm(A:,i:,u:)]_d']='$Mod(Roys)_{inv}$'
# Paper_name_map['Trend[roys_root_lin_norm(A:,i:,u:)]_k']='$Mod(Roys)_{part}$'
# Paper_name_map['Trend[hotelling_lin_norm(A:,i:,u:)]_d']='$Mod(Hotel)_{inv}$'
# Paper_name_map['Trend[hotelling_lin_norm(A:,i:,u:)]_k']='$Mod(Hotel)_{part}$'
# Paper_name_map['Divergence[within_covariance_norm(A:,i:,u:)]']='$Div(WCC)$'
# Paper_name_map['Trend[within_covariance_norm(A:,i:,u:)]_d']='$Mod(WCC)_{inv}$'
# Paper_name_map['Trend[within_covariance_norm(A:,i:,u:)]_k']='$Mod(WCC)_{part}$'
# Paper_name_map['Divergence[within_variance_norm(A:,i:,u:)]']='$Div(WCV)$'
# Paper_name_map['Trend[within_variance_norm(A:,i:,u:)]_d']='$Mod(WCV)_{inv}$'
# Paper_name_map['Trend[within_variance_norm(A:,i:,u:)]_k']='$Mod(WCV)_{part}$'
# Paper_name_map['Divergence[sam_wilks_lin_norm(A:,i:,u:)]']='$Div(Wilks)$'
# Paper_name_map['Trend[sam_wilks_lin_norm(A:,i:,u:)]_d']='$Mod(Wilks)_{inv}$'
# Paper_name_map['Trend[sam_wilks_lin_norm(A:,i:,u:)]_k']='$Mod(Wilks)_{part}$'
# Paper_name_map['Divergence[between_covariance_norm(A:,i:,u:)]']='$Div(BCC)$'
# Paper_name_map['Trend[between_covariance_norm(A:,i:,u:)]_d']='$Mod(BCC)_{inv}$'
# Paper_name_map['Trend[between_covariance_norm(A:,i:,u:)]_k']='$Mod(BCC)_{part}$'
# Paper_name_map['Divergence[between_variance_norm(A:,i:,u:)]']='$Div(BCV)$'
# Paper_name_map['Trend[between_variance_norm(A:,i:,u:)]_d']='$Mod(BCV)_{inv}$'
# Paper_name_map['Trend[between_variance_norm(A:,i:,u:)]_k']='$Mod(BCV)_{part}$'
# Paper_name_map['Trend[Between_Within_Det_ratio_norm(A:,i:,u:)]_d']='$Mod(Det(W^{-1}B))_{inv}$'
# Paper_name_map['Trend[Between_Within_Det_ratio_norm(A:,i:,u:)]_k']='$Mod(Det(W^{-1}B))_{part}$'
# Paper_name_map['Trend[Between_Within_Tr_ratio_norm(A:,i:,u:)]_d']='$Mod(Tr(W^{-1}B))_{inv}$'
# Paper_name_map['Trend[Between_Within_Tr_ratio_norm(A:,i:,u:)]_k']='$Mod(Tr(W^{-1}B))_{part}$'
# Paper_name_map['Trend[total_variance_norm(A:,i:,u:)]_d']='$Mod(TV)_{inv}$'
# Paper_name_map['Trend[total_variance_norm(A:,i:,u:)]_k']='$Mod(TV)_{part}$'
# Paper_name_map['Trend[total_covariance_norm(A:,i:,u:)]_d']='$Mod(TC)_{inv}$'
# Paper_name_map['Trend[total_covariance_norm(A:,i:,u:)]_k']='$Mod(TC)_{part}$'




Paper_name_map['LOC_columns']='Inter-Vowel Dispersion'
Paper_name_map['LOC_columns_Intra']='Intra-Vowel Dispersion'
Paper_name_map['DEP_columns']='formant dependency'
Paper_name_map['Phonation_Trend_D_cols']='Mod[P]_{d}'
Paper_name_map['Phonation_Trend_K_cols']='Mod[P]_{k}'
Paper_name_map['Phonation_Proximity_cols']='Proximity[P]'
Paper_name_map['Phonation_Convergence_cols']='Convergence[P]'
Paper_name_map['Phonation_Syncrony_cols']='Syncrony[P]'
Paper_name_map['LOCDEP_Trend_D_cols']='Mod[VSC]_{d}'
Paper_name_map['LOCDEP_Trend_K_cols']='Mod[VSC]_{k}'
Paper_name_map['LOCDEP_Proximity_cols']='Proximity[VSC]'
Paper_name_map['LOCDEP_Convergence_cols']='Convergence[VSC]'
Paper_name_map['LOCDEP_Syncrony_cols']='Syncrony[VSC]'


# Label
Paper_name_map['ADOS_C']='$ADOS_{comm}$'


Inverse_Paper_name_map={v:k for k,v in Paper_name_map.items()}





