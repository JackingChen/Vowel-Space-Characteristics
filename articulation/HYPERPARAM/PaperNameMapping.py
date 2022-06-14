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

Paper_name_map={}    
Paper_name_map['Divergence[VSA2]']='$Div(VSA)$'
Paper_name_map['Trend[VSA2]_d']='$Mod(VSA)_{inv}$'
Paper_name_map['Trend[VSA2]_k']='$Mod(VSA)_{part}$'
Paper_name_map['Divergence[FCR2]']='$Div(FCR)$'
Paper_name_map['Trend[FCR2]_d']='$Mod(FCR)_{inv}$'
Paper_name_map['Trend[FCR2]_k']='$Mod(FCR)_{part}$'
Paper_name_map['Divergence[pear_12]']='$Div(PearF1F2)$'
Paper_name_map['Trend[pear_12]_d']='$Mod(PearF1F2)_{inv}$'
Paper_name_map['Trend[pear_12]_k']='$Mod(PearF1F2)_{part}$'
Paper_name_map['Divergence[spear_12]']='$Div(SpearF1F2)$'
Paper_name_map['Trend[spear_12]_d']='$Mod(SpearF1F2)_{inv}$'
Paper_name_map['Trend[spear_12]_k']='$Mod(SpearF1F2)_{part}$'
Paper_name_map['Divergence[kendall_12]']='$Div(KendallF1F2)$'
Paper_name_map['Trend[kendall_12]_d']='$Mod(KendallF1F2)_{inv}$'
Paper_name_map['Trend[kendall_12]_k']='$Mod(KendallF1F2)_{part}$'
Paper_name_map['Divergence[dcorr_12]']='$Div(DCorrF1F2)$'
Paper_name_map['Trend[dcorr_12]_d']='$Mod(DCorrF1F2)_{inv}$'
Paper_name_map['Trend[dcorr_12]_k']='$Mod(DCorrF1F2)_{part}$'

Paper_name_map['Divergence[pillai_lin_norm(A:,i:,u:)]']='$Div(Pillai)$'
Paper_name_map['Trend[pillai_lin_norm(A:,i:,u:)]_d']='$Mod(Pillai)_{inv}$'
Paper_name_map['Trend[pillai_lin_norm(A:,i:,u:)]_k']='$Mod(Pillai)_{part}$'
Paper_name_map['Trend[roys_root_lin_norm(A:,i:,u:)]_d']='$Mod(Roys)_{inv}$'
Paper_name_map['Trend[roys_root_lin_norm(A:,i:,u:)]_k']='$Mod(Roys)_{part}$'
Paper_name_map['Trend[hotelling_lin_norm(A:,i:,u:)]_d']='$Mod(Hotel)_{inv}$'
Paper_name_map['Trend[hotelling_lin_norm(A:,i:,u:)]_k']='$Mod(Hotel)_{part}$'
Paper_name_map['Divergence[within_covariance_norm(A:,i:,u:)]']='$Div(WCC)$'
Paper_name_map['Trend[within_covariance_norm(A:,i:,u:)]_d']='$Mod(WCC)_{inv}$'
Paper_name_map['Trend[within_covariance_norm(A:,i:,u:)]_k']='$Mod(WCC)_{part}$'
Paper_name_map['Divergence[within_variance_norm(A:,i:,u:)]']='$Div(WCV)$'
Paper_name_map['Trend[within_variance_norm(A:,i:,u:)]_d']='$Mod(WCV)_{inv}$'
Paper_name_map['Trend[within_variance_norm(A:,i:,u:)]_k']='$Mod(WCV)_{part}$'
Paper_name_map['Divergence[sam_wilks_lin_norm(A:,i:,u:)]']='$Div(Wilks)$'
Paper_name_map['Trend[sam_wilks_lin_norm(A:,i:,u:)]_d']='$Mod(Wilks)_{inv}$'
Paper_name_map['Trend[sam_wilks_lin_norm(A:,i:,u:)]_k']='$Mod(Wilks)_{part}$'
Paper_name_map['Divergence[between_covariance_norm(A:,i:,u:)]']='$Div(BCC)$'
Paper_name_map['Trend[between_covariance_norm(A:,i:,u:)]_d']='$Mod(BCC)_{inv}$'
Paper_name_map['Trend[between_covariance_norm(A:,i:,u:)]_k']='$Mod(BCC)_{part}$'
Paper_name_map['Divergence[between_variance_norm(A:,i:,u:)]']='$Div(BCV)$'
Paper_name_map['Trend[between_variance_norm(A:,i:,u:)]_d']='$Mod(BCV)_{inv}$'
Paper_name_map['Trend[between_variance_norm(A:,i:,u:)]_k']='$Mod(BCV)_{part}$'
Paper_name_map['Trend[Between_Within_Det_ratio_norm(A:,i:,u:)]_d']='$Mod(Det(W^{-1}B))_{inv}$'
Paper_name_map['Trend[Between_Within_Det_ratio_norm(A:,i:,u:)]_k']='$Mod(Det(W^{-1}B))_{part}$'
Paper_name_map['Trend[Between_Within_Tr_ratio_norm(A:,i:,u:)]_d']='$Mod(Tr(W^{-1}B))_{inv}$'
Paper_name_map['Trend[Between_Within_Tr_ratio_norm(A:,i:,u:)]_k']='$Mod(Tr(W^{-1}B))_{part}$'
Paper_name_map['Trend[total_variance_norm(A:,i:,u:)]_d']='$Mod(TV)_{inv}$'
Paper_name_map['Trend[total_variance_norm(A:,i:,u:)]_k']='$Mod(TV)_{part}$'
Paper_name_map['Trend[total_covariance_norm(A:,i:,u:)]_d']='$Mod(TC)_{inv}$'
Paper_name_map['Trend[total_covariance_norm(A:,i:,u:)]_k']='$Mod(TC)_{part}$'
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




Inverse_Paper_name_map={v:k for k,v in Paper_name_map.items()}





