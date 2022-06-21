#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:56:45 2021

@author: jackchen


    This script is only for TBMEA1 


    2022/04/19: should add feature combination code with a form of class
"""

import os, sys
import pandas as pd
import numpy as np

import glob
import pickle

from scipy.stats import spearmanr,pearsonr 
from sklearn.model_selection import LeaveOneOut

from addict import Dict
# import functions
import argparse
from scipy.stats import zscore

from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.linear_model import ElasticNet
import sklearn.svm
import torch
import re
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import recall_score, make_scorer

from articulation.HYPERPARAM import phonewoprosody, Label
import articulation.HYPERPARAM.FeatureSelect as FeatSel
import articulation.HYPERPARAM.PaperNameMapping as PprNmeMp

import articulation.articulation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from tqdm import tqdm
def Assert_labelfeature(feat_name,lab_name):
    # =============================================================================
    #     To check if the label match with feature
    # =============================================================================
    for i,n in enumerate(feat_name):
        assert feat_name[i] == lab_name[i]

def FilterFile_withinManualName(files,Manual_choosen_feature):
    files_manualChoosen=[f  for f in files if os.path.basename(f).split(".")[0]  in Manual_choosen_feature]
    return files_manualChoosen

def Merge_dfs(df_1, df_2):
    return pd.merge(df_1,df_2,left_index=True, right_index=True)

def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser()
    parser.add_argument('--Feature_mode', default='Customized_feature',
                        help='what kind of data you want to get')
    parser.add_argument('--preprocess', default=True,
                        help='')
    parser.add_argument('--start_point', default=-1,
                        help='In case that the program stop at certain point, we can resume the progress by setting this variable')
    parser.add_argument('--experiment', default='gop_exp_ADOShappyDAAIKidallDeceiptformosaCSRC',
                        help='If the mode is set to Session_phone_phf, you may need to determine the experiment used to generate the gop feature')
    parser.add_argument('--pseudo', default=False,
                        help='what kind of data you want to get')
    parser.add_argument('--suffix', default="",
                        help='what kind of data you want to get')
    parser.add_argument('--FS_method_str', default=None,
                        help='Feature selection')
    parser.add_argument('--UseManualCtxFeat', default=True,
                        help='')
    parser.add_argument('--Plot', default=False,
                        help='')
    parser.add_argument('--selectModelScoring', default='neg_mean_squared_error',
                        help='')
    parser.add_argument('--Mergefeatures', default=False,
                        help='')
    parser.add_argument('--knn_weights', default='uniform',
                            help='path of the base directory')
    parser.add_argument('--knn_neighbors', default=2,  type=int,
                            help='path of the base directory')
    parser.add_argument('--Reorder_type', default='DKIndividual',
                            help='[DKIndividual, DKcriteria]')
    parser.add_argument('--FeatureComb_mode', default='baselineFeats',
                            help='[Add_UttLvl_feature, feat_comb3, feat_comb5, feat_comb6,feat_comb7, baselineFeats,Comb_dynPhonation,Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation, Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation_Add_UttLvl_feature]')
    args = parser.parse_args()
    return args


args = get_args()
start_point=args.start_point
experiment=args.experiment
knn_weights=args.knn_weights
knn_neighbors=args.knn_neighbors
Reorder_type=args.Reorder_type
# Add_UttLvl_feature=args.Add_UttLvl_feature
# =============================================================================
# Feature
# columns=[
    # 'FCR2+AUINum',
    # 'VSA2+AUINum',
    # 'FCR2*AUINum',
    # 'VSA2*AUINum',
    # 'FCR2',
    # 'VSA2',
    # 'between_covariance_norm(A:,i:,u:)', 
    # 'between_variance_norm(A:,i:,u:)',
    # 'within_covariance_norm(A:,i:,u:)', 
    # 'within_variance_norm(A:,i:,u:)',
    # 'total_covariance_norm(A:,i:,u:)', 
    # 'total_variance_norm(A:,i:,u:)',
    # 'sam_wilks_lin_norm(A:,i:,u:)', 
    # 'pillai_lin_norm(A:,i:,u:)',
    # 'hotelling_lin_norm(A:,i:,u:)', 
    # 'roys_root_lin_norm(A:,i:,u:)',
    # 'Between_Within_Det_ratio_norm(A:,i:,u:)',
    # 'Between_Within_Tr_ratio_norm(A:,i:,u:)'
    # 'pear_12',
    # 'spear_12',
    # 'kendall_12',
    # 'dcorr_12'
    # 'u_num+i_num+a_num',
    # ]
columns=[
'intensity_mean_mean(A:,i:,u:)', 'meanF0_mean(A:,i:,u:)',
       'stdevF0_mean(A:,i:,u:)', 'hnr_mean(A:,i:,u:)',
       'localJitter_mean(A:,i:,u:)', 'localabsoluteJitter_mean(A:,i:,u:)',
       'rapJitter_mean(A:,i:,u:)', 'ddpJitter_mean(A:,i:,u:)',
       'localShimmer_mean(A:,i:,u:)', 'localdbShimmer_mean(A:,i:,u:)',
       # 'intensity_mean_var(A:,i:,u:)', 'meanF0_var(A:,i:,u:)',
       # 'stdevF0_var(A:,i:,u:)', 'hnr_var(A:,i:,u:)',
       # 'localJitter_var(A:,i:,u:)', 'localabsoluteJitter_var(A:,i:,u:)',
       # 'rapJitter_var(A:,i:,u:)', 'ddpJitter_var(A:,i:,u:)',
       # 'localShimmer_var(A:,i:,u:)', 'localdbShimmer_var(A:,i:,u:)',
       # 'intensity_mean_max(A:,i:,u:)', 'meanF0_max(A:,i:,u:)',
       # 'stdevF0_max(A:,i:,u:)', 'hnr_max(A:,i:,u:)',
       # 'localJitter_max(A:,i:,u:)', 'localabsoluteJitter_max(A:,i:,u:)',
       # 'rapJitter_max(A:,i:,u:)', 'ddpJitter_max(A:,i:,u:)',
       # 'localShimmer_max(A:,i:,u:)', 'localdbShimmer_max(A:,i:,u:)'
]
# 這些會有虛數
# 'Norm(B_CRatio)_sam_wilks_DKRaito', 'Norm(B_CRatio)_pillai_DKRaito',
# 'Norm(B_CRatio)_hotelling_DKRaito', 'Norm(B_CRatio)_roys_root_DKRaito',
# 'Norm(B_CRatio)_Det_DKRaito', 'Norm(B_CRatio)_Tr_DKRaito',
# columns=[
#     'Norm(WC)_sam_wilks_DKRaito', 'Norm(WC)_pillai_DKRaito',
#     'Norm(WC)_hotelling_DKRaito', 'Norm(WC)_roys_root_DKRaito',
#     'Norm(WC)_Det_DKRaito', 'Norm(WC)_Tr_DKRaito',
#     'Norm(BC)_sam_wilks_DKRaito', 'Norm(BC)_pillai_DKRaito',
#     'Norm(BC)_hotelling_DKRaito', 'Norm(BC)_roys_root_DKRaito',
#     'Norm(BC)_Det_DKRaito', 'Norm(BC)_Tr_DKRaito',

#     'Norm(TotalVar)_sam_wilks_DKRaito', 'Norm(TotalVar)_pillai_DKRaito',
#     'Norm(TotalVar)_hotelling_DKRaito', 'Norm(TotalVar)_roys_root_DKRaito',
#     'Norm(TotalVar)_Det_DKRaito', 'Norm(TotalVar)_Tr_DKRaito'
# ]

# Comb=Dict()
# Comb['LOC_columns']=FeatSel.LOC_columns
# Comb['DEP_columns']=FeatSel.DEP_columns
# Comb['LOC_columns+DEP_columns']=FeatSel.LOC_columns+FeatSel.DEP_columns

# Comb['Utt_prosodyF0']=FeatSel.Utt_prosodyF0
# Comb['Utt_prosodyF0+LOC_columns']=FeatSel.Utt_prosodyF0+FeatSel.LOC_columns
# Comb['Utt_prosodyF0+DEP_columns']=FeatSel.Utt_prosodyF0+FeatSel.DEP_columns
# Comb['Utt_prosodyF0+LOC_columns+DEP_columns']=FeatSel.Utt_prosodyF0+FeatSel.LOC_columns+FeatSel.DEP_columns

# Comb['Utt_VoiceQuality']=FeatSel.Utt_VoiceQuality
# Comb['Utt_VoiceQuality+LOC_columns']=FeatSel.Utt_VoiceQuality+FeatSel.LOC_columns
# Comb['Utt_VoiceQuality+DEP_columns']=FeatSel.Utt_VoiceQuality+FeatSel.DEP_columns
# Comb['Utt_VoiceQuality+LOC_columns+DEP_columns']=FeatSel.Utt_VoiceQuality+FeatSel.LOC_columns+FeatSel.DEP_columns

# Comb['Utt_energy']=FeatSel.Utt_energy
# Comb['Utt_energy+LOC_columns']=FeatSel.Utt_energy+FeatSel.LOC_columns 
# Comb['Utt_energy+DEP_columns']=FeatSel.Utt_energy+FeatSel.DEP_columns 
# Comb['Utt_energy+LOC_columns+DEP_columns']=FeatSel.Utt_energy+FeatSel.LOC_columns+FeatSel.DEP_columns

# Comb['Utt_prosodyF0+Utt_energy']=FeatSel.Utt_prosodyF0+FeatSel.Utt_energy 
# Comb['Utt_prosodyF0+Utt_energy+LOC_columns']=FeatSel.Utt_prosodyF0+FeatSel.Utt_energy+FeatSel.LOC_columns
# Comb['Utt_prosodyF0+Utt_energy+DEP_columns']=FeatSel.Utt_prosodyF0+FeatSel.Utt_energy+FeatSel.DEP_columns
# Comb['Utt_prosodyF0+Utt_energy+LOC_columns+DEP_columns']=FeatSel.Utt_prosodyF0+FeatSel.Utt_energy+FeatSel.LOC_columns+FeatSel.DEP_columns

# Comb['Utt_VoiceQuality+Utt_energy']=FeatSel.Utt_VoiceQuality+FeatSel.Utt_energy 
# Comb['Utt_VoiceQuality+Utt_energy+LOC_columns']=FeatSel.Utt_VoiceQuality+FeatSel.Utt_energy+FeatSel.LOC_columns
# Comb['Utt_VoiceQuality+Utt_energy+DEP_columns']=FeatSel.Utt_VoiceQuality+FeatSel.Utt_energy+FeatSel.DEP_columns
# Comb['Utt_VoiceQuality+Utt_energy+LOC_columns+DEP_columns']=FeatSel.Utt_VoiceQuality+FeatSel.Utt_energy+FeatSel.LOC_columns+FeatSel.DEP_columns

# Comb['Utt_prosodyF0+Utt_VoiceQuality']=FeatSel.Utt_prosodyF0+FeatSel.Utt_VoiceQuality 
# Comb['Utt_prosodyF0+Utt_VoiceQuality+LOC_columns']=FeatSel.Utt_prosodyF0+FeatSel.Utt_VoiceQuality+FeatSel.LOC_columns
# Comb['Utt_prosodyF0+Utt_VoiceQuality+DEP_columns']=FeatSel.Utt_prosodyF0+FeatSel.Utt_VoiceQuality+FeatSel.DEP_columns
# Comb['Utt_prosodyF0+Utt_VoiceQuality+LOC_columns+DEP_columns']=FeatSel.Utt_prosodyF0+FeatSel.Utt_VoiceQuality+FeatSel.LOC_columns+FeatSel.DEP_columns

# Comb['Utt_prosodyF0+Utt_energy+Utt_VoiceQuality']=FeatSel.Utt_prosodyF0+FeatSel.Utt_energy+FeatSel.Utt_VoiceQuality
# Comb['Utt_prosodyF0+Utt_energy+Utt_VoiceQuality+LOC_columns']=FeatSel.Utt_prosodyF0+FeatSel.Utt_energy+FeatSel.Utt_VoiceQuality+FeatSel.LOC_columns
# Comb['Utt_prosodyF0+Utt_energy+Utt_VoiceQuality+DEP_columns']=FeatSel.Utt_prosodyF0+FeatSel.Utt_energy+FeatSel.Utt_VoiceQuality+FeatSel.DEP_columns
# Comb['Utt_prosodyF0+Utt_energy+Utt_VoiceQuality+LOC_columns+DEP_columns']=FeatSel.Utt_prosodyF0+FeatSel.Utt_energy+FeatSel.Utt_VoiceQuality+ FeatSel.LOC_columns+ FeatSel.DEP_columns



# 用comb來組合multifeature fusion
# 用[ [col] for col in columns]  來跑單feautre實驗
# featuresOfInterest=[ [col] for col in columns] 
# featuresOfInterest=[ [col] + ['u_num+i_num+a_num'] for col in columns]
# featuresOfInterest=[ [col] for col in columns]
# featuresOfInterest=[ Comb[k] for k in Comb.keys()]
Top_ModuleColumn_mapping_dict={}
Top_ModuleColumn_mapping_dict['Add_UttLvl_feature']=FeatSel.Columns_comb2.copy()
Top_ModuleColumn_mapping_dict['feat_comb']=FeatSel.Columns_comb.copy()
Top_ModuleColumn_mapping_dict['feat_comb3']=FeatSel.Columns_comb3.copy()
Top_ModuleColumn_mapping_dict['feat_comb5']=FeatSel.Columns_comb5.copy()
Top_ModuleColumn_mapping_dict['feat_comb6']=FeatSel.Columns_comb6.copy()
Top_ModuleColumn_mapping_dict['feat_comb7']=FeatSel.Columns_comb7.copy()
Top_ModuleColumn_mapping_dict['feat_comb8']=FeatSel.Columns_comb8.copy()
Top_ModuleColumn_mapping_dict['Comb_dynPhonation']=FeatSel.Comb_dynPhonation.copy()
Top_ModuleColumn_mapping_dict['Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation']=FeatSel.Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation.copy()
Top_ModuleColumn_mapping_dict['Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation_Add_UttLvl_feature']=FeatSel.Comb_Utt_feature_staticLOCDEP_dynamicLOCDEP_dynamicphonation.copy()
Top_ModuleColumn_mapping_dict['baselineFeats']=FeatSel.Baseline_comb.copy()

featuresOfInterest=Top_ModuleColumn_mapping_dict[args.FeatureComb_mode]


# label_choose=['ADOS_C','Multi1','Multi2','Multi3','Multi4']
# label_choose=['ADOS_S','ADOS_C']
# label_choose=['ADOS_D']
label_choose=['ADOS_C']
# label_choose=['ADOS_S']
# label_choose=['ADOS_cate','ASDTD']

pearson_scorer = make_scorer(pearsonr, greater_is_better=False)

df_formant_statistics_CtxPhone_collect_dict=Dict()
# =============================================================================

class ADOSdataset():
    def __init__(self,):
        self.featurepath='Features'            
        self.N=2
        self.LabelType=Dict()
        self.LabelType['ADOS_S']='regression'
        self.LabelType['ADOS_C']='regression'
        self.LabelType['ADOS_D']='regression'
        self.LabelType['ADOS_cate']='classification'
        self.LabelType['ASDTD']='classification'
        self.Fractionfeatures_str='Features/artuculation_AUI/Vowels/Fraction/*.pkl'    
        self.FeatureCombs=Dict()
        # self.FeatureCombs['TD_normal vs ASDSevere_agesexmatch']=['df_formant_statistic_TD_normal', 'df_formant_statistic_agesexmatch_ASDSevere']
        # self.FeatureCombs['TD_normal vs ASDMild_agesexmatch']=['df_formant_statistic_TD_normal', 'df_formant_statistic_agesexmatch_ASDMild']
        # self.FeatureCombs['Notautism vs ASD']=['df_formant_statistic_77_Notautism', 'df_formant_statistic_77_ASD']
        # self.FeatureCombs['ASD vs Autism']=['df_formant_statistic_77_ASD', 'df_formant_statistic_77_Autism']
        # self.FeatureCombs['Notautism vs Autism']=['df_formant_statistic_77_Notautism', 'df_formant_statistic_77_Autism']
    
        # self._FeatureBuild()
    def Get_FormantAUI_feat(self,label_choose,pickle_path,featuresOfInterest=['MSB_f1','MSB_f2','MSB_mix'],filterbyNum=True,**kwargs):
        self.featuresOfInterest=featuresOfInterest
        arti=articulation.articulation.Articulation()
        if not kwargs and len(pickle_path)>0:
            df_tmp=pickle.load(open(pickle_path,"rb")).sort_index()
            # df_tmp=pickle.load(open(pickle_path,"rb"))
        elif len(kwargs)>0: # usage Get_FormantAUI_feat(...,key1=values1):
            for k, v in kwargs.items(): #there will be only one element
                df_tmp=kwargs[k].sort_index()

        if filterbyNum:
            df_tmp=arti.BasicFilter_byNum(df_tmp,N=self.N)
        
        # if label_choose not in df_tmp.columns:
        #     # print("len(df_tmp): ", len(df_tmp))
        #     # print("Feature name = ", os.path.basename(pickle_path))
        #     for people in df_tmp.index:
        #         lab=Label.label_raw[label_choose][Label.label_raw['name']==people]
        #         df_tmp.loc[people,'ADOS']=lab.values
        #     df_y=df_tmp['ADOS'] #Still keep the form of dataframe
        # else:
        #     df_y=df_tmp[label_choose] #Still keep the form of dataframe
        
        # Always update the label from Label
        for people in df_tmp.index:
            lab=Label.label_raw[label_choose][Label.label_raw['name']==people]
            df_tmp.loc[people,'ADOS']=lab.values
        df_y=df_tmp['ADOS'] #Still keep the form of dataframe
        
        
        feature_array=df_tmp[featuresOfInterest]
        
            
        LabType=self.LabelType[label_choose]
        return feature_array, df_y, LabType
    def _FeatureBuild(self):
        Features=Dict()
        Features_comb=Dict()
        files = glob.glob(self.Fractionfeatures_str)
        for file in files:
            feat_name=os.path.basename(file).replace(".pkl","")
            df_tmp=pickle.load(open(file,"rb")).sort_index()
            Features[feat_name]=df_tmp
        for keys in self.FeatureCombs.keys():
            combF=[Features[k] for k in self.FeatureCombs[keys]]
            Features_comb[keys]=pd.concat(combF)
        
        self.Features_comb=Features_comb


ados_ds=ADOSdataset()
ErrorFeat_bookeep=Dict()
Session_level_all=Dict()


if 'Add_UttLvl_feature' in args.FeatureComb_mode :
    Merge_feature_path='RegressionMerged_dfs/ADDed_UttFeat/{knn_weights}_{knn_neighbors}_{Reorder_type}/ASD_DOCKID/'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,Reorder_type=Reorder_type)
else:
    Merge_feature_path='RegressionMerged_dfs/{knn_weights}_{knn_neighbors}_{Reorder_type}/ASD_DOCKID/'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,Reorder_type=Reorder_type)

# ChooseData_manual=['static_feautre_LOC','dynamic_feature_LOC','dynamic_feature_phonation']
# ChooseData_manual=['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation']
# ChooseData_manual=['Utt_features+static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation']
ChooseData_manual=None

# for feature_paths in [Merge_feature_path]:
if ChooseData_manual==None:
    files = glob.glob(ados_ds.featurepath +'/'+ Merge_feature_path+'/*.pkl')
else:
    files=[]
    for d in ChooseData_manual:
        files.append(ados_ds.featurepath +'/'+ Merge_feature_path+'/{}.pkl'.format(d))

# load features from file
for file in files: #iterate over features
    feat_=os.path.basename(file).split(".")[0]  
    
    if type(featuresOfInterest)==dict or type(featuresOfInterest)==Dict:
        column_dict=featuresOfInterest[feat_]
    elif type(featuresOfInterest)==list:
        column_dict=featuresOfInterest
    else:
        raise KeyError()
    
    
    for key,feat_col in tqdm(column_dict.items()):
        # if len(feat_col)==1:
        #     feat_col_ = list([feat_col]) # ex: ['MSB_f1']
        # else:
        feat_col_ = list(feat_col) # ex: ['MSB_f1']
        for lab_ in label_choose:
            X,y, featType=ados_ds.Get_FormantAUI_feat(label_choose=lab_,pickle_path=file,featuresOfInterest=feat_col_,filterbyNum=False)
            Item_name="{feat}::{lab}".format(feat='-'.join([feat_]+[key]),lab=lab_)
            Session_level_all[Item_name].X, \
                Session_level_all[Item_name].y, \
                    Session_level_all[Item_name].feattype = X,y, featType
        
        assert y.isna().any() !=True

# =============================================================================
'''

    Feature merging function
    
    Ths slice of code provide user to manually make functions to combine df_XXX_infos

'''
# =============================================================================
if args.Mergefeatures:
    Merg_filepath={}
    Merg_filepath['static_feautre_LOC']='Features/artuculation_AUI/Vowels/Formants/Formant_AUI_tVSAFCRFvals_KID_FromASD_DOCKID.pkl'
    Merg_filepath['static_feautre_phonation']='Features/artuculation_AUI/Vowels/Phonation/Phonation_meanvars_KID_FromASD_DOCKID.pkl'
    Merg_filepath['dynamic_feature_LOC']='Features/artuculation_AUI/Interaction/Formants/Syncrony_measure_of_variance_DKIndividual_ASD_DOCKID.pkl'
    Merg_filepath['dynamic_feature_phonation']='Features/artuculation_AUI/Interaction/Phonation/Syncrony_measure_of_variance_phonation_ASD_DOCKID.pkl'
    
    merge_out_path='Features/RegressionMerged_dfs/'
    if not os.path.exists(merge_out_path):
        os.makedirs(merge_out_path)
    
    df_infos_dict=Dict()
    for keys, paths in Merg_filepath.items():
        df_infos_dict[keys]=pickle.load(open(paths,"rb")).sort_index()
    
    Merged_df_dict=Dict()
    comb1 = list(combinations(list(Merg_filepath.keys()), 1))
    comb2 = list(combinations(list(Merg_filepath.keys()), 2))
    for c in comb1:
        e1=c[0]
        Merged_df_dict[e1]=df_infos_dict[e1]
        OutPklpath=merge_out_path+ e1 + ".pkl"
        pickle.dump(Merged_df_dict[e1],open(OutPklpath,"wb"))
        
        
    for c in comb2:
        e1, e2=c
        Merged_df_dict['+'.join(c)]=Merge_dfs(df_infos_dict[e1],df_infos_dict[e2])
        
        OutPklpath=merge_out_path+'+'.join(c)+".pkl"
        pickle.dump(Merged_df_dict['+'.join(c)],open(OutPklpath,"wb"))
        
    
# =============================================================================
# Model parameters
# =============================================================================
# C_variable=np.array([0.01, 0.1,0.5,1.0,10.0, 50.0, 100.0, 1000.0])
# C_variable=np.array(np.arange(0.1,1.1,0.2))
# C_variable=np.array([0.1,0.5,0.9])
# epsilon=np.array(np.arange(0.1,1.5,0.1) )
epsilon=np.array([0.001,0.01,0.1,0.5,1,5,10.0,25,50,75,100])
# C_variable=np.array([0.001,0.01,0.1,1,5,10.0,25,50,75,100])
# epsilon=np.array([0.01, 0.1,0.5,1.0,10.0, 50.0, 100.0, 1000.0])
# C_variable=np.array([0.001,0.01,10.0,50,100] + list(np.arange(0.1,1.5,0.2))  )
# C_variable=np.array([0.001,0.01,10.0,50,100] + list(np.arange(0.1,1.5,0.1))  )
n_estimator=[2, 4, 8, 16, 32, 64]


Classifier={}
loo=LeaveOneOut()


# CV_settings=loo
CV_settings=10
# CV_settings=2
# skf = StratifiedKFold(n_splits=CV_settings)


'''

    Regressor

'''
###############################################################################
# Classifier['EN']={'model':ElasticNet(random_state=0),\
#                   'parameters':{'model__alpha':np.arange(0,1,0.25),\
#                                 'model__l1_ratio': np.arange(0,1,0.25)}} #Just a initial value will be changed by parameter tuning
                                                    ## l1_ratio = 1 is the lasso penalty
# Classifier['EN']={'model':ElasticNet(random_state=0),\
#                   'parameters':{'alpha':[0.25,],\
#                                 'l1_ratio': [0.5]}} #Just a initial value will be changed by parameter tuning
    
    
# from sklearn.neural_network import MLPRegressor
# Classifier['MLP']={'model':MLPRegressor(),\
#                   'parameters':{'random_state':[1],\
#                                 'hidden_layer_sizes':[(40,),(60,),(80,),(100)],\
#                                 'activation':['relu'],\
#                                 'solver':['adam'],\
#                                 'early_stopping':[True],\
#                                 # 'max_iter':[1000],\
#                                 # 'penalty':['elasticnet'],\
#                                 # 'l1_ratio':[0.25,0.5,0.75],\
#                                 }}

Classifier['SVR']={'model':sklearn.svm.SVR(),\
                  'parameters':{
                    'model__epsilon': epsilon,\
                    # 'model__C':C_variable,\
                    'model__kernel': ['rbf'],\
                    # 'gamma': ['auto'],\
                                }}

# Classifier['LinR']={'model':sklearn.linear_model.LinearRegression(),\
#                   'parameters':{'fit_intercept':[True],\
#                                 }}
###############################################################################
    
    
# Classifier['XGBoost']={'model':xgboost.sklearn.XGBRegressor(),\
#                   'parameters':{'fit_intercept':[True,False],\
#                                 }}    
    
# Classifier['SVR']={'model':sklearn.svm.SVR(),\
#                   'parameters':{'C':[50],\
#                     'kernel': ['rbf'],\
#                                 }}    
    
# Classifier['EN']={'model':ElasticNet(random_state=0),\
#               'parameters':{'alpha':[0.1, 0.5, 1],\
#                             'l1_ratio': [0.1, 0.5, 1]}} #Just a initial value will be changed by parameter tuning
    
#                                                    # l1_ratio = 1 is the lasso penalty

# Classifier['EN']={'model':ElasticNet(random_state=0),\
#               'parameters':{'alpha':[0.5],\
#                             'l1_ratio': [0.5]}} #Just a initial value will be changed by parameter tuning
#                                                   # l1_ratio = 1 is the lasso penalty





# =============================================================================
'''

    BookKeep area

'''
Best_predict_optimize={}

df_best_result_r2=pd.DataFrame([])
df_best_result_pear=pd.DataFrame([])
df_best_result_spear=pd.DataFrame([])
df_best_cross_score=pd.DataFrame([])
df_best_result_UAR=pd.DataFrame([])
df_best_result_AUC=pd.DataFrame([])
df_best_result_f1=pd.DataFrame([])
df_best_result_allThreeClassifiers=pd.DataFrame([])
# =============================================================================
Result_path="RESULTS/Fusion_result/{}/".format(args.FeatureComb_mode)


if not os.path.exists(Result_path):
    os.makedirs(Result_path)
final_result_file="_ADOS_{}.xlsx".format(args.suffix)

import warnings
warnings.filterwarnings("ignore")
count=0
OutFeature_dict=Dict()
Best_param_dict=Dict()

for clf_keys, clf in Classifier.items(): #Iterate among different classifiers 
    writer_clf = pd.ExcelWriter(Result_path+"/"+clf_keys+"_"+args.Feature_mode+"_"+final_result_file, engine = 'xlsxwriter')
    for feature_lab_str, features in Session_level_all.items():
        feature_keys, label_keys= feature_lab_str.split("::")
        feature_rawname=feature_keys[feature_keys.find('-')+1:]
        feature_filename=feature_keys[:feature_keys.find('-')]
        if feature_rawname in PprNmeMp.Paper_name_map.keys():
            featurename_paper=PprNmeMp.Paper_name_map[feature_rawname]
            feature_keys=feature_keys.replace(feature_rawname,featurename_paper)
            
        Labels = Session_level_all.X[feature_keys]
        print("=====================Cross validation start==================")
        pipe = Pipeline(steps=[('scalar',StandardScaler()),("model", clf['model'])])
        p_grid=clf['parameters']
        Gclf = GridSearchCV(estimator=pipe, param_grid=p_grid, scoring=args.selectModelScoring, cv=CV_settings, refit=True, n_jobs=-1)
        # Score=cross_val_score(Gclf, features.X, features.y, cv=CV_settings, scoring=pearson_scorer) 
        CVpredict=cross_val_predict(Gclf, features.X, features.y, cv=CV_settings)           
        Gclf.fit(features.X,features.y)
        # if clf_keys == "EN":
        #     print('The coefficient of best estimator is: ',Gclf.best_estimator_.coef_)
        
        print("The best score with scoring parameter: 'r2' is", Gclf.best_score_)
        print("The best parameters are :", Gclf.best_params_)
        best_parameters=Gclf.best_params_
        best_score=Gclf.best_score_
        best_parameters.update({'best_score':best_score})
        Best_param_dict[feature_lab_str]=best_parameters
        cv_results_info=Gclf.cv_results_

        
        
        if features.feattype == 'regression':
            r2=r2_score(features.y,CVpredict )
            n,p=features.X.shape
            r2_adj=1-(1-r2)*(n-1)/(n-p-1)
            MSE=sklearn.metrics.mean_squared_error(features.y.values.ravel(),CVpredict)
            pearson_result, pearson_p=pearsonr(features.y,CVpredict )
            spear_result, spearman_p=spearmanr(features.y,CVpredict )
            print('Feature {0}, label {1} ,spear_result {2}'.format(feature_keys, label_keys,spear_result))
        elif features.feattype == 'classification':
            n,p=features.X.shape
            UAR=recall_score(features.y, CVpredict, average='macro')
            AUC=roc_auc_score(features.y, CVpredict)
            f1Score=f1_score(features.y, CVpredict, average='macro')
            print('Feature {0}, label {1} ,UAR {2}'.format(feature_keys, label_keys,UAR))
        
        if args.Plot and p <2:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 10), sharey=True)
            kernel_label = [clf_keys]
            model_color = ['m']
            # axes.plot((features.X - min(features.X) )/ max(features.X), Gclf.best_estimator_.fit(features.X,features.y).predict(features.X), color=model_color[0],
            #               label='CV Predict')
            axes.scatter((features.X.values - min(features.X.values) )/ max(features.X.values), CVpredict, 
                         facecolor="none", edgecolor="k", s=150,
                         label='{}'.format(feature_lab_str)
                         )
            axes.scatter((features.X.values - min(features.X.values) )/ max(features.X.values), features.y, 
                         facecolor="none", edgecolor="r", s=50,
                         label='Real Y')
            axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)
            
            Plot_path='./Plot/LinearRegress/'
            if not os.path.exists(Plot_path):
                os.makedirs(Plot_path)
            plot_file=Plot_path+"/{0}_{1}.png".format(clf_keys,feature_lab_str)
            plt.savefig(plot_file, dpi=200) 
        
        # =============================================================================
        '''
            Inspect the best result
        '''
        # =============================================================================
        # Best_predict_optimize[label_keys]=pd.DataFrame(np.vstack((CVpredict,features.y)).T,columns=['y_pred','y'])
        # excel_path='./Statistics/prediction_result'
        # if not os.path.exists(excel_path):
        #     os.makedirs(excel_path)
        # excel_file=excel_path+"/{0}_{1}.xlsx"
        # writer = pd.ExcelWriter(excel_file.format(clf_keys,feature_keys.replace(":","")), engine = 'xlsxwriter')
        # for label_name in  Best_predict_optimize.keys():
        #     Best_predict_optimize[label_name].to_excel(writer,sheet_name=label_name.replace("/","_"))
        # writer.save()
                                
        # ================================================      =============================
        if features.feattype == 'regression':
            df_best_result_r2.loc[feature_keys,label_keys]='{0}/{1}'.format(np.round(r2_adj,3),np.round(np.nan,6))
            df_best_result_pear.loc[feature_keys,label_keys]='{0}/{1}'.format(np.round(pearson_result,3),np.round(pearson_p,6))
            df_best_result_spear.loc[feature_keys,label_keys]='{0}/{1}'.format(np.round(spear_result,3),np.round(spearman_p,6))
            df_best_result_spear.loc[feature_keys,'de-zero_num']=len(features.X)
            # df_best_cross_score.loc[feature_keys,label_keys]=Score.mean()
            df_best_result_allThreeClassifiers.loc[feature_keys,'{0}/{1} (MSE/pear/spear)'.format(label_keys,clf_keys)]\
                        ='{0}/{1}/{2}'.format(np.round(MSE,3),np.round(pearson_result,3),np.round(spear_result,3))

        elif features.feattype == 'classification':
            df_best_result_UAR.loc[feature_keys,label_keys]='{0}'.format(UAR)
            df_best_result_AUC.loc[feature_keys,label_keys]='{0}'.format(AUC)
            df_best_result_f1.loc[feature_keys,label_keys]='{0}'.format(f1Score)
            # df_best_result_allThreeClassifiers.loc[feature_keys,'{0}/{1} (UAR/AUC/f1score)'.format(label_keys,clf_keys)]\
            #             ='{0}/{1}/{2}'.format(np.round(UAR,3),np.round(AUC,3),np.round(f1Score,3))
            df_best_result_allThreeClassifiers.loc[feature_keys,'{0}/{1}'.format(label_keys,clf_keys)]\
                        ='{0}'.format(np.round(UAR,3))
    
    if features.feattype == 'regression':
        df_best_result_r2.to_excel(writer_clf,sheet_name="R2_adj")
        df_best_result_pear.to_excel(writer_clf,sheet_name="pear")
        df_best_result_spear.to_excel(writer_clf,sheet_name="spear")
        df_best_result_spear.to_csv(Result_path+"/"+clf_keys+"_"+args.Feature_mode+"_spearman.csv")
    elif features.feattype == 'classification':
        df_best_result_UAR.to_excel(writer_clf,sheet_name="UAR")
        df_best_result_AUC.to_excel(writer_clf,sheet_name="AUC")
        df_best_result_f1.to_excel(writer_clf,sheet_name="f1")

writer_clf.save()
print(df_best_result_allThreeClassifiers)
df_best_result_allThreeClassifiers.to_excel(Result_path+"/"+"Regression_{knn_weights}_{knn_neighbors}_{Reorder_type}.xlsx".format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,Reorder_type=Reorder_type))
print('generated at',Result_path+"/"+"Regression_{knn_weights}_{knn_neighbors}_{Reorder_type}.xlsx".format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,Reorder_type=Reorder_type) )
