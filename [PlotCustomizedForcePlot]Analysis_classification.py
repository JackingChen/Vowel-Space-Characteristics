#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:56:45 2021

@author: jackchen


    This is a branch of MainClassification that run specific experiments
    
    Include SHAP values in this script

"""

import os, sys
import pandas as pd
import numpy as np

import glob
import pickle

from scipy.stats import spearmanr,pearsonr 
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier

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

from articulation.HYPERPARAM import phonewoprosody, Label
from articulation.HYPERPARAM.PeopleSelect import SellectP_define
import articulation.HYPERPARAM.FeatureSelect as FeatSel

import articulation.articulation
from sklearn.metrics import f1_score,recall_score,roc_auc_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from itertools import combinations
import shap
import articulation.HYPERPARAM.PaperNameMapping as PprNmeMp
import inspect
import seaborn as sns

from articulation.HYPERPARAM.PlotFigureVars import *

def Find_Experiment_actualindex(Total_FeatComb_idxs,search_string):
    # usage:
    # e.x. :
    # search_string='Phonation_Trend_K_cols+Phonation_Syncrony_cols+Phonation_Trend_D_cols'
    # Total_FeatComb_idxs=FeatSel.Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation'].keys()
    # Find_Experiment_actualindex(Total_FeatComb_idxs,search_string)
    for FC_str in Total_FeatComb_idxs:
        if ''.join(sorted(FC_str.split("+"))) == ''.join(sorted(search_string.split("+"))):
            print(FC_str)



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

def Add_label(df_formant_statistic,Label,label_choose='ADOS_cate_C'):
    for people in df_formant_statistic.index:
        bool_ind=Label.label_raw['name']==people
        df_formant_statistic.loc[people,label_choose]=Label.label_raw.loc[bool_ind,label_choose].values
    return df_formant_statistic
def Swap2PaperName(feature_rawname,PprNmeMp):
    if feature_rawname in PprNmeMp.Paper_name_map.keys():
        featurename_paper=PprNmeMp.Paper_name_map[feature_rawname]
        feature_keys=featurename_paper
    else: 
        feature_keys=feature_rawname
    return feature_keys

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def Return_ALigned_dfXtestndfShapValues(Feature_SHAP_info_dict,logit_number=1):
    #Inputs
    # logit_numberit_number=1
    # Feature_SHAP_info_dict=Proposed_changed_info_dict
    ###################################
    df_XTest_stacked=pd.DataFrame()
    df_ShapValues_stacked=pd.DataFrame()
    for people in Feature_SHAP_info_dict.keys():
        df_XTest=Feature_SHAP_info_dict[people]['XTest']
        df_ShapValues=Feature_SHAP_info_dict[people]['shap_values'].loc[logit_number]
        df_ShapValues.name=df_XTest.name
        df_XTest_stacked=pd.concat([df_XTest_stacked,df_XTest],axis=1)
        df_ShapValues_stacked=pd.concat([df_ShapValues_stacked,df_ShapValues],axis=1)
    return df_XTest_stacked,df_ShapValues_stacked

def Calculate_XTestShape_correlation(Feature_SHAP_info_dict,logit_number=1):
    df_XTest_stacked,df_ShapValues_stacked=Return_ALigned_dfXtestndfShapValues(Feature_SHAP_info_dict,logit_number=logit_number)
    Correlation_XtestnShap={}
    for features in df_XTest_stacked.index:
        r,p=pearsonr(df_XTest_stacked.loc[features],df_ShapValues_stacked.loc[features])
        Correlation_XtestnShap[features]=r
    df_Correlation_XtestnShap=pd.DataFrame.from_dict(Correlation_XtestnShap,orient='index')
    df_Correlation_XtestnShap.columns=['correlation w logit:{}'.format(logit_number)]
    return df_Correlation_XtestnShap

def Prepare_data_for_summaryPlot(SHAPval_info_dict, feature_columns=None, PprNmeMp=None):
    keys_bag=[]
    XTest_dict={}
    shap_values_0_bag=[]
    shap_values_1_bag=[]
    for people in sorted(SHAPval_info_dict.keys()):
        keys_bag.append(people)
        if not feature_columns == None:
            df_=SHAPval_info_dict[people]['XTest'][feature_columns]
            df_shape_value=SHAPval_info_dict[people]['shap_values'][feature_columns]
        else:
            df_=SHAPval_info_dict[people]['XTest']
            df_shape_value=SHAPval_info_dict[people]['shap_values']
        # if not SumCategorical_feats == None:
        #     for k,values in SumCategorical_feats.items():
        #         df_[k]=df_.loc[values].sum()
        XTest_dict[people]=df_        
        shap_values_0_bag.append(df_shape_value.loc[0].values)
        shap_values_1_bag.append(df_shape_value.loc[1].values)
    shap_values_0_array=np.vstack(shap_values_0_bag)
    shap_values_1_array=np.vstack(shap_values_1_bag)
    shap_values=[shap_values_0_array,shap_values_1_array]
    df_XTest=pd.DataFrame.from_dict(XTest_dict,orient='index')
    if PprNmeMp!=None:
        df_XTest.columns=[Swap2PaperName(k,PprNmeMp) for k in df_XTest.columns]
    return shap_values, df_XTest, keys_bag

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
    parser.add_argument('--Print_Analysis_grp_Manual_select', default=True,
                        help='')
    parser.add_argument('--Plot', default=True,
                        help='')
    parser.add_argument('--selectModelScoring', default='recall_macro',
                        help='[recall_macro,accuracy]')
    parser.add_argument('--Mergefeatures', default=False,
                        help='')
    parser.add_argument('--logit_number', default=1,
                        help='')
    parser.add_argument('--knn_weights', default='uniform',
                            help='path of the base directory')
    parser.add_argument('--knn_neighbors', default=2,  type=int,
                            help='path of the base directory')
    parser.add_argument('--Reorder_type', default='DKIndividual',
                            help='[DKIndividual, DKcriteria]')
    parser.add_argument('--FeatureComb_mode', default='Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation',
                            help='[Add_UttLvl_feature, feat_comb3, feat_comb5, feat_comb6,feat_comb7, baselineFeats,Comb_dynPhonation,Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation]')
    args = parser.parse_args()
    return args
args = get_args()
start_point=args.start_point
experiment=args.experiment
knn_weights=args.knn_weights
knn_neighbors=args.knn_neighbors
Reorder_type=args.Reorder_type
logit_number=args.logit_number
#%%
# =============================================================================

#[tmp] should remove soon
# Check the input data  are all right

# sellect_people_define=SellectP_define()
# selected_columns=['Divergence[sam_wilks_lin_norm(A:,i:,u:)]',\
#                                               'Divergence[within_variance_norm(A:,i:,u:)]_var_p1',\
#                                               'Divergence[between_covariance_norm(A:,i:,u:)]_var_p1']
    
# Aaa_data=ados_ds.Features_comb['TD_normal vs ASDMild_agesexmatch -> FeatCoor']


# Aaa_df_formant_statistic_agesexmatch_ASDMild=\
#     Aaa_data.loc[sellect_people_define.MildASD_age_sex_match_ver2,selected_columns]
# df_formant_statistic_TD_normal=\
#     Aaa_data.loc[sellect_people_define.TD_normal_ver2,selected_columns]

# =============================================================================

Session_level_all=Dict()
# Discriminative analysis Main
columns=[
    'Divergence[within_covariance_norm(A:,i:,u:)]',
    'Divergence[within_variance_norm(A:,i:,u:)]',    
    'Divergence[between_covariance_norm(A:,i:,u:)]',    
    'Divergence[between_variance_norm(A:,i:,u:)]',    
    'Divergence[total_covariance_norm(A:,i:,u:)]', 
    'Divergence[total_variance_norm(A:,i:,u:)]',
    'Divergence[sam_wilks_lin_norm(A:,i:,u:)]',    
    'Divergence[pillai_lin_norm(A:,i:,u:)]',
    'Divergence[roys_root_lin_norm(A:,i:,u:)]',
    'Divergence[hotelling_lin_norm(A:,i:,u:)]',
    'Divergence[Between_Within_Det_ratio_norm(A:,i:,u:)]',
    'Divergence[Between_Within_Tr_ratio_norm(A:,i:,u:)]',
    'Divergence[within_covariance_norm(A:,i:,u:)]_var_p1',
    'Divergence[within_variance_norm(A:,i:,u:)]_var_p1',
    'Divergence[between_covariance_norm(A:,i:,u:)]_var_p1',
    'Divergence[between_variance_norm(A:,i:,u:)]_var_p1',
    'Divergence[total_covariance_norm(A:,i:,u:)]_var_p1', 
    'Divergence[total_variance_norm(A:,i:,u:)]_var_p1',
    'Divergence[sam_wilks_lin_norm(A:,i:,u:)]_var_p1',
    'Divergence[pillai_lin_norm(A:,i:,u:)]_var_p1',
    'Divergence[roys_root_lin_norm(A:,i:,u:)]_var_p1',
    'Divergence[hotelling_lin_norm(A:,i:,u:)]_var_p1',
    'Divergence[Between_Within_Det_ratio_norm(A:,i:,u:)]_var_p1',
    'Divergence[Between_Within_Tr_ratio_norm(A:,i:,u:)]_var_p1',
    'Divergence[within_covariance_norm(A:,i:,u:)]_var_p2',    
    'Divergence[within_variance_norm(A:,i:,u:)]_var_p2',    
    'Divergence[between_covariance_norm(A:,i:,u:)]_var_p2',    
    'Divergence[between_variance_norm(A:,i:,u:)]_var_p2',
    'Divergence[total_covariance_norm(A:,i:,u:)]_var_p2', 
    'Divergence[total_variance_norm(A:,i:,u:)]_var_p2',
    'Divergence[sam_wilks_lin_norm(A:,i:,u:)]_var_p2',
    'Divergence[pillai_lin_norm(A:,i:,u:)]_var_p2',
    'Divergence[roys_root_lin_norm(A:,i:,u:)]_var_p2',
    'Divergence[hotelling_lin_norm(A:,i:,u:)]_var_p2',
    'Divergence[Between_Within_Det_ratio_norm(A:,i:,u:)]_var_p2',
    'Divergence[Between_Within_Tr_ratio_norm(A:,i:,u:)]_var_p2',
    ]

# columns=[
#     'VSA2',
#     'FCR2',
#     'within_covariance_norm(A:,i:,u:)',
#     'between_covariance_norm(A:,i:,u:)',
#     'within_variance_norm(A:,i:,u:)',
#     'between_variance_norm(A:,i:,u:)',
#     'total_covariance_norm(A:,i:,u:)',
#     'total_variance_norm(A:,i:,u:)',
#     'sam_wilks_lin_norm(A:,i:,u:)',
#     'pillai_lin_norm(A:,i:,u:)',
#     'Between_Within_Det_ratio_norm(A:,i:,u:)',
#     'Between_Within_Tr_ratio_norm(A:,i:,u:)',
# ]
# columns=[
# 'Norm(WC)_sam_wilks_DKRaito', 
# 'Norm(WC)_pillai_DKRaito',
# 'Norm(WC)_hotelling_DKRaito', 'Norm(WC)_roys_root_DKRaito',
# 'Norm(WC)_Det_DKRaito', 'Norm(WC)_Tr_DKRaito',
# 'Norm(BC)_sam_wilks_DKRaito', 'Norm(BC)_pillai_DKRaito',
# 'Norm(BC)_hotelling_DKRaito', 'Norm(BC)_roys_root_DKRaito',
# 'Norm(BC)_Det_DKRaito', 'Norm(BC)_Tr_DKRaito',
# 'Norm(TotalVar)_sam_wilks_DKRaito', 'Norm(TotalVar)_pillai_DKRaito',
# 'Norm(TotalVar)_hotelling_DKRaito', 'Norm(TotalVar)_roys_root_DKRaito',
# 'Norm(TotalVar)_Det_DKRaito', 'Norm(TotalVar)_Tr_DKRaito',
# ]
# Discriminative analysis: Side exp
# columns=[
# 'VSA1',
# 'FCR',
# 'within_covariance(A:,i:,u:)',
# 'between_covariance(A:,i:,u:)',
# 'within_variance(A:,i:,u:)',
# 'between_variance(A:,i:,u:)',
# 'sam_wilks_lin(A:,i:,u:)',
# 'pillai_lin(A:,i:,u:)',
# ]


featuresOfInterest_manual=[ [col] for col in columns]
# featuresOfInterest_manual=[ [col] + ['u_num+i_num+a_num'] for col in columns]


# label_choose=['ADOS_C','Multi1','Multi2','Multi3','Multi4']
label_choose=['ADOS_C']
# label_choose=['ADOS_cate','ASDTD']
# FeatureLabelMatch_manual=[['TD_normal vs ASDSevere_agesexmatch','ASDTD'],
#                     ['TD_normal vs ASDMild_agesexmatch','ASDTD'],
#                     ['Notautism vs ASD','ADOS_cate'],
#                     ['ASD vs Autism','ADOS_cate'],
#                     ['Notautism vs Autism','ADOS_cate']]
# FeatureLabelMatch_manual=[['TD_normal vs ASDSevere_agesexmatch','ASDTD'],
#                     ['TD_normal vs ASDMild_agesexmatch','ASDTD'],
#                     ]
# FeatureLabelMatch_manual=[
                    # ['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_kid','ASDTD'],
                    # ['TD_normal vs ASDMild_agesexmatch >> FeatLOC_kid','ASDTD'],
                    # ['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_doc','ASDTD'],
                    # ['TD_normal vs ASDMild_agesexmatch >> FeatLOC_doc','ASDTD'],
                    # ['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_DKRatio','ASDTD'],#注意DKRatio的columns跟別人不一樣，不過是可以統一的
                    # ['TD_normal vs ASDMild_agesexmatch >> FeatLOC_DKRatio','ASDTD'],
                    # ['TD_normal vs ASDSevere_agesexmatch >> FeatCoor','ASDTD'],#注意DKRatio的columns跟別人不一樣，不過是可以統一的
                    # ['TD_normal vs ASDMild_agesexmatch >> FeatCoor','ASDTD'],
                    # ]

df_formant_statistics_CtxPhone_collect_dict=Dict()

# =============================================================================

class ADOSdataset():
    def __init__(self,knn_weights,knn_neighbors,Reorder_type,FeatureComb_mode):
        self.featurepath='Features'            
        self.N=2
        self.LabelType=Dict()
        self.LabelType['ADOS_C']='regression'
        self.LabelType['ADOS_cate_C']='classification'
        self.LabelType['ASDTD']='classification'
        self.Fractionfeatures_str='Features/artuculation_AUI/Vowels/Fraction/*.pkl'    
        self.FeatureComb_mode=FeatureComb_mode
        if self.FeatureComb_mode == 'Add_UttLvl_feature':
            self.File_root_path='Features/ClassificationMerged_dfs/ADDed_UttFeat/{knn_weights}_{knn_neighbors}_{Reorder_type}/'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,Reorder_type=Reorder_type)
            self.Merge_feature_path=self.File_root_path+'{dataset_role}/*.pkl'.format(dataset_role='ASD_DOCKID')
        else:
            self.File_root_path='Features/ClassificationMerged_dfs/{knn_weights}_{knn_neighbors}_{Reorder_type}/'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,Reorder_type=Reorder_type)
            self.Merge_feature_path=self.File_root_path+'{dataset_role}/*.pkl'.format(dataset_role='ASD_DOCKID')
        self.Merge_feature_path='Features/ClassificationMerged_dfs/distance_2_DKIndividual/{dataset_role}/*.pkl'.format(dataset_role='ASD_DOCKID')
        
        self.Top_ModuleColumn_mapping_dict={}
        self.Top_ModuleColumn_mapping_dict['Add_UttLvl_feature']=FeatSel.Columns_comb2.copy()
        self.Top_ModuleColumn_mapping_dict['feat_comb']=FeatSel.Columns_comb.copy()
        self.Top_ModuleColumn_mapping_dict['feat_comb3']=FeatSel.Columns_comb3.copy()
        self.Top_ModuleColumn_mapping_dict['feat_comb5']=FeatSel.Columns_comb5.copy()
        self.Top_ModuleColumn_mapping_dict['feat_comb6']=FeatSel.Columns_comb6.copy()
        self.Top_ModuleColumn_mapping_dict['feat_comb7']=FeatSel.Columns_comb7.copy()
        self.Top_ModuleColumn_mapping_dict['Comb_dynPhonation']=FeatSel.Comb_dynPhonation.copy()
        self.Top_ModuleColumn_mapping_dict['Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation']=FeatSel.Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation.copy()
        
        self.Top_ModuleColumn_mapping_dict['baselineFeats']=FeatSel.Baseline_comb.copy()
        
        self.FeatureCombs_manual=Dict()
        self.FeatureCombs_manual['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_kid']=['df_formant_statistic_TD_normal_kid', 'df_formant_statistic_agesexmatch_ASDSevereGrp_kid']
        self.FeatureCombs_manual['TD_normal vs ASDMild_agesexmatch >> FeatLOC_kid']=['df_formant_statistic_TD_normal_kid', 'df_formant_statistic_agesexmatch_ASDMildGrp_kid']
        self.FeatureCombs_manual['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_doc']=['df_formant_statistic_TD_normal_doc', 'df_formant_statistic_agesexmatch_ASDSevereGrp_doc']
        self.FeatureCombs_manual['TD_normal vs ASDMild_agesexmatch >> FeatLOC_doc']=['df_formant_statistic_TD_normal_doc', 'df_formant_statistic_agesexmatch_ASDMildGrp_doc']
        self.FeatureCombs_manual['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_DKRatio']=['df_formant_statistic_TD_normalGrp_DKRatio', 'df_formant_statistic_agesexmatch_ASDSevereGrp_DKRatio']
        self.FeatureCombs_manual['TD_normal vs ASDMild_agesexmatch >> FeatLOC_DKRatio']=['df_formant_statistic_TD_normalGrp_DKRatio', 'df_formant_statistic_agesexmatch_ASDMildGrp_DKRatio']
        
        self.FeatureCombs_manual['TD_normal vs ASDSevere_agesexmatch >> FeatCoor']=['df_syncrony_statistic_TD_normalGrp', 'df_syncrony_statistic_agesexmatch_ASDSevereGrp']
        self.FeatureCombs_manual['TD_normal vs ASDMild_agesexmatch >> FeatCoor']=['df_syncrony_statistic_TD_normalGrp', 'df_syncrony_statistic_agesexmatch_ASDMildGrp']
        # self.FeatureCombs_manual['Notautism vs ASD']=['df_formant_statistic_77_Notautism', 'df_formant_statistic_77_ASD']
        # self.FeatureCombs_manual['ASD vs Autism']=['df_formant_statistic_77_ASD', 'df_formant_statistic_77_Autism']
        # self.FeatureCombs_manual['Notautism vs Autism']=['df_formant_statistic_77_Notautism', 'df_formant_statistic_77_Autism']
    
        # self._FeatureBuild_single()
        self._FeatureBuild_Module()
    def Get_FormantAUI_feat(self,label_choose,pickle_path,featuresOfInterest=['MSB_f1','MSB_f2','MSB_mix'],filterbyNum=True,**kwargs):
        arti=articulation.articulation.Articulation()
        
        
        #如果path有放的話字串的話，就使用path的字串，不然就使用「feat_」等於的東西，在function裡面會以kwargs的形式出現
        
        if not kwargs and len(pickle_path)>0:
            df_tmp=pickle.load(open(pickle_path,"rb")).sort_index()
        elif len(kwargs)>0: # usage Get_FormantAUI_feat(...,key1=values1):
            for k, v in kwargs.items(): #there will be only one element
                df_tmp=kwargs[k].sort_index()

        if filterbyNum:
            df_tmp=arti.BasicFilter_byNum(df_tmp,N=self.N)

        if label_choose not in df_tmp.columns:
            for people in df_tmp.index:
                lab=Label.label_raw[label_choose][Label.label_raw['name']==people]
                df_tmp.loc[people,'ADOS']=lab.values
            df_y=df_tmp['ADOS'] #Still keep the form of dataframe
        else:
            df_y=df_tmp[label_choose] #Still keep the form of dataframe
        
        
        feature_array=df_tmp[featuresOfInterest]
        
            
        LabType=self.LabelType[label_choose]
        return feature_array, df_y, LabType
    def _FeatureBuild_single(self):
        Features=Dict()
        Features_comb=Dict()
        files = glob.glob(self.Fractionfeatures_str)
        for file in files:
            feat_name=os.path.basename(file).replace(".pkl","")
            df_tmp=pickle.load(open(file,"rb")).sort_index()
            Features[feat_name]=df_tmp
        for keys in self.FeatureCombs_manual.keys():
            combF=[Features[k] for k in self.FeatureCombs_manual[keys]]
            Features_comb[keys]=pd.concat(combF)
        
        self.Features_comb_single=Features_comb
    def _FeatureBuild_Module(self):
        Labels_add=['ASDTD']
        ModuledFeatureCombination=self.Top_ModuleColumn_mapping_dict[self.FeatureComb_mode]
        
        sellect_people_define=SellectP_define()
        #Loading features from ASD        
        Features_comb=Dict()
        IterateFilesFullPaths = glob.glob(self.Merge_feature_path)
        

        if self.FeatureComb_mode in ['feat_comb3','feat_comb5','feat_comb6','feat_comb7','Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation']:
            DfCombFilenames=['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation.pkl']
        if self.FeatureComb_mode == 'Comb_dynPhonation':
            DfCombFilenames=['dynamic_feature_phonation.pkl']
        elif self.FeatureComb_mode == 'baselineFeats':
             DfCombFilenames=['{}.pkl'.format(Dataname) for Dataname in ModuledFeatureCombination.keys()]
        else:
            DfCombFilenames=[os.path.basename(f) for f in IterateFilesFullPaths]
        File_ASD_paths=[self.File_root_path+"ASD_DOCKID/"+f for f in DfCombFilenames]
        File_TD_paths=[self.File_root_path+"TD_DOCKID/"+f for f in DfCombFilenames]
        
        
        df_Top_Check_length=pd.DataFrame()
        for file_ASD, file_TD in zip(File_ASD_paths,File_TD_paths):
            if not os.path.exists(file_ASD) or not os.path.exists(file_TD):
                raise FileExistsError()
            
            assert os.path.basename(file_ASD) == os.path.basename(file_TD)
            filename=os.path.basename(file_ASD)
            k_FeatTypeLayer1=filename.replace(".pkl","")
            df_feature_ASD=pickle.load(open(file_ASD,"rb")).sort_index()
            df_feature_TD=pickle.load(open(file_TD,"rb")).sort_index()
            df_feature_ASD['ASDTD']=sellect_people_define.ASDTD_label['ASD']
            df_feature_TD['ASDTD']=sellect_people_define.ASDTD_label['TD']
            
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
            
            df_feauture_ASDgrp_dict={}
            df_feauture_ASDgrp_dict['df_feature_ASD']=df_feature_ASD
            
            # df_feauture_ASDgrp_dict['df_feature_Minimal_CSS']=df_feature_ASD[filter_Minimal_TCSS]
            # df_feauture_ASDgrp_dict['df_feature_low_CSS']=df_feature_ASD[filter_low_TCSS]
            # df_feauture_ASDgrp_dict['df_feature_moderate_CSS']=df_feature_ASD[filter_moderate_TCSS]
            # df_feauture_ASDgrp_dict['df_feature_high_CSS']=df_feature_ASD[filter_high_TCSS]
            df_feauture_ASDgrp_dict['df_feature_lowMinimal_CSS']=df_feature_ASD[filter_low_TCSS | filter_Minimal_TCSS]
            df_feauture_ASDgrp_dict['df_feature_moderate_CSS']=df_feature_ASD[filter_moderate_TCSS]
            df_feauture_ASDgrp_dict['df_feature_high_CSS']=df_feature_ASD[filter_high_TCSS]
            
            # df_feauture_ASDgrp_dict['df_feature_Notautism_TC']=df_feature_ASD[filter_Notautism_TC]
            # df_feauture_ASDgrp_dict['df_feature_ASD_TC']=df_feature_ASD[filter_ASD_TC]
            # df_feauture_ASDgrp_dict['df_feature_NotautismandASD_TC']=df_feature_ASD[filter_Notautism_TC | filter_ASD_TC]
            # df_feauture_ASDgrp_dict['df_feature_Autism_TC']=df_feature_ASD[filter_Autism_TC]
            
            # df_feauture_ASDgrp_dict['df_feature_Notautism_TS']=df_feature_ASD[filter_Notautism_TS]
            # df_feauture_ASDgrp_dict['df_feature_ASD_TS']=df_feature_ASD[filter_ASD_TS]
            # df_feauture_ASDgrp_dict['df_feature_NotautismandASD_TS']=df_feature_ASD[filter_Notautism_TS | filter_ASD_TS]
            # df_feauture_ASDgrp_dict['df_feature_Autism_TS']=df_feature_ASD[filter_Autism_TS]
            
            # df_feauture_ASDgrp_dict['df_feature_Notautism_TSC']=df_feature_ASD[filter_Notautism_TSC]
            # df_feauture_ASDgrp_dict['df_feature_ASD_TSC']=df_feature_ASD[filter_ASD_TSC]
            # df_feauture_ASDgrp_dict['df_feature_NotautismandASD_TSC']=df_feature_ASD[filter_Notautism_TSC | filter_ASD_TSC]
            # df_feauture_ASDgrp_dict['df_feature_Autism_TSC']=df_feature_ASD[filter_Autism_TSC]
            
            #Check the length of each paired comparison, should be stored on the top of for loop
            
            Tmp_Numcmp_dict={}
            for key in df_feauture_ASDgrp_dict.keys():
                Numcmp_str='ASD({0}) vs TD({1})'.format(len(df_feauture_ASDgrp_dict[key]),len(df_feature_TD))
                Tmp_Numcmp_dict[key]=Numcmp_str
                
            
            df_Tmp_Numcmp_list=pd.DataFrame.from_dict(Tmp_Numcmp_dict,orient='index')
            df_Tmp_Numcmp_list.columns=[k_FeatTypeLayer1]

            if len(df_Top_Check_length)==0:
                df_Top_Check_length=df_Tmp_Numcmp_list
            else:
                df_Top_Check_length=Merge_dfs(df_Top_Check_length,df_Tmp_Numcmp_list)
            # 手動執行到這邊，從for 上面
            
            
            for k_FeatTypeLayer2 in ModuledFeatureCombination[k_FeatTypeLayer1].keys():
                colums_sel=ModuledFeatureCombination[k_FeatTypeLayer1][k_FeatTypeLayer2]
                
                # 1. Set ASD vs TD experiment
                for k_ASDgrp in df_feauture_ASDgrp_dict.keys():
                    df_ASD_subgrp=df_feauture_ASDgrp_dict[k_ASDgrp].copy()[colums_sel+Labels_add]
                    df_TD_subgrp=df_feature_TD.copy()[colums_sel+Labels_add]
                    
                    experiment_str="{TD_name} vs {ASD_name} >> {feature_type}".format(TD_name='TD',ASD_name=k_ASDgrp,feature_type=k_FeatTypeLayer2)
                    Features_comb[experiment_str]=pd.concat([df_ASD_subgrp,df_TD_subgrp],axis=0)
                # 2. Set ASDsevere vs ASDmild experiment
                # experiment_str="{ASDsevere_name} vs {ASDmild_name} >> {feature_type}".format(ASDsevere_name='df_feature_moderatehigh_CSS',ASDmild_name='df_feature_lowMinimal_CSS',feature_type=k_FeatTypeLayer2)
                # df_ASDsevere_subgrp=df_feauture_ASDgrp_dict['df_feature_moderatehigh_CSS'].copy()
                # df_ASDmild_subgrp=df_feauture_ASDgrp_dict['df_feature_lowMinimal_CSS'].copy()
                # df_ASDsevere_subgrp['ASDsevereMild']=sellect_people_define.ASDsevereMild_label['ASDsevere']
                # df_ASDmild_subgrp['ASDsevereMild']=sellect_people_define.ASDsevereMild_label['ASDmild']
                # Features_comb[experiment_str]=pd.concat([df_ASDsevere_subgrp,df_ASDmild_subgrp],axis=0)
        self.Features_comb_multi=Features_comb

# =============================================================================
'''

    Feature merging function
    
    Ths slice of code provide user to manually make functions to combine df_XXX_infos

'''
# =============================================================================

ados_ds=ADOSdataset(knn_weights,knn_neighbors,Reorder_type,FeatureComb_mode=args.FeatureComb_mode)
ErrorFeat_bookeep=Dict()


FeatureLabelMatch_manual=[

    ['TD vs df_feature_lowMinimal_CSS >> LOC_columns+Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols', 'ASDTD'],
    ['TD vs df_feature_lowMinimal_CSS >> LOC_columns+Phonation_Trend_K_cols+Phonation_Syncrony_cols', 'ASDTD'],
    ['TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols', 'ASDTD'],
    ['TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_K_cols+Phonation_Syncrony_cols', 'ASDTD'],


    ['TD vs df_feature_moderate_CSS >> LOCDEP_Trend_D_cols+Phonation_Proximity_cols', 'ASDTD'],
    ['TD vs df_feature_moderate_CSS >> Phonation_Proximity_cols', 'ASDTD'],
    

    ['TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_K_cols+Phonation_Syncrony_cols', 'ASDTD'],
    ['TD vs df_feature_high_CSS >> DEP_columns+Phonation_Trend_D_cols+Phonation_Proximity_cols', 'ASDTD'],
    ['TD vs df_feature_high_CSS >> Phonation_Trend_D_cols+Phonation_Proximity_cols', 'ASDTD'],
    ['TD vs df_feature_high_CSS >> DEP_columns+Phonation_Proximity_cols', 'ASDTD'],
    ['TD vs df_feature_high_CSS >> Phonation_Proximity_cols', 'ASDTD'],


 ]
    
# FeatSel 掌管該出現的columns
# ados_ds.Features_comb_multi 掌管load進來的data


Top_ModuleColumn_mapping_dict={}
Top_ModuleColumn_mapping_dict['Add_UttLvl_feature']={ e2_str:FeatSel.Columns_comb2[e_str][e2_str] for e_str in FeatSel.Columns_comb2.keys() for e2_str in FeatSel.Columns_comb2[e_str].keys()}
Top_ModuleColumn_mapping_dict['feat_comb3']=ModuleColumn_mapping={ e2_str:FeatSel.Columns_comb3[e_str][e2_str] for e_str in FeatSel.Columns_comb3.keys() for e2_str in FeatSel.Columns_comb3[e_str].keys()}
Top_ModuleColumn_mapping_dict['feat_comb5']=ModuleColumn_mapping={ e2_str:FeatSel.Columns_comb5[e_str][e2_str] for e_str in FeatSel.Columns_comb5.keys() for e2_str in FeatSel.Columns_comb5[e_str].keys()}
Top_ModuleColumn_mapping_dict['feat_comb6']=ModuleColumn_mapping={ e2_str:FeatSel.Columns_comb6[e_str][e2_str] for e_str in FeatSel.Columns_comb6.keys() for e2_str in FeatSel.Columns_comb6[e_str].keys()}
Top_ModuleColumn_mapping_dict['feat_comb7']=ModuleColumn_mapping={ e2_str:FeatSel.Columns_comb7[e_str][e2_str] for e_str in FeatSel.Columns_comb7.keys() for e2_str in FeatSel.Columns_comb7[e_str].keys()}
Top_ModuleColumn_mapping_dict['Comb_dynPhonation']=ModuleColumn_mapping={ e2_str:FeatSel.Comb_dynPhonation[e_str][e2_str] for e_str in FeatSel.Comb_dynPhonation.keys() for e2_str in FeatSel.Comb_dynPhonation[e_str].keys()}
Top_ModuleColumn_mapping_dict['Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation']=ModuleColumn_mapping={ e2_str:FeatSel.Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation[e_str][e2_str] for e_str in FeatSel.Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation.keys() for e2_str in FeatSel.Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation[e_str].keys()}
Top_ModuleColumn_mapping_dict['feat_comb']=ModuleColumn_mapping={ e2_str:FeatSel.Columns_comb[e_str][e2_str] for e_str in FeatSel.Columns_comb.keys() for e2_str in FeatSel.Columns_comb[e_str].keys()}


ModuleColumn_mapping=Top_ModuleColumn_mapping_dict[args.FeatureComb_mode]
# =============================================================================
'''

    Here starts to load features to Session_level_all dict

'''
# =============================================================================
for exp_str, lab_ in FeatureLabelMatch_manual:
    comparison_pair=exp_str.split(" >> ")[0]
    ModuleColumn_str=exp_str.split(" >> ")[-1]
    featuresOfInterest=[ModuleColumn_mapping[ModuleColumn_str]]
    # feat_=key
    for feat_col in featuresOfInterest:
        feat_col_ = list(feat_col) # ex: ['MSB_f1']
        if len(feat_col) > 144: # 144 is the limit of the filename
            key=feat_col_
        else:
            key=[feat_col_]
        # X,y, featType=ados_ds.Get_FormantAUI_feat(\
        #     label_choose=lab_,pickle_path='',featuresOfInterest=feat_col_,filterbyNum=False,\
        #     feat_=ados_ds.Features_comb_single[feat_])
        
        X,y, featType=ados_ds.Get_FormantAUI_feat(\
            label_choose=lab_,pickle_path='',featuresOfInterest=feat_col,filterbyNum=False,\
            feat_=ados_ds.Features_comb_multi[exp_str])
        
        if X.isnull().values.any() or y.isnull().values.any():
            print("Feat: ",key,'Contains nan')
            ErrorFeat_bookeep['{0} {1} {2}'.format(exp_str,lab_,key)].X=X
            ErrorFeat_bookeep['{0} {1} {2}'.format(exp_str,lab_,key)].y=y
            continue
        
        Item_name="{feat}::{lab}".format(feat=' >> '.join([comparison_pair,ModuleColumn_str]),lab=lab_)
        Session_level_all[Item_name].X, \
            Session_level_all[Item_name].y, \
                Session_level_all[Item_name].feattype = X,y, featType

# =============================================================================
# Model parameters
# =============================================================================
# C_variable=np.array([0.0001, 0.01, 0.1,0.5,1.0,10.0, 50.0, 100.0, 1000.0])
# C_variable=np.array([0.01, 0.1,0.5,1.0,10.0, 50.0, 100.0])
# C_variable=np.array(np.arange(0.1,1.5,0.1))
C_variable=np.array([0.001,0.01,0.1,1,5,10.0,25,50,75,100])
# C_variable=np.array([0.001,0.01,10.0,50,100] + list(np.arange(0.1,1.5,0.2))  )
# C_variable=np.array([0.01, 0.1,0.5,1.0, 5.0])
n_estimator=[ 32, 50, 64, 100 ,128, 256]

'''

    Classifier

'''

# This is the closest 
Classifier={}
Classifier['SVC']={'model':sklearn.svm.SVC(),\
                  'parameters':{'model__random_state':[1],\
                      'model__C':C_variable,\
                    'model__kernel': ['rbf'],\
                      # 'model__gamma':['auto'],\
                    'model__probability':[True],\
                                }}

   

loo=LeaveOneOut()
# CV_settings=loo
CV_settings=10

# =============================================================================
# Outputs
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
Result_path="RESULTS/"
if not os.path.exists(Result_path):
    os.makedirs(Result_path)
final_result_file="_ADOS_{}.xlsx".format(args.suffix)

import warnings
warnings.filterwarnings("ignore")
count=0
OutFeature_dict=Dict()
Best_param_dict=Dict()
sellect_people_define=SellectP_define()

# ''' 要手動執行一次從Incorrect2Correct_indexes和Correct2Incorrect_indexes決定哪些indexes 需要算shap value 再在這邊指定哪些fold需要停下來算SHAP value '''
SHAP_inspect_idxs_manual=None # None means calculate SHAP value of all people
# SHAP_inspect_idxs_manual=[] # empty list means we do not execute shap function
# SHAP_inspect_idxs_manual=sorted(list(set([14, 21]+[]+[24, 28, 30, 31, 39, 41, 45]+[22, 23, 27, 47, 58]+[6, 13, 19, 23, 24, 25]+[28, 35, 38, 45])))

for clf_keys, clf in Classifier.items(): #Iterate among different classifiers 
    writer_clf = pd.ExcelWriter(Result_path+"/"+clf_keys+"_"+args.Feature_mode+"_"+final_result_file, engine = 'xlsxwriter')
    for feature_lab_str, features in Session_level_all.items():

        feature_keys, label_keys= feature_lab_str.split("::")
        feature_rawname=feature_keys[feature_keys.find('-')+1:]
        # if feature_rawname in paper_name_map.keys():
        #     featurename_paper=paper_name_map[feature_rawname]
        #     feature_keys=feature_keys.replace(feature_rawname,featurename_paper)
        
        if SHAP_inspect_idxs_manual != None:
            SHAP_inspect_idxs=SHAP_inspect_idxs_manual
        else:
            SHAP_inspect_idxs=range(len(features.y))
        
        Labels = Session_level_all.X[feature_keys]
        print("=====================Cross validation start==================")
        pipe = Pipeline(steps=[('scalar',StandardScaler()),("model", clf['model'])])
        p_grid=clf['parameters']
        Gclf = GridSearchCV(estimator=pipe, param_grid=p_grid, scoring=args.selectModelScoring, cv=CV_settings, refit=True, n_jobs=-1)
        Gclf_manual = GridSearchCV(estimator=pipe, param_grid=p_grid, scoring=args.selectModelScoring, cv=CV_settings, refit=True, n_jobs=-1)
        
        CVpredict=cross_val_predict(Gclf, features.X, features.y, cv=CV_settings)  
        
        # The cv is as the one in cross_val_predict function
        cv = sklearn.model_selection.check_cv(CV_settings,features.y,classifier=sklearn.base.is_classifier(Gclf))
        splits = list(cv.split(features.X, features.y, groups=None))
        test_indices = np.concatenate([test for _, test in splits])

        
        CVpredict_manual=np.zeros(len(features.y))
        for i, (train_index, test_index) in enumerate(splits):
            X_train, X_test = features.X.iloc[train_index], features.X.iloc[test_index]
            y_train, y_test = features.y.iloc[train_index], features.y.iloc[test_index]
            Gclf_manual.fit(X_train,y_train)
            result_bestmodel=Gclf_manual.predict(X_test)
            
            CVpredict_manual[test_index]=result_bestmodel
            
            # result_bestmodel_fitted_again=best_model_fittedagain.predict(X_test_encoded)
            CVpred_fromFunction=CVpredict[test_index]
            
            # SHAP value generating
            # logit_number=0
            # inspect_sample=0
            # If the indexes we want to examine are in that fold, store the whole fold
            # 先把整個fold記錄下來然後在analysis area再拆解
            SHAP_exam_lst=[i for i in test_index if i in SHAP_inspect_idxs]
            if len(SHAP_exam_lst) != 0:
                explainer = shap.KernelExplainer(Gclf_manual.predict_proba, X_train)
                shap_values = explainer.shap_values(X_test)
                Session_level_all[feature_lab_str]['SHAP_info']['_'.join(test_index.astype(str))].explainer_expected_value=explainer.expected_value
                Session_level_all[feature_lab_str]['SHAP_info']['_'.join(test_index.astype(str))].shap_values=shap_values # shap_values= [logit, index, feature]
                Session_level_all[feature_lab_str]['SHAP_info']['_'.join(test_index.astype(str))].XTest=X_test
                Session_level_all[feature_lab_str]['Logit{}_predictproba'.format(logit_number)]['_'.join(test_index.astype(str))].predictproba=Gclf_manual.predict_proba(X_test)[:,logit_number]
                Session_level_all[feature_lab_str]['Logit{}_predictproba'.format(logit_number)]['_'.join(test_index.astype(str))].decisionfunc=Gclf_manual.decision_function(X_test)
                # Session_level_all[feature_lab_str]['SHAP_info']['_'.join(test_index)].testIndex=test_index
            # shap.force_plot(explainer.expected_value[logit_number], shap_values[logit_number][inspect_sample,:], X_test.iloc[inspect_sample,:], matplotlib=True,show=False)
            
            
            # assert (result_bestmodel==result_bestmodel_fitted_again).all()
            assert (result_bestmodel==CVpred_fromFunction).all()
        
            
        assert (CVpredict_manual==CVpredict).all()
        Session_level_all[feature_lab_str]['y_pred']=CVpredict_manual
        Session_level_all[feature_lab_str]['y_true']=features.y
        
        Gclf.fit(features.X,features.y)
        if clf_keys == "EN":
            print('The coefficient of best estimator is: ',Gclf.best_estimator_.coef_)
        
        print("The best score with scoring parameter: 'r2' is", Gclf.best_score_)
        print("The best parameters are :", Gclf.best_params_)
        best_parameters=Gclf.best_params_
        best_score=Gclf.best_score_
        best_parameters.update({'best_score':best_score})
        Best_param_dict[feature_lab_str]=best_parameters
        cv_results_info=Gclf.cv_results_

        num_ASD=len(np.where(features.y==sellect_people_define.ASDTD_label['ASD'])[0])
        num_TD=len(np.where(features.y==sellect_people_define.ASDTD_label['TD'])[0])
        
        if features.feattype == 'regression':
            r2=r2_score(features.y,CVpredict )
            n,p=features.X.shape
            r2_adj=1-(1-r2)*(n-1)/(n-p-1)
            pearson_result, pearson_p=pearsonr(features.y,CVpredict )
            spear_result, spearman_p=spearmanr(features.y,CVpredict )
            print('Feature {0}, label {1} ,spear_result {2}'.format(feature_keys, label_keys,spear_result))
        elif features.feattype == 'classification':
            n,p=features.X.shape
            CM=confusion_matrix(features.y, CVpredict)
            Session_level_all[feature_lab_str]['Confusion_matrix']=pd.DataFrame(CM,\
                                                                    index=['y_true_{}'.format(ii) for ii in range(CM.shape[0])],\
                                                                    columns=['y_pred_{}'.format(ii) for ii in range(CM.shape[1])])
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
        Best_predict_optimize[label_keys]=pd.DataFrame(np.vstack((CVpredict,features.y)).T,columns=['y_pred','y'])
        excel_path='./Statistics/prediction_result'
        if not os.path.exists(excel_path):
            os.makedirs(excel_path)
        excel_file=excel_path+"/{0}_{1}.xlsx"
        writer = pd.ExcelWriter(excel_file.format(clf_keys,feature_keys.replace(":","")), engine = 'xlsxwriter')
        for label_name in  Best_predict_optimize.keys():
            Best_predict_optimize[label_name].to_excel(writer,sheet_name=label_name.replace("/","_"))
        writer.save()
                                
        # ================================================      =============================
        if features.feattype == 'regression':
            df_best_result_r2.loc[feature_keys,label_keys]='{0}/{1}'.format(np.round(r2_adj,3),np.round(np.nan,6))
            df_best_result_pear.loc[feature_keys,label_keys]='{0}/{1}'.format(np.round(pearson_result,3),np.round(pearson_p,6))
            df_best_result_spear.loc[feature_keys,label_keys]='{0}/{1}'.format(np.round(spear_result,3),np.round(spearman_p,6))
            df_best_result_spear.loc[feature_keys,'de-zero_num']=len(features.X)
            # df_best_cross_score.loc[feature_keys,label_keys]=Score.mean()
            df_best_result_allThreeClassifiers.loc[feature_keys,'{0}/{1} (R2adj/pear/spear)'.format(label_keys,clf_keys)]\
                        ='{0}/{1}/{2}'.format(np.round(r2_adj,3),np.round(pearson_result,3),np.round(spear_result,3))

        elif features.feattype == 'classification':
            df_best_result_UAR.loc[feature_keys,label_keys]='{0}'.format(UAR)
            df_best_result_AUC.loc[feature_keys,label_keys]='{0}'.format(AUC)
            df_best_result_f1.loc[feature_keys,label_keys]='{0}'.format(f1Score)
            # df_best_result_allThreeClassifiers.loc[feature_keys,'{0}/{1} (UAR/AUC/f1score)'.format(label_keys,clf_keys)]\
            #             ='{0}/{1}/{2}'.format(np.round(UAR,3),np.round(AUC,3),np.round(f1Score,3))
            df_best_result_allThreeClassifiers.loc[feature_keys,'{0}/{1}'.format(label_keys,clf_keys)]\
                        ='{0}'.format(np.round(UAR,3))
            df_best_result_allThreeClassifiers.loc[feature_keys,'num_ASD']\
                        ='{0}'.format(num_ASD)
            df_best_result_allThreeClassifiers.loc[feature_keys,'num_TD']\
                        ='{0}'.format(num_TD)
            df_best_result_allThreeClassifiers.loc[feature_keys,'f1']\
                        ='{0}'.format(np.round(f1Score,3))
        count+=1
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
df_best_result_allThreeClassifiers.to_excel(Result_path+"/"+"Classification_"+args.Feature_mode+"_3clsferRESULT.xlsx")
print(df_best_result_allThreeClassifiers)

# Change to paper name
df_allThreeClassifiers_paperName=df_best_result_allThreeClassifiers.copy()
index_bag=[]
for exp_str in df_best_result_allThreeClassifiers.index:
    experiment_name, feature_name=exp_str.split(" >> ")
    paper_idx='+'.join([Swap2PaperName(n, PprNmeMp) for n in feature_name.split("+")])
    index_bag.append(paper_idx)
df_allThreeClassifiers_paperName.index=index_bag

#%%
# =============================================================================
'''

    Analysis part

'''
def Organize_Needed_SHAP_info(Incorrect2Correct_indexes, Session_level_all, proposed_expstr):
    Incorrect2Correct_info_dict=Dict()
    for tst_idx in Incorrect2Correct_indexes:
        for key, values in Session_level_all[proposed_expstr]['SHAP_info'].items():
            test_fold_idx=[int(k) for k in key.split("_")]
            for i,ii in enumerate(test_fold_idx): #ii is the index of the sample, i is the position of this sample in this test fold
                if tst_idx == ii:
                    Incorrect2Correct_info_dict[tst_idx]['XTest']=values['XTest'].iloc[i,:]
                    Incorrect2Correct_info_dict[tst_idx]['explainer_expected_value']=values['explainer_expected_value']
                    shap_values_array=[array[i,:] for array in values['shap_values']]
                    df_shap_values=pd.DataFrame(shap_values_array,columns=Incorrect2Correct_info_dict[tst_idx]['XTest'].index)
                    Incorrect2Correct_info_dict[tst_idx]['shap_values']=df_shap_values
                    print("testing sample ", ii, "is in the ", i, "position of test fold", key)
                    assert (Incorrect2Correct_info_dict[tst_idx]['XTest'] == Session_level_all[proposed_expstr]['X'].iloc[tst_idx]).all()
                    # print("It's feature value captured is", Incorrect2Correct_info_dict[tst_idx]['XTest'])
                    # print("It's original X value is", Session_level_all[proposed_expstr]['X'].iloc[tst_idx])
                    # print("See if they match")
    return Incorrect2Correct_info_dict
def Get_Model_Type12Errors(model_str='baseline', tureLab_str='y_true'):
    # Positive = ASD
    Type1Err= ( df_Y_pred[model_str]  == sellect_people_define.ASDTD_label['ASD']) & ( df_Y_pred[tureLab_str] == sellect_people_define.ASDTD_label['TD']  )
    Type2Err= ( df_Y_pred[model_str]  == sellect_people_define.ASDTD_label['TD'] ) & ( df_Y_pred[tureLab_str] == sellect_people_define.ASDTD_label['ASD']  )
    return Type1Err, Type2Err
# =============================================================================

'''

    Part 1: Check incorrect to correct and correct to incorrect

'''

if args.Print_Analysis_grp_Manual_select == True:
    count=0
    for exp_lst in FeatureLabelMatch_manual:
        exp_lst_str='::'.join(exp_lst)
        if count < len(FeatureLabelMatch_manual)/2:
            print("proposed_expstr='{}'".format(exp_lst_str))
        else:
            print("baseline_expstr='{}'".format(exp_lst_str))
        count+=1
    

############################################################
# Low Minimal
# proposed_expstr='TD vs df_feature_lowMinimal_CSS >> LOC_columns+Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD'
# baseline_expstr='TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD'
[14, 21]
[]

# proposed_expstr='TD vs df_feature_lowMinimal_CSS >> LOC_columns+Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD'
# baseline_expstr='TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD'
# [1, 14, 15, 21]
# []

# proposed_expstr='TD vs df_feature_lowMinimal_CSS >> LOC_columns+Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD'
# baseline_expstr='TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD'
[12, 21]
[4 , 11]

# proposed_expstr='TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD'
# baseline_expstr='TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD'
# []
# [1, 15]
############################################################
# Moderate
proposed_expstr='TD vs df_feature_moderate_CSS >> LOCDEP_Trend_D_cols+Phonation_Proximity_cols::ASDTD'
baseline_expstr='TD vs df_feature_moderate_CSS >> Phonation_Proximity_cols::ASDTD'
[24, 28, 30, 31, 39, 41, 45]
[22, 23, 27, 47, 58]


############################################################
# high
# proposed_expstr='TD vs df_feature_high_CSS >> DEP_columns+Phonation_Trend_D_cols+Phonation_Proximity_cols::ASDTD'
# baseline_expstr='TD vs df_feature_high_CSS >> Phonation_Proximity_cols::ASDTD'
[6, 13, 19, 23, 24, 25]
[28, 35, 38, 45]

# proposed_expstr='TD vs df_feature_high_CSS >> DEP_columns+Phonation_Trend_D_cols+Phonation_Proximity_cols::ASDTD'
# proposed_expstr='TD vs df_feature_high_CSS >> Phonation_Trend_D_cols+Phonation_Proximity_cols::ASDTD'

# proposed_expstr='TD vs df_feature_high_CSS >> Phonation_Trend_D_cols+Phonation_Proximity_cols::ASDTD'
# baseline_expstr='TD vs df_feature_high_CSS >> Phonation_Proximity_cols::ASDTD'
[13, 19, 23, 24, 25]
[2, 3, 28, 29, 35, 36, 38, 44]

# proposed_expstr='TD vs df_feature_high_CSS >> DEP_columns+Phonation_Proximity_cols::ASDTD'
# baseline_expstr='TD vs df_feature_high_CSS >> Phonation_Proximity_cols::ASDTD'
experiment_title=baseline_expstr[re.search("df_feature_",baseline_expstr).end():re.search("_CSS >> ",baseline_expstr).start()]

proposed_featset_lst=proposed_expstr[re.search(" >> ",proposed_expstr).end():re.search("::",proposed_expstr).start()].split("+")
baseline_featset_lst=baseline_expstr[re.search(" >> ",baseline_expstr).end():re.search("::",baseline_expstr).start()].split("+")
Additional_featureSet=set(proposed_featset_lst) - set(baseline_featset_lst)
print("For Task", experiment_title, " additional feature sets are", Additional_featureSet)
# =============================================================================
# Error type analyses
# =============================================================================
# df_compare_pair=pd.DataFrame(list())
Y_pred_lst=[
Session_level_all[proposed_expstr]['y_pred'],
Session_level_all[baseline_expstr]['y_pred'],
Session_level_all[proposed_expstr]['y_true'],
Session_level_all[proposed_expstr]['y_true'].index,
]
assert (Session_level_all[proposed_expstr]['y_true'] == Session_level_all[baseline_expstr]['y_true']).all()

df_Y_pred=pd.DataFrame(Y_pred_lst[:-1],index=['proposed','baseline','y_true']).T
df_Y_pred_withName=pd.DataFrame(Y_pred_lst,index=['proposed','baseline','y_true','name']).T
df_Index2Name_mapping=df_Y_pred_withName['name']


Incorrect=df_Y_pred['baseline'] != df_Y_pred['y_true']
Correct=df_Y_pred['proposed'] == df_Y_pred['y_true']
Incorrect2Correct= Correct & Incorrect


Incorrect=df_Y_pred['baseline'] == df_Y_pred['y_true']
Correct=df_Y_pred['proposed'] != df_Y_pred['y_true']
Correct2Incorrect= Correct & Incorrect

Incorrect2Correct_indexes=list(df_Y_pred[Incorrect2Correct].index)
Correct2Incorrect_indexes=list(df_Y_pred[Correct2Incorrect].index)
print('Incorrect2Correct_indexes: ', Incorrect2Correct_indexes)
print('Correct2Incorrect_indexes: ', Correct2Incorrect_indexes)


Ones=df_Y_pred['baseline'] ==sellect_people_define.ASDTD_label['ASD']
Twos=df_Y_pred['proposed'] ==sellect_people_define.ASDTD_label['TD']
Ones2Twos=  Ones & Twos

Twos=df_Y_pred['baseline'] ==sellect_people_define.ASDTD_label['TD']
Ones=df_Y_pred['proposed'] ==sellect_people_define.ASDTD_label['ASD']
Twos2Ones=  Ones & Twos

Ones2Twos_indexes=list(df_Y_pred[Ones2Twos].index)
Twos2Ones_indexes=list(df_Y_pred[Twos2Ones].index)

assert len(Ones2Twos_indexes+Twos2Ones_indexes) == len(Incorrect2Correct_indexes+Correct2Incorrect_indexes)

ASDTD2Logit_map={
    'TD': sellect_people_define.ASDTD_label['TD']-1,
    'ASD': sellect_people_define.ASDTD_label['ASD']-1,
    }

quadrant1_indexes=intersection(Correct2Incorrect_indexes, Ones2Twos_indexes)
quadrant2_indexes=intersection(Incorrect2Correct_indexes, Ones2Twos_indexes)
quadrant3_indexes=intersection(Correct2Incorrect_indexes, Twos2Ones_indexes)
quadrant4_indexes=intersection(Incorrect2Correct_indexes, Twos2Ones_indexes)

Type1Err_dict, Type2Err_dict={}, {}
for model_str in ['baseline', 'proposed']:
    Type1Err_dict[model_str], Type2Err_dict[model_str] = Get_Model_Type12Errors(model_str=model_str, tureLab_str='y_true')

model_str='baseline'
Type1Err_baseline_indexes=list(df_Y_pred[Type1Err_dict[model_str]].index)
Type2Err_baseline_indexes=list(df_Y_pred[Type2Err_dict[model_str]].index)
model_str='proposed'
Type1Err_proposed_indexes=list(df_Y_pred[Type1Err_dict[model_str]].index)
Type2Err_proposed_indexes=list(df_Y_pred[Type2Err_dict[model_str]].index)


All_indexes=list(set(Type1Err_baseline_indexes+Type2Err_baseline_indexes+Type1Err_proposed_indexes+Type2Err_proposed_indexes))
'''

    Part 2: Check the SHAP values based on indexes in part 1
    
    先紀錄，再執行分析和畫圖

'''
Manual_inspect_idxs=[2, 3, 29, 36, 44]
##############################################

selected_idxs=Ones2Twos_indexes+Twos2Ones_indexes
Baseline_changed_info_dict=Organize_Needed_SHAP_info(selected_idxs, Session_level_all, baseline_expstr)
Proposed_changed_info_dict=Organize_Needed_SHAP_info(selected_idxs, Session_level_all, proposed_expstr)

Baseline_totalPoeple_info_dict=Organize_Needed_SHAP_info(df_Y_pred.index, Session_level_all, baseline_expstr)
Proposed_totalPoeple_info_dict=Organize_Needed_SHAP_info(df_Y_pred.index, Session_level_all, proposed_expstr)

#%%
# 畫炫炮的錯誤型態分析 (Changed smaples的logit 1 decision function 的移動)
def Organize_Needed_decisionProb(Incorrect2Correct_indexes, Session_level_all, proposed_expstr):
    Incorrect2Correct_info_dict=Dict()
    for tst_idx in Incorrect2Correct_indexes:
        for key, values in Session_level_all[proposed_expstr]['Logit{}_predictproba'.format(logit_number)].items():
            test_fold_idx=[int(k) for k in key.split("_")]
            for i,ii in enumerate(test_fold_idx): #ii is the index of the sample, i is the position of this sample in this test fold
                if tst_idx == ii:
                    Incorrect2Correct_info_dict[tst_idx]['predictproba']=values['predictproba'][i]
                    Incorrect2Correct_info_dict[tst_idx]['decisionfunc']=values['decisionfunc'][i]

    return Incorrect2Correct_info_dict
# fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10,5))



# step 1: prepare data
selected_idxs=Ones2Twos_indexes+Twos2Ones_indexes
Baseline_changed_decision_info_dict=Organize_Needed_decisionProb(selected_idxs, Session_level_all, baseline_expstr)
Proposed_changed_decision_info_dict=Organize_Needed_decisionProb(selected_idxs, Session_level_all, proposed_expstr)
Baseline_total_decision_info_dict=Organize_Needed_decisionProb(df_Y_pred.index, Session_level_all, baseline_expstr)
Proposed_total_decision_info_dict=Organize_Needed_decisionProb(df_Y_pred.index, Session_level_all, proposed_expstr)


df_Proposed_changed_decision_info_dict=pd.DataFrame.from_dict(Proposed_changed_decision_info_dict,orient='index')
df_Baseline_changed_decision_info_dict=pd.DataFrame.from_dict(Baseline_changed_decision_info_dict,orient='index')
df_Proposed_total_decision_info_dict=pd.DataFrame.from_dict(Proposed_total_decision_info_dict,orient='index')
df_Baseline_total_decision_info_dict=pd.DataFrame.from_dict(Baseline_total_decision_info_dict,orient='index')
Sample_idxs_array=df_Baseline_changed_decision_info_dict.index.values

df_Y_true=df_Y_pred.loc[df_Baseline_changed_decision_info_dict.index]['y_true']
df_Y_true_ASD_bool=df_Y_true==sellect_people_define.ASDTD_label['ASD']
df_Y_true_TD_bool=df_Y_true==sellect_people_define.ASDTD_label['TD']
Incorrect_baseline=df_Y_pred['baseline'] != df_Y_pred['y_true']
Incorrect_proposed=df_Y_pred['proposed'] != df_Y_pred['y_true']
Correct_baseline=df_Y_pred['baseline'] == df_Y_pred['y_true']
Correct_proposed=df_Y_pred['proposed'] == df_Y_pred['y_true']

fig, ax = plt.subplots()

# decision function 負的表示predict logit 0, 正的表示logit 1
Baseline_x= df_Baseline_changed_decision_info_dict['predictproba'].values
Baseline_y_decisionfunc= df_Baseline_changed_decision_info_dict['decisionfunc'].abs().values.copy()
Baseline_y= df_Baseline_changed_decision_info_dict['decisionfunc'].abs().values.copy()
Baseline_total_x= df_Baseline_total_decision_info_dict['predictproba'].values
Baseline_total_y_decisionfunc= df_Baseline_total_decision_info_dict['decisionfunc'].abs().values.copy()
Baseline_total_y= df_Baseline_total_decision_info_dict['decisionfunc'].abs().values.copy()
# 如果是TD decision function是正的y軸就是正的，decision function是負的y軸就是負的
# 如果是ASD decision function是正的y軸就是負的，decision function是負的y軸就是正的
Baseline_y[Incorrect2Correct.loc[df_Baseline_changed_decision_info_dict.index]]\
    =-Baseline_y_decisionfunc[Incorrect2Correct.loc[df_Baseline_changed_decision_info_dict.index]]
Baseline_y[Correct2Incorrect.loc[df_Baseline_changed_decision_info_dict.index]]\
    =Baseline_y_decisionfunc[Correct2Incorrect.loc[df_Baseline_changed_decision_info_dict.index]]

df_Baseline_total_decision_info_dict.loc[Incorrect_baseline]

Baseline_total_y[Incorrect_baseline.values]=-Baseline_total_y_decisionfunc[Incorrect_baseline.values]
Baseline_total_y[Correct_baseline.values]=Baseline_total_y_decisionfunc[Correct_baseline.values]



Proposed_x= df_Proposed_changed_decision_info_dict['predictproba'].values
Proposed_y_decisionfunc= df_Proposed_changed_decision_info_dict['decisionfunc'].abs().values.copy()
Proposed_y= df_Proposed_changed_decision_info_dict['decisionfunc'].abs().values.copy()
Proposed_total_x= df_Proposed_total_decision_info_dict['predictproba'].values
Proposed_total_y_decisionfunc= df_Proposed_total_decision_info_dict['decisionfunc'].abs().values.copy()
Proposed_total_y= df_Proposed_total_decision_info_dict['decisionfunc'].abs().values.copy()

Proposed_y[Incorrect2Correct.loc[df_Baseline_changed_decision_info_dict.index]]=\
    Proposed_y_decisionfunc[Incorrect2Correct.loc[df_Baseline_changed_decision_info_dict.index]]
Proposed_y[Correct2Incorrect.loc[df_Baseline_changed_decision_info_dict.index]]=\
    -Proposed_y_decisionfunc[Correct2Incorrect.loc[df_Baseline_changed_decision_info_dict.index]]

Proposed_total_y[Incorrect_proposed.values]=-Proposed_total_y_decisionfunc[Incorrect_proposed.values]
Proposed_total_y[Correct_proposed.values]=Proposed_total_y_decisionfunc[Correct_proposed.values]

# Proposed_total_y[df_Proposed_total_decision_info_dict.loc[Incorrect_proposed]]\
#     =-df_Proposed_total_decision_info_dict.loc[Incorrect_proposed]
# Proposed_total_y[df_Proposed_total_decision_info_dict.loc[Incorrect_proposed]]\
#     =df_Proposed_total_decision_info_dict.loc[Correct_proposed]

Total_y=list(Baseline_y)+list(Proposed_y)
Total_x=list(Baseline_x)+list(Proposed_x)

y_max=np.max(Total_y)
y_min=np.min(Total_y)
x_max=np.max(Total_x)
x_min=np.min(Total_x)
x_middle=(x_max+x_min)/2
y_middle=(y_max+y_min)/2

# ax.annotate("", xy=(起點x, 起點y), xytext=(終點x, 終點y),arrowprops=dict(arrowstyle="->"))
for B_x, B_y, P_x, P_y,idx in zip(Baseline_x,Baseline_y,Proposed_x,Proposed_y,Sample_idxs_array):
    ax.annotate("", xy=(B_x, B_y), xytext=(P_x, P_y),arrowprops=dict(arrowstyle="<-",alpha=.4))
    # ax.text((B_x+P_x)/2, (B_y+P_y)/2, str(idx), fontsize=12)
    # ax.text(B_x, B_y, str(idx), fontsize=12)
    # ax.text(P_x, P_y, str(idx), fontsize=12)

plt.scatter(Baseline_x, Baseline_y, c='b', alpha=1)
plt.scatter(Proposed_x, Proposed_y, c='r', alpha=1)

plt.scatter(df_Baseline_total_decision_info_dict.predictproba, Baseline_total_y, c='b', alpha=.05)
plt.scatter(df_Proposed_total_decision_info_dict.predictproba, Proposed_total_y, c='r', alpha=.05)

# plt.scatter(df_Baseline_total_decision_info_dict.predictproba, df_Baseline_total_decision_info_dict.abs().decisionfunc, c='b', alpha=.5)
# plt.scatter(df_Proposed_total_decision_info_dict.predictproba, df_Proposed_total_decision_info_dict.abs().decisionfunc, c='r', alpha=.5)

ax.annotate('',xy=(0, 0), xytext=(1, 0),arrowprops=dict(arrowstyle="<->",alpha=1,))                                                                     
ax.annotate('',xy=(0.5, y_min), xytext=(0.5, y_max),arrowprops=dict(arrowstyle="<->",alpha=1,))
margin_y=(y_max-y_min)/10
margin_x=(1-0)/20

ax.text(0, y_middle-margin_y, 'ASD', fontsize=12)
ax.text(1-margin_x, y_middle-margin_y, 'TD', fontsize=12)
    
ax.text(0.5, y_min, 'Incorrect', fontsize=12)
ax.text(0.5, y_max-margin_y, 'Correct', fontsize=12)   
fig.patch.set_visible(True)
ax.axis('off')
# plt.ylim(-1.5,1.5)
plt.xlim(-0,1.1)
plt.title(experiment_title)
plt.show()



shap_values, df_XTest, keys=Prepare_data_for_summaryPlot(Proposed_changed_info_dict,\
                                                          feature_columns=None,\
                                                          PprNmeMp=PprNmeMp) 



shap.summary_plot(shap_values[1], df_XTest,feature_names=df_XTest.columns,show=False, max_display=8)
plt.title(experiment_title)
plt.show()


#%%
# 不重要的宣告放這邊
import string
import random
import json
from matplotlib import lines
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
class BaseVisualizer:
    pass 
def ensure_not_numpy(x):
    if isinstance(x, bytes):
        return x.decode()
    elif isinstance(x, np.str):
        return str(x)
    elif isinstance(x, np.generic):
        return float(x.item())
    else:
        return x
err_msg = """
<div style='color: #900; text-align: center;'>
  <b>Visualization omitted, Javascript library not loaded!</b><br>
  Have you run `initjs()` in this notebook? If this notebook was from another
  user you must also trust this notebook (File -> Trust notebook). If you are viewing
  this notebook on github the Javascript has been stripped for security. If you are using
  JupyterLab this error is because a JupyterLab extension has not yet been written.
</div>"""
def id_generator(size=20, chars=string.ascii_uppercase + string.digits):
    return "i"+''.join(random.choice(chars) for _ in range(size))
class Explanation:
    def __init__(self):
        pass
class AdditiveExplanation(Explanation):
    def __init__(self, base_value, out_value, effects, effects_var, instance, link, model, data):
        self.base_value = base_value
        self.out_value = out_value
        self.effects = effects
        self.effects_var = effects_var
        assert isinstance(instance, Instance)
        self.instance = instance
        assert isinstance(link, Link)
        self.link = link
        assert isinstance(model, Model)
        self.model = model
        assert isinstance(data, Data)
        self.data = data

def format_data(data):
    """Format data."""
    # Format negative features
    neg_features = np.array([[data['features'][x]['effect'],
                              data['features'][x]['value'],
                              data['featureNames'][x]]
                             for x in data['features'].keys() if data['features'][x]['effect'] < 0])
    
    neg_features = np.array(sorted(neg_features, key=lambda x: float(x[0]), reverse=False))
    
    # Format postive features
    pos_features = np.array([[data['features'][x]['effect'],
                              data['features'][x]['value'],
                              data['featureNames'][x]]
                             for x in data['features'].keys() if data['features'][x]['effect'] >= 0])
    pos_features = np.array(sorted(pos_features, key=lambda x: float(x[0]), reverse=True))
    
    # Define link function
    if data['link'] == 'identity':
        convert_func = lambda x: x
    elif data['link'] == 'logit':
        convert_func = lambda x: 1 / (1 + np.exp(-x))
    else:
        assert False, "ERROR: Unrecognized link function: " + str(data['link'])
    
    # Convert negative feature values to plot values
    neg_val = data['outValue']
    for i in neg_features:
        val = float(i[0])
        neg_val = neg_val + np.abs(val)
        i[0] = convert_func(neg_val)
    if len(neg_features) > 0:
        total_neg = np.max(neg_features[:, 0].astype(float)) - \
                    np.min(neg_features[:, 0].astype(float))
    else:
        total_neg = 0
    
    # Convert positive feature values to plot values
    pos_val = data['outValue']
    for i in pos_features:
        val = float(i[0])
        pos_val = pos_val - np.abs(val)
        i[0] = convert_func(pos_val)
        
    if len(pos_features) > 0:
        total_pos = np.max(pos_features[:, 0].astype(float)) - \
                    np.min(pos_features[:, 0].astype(float))
    else:
        total_pos = 0
    
    # Convert output value and base value
    data['outValue'] = convert_func(data['outValue'])
    data['baseValue'] = convert_func(data['baseValue'])
    
    return neg_features, total_neg, pos_features, total_pos
def update_axis_limits(ax, total_pos, pos_features, total_neg,
                       neg_features, base_value,
                       ylim=(-0.5,0.15),
                       ):
    ax.set_ylim(ylim)
    padding = np.max([np.abs(total_pos) * 0.2,
                      np.abs(total_neg) * 0.2])
    
    if len(pos_features) > 0:
        min_x = min(np.min(pos_features[:, 0].astype(float)), base_value) - padding
    else:
        min_x = 0
    if len(neg_features) > 0:
        max_x = max(np.max(neg_features[:, 0].astype(float)), base_value) + padding
    else:
        max_x = 0
    ax.set_xlim(min_x, max_x)

    plt.tick_params(top=True, bottom=False, left=False, right=False, labelleft=False,
                    labeltop=True, labelbottom=False)
    plt.locator_params(axis='x', nbins=12)

    for key, spine in zip(plt.gca().spines.keys(), plt.gca().spines.values()):
        if key != 'top':
            spine.set_visible(False)
def draw_bars(out_value, features, feature_type, width_separators, width_bar):
    """Draw the bars and separators."""
    rectangle_list = []
    separator_list = []
    
    pre_val = out_value
    for index, features in zip(range(len(features)), features):
        if feature_type == 'positive':
            left_bound = float(features[0])
            right_bound = pre_val
            pre_val = left_bound
            
            separator_indent = np.abs(width_separators)
            separator_pos = left_bound
            colors = ['#FF0D57', '#FFC3D5']
        else:
            left_bound = pre_val
            right_bound = float(features[0])
            pre_val = right_bound
            
            separator_indent = - np.abs(width_separators)
            separator_pos = right_bound
            colors = ['#1E88E5', '#D1E6FA']
        
        # Create rectangle
        if index == 0:
            if feature_type == 'positive':
                points_rectangle = [[left_bound, 0],
                                    [right_bound, 0],
                                    [right_bound, width_bar],
                                    [left_bound, width_bar],
                                    [left_bound + separator_indent, (width_bar / 2)]
                                    ]
            else:
                points_rectangle = [[right_bound, 0],
                                    [left_bound, 0],
                                    [left_bound, width_bar],
                                    [right_bound, width_bar],
                                    [right_bound + separator_indent, (width_bar / 2)]
                                    ]
        
        else:
            points_rectangle = [[left_bound, 0],
                                [right_bound, 0],
                                [right_bound + separator_indent * 0.90, (width_bar / 2)],
                                [right_bound, width_bar],
                                [left_bound, width_bar],
                                [left_bound + separator_indent * 0.90, (width_bar / 2)]]

        line = plt.Polygon(points_rectangle, closed=True, fill=True,
                           facecolor=colors[0], linewidth=0)
        rectangle_list += [line]

        # Create seperator
        points_separator = [[separator_pos, 0],
                            [separator_pos + separator_indent, (width_bar / 2)],
                            [separator_pos, width_bar]]
        
        line = plt.Polygon(points_separator, closed=None, fill=None,
                           edgecolor=colors[1], lw=3)
        separator_list += [line]

    return rectangle_list, separator_list
def draw_labels(fig, ax, out_value, features, feature_type, offset_text, total_effect=0, min_perc=0.05, text_rotation=0):
    start_text = out_value
    pre_val = out_value
    
    # Define variables specific to positive and negative effect features
    if feature_type == 'positive':
        colors = ['#FF0D57', '#FFC3D5']
        alignement = 'right'
        sign = 1
    else:
        colors = ['#1E88E5', '#D1E6FA']
        alignement = 'left'
        sign = -1
    
    # Draw initial line
    if feature_type == 'positive':
        x, y = np.array([[pre_val, pre_val], [0, -0.18]])
        line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
        line.set_clip_on(False)
        ax.add_line(line)
        start_text = pre_val
    
    box_end = out_value
    val = out_value
    for feature in features:
        # Exclude all labels that do not contribute at least 10% to the total
        feature_contribution = np.abs(float(feature[0]) - pre_val) / np.abs(total_effect)
        if feature_contribution < min_perc:
            break
        
        # Compute value for current feature
        val = float(feature[0])
        
        # Draw labels.
        if feature[1] == "":
            text = feature[2]
        else:
            text = feature[2] + ' = ' + feature[1]

        if text_rotation != 0:
            va_alignment = 'top'
        else:
            va_alignment = 'baseline'

        text_out_val = plt.text(start_text - sign * offset_text,
                                -0.15, text,
                                fontsize=12, color=colors[0],
                                horizontalalignment=alignement,
                                va=va_alignment,
                                rotation=text_rotation)
        text_out_val.set_bbox(dict(facecolor='none', edgecolor='none'))
        
        # We need to draw the plot to be able to get the size of the
        # text box
        fig.canvas.draw()
        box_size = text_out_val.get_bbox_patch().get_extents()\
                               .transformed(ax.transData.inverted())
        if feature_type == 'positive':
            box_end_ = box_size.get_points()[0][0]
        else:
            box_end_ = box_size.get_points()[1][0]
        
        # If the feature goes over the side of the plot, we remove that label
        # and stop drawing labels
        if box_end_ > ax.get_xlim()[1]:
            text_out_val.remove()
            break
        
        # Create end line
        if (sign * box_end_) > (sign * val):
            x, y = np.array([[val, val], [0, -0.18]])
            line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = val
            box_end = val

        else:
            box_end = box_end_ - sign * offset_text
            x, y = np.array([[val, box_end, box_end],
                             [0, -0.08, -0.18]])
            line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = box_end
        
        # Update previous value
        pre_val = float(feature[0])
            
    
    # Create line for labels
    extent_shading = [out_value, box_end, 0, -0.31]
    path = [[out_value, 0], [pre_val, 0], [box_end, -0.08],
            [box_end, -0.2], [out_value, -0.2],
            [out_value, 0]]
    
    path = Path(path)
    patch = PathPatch(path, facecolor='none', edgecolor='none')
    ax.add_patch(patch) 
    
    # Extend axis if needed
    lower_lim, upper_lim = ax.get_xlim()
    if (box_end < lower_lim):
        ax.set_xlim(box_end, upper_lim)
    
    if (box_end > upper_lim):
        ax.set_xlim(lower_lim, box_end)
        
    # Create shading
    if feature_type == 'positive':
        colors = np.array([(255, 13, 87), (255, 255, 255)]) / 255.
    else:
        colors = np.array([(30, 136, 229), (255, 255, 255)]) / 255.
    import matplotlib
    cm = matplotlib.colors.LinearSegmentedColormap.from_list('cm', colors)
    
    _, Z2 = np.meshgrid(np.linspace(0, 10), np.linspace(-10, 10))
    im = plt.imshow(Z2, interpolation='quadric', cmap=cm,
                    vmax=0.01, alpha=0.3,
                    origin='lower', extent=extent_shading,
                    clip_path=patch, clip_on=True, aspect='auto')
    im.set_clip_path(patch)
    
    return fig, ax
def draw_higher_lower_element(out_value, offset_text, fontsize=13):
    plt.text(out_value - offset_text, 0.405, 'higher',
             fontsize=fontsize, color='#FF0D57',
             horizontalalignment='right')

    plt.text(out_value + offset_text, 0.405, 'lower',
             fontsize=fontsize, color='#1E88E5',
             horizontalalignment='left')
    
    plt.text(out_value, 0.4, r'$\leftarrow$',
             fontsize=fontsize, color='#1E88E5',
             horizontalalignment='center')
    
    plt.text(out_value, 0.425, r'$\rightarrow$',
             fontsize=fontsize, color='#FF0D57',
             horizontalalignment='center')
def draw_base_element(base_value, ax):
    x, y = np.array([[base_value, base_value], [0.13, 0.25]])
    line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
    line.set_clip_on(False)
    ax.add_line(line)
    
    text_out_val = plt.text(base_value, 0.33, 'base value',
                            fontsize=12, alpha=0.5,
                            horizontalalignment='center')
    text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))
def draw_output_element(out_name, out_value, ax,
                        outValue_y_position=0.25,
                        ):
    # Add output value
    x, y = np.array([[out_value, out_value], [0, 0.24]])
    line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
    line.set_clip_on(False)
    ax.add_line(line)
    
    font0 = FontProperties()
    font = font0.copy()
    font.set_weight('bold')
    text_out_val = plt.text(out_value, outValue_y_position, '{0:.2f}'.format(out_value),
                            fontproperties=font,
                            fontsize=14,
                            horizontalalignment='center')
    text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))
    
    text_out_val = plt.text(out_value, 0.33, out_name,
                            fontsize=12, alpha=0.5,
                            horizontalalignment='center')
    text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))
SHAP_save_path_root="SHAP_figures/{quadrant}/"


import shutil
from collections import Counter
shutil.rmtree(SHAP_save_path_root.format(quadrant=""), ignore_errors = True)

Analysis_grp_bool=False
N=5
Xtest_dict={}
expected_value_lst=[]
UsePaperName_bool=True
Quadrant_FeatureImportance_dict={}
Quadrant_feature_AddedTopFive_dict={}
Quadrant_feature_AddedFeatureImportance_dict={}
Manual_inspect_idxs=[24]
# for Analysis_grp_str in ['Manual_inspect_idxs','quadrant1_indexes','quadrant2_indexes','quadrant3_indexes','quadrant4_indexes']:
# for Analysis_grp_str in ['quadrant1_indexes','quadrant2_indexes','quadrant3_indexes','quadrant4_indexes']:
# for Analysis_grp_str in ['quadrant2_indexes']:
for Analysis_grp_str in ['Manual_inspect_idxs']:
    Analysis_grp_indexes=vars()[Analysis_grp_str]
    df_shap_values_stacked=pd.DataFrame([])
    for Inspect_samp in Analysis_grp_indexes:
        shap_info=Proposed_changed_info_dict[Inspect_samp]
        
        expected_value=shap_info['explainer_expected_value'][logit_number]
        shap_values=shap_info['shap_values'].loc[logit_number].values
        df_shap_values=shap_info['shap_values'].loc[[logit_number]]
        df_shap_values.index=[Inspect_samp]
        
        df_shap_values_T=df_shap_values.T
        
        Xtest=shap_info['XTest']
        
        Xtest_dict[Inspect_samp]=Xtest
        df_shap_values_stacked=pd.concat([df_shap_values_stacked,df_shap_values],)
        expected_value_lst.append(expected_value)
        
        Xtest.index=[ Swap2PaperName(name,PprNmeMp) for name in Xtest.index]
        # Xtest.index=[ repr(Swap2PaperName(name,PprNmeMp)) for name in Xtest.index]
        
        
        # 這個部份跑TASLP的Fig.7 也就是說明有些不顯著的feature卻shap value很高
        SHAP_save_path=SHAP_save_path_root.format(quadrant=Analysis_grp_str)
        if not os.path.exists(SHAP_save_path):
            os.makedirs(SHAP_save_path)
        # ============================================================================= 
        # 魔改force plot
        # fig = shap.force_plot(expected_value, df_shap_values.values, Xtest.round(2).T, figsize=(8, 3) ,matplotlib=True,show=False)
        
        from shap.plots._labels import labels
        from shap.utils._legacy import convert_to_link, Instance, Model, Data, DenseData, Link
        from shap.plots._force_matplotlib import draw_additive_plot
        import scipy as sp
        
        
        class AdditiveForceVisualizer(BaseVisualizer):
            def __init__(self, e, plot_cmap="RdBu"):
                assert isinstance(e, AdditiveExplanation), \
                    "AdditiveForceVisualizer can only visualize AdditiveExplanation objects!"

                # build the json data
                features = {}
                for i in filter(lambda j: e.effects[j] != 0, range(len(e.data.group_names))):
                    features[i] = {
                        "effect": ensure_not_numpy(e.effects[i]),
                        "value": ensure_not_numpy(e.instance.group_display_values[i])
                    }
                self.data = {
                    "outNames": e.model.out_names,
                    "baseValue": ensure_not_numpy(e.base_value),
                    "outValue": ensure_not_numpy(e.out_value),
                    "link": str(e.link),
                    "featureNames": e.data.group_names,
                    "features": features,
                    "plot_cmap": plot_cmap
                }

            def html(self, label_margin=20):
                # assert have_ipython, "IPython must be installed to use this visualizer! Run `pip install ipython` and then restart shap."
                self.data["labelMargin"] = label_margin
                return """
        <div id='{id}'>{err_msg}</div>
         <script>
           if (window.SHAP) SHAP.ReactDom.render(
            SHAP.React.createElement(SHAP.AdditiveForceVisualizer, {data}),
            document.getElementById('{id}')
          );
        </script>""".format(err_msg=err_msg, data=json.dumps(self.data), id=id_generator())
            
            def matplotlib(self, figsize, show, text_rotation, min_perc=0.05):
                fig = draw_additive_plot(self.data,
                                         figsize=figsize,
                                         show=show,
                                         text_rotation=text_rotation,
                                         min_perc=min_perc)
                
                return fig
            
            def _repr_html_(self):
                return self.html()

        
        
        def verify_valid_cmap(cmap):
            assert (isinstance(cmap, str) or isinstance(cmap, list) or str(type(cmap)).endswith("unicode'>")
                ),"Plot color map must be string or list! not: " + str(type(cmap))
            if isinstance(cmap, list):
                assert (len(cmap) > 1), "Color map must be at least two colors."
                _rgbstring = re.compile(r'#[a-fA-F0-9]{6}$')
                for color in cmap:
                     assert(bool(_rgbstring.match(color))),"Invalid color found in CMAP."

            return cmap
        
        def visualize(e, plot_cmap="RdBu", matplotlib=False, figsize=(20,3), show=True,
                      ordering_keys=None, ordering_keys_time_format=None, text_rotation=0, min_perc=0.05):
            plot_cmap = verify_valid_cmap(plot_cmap)
            if isinstance(e, AdditiveExplanation):
                if matplotlib:
                    return AdditiveForceVisualizer(e, plot_cmap=plot_cmap).matplotlib(figsize=figsize,
                                                                            show=show,
                                                                            text_rotation=text_rotation,
                                                                            min_perc=min_perc)
            # 如果code出錯是因為應該要跑到這邊的話，就把槓掉的部份取消掉
            #     else:
            #         return AdditiveForceVisualizer(e, plot_cmap=plot_cmap)
            # elif isinstance(e, Explanation):
            #     if matplotlib:
            #         assert False, "Matplotlib plot is only supported for additive explanations"
            #     else:
            #         return SimpleListVisualizer(e)
            # elif isinstance(e, Sequence) and len(e) > 0 and isinstance(e[0], AdditiveExplanation):
            #     if matplotlib:
            #         assert False, "Matplotlib plot is only supported for additive explanations"
            #     else:
            #         return AdditiveForceArrayVisualizer(e, plot_cmap=plot_cmap, ordering_keys=ordering_keys, ordering_keys_time_format=ordering_keys_time_format)
            # else:
            #     assert False, "visualize() can only display Explanation objects (or arrays of them)!"
        
        # base_value, shap_values, features =\
        #     expected_value, df_shap_values.values, Xtest.round(2).T
        base_value, shap_values, feature_names =\
            expected_value, df_shap_values.values, list(Xtest.index)
        
        # features=Xtest.round(2).T
        # feature_names=None
        out_names=None
        link="identity"
        plot_cmap="RdBu"
        ordering_keys=None
        ordering_keys_time_format=None
        text_rotation=0
        contribution_threshold=0.075
        matplotlib=True
        show=False
        figsize=(8, 3) 
        ##################################  參數宣告完畢 ######################
        # support passing an explanation object
        if str(type(base_value)).endswith("Explanation'>"):
            shap_exp = base_value
            base_value = shap_exp.base_values
            shap_values = shap_exp.values
            if features is None:
                if shap_exp.display_data is None:
                    features = shap_exp.data
                else:
                    features = shap_exp.display_data
            if sp.sparse.issparse(features):
                features = features.toarray().flatten()
            if feature_names is None:
                feature_names = shap_exp.feature_names
            # if out_names is None: # TODO: waiting for slicer support of this
            #     out_names = shap_exp.output_names

        # auto unwrap the base_value
        if isinstance(base_value, np.ndarray):
            if len(base_value) == 1:
                base_value = base_value[0]
            elif len(base_value) > 1 and np.all(base_value == base_value[0]):
                base_value = base_value[0]

        if (isinstance(base_value, np.ndarray) or type(base_value) == list):
            if not isinstance(shap_values, list) or len(shap_values) != len(base_value):
                raise Exception("In v0.20 force_plot now requires the base value as the first parameter! " \
                                "Try shap.force_plot(explainer.expected_value, shap_values) or " \
                                "for multi-output models try " \
                                "shap.force_plot(explainer.expected_value[0], shap_values[0]).")


        assert not type(shap_values) == list, "The shap_values arg looks looks multi output, try shap_values[i]."

        link = convert_to_link(link)

        if type(shap_values) != np.ndarray:
            fig= visualize(shap_values)
            print("[DEBUG] force function stop at if type(shap_values) != np.ndarray" , )

        # convert from a DataFrame or other types
        if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
            if feature_names is None:
                feature_names = list(features.columns)
            features = features.values
        elif str(type(features)) == "<class 'pandas.core.series.Series'>":
            if feature_names is None:
                feature_names = list(features.index)
            features = features.values
        elif isinstance(features, list):
            if feature_names is None:
                feature_names = features
            features = None
        elif features is not None and len(features.shape) == 1 and feature_names is None:
            feature_names = features
            features = None

        if len(shap_values.shape) == 1:
            shap_values = np.reshape(shap_values, (1, len(shap_values)))

        if out_names is None:
            out_names = ["f(x)"]
        elif type(out_names) == str:
            out_names = [out_names]

        if shap_values.shape[0] == 1:
            if feature_names is None:
                feature_names = [labels['FEATURE'] % str(i) for i in range(shap_values.shape[1])]
            if features is None:
                features = ["" for _ in range(len(feature_names))]
            if type(features) == np.ndarray:
                features = features.flatten()

            # check that the shape of the shap_values and features match
            if len(features) != shap_values.shape[1]:
                msg = "Length of features is not equal to the length of shap_values!"
                if len(features) == shap_values.shape[1] - 1:
                    msg += " You might be using an old format shap_values array with the base value " \
                           "as the last column. In this case just pass the array without the last column."
                raise Exception(msg)

        instance = Instance(np.zeros((1, len(feature_names))), features)
        e = AdditiveExplanation(
            base_value,
            np.sum(shap_values[0, :]) + base_value,
            shap_values[0, :],
            None,
            instance,
            link,
            Model(None, out_names),
            DenseData(np.zeros((1, len(feature_names))), list(feature_names))
        )
        
        
        # =============================================================================
        ''' 夢的第二層，在visualize裡面 '''         

        # fig = visualize(e,
        #                  plot_cmap,
        #                  matplotlib,
        #                  figsize=figsize,
        #                  show=show,
        #                  text_rotation=text_rotation,
        #                  min_perc=contribution_threshold)
        
        
        # def visualize(e, plot_cmap="RdBu", matplotlib=False, figsize=(20,3), show=True,
        #               ordering_keys=None, ordering_keys_time_format=None, text_rotation=0, min_perc=0.05):
            
        e, plot_cmap, matplotlib, figsize, show=\
            e, plot_cmap, matplotlib, figsize, show
                      

        ordering_keys=None 
        ordering_keys_time_format=None
        
        text_rotation, min_perc=\
            text_rotation, contribution_threshold
        
        
        plot_cmap = verify_valid_cmap(plot_cmap)
        if isinstance(e, AdditiveExplanation):
            if matplotlib:
                # fig= AdditiveForceVisualizer(e, plot_cmap=plot_cmap).matplotlib(figsize=figsize,
                #                                                         show=show,
                #                                                         text_rotation=text_rotation,
                #                                                         min_perc=min_perc)
                AddForcVis=AdditiveForceVisualizer(e, plot_cmap=plot_cmap)
                # =============================================================================
                ''' 夢的第三層，在matplotlib裡面 '''
                fig=AddForcVis.matplotlib(figsize=figsize,
                                        show=show,
                                        text_rotation=text_rotation,
                                        min_perc=min_perc)
                
                
                figsize, show, text_rotation, min_perc=\
                    figsize, show, text_rotation, min_perc
                
                
                # =============================================================================
                ''' 夢的第四層，在draw_additive_plot裡面 '''
                fig = draw_additive_plot(AddForcVis.data,
                                         figsize=figsize,
                                         show=show,
                                         text_rotation=text_rotation,
                                         min_perc=min_perc)
                data, figsize, show, text_rotation, min_perc=\
                    AddForcVis.data, figsize, show, text_rotation, min_perc
                    
                """Draw additive plot."""
                # Turn off interactive plot
                if show is False:
                    plt.ioff()
                
                # Format data
                neg_features, total_neg, pos_features, total_pos = format_data(data)
                
                # Compute overall metrics
                base_value = data['baseValue']
                out_value = data['outValue']
                offset_text = (np.abs(total_neg) + np.abs(total_pos)) * 0.04
                
                # Define plots
                fig, ax = plt.subplots(figsize=figsize)
                
                # 這邊畫圖的邊界 (x_lim, y_lim)
                # Compute axis limit
                update_axis_limits(ax, total_pos, pos_features, total_neg,
                                   neg_features, base_value,
                                   ylim=(-0.2,0.15))
                
                # Define width of bar
                width_bar = 0.1
                width_separators = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 200
                
                # Create bar for negative shap values
                rectangle_list, separator_list = draw_bars(out_value, neg_features, 'negative',
                                                           width_separators, width_bar)
                for i in rectangle_list:
                    ax.add_patch(i)
                
                for i in separator_list:
                    ax.add_patch(i)
                
                # Create bar for positive shap values
                rectangle_list, separator_list = draw_bars(out_value, pos_features, 'positive',
                                                           width_separators, width_bar)
                for i in rectangle_list:
                    ax.add_patch(i)
                
                for i in separator_list:
                    ax.add_patch(i)

                # Add labels
                total_effect = np.abs(total_neg) + total_pos
                fig, ax = draw_labels(fig, ax, out_value, neg_features, 'negative',
                                      offset_text, total_effect, min_perc=min_perc, text_rotation=text_rotation)
                
                fig, ax = draw_labels(fig, ax, out_value, pos_features, 'positive',
                                      offset_text, total_effect, min_perc=min_perc, text_rotation=text_rotation)
                
                
                
                # higher lower legend
                draw_higher_lower_element(out_value, offset_text, fontsize=TEXTFONTSIZE)
                
                # Debug用的
                # import matplotlib
                # print("Font family", matplotlib.rcParams['font.family'])
                
                
                # 這個函數會畫base value的label在圖上面
                # draw_base_element(base_value, ax)
                
                # Add output label
                out_names = "Probability of predicting TD"
                # out_names = data['outNames'][0]
                draw_output_element(out_names, out_value, ax,
                                    outValue_y_position=0.20)
                
                
                # Scale axis
                if data['link'] == 'logit':
                    plt.xscale('logit')
                    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                    ax.ticklabel_format(style='plain')

                if show:
                    plt.show()
                else:
                    fig= plt.gcf()
                
                # =============================================================================
                # =============================================================================
                
                
                
                
        ''' 夢的第二層，在visualize裡面 '''
        # =============================================================================
        # =============================================================================
        
        
        
        
        fig=plt.gcf()
        # equal_textwidth=4.7747/3 #the size of latex textwidth
        equal_textwidth=4.7747 #the size of latex textwidth
        figure_ratio=0.6
        figsize_Textextwidth_tmp=(equal_textwidth, 0.8)
        figsize_Textextwidth=tuple(ti/figure_ratio for ti in figsize_Textextwidth_tmp)
        fig.set_size_inches(figsize_Textextwidth) #這個比例很好，三個圖都這樣fix住
        
        plt.savefig("images/SHAP_discussion_{sample}.png".format(sample=Inspect_samp),dpi=400, bbox_inches='tight')
        plt.savefig(SHAP_save_path+"{sample}.png".format(sample=Inspect_samp),dpi=150, bbox_inches='tight')
        # plt.savefig(SHAP_save_path+"{sample}.png".format(sample=Inspect_samp),dpi=150)
        
        Lists_of_addedFeatures=[getattr(FeatSel,k)  for k in Additional_featureSet]
        Lists_of_addedFeatures_flatten=[e for ee in Lists_of_addedFeatures for e in ee]
        df_FeatureImportance_AddedFeatures=df_shap_values[Lists_of_addedFeatures_flatten].T
        df_FeatureImportance_AddedFeatures_absSorted=df_FeatureImportance_AddedFeatures.abs()[Inspect_samp].sort_values(ascending=False)
        
        
        
        df_addedFeatures_TopN=df_FeatureImportance_AddedFeatures.loc[df_FeatureImportance_AddedFeatures_absSorted.head(N).index]
        # Quadrant_feature_AddedTopFive_dict[Inspect_samp]=df_addedFeatures_TopN
        Quadrant_feature_AddedFeatureImportance_dict[Inspect_samp]=df_FeatureImportance_AddedFeatures[Inspect_samp].sort_values(ascending=False)
        
    if UsePaperName_bool==False:
            df_shap_values_stacked.columns=[Swap2PaperName(idx, PprNmeMp) for idx in df_shap_values_stacked.columns]
    
    
    
    ''' Analyses of per quadrants '''
    if len(Analysis_grp_indexes)>0 and Analysis_grp_bool == True:
        # =============================================================================
        # Feature importance
        # =============================================================================
        df_shap_values_stacked_T=df_shap_values_stacked.T
        df_FeatureImportance=df_shap_values_stacked_T.abs().sum(axis=1).sort_values(ascending=False)

        df_TopN=df_FeatureImportance.head(N)
        Quadrant_FeatureImportance_dict[Analysis_grp_str]=df_FeatureImportance
        df_shap_values_stacked[df_FeatureImportance.index].T
        
        # =============================================================================
        # Count Top N occurence and plot histogram
        N=5
        # =============================================================================
        df_shap_values_stacked_T=df_shap_values_stacked.T
        N_totalpeople=len(df_shap_values_stacked_T.columns)
        TopN_dict={}
        LowN_dict={}
        for Inspect_samp in df_shap_values_stacked_T.columns:
            LowestN=df_shap_values_stacked_T[[Inspect_samp]].sort_values(by=[Inspect_samp])[:N].index
            TopN=df_shap_values_stacked_T[[Inspect_samp]].sort_values(by=[Inspect_samp])[-N:].index
            TopN_dict[Inspect_samp]=TopN
            LowN_dict[Inspect_samp]=LowestN
        df_TopN_dict=pd.DataFrame.from_dict(TopN_dict)
        df_LowN_dict=pd.DataFrame.from_dict(LowN_dict)
    
        TopN_total_feat_array=df_TopN_dict.to_numpy().flatten()
        TopN_total_feat_counts_dict = Counter(TopN_total_feat_array)
        
        df_TopN_total_feat_counts = pd.DataFrame.from_dict(TopN_total_feat_counts_dict, orient='index')
        df_TopN_total_feat_counts.columns=['feature_counts']
        df_TopN_total_feat_percentage=df_TopN_total_feat_counts/N_totalpeople*100
        ax = df_TopN_total_feat_percentage.plot(kind='bar')
        ax.set_ylabel("percentage")
        ax.set_xlabel("feature")


#%%
def ReorganizeFeatures4SummaryPlot(shap_values_logit, df_XTest,FeatureSet_lst):
    # step1 convert shapvalue to df_shapvalues
    df_shap_values=pd.DataFrame(shap_values_logit,columns=df_XTest.columns)
    # step2 Categorize according to FeatureSet_lst
    df_Reorganized_shap_values=pd.DataFrame()
    df_Reorganized_XTest=pd.DataFrame()
    for FSL in FeatureSet_lst:
        FSL_papercolumns=[Swap2PaperName(k,PprNmeMp) for k in FeatSel.CategoricalName2cols[FSL]]
    
        df_Reorganized_shap_values=pd.concat([df_Reorganized_shap_values,df_shap_values[FSL_papercolumns]],axis=1)
        df_Reorganized_XTest=pd.concat([df_Reorganized_XTest,df_XTest[FSL_papercolumns]],axis=1)
    assert df_Reorganized_shap_values.shape == df_Reorganized_XTest.shape
    return df_Reorganized_shap_values.values, df_Reorganized_XTest


# =============================================================================
# 
#%%
# Show feature importance TopN
shap_values, df_XTest, keys=Prepare_data_for_summaryPlot(Proposed_totalPoeple_info_dict,\
                                                          feature_columns=None,\
                                                          PprNmeMp=PprNmeMp)


df_shapValues=pd.DataFrame(shap_values[logit_number],columns=df_XTest.columns)
df_shapValues_sum=df_shapValues.abs().sum(axis=0)
df_shapValues_sum_norm=df_shapValues_sum/df_shapValues_sum.sum() * 100
# df_shapValues_sum_norm_percentage=\
#     (df_shapValues_sum_norm-df_shapValues_sum_norm.min())/(df_shapValues_sum_norm.max()-df_shapValues_sum_norm.min())
df_shapValues_sum_normTopN=df_shapValues_sum_norm.sort_values(ascending=False).iloc[:N]
df_shapValues_sum_normSorted=df_shapValues_sum_norm.sort_values(ascending=False)
df_shapValues_sum_normSortedForPlotting=pd.DataFrame(df_shapValues_sum_normSorted.T.values,columns=['shap_val'],index=df_shapValues_sum_normSorted.index)
switch2paperName_bool=True
if switch2paperName_bool==True:
    df_shapValues_sum_normSortedForPlotting.index=[Swap2PaperName(n, PprNmeMp) for n in df_shapValues_sum_normSortedForPlotting.index]

# sns.histplot(df_shapValues_sum_normSortedForPlotting, x="shap_val", element="poly")
plt.figure(figsize=(16,8))
x=[]
count=1
for n in df_shapValues_sum_normSortedForPlotting.index:
    if count< 5:
        x.append(Swap2PaperName(n, PprNmeMp))
    else:
        x.append('{}'.format(count))
    count+=1

y3 = df_shapValues_sum_normSortedForPlotting.values.reshape(-1)
ax = sns.barplot(x=x, y=y3, palette="deep")#, ax=ax3)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
#%%
# =============================================================================

# inspect_featuresets='Trend[LOCDEP]_d'  #Trend[LOCDEP]d + Proximity[phonation]
# inspect_featuresets='LOCDEP_Trend_D_cols'  #LOC_columns + Syncrony[phonation]
inspect_featuresets='Vowel_dispersion'
shap_values, df_XTest, keys=Prepare_data_for_summaryPlot(Proposed_changed_info_dict,\
                                                          feature_columns=(FeatSel.CategoricalName2cols)[inspect_featuresets],\
                                                          PprNmeMp=PprNmeMp)


# FeatureSet_lst=['Trend[Vowel_dispersion_inter__vowel_centralization]_d','Trend[Vowel_dispersion_inter__vowel_dispersion]_d',\
#                 'Trend[Vowel_dispersion_intra]_d','Trend[formant_dependency]_d']     #Trend[LOCDEP]d + P roximity[phonation]
# FeatureSet_lst=['Vowel_dispersion_inter__vowel_centralization','Vowel_dispersion_inter__vowel_dispersion',\
#                 ]     #LOC + P roximity[phonation]

FeatureSet_lst=[
'Vowel_dispersion_inter__vowel_centralization','Vowel_dispersion_inter__vowel_dispersion',
]    # 'LOC_columns', 'Phonation_Trend_D_cols'
###########################################
# 檢查feature值與model output的方向關係
# 你可以看圖（summary plot）
shap_values, df_XTest, keys=Prepare_data_for_summaryPlot(Proposed_totalPoeple_info_dict,\
                                                          feature_columns=FeatSel.CategoricalName2cols[inspect_featuresets],\
                                                          PprNmeMp=PprNmeMp)


Reorganized_shap_values, df_Reorganized_XTest=ReorganizeFeatures4SummaryPlot(shap_values[1], df_XTest, FeatureSet_lst)
shap.summary_plot(Reorganized_shap_values, df_Reorganized_XTest,show=False, max_display=len(df_XTest.columns),sort=False)
plt.title(experiment_title)
plt.show()

# 也可以看Correlation 值
df_Pcorrela_FeaturesnSHAPVal=Calculate_XTestShape_correlation(Proposed_totalPoeple_info_dict,logit_number=logit_number)


# 看Feature 值 & SHAP值
inspect_idxs=quadrant1_indexes + quadrant3_indexes
Feature_SHAP_info_dict=Organize_Needed_SHAP_info(inspect_idxs, Session_level_all, proposed_expstr)
df_XTest_stacked,df_ShapValues_stacked=Return_ALigned_dfXtestndfShapValues(Feature_SHAP_info_dict,logit_number=logit_number)

# Scatter plot 

df_X_featureRank=pd.DataFrame()
for col in df_XTest.columns:
    df_rank_features=df_XTest[col].argsort()  #從小排到大
    df_rank=pd.Series(df_rank_features.index.values, index=df_rank_features ,name=df_rank_features.name).sort_index()
    
    df_X_featureRank=pd.concat([df_X_featureRank,df_rank],axis=1)
    df_X_featureRank_T=df_X_featureRank.T

#%%
#////////////////////////////////////////////////////////////////////////


# //////
# def SummaryPlot_Category(Poeple_info_dict,proposed_expstr,FeatureSet_lst,experiment_title,module):
#     ##################################### 統合feautre set values
#     # df_Baseline_shap_values=Get_Inspected_SHAP_df(Baseline_totalPoeple_info_dict,logits=[ASDTD2Logit_map['TD']]) [ASDTD2Logit_map['TD']]
#     # df_Proposed_shap_values=Get_Inspected_SHAP_df(Poeple_info_dict,logits=[ASDTD2Logit_map['TD']]) [ASDTD2Logit_map['TD']]
#     shap_values, df_XTest, keys=Prepare_data_for_summaryPlot(Poeple_info_dict,\
#                                                               feature_columns=None,\
#                                                               PprNmeMp=None)
    
#     df_shap_values=pd.DataFrame(shap_values[ASDTD2Logit_map['TD']],columns=df_XTest.columns,index=df_XTest.index)
    
#     df_FeaSet_avg_Comparison_proposed=Calculate_sum_of_SHAP_vals(df_shap_values,FeatSel_module=module,FeatureSet_lst=FeatureSet_lst)
#     shap.summary_plot(df_FeaSet_avg_Comparison_proposed.drop(index=['Average','abs_Average']).values,\
#                       feature_names=df_FeaSet_avg_Comparison_proposed.columns,show=False, max_display=10)
#     plt.title(experiment_title)
#     plt.show()
# SummaryPlot_Category(Proposed_changed_info_dict,proposed_expstr,FeatureSet_lst,experiment_title,module=FeatSel.CategoricalName2cols)

#%%
# shap_values, df_XTest, keys=Prepare_data_for_summaryPlot(Proposed_totalPoeple_info_dict,\
#                                                           feature_columns=FeatSel.CategoricalName2cols['Trend[LOCDEP]_d'],\
#                                                           PprNmeMp=None)
# print(Get_Top_10_abs_SHAP_columns(shap_values[1],df_XTest).index)
def Calculate_sum_of_SHAP_vals(df_values,FeatSel_module,FeatureSet_lst=['Phonation_Proximity_cols']):
    df_FeaSet_avg_Comparison=pd.DataFrame([],columns=FeatureSet_lst)
    for feat_set in FeatureSet_lst:
        if inspect.ismodule(FeatSel_module): #FeatSel_module is a python module
            feature_cols = getattr(FeatSel_module, feat_set)
        elif type(FeatSel_module) == dict:
            feature_cols = FeatSel_module[feat_set]
        else:
            raise TypeError()
        df_FeaSet_avg_Comparison[feat_set]=df_values.loc[feature_cols,:].sum()
    return df_FeaSet_avg_Comparison


def Get_Inspected_SHAP_df(Info_dict,logits=[0,1]):
    Top_shap_values_collect=Dict()
    for logit_number in logits:
        Top_shap_values_collect[logit_number]=pd.DataFrame()
        
        for Inspect_samp in Info_dict.keys():
            shap_info=Info_dict[Inspect_samp]
            df_shap_values=shap_info['shap_values'].loc[[logit_number]].T
            df_shap_values.columns=[Inspect_samp]
            Top_shap_values_collect[logit_number]=pd.concat([Top_shap_values_collect[logit_number],df_shap_values],axis=1)
            # df_shap_values.columns=[]
            # Xtest=shap_info['XTest']
            # column2idx_dict={idx:str(i) for i,idx in enumerate(Xtest.index)}
            # df_column2idx_dict=pd.DataFrame.from_dict(column2idx_dict,orient='index')
            # df_shap_values['feature_idxs']=df_column2idx_dict
        Top_shap_values_collect[logit_number]['Average']=Top_shap_values_collect[logit_number].mean(axis=1)
        Top_shap_values_collect[logit_number]['abs_Average']=Top_shap_values_collect[logit_number].abs().mean(axis=1)
        Top_shap_values_collect[logit_number]=Top_shap_values_collect[logit_number].sort_values(by='Average')
    return Top_shap_values_collect

Baseline_info_dict=Organize_Needed_SHAP_info(Incorrect2Correct_indexes+Correct2Incorrect_indexes, Session_level_all, baseline_expstr)
Proposed_info_dict=Organize_Needed_SHAP_info(Incorrect2Correct_indexes+Correct2Incorrect_indexes, Session_level_all, proposed_expstr)


Baseline_shap_values=Get_Inspected_SHAP_df(Baseline_info_dict,logits=[ASDTD2Logit_map['TD']]) [ASDTD2Logit_map['TD']]
Proposed_shap_values=Get_Inspected_SHAP_df(Proposed_info_dict,logits=[ASDTD2Logit_map['TD']]) [ASDTD2Logit_map['TD']]
# =============================================================================
# 分析統計整體SHAP value的平均
#Baseline model    
baseline_featset_lst=baseline_expstr[re.search(" >> ",baseline_expstr).end():re.search("::",baseline_expstr).start()].split("+")
proposed_featset_lst=proposed_expstr[re.search(" >> ",proposed_expstr).end():re.search("::",proposed_expstr).start()].split("+")


df_FeaSet_avg_Comparison_baseline=Calculate_sum_of_SHAP_vals(Baseline_shap_values,FeatSel_module=FeatSel,FeatureSet_lst=baseline_featset_lst)
df_FeaSet_avg_Comparison_proposed=Calculate_sum_of_SHAP_vals(Proposed_shap_values,FeatSel_module=FeatSel,FeatureSet_lst=proposed_featset_lst)

assert (df_FeaSet_avg_Comparison_proposed.loc[Ones2Twos_indexes,proposed_featset_lst[0]] >= 0).all()
assert (df_FeaSet_avg_Comparison_proposed.loc[Twos2Ones_indexes,proposed_featset_lst[0]] <= 0).all()



# =============================================================================
'''

    Check feature importance

'''
# =============================================================================

Proposed_changed_info_dict=Organize_Needed_SHAP_info(Incorrect2Correct_indexes+Correct2Incorrect_indexes, Session_level_all, proposed_expstr)
Proposed_All_info_dict=Organize_Needed_SHAP_info(df_Y_pred.index, Session_level_all, proposed_expstr)

Proposed_changed_shap_values=Get_Inspected_SHAP_df(Proposed_changed_info_dict,logits=[ASDTD2Logit_map['TD']]) [ASDTD2Logit_map['TD']]
Proposed_All_shap_values=Get_Inspected_SHAP_df(Proposed_All_info_dict,logits=[ASDTD2Logit_map['TD']]) [ASDTD2Logit_map['TD']]

df_catagorical_featImportance=pd.DataFrame()
df_catagorical_featImportance.name='CategoricalFeatureImportance'
for shap_valdfs in ['Proposed_changed_shap_values','Proposed_All_shap_values']:
    for key in FeatSel.CategoricalName2cols.keys():
        #feature importance is defined as absolute average of SHAP values 
        #refer to https://christophm.github.io/interpretable-ml-book/shap.html#shap-feature-importance
        df_catagorical_featImportance.loc[key,shap_valdfs]=vars()[shap_valdfs].loc[FeatSel.CategoricalName2cols[key],'abs_Average'].round(2).sum()
print(df_catagorical_featImportance)


df_FeaSet_avg_Comparison_baseline=Calculate_sum_of_SHAP_vals(Baseline_shap_values,FeatSel_module=FeatSel,FeatureSet_lst=baseline_featset_lst)
df_FeaSet_avg_Comparison_proposed=Calculate_sum_of_SHAP_vals(Proposed_shap_values,FeatSel_module=FeatSel,FeatureSet_lst=proposed_featset_lst)


# df_FeaSet_deltaOne2Twos_avg_Comparison=df_FeaSet_avg_Comparison_proposed.loc[Ones2Twos_indexes,baseline_featset_lst] - \
#     df_FeaSet_avg_Comparison_baseline.loc[Ones2Twos_indexes,baseline_featset_lst]
# df_FeaSet_deltaTwos2Ones_avg_Comparison=df_FeaSet_avg_Comparison_proposed.loc[Twos2Ones_indexes,baseline_featset_lst] - \
#     df_FeaSet_avg_Comparison_baseline.loc[Twos2Ones_indexes,baseline_featset_lst]
# assert (df_FeaSet_deltaOne2Twos_avg_Comparison >=0).values.all()
# assert (df_FeaSet_deltaTwos2Ones_avg_Comparison <=0).values.all()
# df_values=df_Type1Err_proposed_shap_values
# #Proposed model
# proposed_featset_lst=proposed_expstr[re.search(" >> ",proposed_expstr).end():re.search("::",proposed_expstr).start()].split("+")
# for feat_set in proposed_featset_lst:
#     feature_cols = getattr(FeatSel, feat_set)
#     df_FeaSet_avg_Comparison.loc[feat_set,'Average_proposed']=df_values.loc[feature_cols,'Average'].sum()
# # =============================================================================


# # 對Type2來說**鑑別性資訊**就是幫助推向實際上是陽性（ASD）的那個logit
# df_Type2Err_baseline_shap_values=Get_Inspected_SHAP_df(Type2Err_baseline_info_dict,logits=[ASDTD2Logit_map['ASD']]) [ASDTD2Logit_map['ASD']]
# df_Type2Err_proposed_shap_values=Get_Inspected_SHAP_df(Type2Err_proposed_info_dict,logits=[ASDTD2Logit_map['ASD']]) [ASDTD2Logit_map['ASD']]


# =============================================================================
''' Plotting area


'''
# =============================================================================
# shap_info=Incorrect2Correct_info_dict[Inspect_samp]


# expected_value=shap_info['explainer_expected_value'][logit_number]
# shap_values=shap_info['shap_values'].loc[logit_number].values
# df_shap_values=shap_info['shap_values'].loc[[logit_number]].T
# Xtest=shap_info['XTest']

# col_idxs=column_index(Xtest, FeatSel.LOCDEP_Trend_D_cols)
# excol_idxs=np.array(list(set(range(len(Xtest))) - set(col_idxs)))


# column2idx_dict={idx:str(i) for i,idx in enumerate(Xtest.index)}
# df_column2idx_dict=pd.DataFrame.from_dict(column2idx_dict,orient='index')
# df_shap_values['feature_idxs']=df_column2idx_dict
# Xtest_numerized=pd.Series(Xtest.values,index=[ column2idx_dict[idxs] for idxs in  Xtest.index ]).round(2)
# Xtest_numerized.name=Xtest.name





# shap.force_plot(expected_value, shap_values, Xtest, matplotlib=True,show=False)
# # need adjustment when you are just showing part of the feature and not using all feautres
# shap.force_plot(expected_value + sum(shap_values[excol_idxs]), shap_values[col_idxs], Xtest.iloc[col_idxs], matplotlib=True,show=False)
def Get_Top_10_abs_SHAP_columns(shap_values,df_XTest):
    ''' Get Top 10 abs SHAP columns '''
    df_XSumSHAPVals=pd.DataFrame(np.abs(shap_values).sum(axis=0).reshape(1,-1),columns=df_XTest.columns).T
    return df_XSumSHAPVals.sort_values(by=0,ascending=False).head(10)
def column_index(df, query_cols):
    # column_index(df, ['peach', 'banana', 'apple'])
    cols = df.index.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]



OriginFSet2PperNmeFSet_dict={
'A':'Phonation_Trend_K_cols+Phonation_Syncrony_cols',
'B':'LOC_columns+Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols',
'C':'Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols',
'D':'Phonation_Proximity_cols',
'E':'LOCDEP_Trend_D_cols+Phonation_Proximity_cols',
'F':'DEP_columns+Phonation_Trend_D_cols+Phonation_Proximity_cols',
'G':'Phonation_Trend_D_cols+Phonation_Proximity_cols',
}


for k,v in OriginFSet2PperNmeFSet_dict.items():
    Ppr_str='+'.join([Swap2PaperName(f, PprNmeMp) for f in v.split('+')])
    print('Model {M}: {_str}'.format(M=k,_str=Ppr_str))