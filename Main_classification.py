#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:56:45 2021

@author: jackchen


    This script is only for TBMEB1 

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
import articulation.articulation
from sklearn.metrics import f1_score,recall_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
def Assert_labelfeature(feat_name,lab_name):
    # =============================================================================
    #     To check if the label match with feature
    # =============================================================================
    for i,n in enumerate(feat_name):
        assert feat_name[i] == lab_name[i]

def FilterFile_withinManualName(files,Manual_choosen_feature):
    files_manualChoosen=[f  for f in files if os.path.basename(f).split(".")[0]  in Manual_choosen_feature]
    return files_manualChoosen

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
    parser.add_argument('--Plot', default=True,
                        help='')
    parser.add_argument('--selectModelScoring', default='recall_macro',
                        help='[recall_macro,accuracy]')
    args = parser.parse_args()
    return args
args = get_args()
start_point=args.start_point
experiment=args.experiment

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


featuresOfInterest=[ [col] for col in columns]
# featuresOfInterest=[ [col] + ['u_num+i_num+a_num'] for col in columns]


# label_choose=['ADOS_C','Multi1','Multi2','Multi3','Multi4']
label_choose=['ADOS_C']
# label_choose=['ADOS_cate','ASDTD']
# FeatureLabelMatch=[['TD_normal vs ASDSevere_agesexmatch','ASDTD'],
#                     ['TD_normal vs ASDMild_agesexmatch','ASDTD'],
#                     ['Notautism vs ASD','ADOS_cate'],
#                     ['ASD vs Autism','ADOS_cate'],
#                     ['Notautism vs Autism','ADOS_cate']]
# FeatureLabelMatch=[['TD_normal vs ASDSevere_agesexmatch','ASDTD'],
#                     ['TD_normal vs ASDMild_agesexmatch','ASDTD'],
#                     ]
FeatureLabelMatch=[
                    # ['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_kid','ASDTD'],
                    # ['TD_normal vs ASDMild_agesexmatch >> FeatLOC_kid','ASDTD'],
                    # ['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_doc','ASDTD'],
                    # ['TD_normal vs ASDMild_agesexmatch >> FeatLOC_doc','ASDTD'],
                    # ['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_DKRatio','ASDTD'],#注意DKRatio的columns跟別人不一樣，不過是可以統一的
                    # ['TD_normal vs ASDMild_agesexmatch >> FeatLOC_DKRatio','ASDTD'],
                    ['TD_normal vs ASDSevere_agesexmatch >> FeatCoor','ASDTD'],#注意DKRatio的columns跟別人不一樣，不過是可以統一的
                    ['TD_normal vs ASDMild_agesexmatch >> FeatCoor','ASDTD'],
                    ]
df_formant_statistics_CtxPhone_collect_dict=Dict()
# =============================================================================

class ADOSdataset():
    def __init__(self,):
        self.featurepath='Features'            
        self.N=2
        self.LabelType=Dict()
        self.LabelType['ADOS_C']='regression'
        self.LabelType['ADOS_cate_C']='classification'
        self.LabelType['ASDTD']='classification'
        self.Fractionfeatures_str='Features/artuculation_AUI/Vowels/Fraction/*.pkl'    
        self.FeatureCombs=Dict()
        self.FeatureCombs['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_kid']=['df_formant_statistic_TD_normal_kid', 'df_formant_statistic_agesexmatch_ASDSevereGrp_kid']
        self.FeatureCombs['TD_normal vs ASDMild_agesexmatch >> FeatLOC_kid']=['df_formant_statistic_TD_normal_kid', 'df_formant_statistic_agesexmatch_ASDMildGrp_kid']
        self.FeatureCombs['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_doc']=['df_formant_statistic_TD_normal_doc', 'df_formant_statistic_agesexmatch_ASDSevereGrp_doc']
        self.FeatureCombs['TD_normal vs ASDMild_agesexmatch >> FeatLOC_doc']=['df_formant_statistic_TD_normal_doc', 'df_formant_statistic_agesexmatch_ASDMildGrp_doc']
        self.FeatureCombs['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_DKRatio']=['df_formant_statistic_TD_normalGrp_DKRatio', 'df_formant_statistic_agesexmatch_ASDSevereGrp_DKRatio']
        self.FeatureCombs['TD_normal vs ASDMild_agesexmatch >> FeatLOC_DKRatio']=['df_formant_statistic_TD_normalGrp_DKRatio', 'df_formant_statistic_agesexmatch_ASDMildGrp_DKRatio']
        
        self.FeatureCombs['TD_normal vs ASDSevere_agesexmatch >> FeatCoor']=['df_syncrony_statistic_TD_normalGrp', 'df_syncrony_statistic_agesexmatch_ASDSevereGrp']
        self.FeatureCombs['TD_normal vs ASDMild_agesexmatch >> FeatCoor']=['df_syncrony_statistic_TD_normalGrp', 'df_syncrony_statistic_agesexmatch_ASDMildGrp']
        # self.FeatureCombs['Notautism vs ASD']=['df_formant_statistic_77_Notautism', 'df_formant_statistic_77_ASD']
        # self.FeatureCombs['ASD vs Autism']=['df_formant_statistic_77_ASD', 'df_formant_statistic_77_Autism']
        # self.FeatureCombs['Notautism vs Autism']=['df_formant_statistic_77_Notautism', 'df_formant_statistic_77_Autism']
    
        self._FeatureBuild()
    def Get_FormantAUI_feat(self,label_choose,pickle_path,featuresOfInterest=['MSB_f1','MSB_f2','MSB_mix'],filterbyNum=True,**kwargs):
        self.featuresOfInterest=featuresOfInterest
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
            y_array=df_tmp['ADOS'].values
        else:
            y_array=df_tmp[label_choose].values
        feature_array=df_tmp[featuresOfInterest].values
        
            
        LabType=self.LabelType[label_choose]
        return feature_array, y_array, LabType
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

Pseudo_CtxDepPhone_path='artuculation_AUI/Pseudo_CtxDepVowels'
CtxDepPhone_path='artuculation_AUI/CtxDepVowels/bkup0729'
Vowel_path='artuculation_AUI/Vowels'
Interactionfeat_path='artuculation_AUI/Interaction'
OtherFeat_path='Other/Static_BasicInfo'
# for feature_paths in [Vowel_path, CtxDepPhone_path, Pseudo_CtxDepPhone_path]:
for feature_paths in [Interactionfeat_path]:
# for feature_paths in [Vowel_path, CtxDepPhone_path]:df_best_result_allThreeClassifiers
    # files = glob.glob(ados_ds.featurepath +'/'+ feature_paths+'/*.pkl')

    for feat_,lab_ in FeatureLabelMatch:
        # feat_=key
        for feat_col in featuresOfInterest:
            feat_col_ = list(feat_col) # ex: ['MSB_f1']
            
            X,y, featType=ados_ds.Get_FormantAUI_feat(label_choose=lab_,pickle_path='',featuresOfInterest=feat_col_,filterbyNum=False,feat_=ados_ds.Features_comb[feat_])
            
            if np.isnan(X).any() or np.isnan(y).any():
                print("Feat: ",feat_col_,'Contains nan')
                ErrorFeat_bookeep['{0} {1} {2}'.format(feat_,lab_,feat_col_)].X=X
                ErrorFeat_bookeep['{0} {1} {2}'.format(feat_,lab_,feat_col_)].y=y
                continue
            
            Item_name="{feat}::{lab}".format(feat='-'.join([feat_]+feat_col_),lab=lab_)
            Session_level_all[Item_name].X, \
                Session_level_all[Item_name].y, \
                    Session_level_all[Item_name].feattype = X,y, featType



paper_name_map={}    
paper_name_map['Divergence[pillai_lin_(A:,i:,u:)]']='$Div(Pillai)$'
paper_name_map['Divergence[pillai_lin_(A:,i:,u:)]_var_p1']='$Inc(Pillai)_{inv}$'
paper_name_map['Divergence[pillai_lin_(A:,i:,u:)]_var_p2']='$Inc(Pillai)_{part}$'
paper_name_map['Divergence[within_covariance_(A:,i:,u:)]']='$Div(WCC)$'
paper_name_map['Divergence[within_covariance_(A:,i:,u:)]_var_p1']='$Inc(WCC)_{inv}$'
paper_name_map['Divergence[within_covariance_(A:,i:,u:)]_var_p2']='$Inc(WCC)_{part}$'
paper_name_map['Divergence[within_variance_(A:,i:,u:)]']='$Div(WCV)$'
paper_name_map['Divergence[within_variance_(A:,i:,u:)]_var_p1']='$Inc(WCV)_{inv}$'
paper_name_map['Divergence[within_variance_(A:,i:,u:)]_var_p2']='$Inc(WCV)_{part}$'
paper_name_map['Divergence[sam_wilks_lin_(A:,i:,u:)]']='$Div(Wilks)$'
paper_name_map['Divergence[sam_wilks_lin_(A:,i:,u:)]_var_p1']='$Inc(Wilks)_{inv}$'
paper_name_map['Divergence[sam_wilks_lin_(A:,i:,u:)]_var_p2']='$Inc(Wilks)_{part}$'
paper_name_map['Divergence[between_covariance_(A:,i:,u:)]']='$Div(BCC)$'
paper_name_map['Divergence[between_covariance_(A:,i:,u:)]_var_p1']='$Inc(BCC)_{inv}$'
paper_name_map['Divergence[between_covariance_(A:,i:,u:)]_var_p2']='$Inc(BCC)_{part}$'
paper_name_map['Divergence[between_variance_(A:,i:,u:)]']='$Div(BCV)$'
paper_name_map['Divergence[between_variance_(A:,i:,u:)]_var_p1']='$Inc(BCV)_{inv}$'
paper_name_map['Divergence[between_variance_(A:,i:,u:)]_var_p2']='$Inc(BCV)_{part}$'
paper_name_map['between_covariance_norm(A:,i:,u:)']='$BCC$'
paper_name_map['between_variance_norm(A:,i:,u:)']='$BCV$'
paper_name_map['within_covariance_norm(A:,i:,u:)']='$WCC$'
paper_name_map['within_variance_norm(A:,i:,u:)']='$WCV$'
paper_name_map['total_covariance_norm(A:,i:,u:)']='$TC$'
paper_name_map['total_variance_norm(A:,i:,u:)']='$TV$'
paper_name_map['sam_wilks_lin_norm(A:,i:,u:)']='$Wilks$'
paper_name_map['pillai_lin_norm(A:,i:,u:)']='$Pillai$'
paper_name_map['hotelling_lin_norm(A:,i:,u:)']='$Hotel$'
paper_name_map['roys_root_lin_norm(A:,i:,u:)']='$Roys$'
paper_name_map['Between_Within_Det_ratio_norm(A:,i:,u:)']='$Det(W^{-1}B)$'
paper_name_map['Between_Within_Tr_ratio_norm(A:,i:,u:)']='$Tr(W^{-1}B)$'


# =============================================================================
# Model parameters
# =============================================================================
# C_variable=np.array([0.0001, 0.01, 0.1,0.5,1.0,10.0, 50.0, 100.0, 1000.0])
# C_variable=np.array([0.01, 0.1,0.5,1.0,10.0, 50.0, 100.0])
C_variable=np.array(np.arange(0.1,1.5,0.1))
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

    

# Classifier['LR']={'model':sklearn.linear_model.LogisticRegression(),\
#                   'parameters':{'model__random_state':[1],\
#                                 'model__C':C_variable,\
#                                 }}
# Classifier['SGD']={'model':sklearn.linear_model.SGDClassifier(),\
#                   'parameters':{'model__random_state':[1],\
#                                 'model__l1_ratio':np.arange(0,1,0.25),\
#                                 }}



# from sklearn.neural_network import MLPClassifier
# Classifier['MLP']={'model':MLPClassifier(),\
#                   'parameters':{'model__random_state':[1],\
#                                 'model__hidden_layer_sizes':[(20),(40),(60,),(80,)],\
#                                 'model__activation':['relu'],\
#                                 'model__batch_size':[1,2,3],\
#                                 # 'model__solver':['sgd'],\
#                                 'model__early_stopping':[True],\
#                                 # 'model__penalty':['elasticnet'],\
#                                 # 'model__l1_ratio':[0.25,0.5,0.75],\
#                                 }}

# Classifier['Ridge']={'model':sklearn.linear_model.RidgeClassifier(),\
#                   'parameters':{'model__random_state':[1],\
#                                 'model__solver':['auto'],\
#                                 # 'model__penalty':['elasticnet'],\
#                                 # 'model__l1_ratio':[0.25,0.5,0.75],\
#                                 }}

# from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
# Classifier['GB']={'model':GradientBoostingClassifier(),\
#                   'parameters':{
#                                 'model__random_state':[1],\
#                                 'model__n_estimators':n_estimator
#                                 }}
# Classifier['AdaBoostClassifier']={'model':AdaBoostClassifier(),\
#                   'parameters':{
#                                 'model__random_state':[1],\
#                                 'model__n_estimators':n_estimator
#                                 }}
# Classifier['DT']={'model':DecisionTreeClassifier(),\
#                   'parameters':{'model__random_state':[1],\
#                                 'model__criterion':['gini','entropy'],
#                                 'model__splitter':['splitter','random'],\
#                                 }}
    
# from sklearn.ensemble import RandomForestClassifier
# Classifier['RF']={'model':RandomForestClassifier(),\
#                   'parameters':{
#                   'model__random_state':[1],\
#                   'model__n_estimators':n_estimator
#                                 }}


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

for clf_keys, clf in Classifier.items(): #Iterate among different classifiers 
    writer_clf = pd.ExcelWriter(Result_path+"/"+clf_keys+"_"+args.Feature_mode+"_"+final_result_file, engine = 'xlsxwriter')
    for feature_lab_str, features in Session_level_all.items():
        feature_keys, label_keys= feature_lab_str.split("::")
        feature_rawname=feature_keys[feature_keys.find('-')+1:]
        if feature_rawname in paper_name_map.keys():
            featurename_paper=paper_name_map[feature_rawname]
            feature_keys=feature_keys.replace(feature_rawname,featurename_paper)
        
        
        Labels = Session_level_all.X[feature_keys]
        print("=====================Cross validation start==================")
        pipe = Pipeline(steps=[('scalar',StandardScaler()),("model", clf['model'])])
        p_grid=clf['parameters']
        Gclf = GridSearchCV(estimator=pipe, param_grid=p_grid, scoring=args.selectModelScoring, cv=CV_settings, refit=True, n_jobs=-1)
        # Score=cross_val_score(Gclf, features.X, features.y, cv=loo) 
        CVpredict=cross_val_predict(Gclf, features.X, features.y, cv=CV_settings)           
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

        
        
        if features.feattype == 'regression':
            r2=r2_score(features.y,CVpredict )
            n,p=features.X.shape
            r2_adj=1-(1-r2)*(n-1)/(n-p-1)
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
            axes.scatter((features.X - min(features.X) )/ max(features.X), CVpredict, 
                         facecolor="none", edgecolor="k", s=150,
                         label='{}'.format(feature_lab_str)
                         )
            axes.scatter((features.X - min(features.X) )/ max(features.X), features.y, 
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
df_best_result_allThreeClassifiers.to_excel(Result_path+"/"+"_"+args.Feature_mode+"_3clsferRESULT.xlsx")
print(df_best_result_allThreeClassifiers)