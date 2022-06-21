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

import articulation.articulation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import shap
import articulation.HYPERPARAM.PaperNameMapping as PprNmeMp
import seaborn as sns
import shutil
from collections import Counter
import shutil
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
    parser.add_argument('--logit_number', default=0,
                        help='')
    parser.add_argument('--Print_Analysis_grp_Manual_select', default=True,
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


# label_choose=['ADOS_C','Multi1','Multi2','Multi3','Multi4']
# label_choose=['ADOS_S','ADOS_C']
# label_choose=['ADOS_D']
label_choose=['ADOS_C']
# label_choose=['ADOS_S']
# label_choose=['ADOS_cate','ASDTD']

pearson_scorer = make_scorer(pearsonr, greater_is_better=False)

df_formant_statistics_CtxPhone_collect_dict=Dict()
#%%
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

# All_combination_keys=[key2 for key in All_combinations.keys() for key2 in sorted(All_combinations[key], key=len, reverse=True)]
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
Top_ModuleColumn_mapping_dict['baselineFeats']=FeatSel.Baseline_comb.copy()

featuresOfInterest=Top_ModuleColumn_mapping_dict[args.FeatureComb_mode]

# =============================================================================
# 1. 如果要做全盤的實驗的話用這一區
# FeatureLabelMatch_manual=[]
# All_combinations=featuresOfInterest
# # All_combinations4=FeatSel.Columns_comb4.copy()
# for key_layer1 in All_combinations.keys():
#     for key_layer2 in All_combinations[key_layer1].keys():
#         if 'Utt_prosodyF0_VoiceQuality_energy' in key_layer2:
#             FeatureLabelMatch_manual.append('{0}-{1}'.format(key_layer1,key_layer2))
    

# 2. 如果要手動設定實驗的話用這一區
FeatureLabelMatch_manual=[
    # Rule: {layer1}-{layer2}
    'static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation-LOC_columns+DEP_columns+LOCDEP_Trend_D_cols+LOCDEP_Syncrony_cols',
    'static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation-Phonation_Trend_K_cols',
    ]
# =============================================================================


if args.FeatureComb_mode == 'Add_UttLvl_feature':
    Merge_feature_path='RegressionMerged_dfs/ADDed_UttFeat/{knn_weights}_{knn_neighbors}_{Reorder_type}/ASD_DOCKID/'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,Reorder_type=Reorder_type)
else:
    Merge_feature_path='RegressionMerged_dfs/{knn_weights}_{knn_neighbors}_{Reorder_type}/ASD_DOCKID/'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,Reorder_type=Reorder_type)

for exp_str in FeatureLabelMatch_manual:
    Layer1Feat, Layer2Feat=exp_str.split("-")
    
    # load features from file
    data=ados_ds.featurepath +'/'+ Merge_feature_path+'{}.pkl'.format(Layer1Feat)
    

    feat_col_ = featuresOfInterest[Layer1Feat][Layer2Feat] # ex: ['MSB_f1']
    for lab_ in label_choose:
        X,y, featType=ados_ds.Get_FormantAUI_feat(label_choose=lab_,pickle_path=data,featuresOfInterest=feat_col_,filterbyNum=False)
        Item_name="{feat}::{lab}".format(feat='-'.join([Layer1Feat]+[Layer2Feat]),lab=lab_)
        Session_level_all[Item_name].X, \
            Session_level_all[Item_name].y, \
                Session_level_all[Item_name].feattype = X,y, featType
    assert len(X.columns) >0
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
        
    
    
    

paper_name_map={}    
paper_name_map['Divergence[pillai_lin_norm(A:,i:,u:)]']='$Div(Norm(Pillai))$'
paper_name_map['Divergence[pillai_lin_norm(A:,i:,u:)]_var_p1']='$Inc(Norm(Pillai))_{inv}$'
paper_name_map['Divergence[pillai_lin_norm(A:,i:,u:)]_var_p2']='$Inc(Norm(Pillai))_{part}$'
paper_name_map['Divergence[within_covariance_norm(A:,i:,u:)]']='$Div(Norm(WCC))$'
paper_name_map['Divergence[within_covariance_norm(A:,i:,u:)]_var_p1']='$Inc(Norm(WCC))_{inv}$'
paper_name_map['Divergence[within_covariance_norm(A:,i:,u:)]_var_p2']='$Inc(Norm(WCC))_{part}$'
paper_name_map['Divergence[within_variance_norm(A:,i:,u:)]']='$Div(Norm(WCV))$'
paper_name_map['Divergence[within_variance_norm(A:,i:,u:)]_var_p1']='$Inc(Norm(WCV))_{inv}$'
paper_name_map['Divergence[within_variance_norm(A:,i:,u:)]_var_p2']='$Inc(Norm(WCV))_{part}$'
paper_name_map['Divergence[sam_wilks_lin_norm(A:,i:,u:)]']='$Div(Norm(Wilks))$'
paper_name_map['Divergence[sam_wilks_lin_norm(A:,i:,u:)]_var_p1']='$Inc(Norm(Wilks))_{inv}$'
paper_name_map['Divergence[sam_wilks_lin_norm(A:,i:,u:)]_var_p2']='$Inc(Norm(Wilks))_{part}$'
paper_name_map['Divergence[between_covariance_norm(A:,i:,u:)]']='$Div(Norm(BCC))$'
paper_name_map['Divergence[between_covariance_norm(A:,i:,u:)]_var_p1']='$Inc(Norm(BCC))_{inv}$'
paper_name_map['Divergence[between_covariance_norm(A:,i:,u:)]_var_p2']='$Inc(Norm(BCC))_{part}$'
paper_name_map['Divergence[between_variance_norm(A:,i:,u:)]']='$Div(Norm(BCV))$'
paper_name_map['Divergence[between_variance_norm(A:,i:,u:)]_var_p1']='$Inc(Norm(BCV))_{inv}$'
paper_name_map['Divergence[between_variance_norm(A:,i:,u:)]_var_p2']='$Inc(Norm(BCV))_{part}$'
paper_name_map['between_covariance_norm(A:,i:,u:)']='$BCC$'
paper_name_map['between_variance_norm(A:,i:,u:)']='$BCV$'
paper_name_map['within_covariance_norm(A:,i:,u:)']='$WCC$'
paper_name_map['within_variance_norm(A:,i:,u:)']='$WCV$'
# paper_name_map['total_covariance_norm(A:,i:,u:)']='$TC$'
# paper_name_map['total_variance_norm(A:,i:,u:)']='$TV$'
paper_name_map['sam_wilks_lin_norm(A:,i:,u:)']='$Wilks$'
paper_name_map['pillai_lin_norm(A:,i:,u:)']='$Pillai$'
paper_name_map['hotelling_lin_norm(A:,i:,u:)']='$Hotel$'
paper_name_map['roys_root_lin_norm(A:,i:,u:)']='$Roys$'
paper_name_map['Between_Within_Det_ratio_norm(A:,i:,u:)']='$Det(B_W)$'
paper_name_map['Between_Within_Tr_ratio_norm(A:,i:,u:)']='$Tr(B_W)$'


# =============================================================================
# Model parameters
# =============================================================================
# C_variable=np.array([0.01, 0.1,0.5,1.0,10.0, 50.0, 100.0, 1000.0])
# C_variable=np.array(np.arange(0.1,1.1,0.2))
# C_variable=np.array([0.1,0.5,0.9])
# epsilon=np.array(np.arange(0.1,1.5,0.1) )
epsilon=np.array([0.001,0.01,0.1,1,5,10.0,25,50,75,100])

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
Result_path="RESULTS/"

if not os.path.exists(Result_path):
    os.makedirs(Result_path)
final_result_file="_ADOS_{}.xlsx".format(args.suffix)

import warnings
warnings.filterwarnings("ignore")
count=0
OutFeature_dict=Dict()
Best_param_dict=Dict()

# ''' 要手動執行一次從Incorrect2Correct_indexes和Correct2Incorrect_indexes決定哪些indexes 需要算shap value 再在這邊指定哪些fold需要停下來算SHAP value '''
# SHAP_inspect_idxs_manual=[]
SHAP_inspect_idxs_manual=None # None means calculate SHAP value of all people
# SHAP_inspect_idxs_manual=[1,3,5] # empty list means we do not execute shap function
for clf_keys, clf in Classifier.items(): #Iterate among different classifiers 
    writer_clf = pd.ExcelWriter(Result_path+"/"+clf_keys+"_"+args.Feature_mode+"_"+final_result_file, engine = 'xlsxwriter')
    for feature_lab_str, features in Session_level_all.items():
        feature_keys, label_keys= feature_lab_str.split("::")
        feature_rawname=feature_keys[feature_keys.find('-')+1:]
        feature_filename=feature_keys[:feature_keys.find('-')]
        if feature_rawname in paper_name_map.keys():
            featurename_paper=paper_name_map[feature_rawname]
            feature_keys=feature_keys.replace(feature_rawname,featurename_paper)
        
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
        # Score=cross_val_score(Gclf, features.X, features.y, cv=CV_settings, scoring=pearson_scorer) 
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
            

            # If the indexes we want to examine are in that fold, store the whole fold
            # 先把整個fold記錄下來然後在analysis area再拆解
            SHAP_exam_lst=[i for i in test_index if i in SHAP_inspect_idxs]
            if len(SHAP_exam_lst) != 0:
                explainer = shap.KernelExplainer(Gclf_manual.predict, X_train)
                # explainer(X_test)
                shap_values = explainer.shap_values(X_test)
                Session_level_all[feature_lab_str]['SHAP_info']['_'.join(test_index.astype(str))].explainer_expected_value=explainer.expected_value
                Session_level_all[feature_lab_str]['SHAP_info']['_'.join(test_index.astype(str))].shap_values=shap_values # shap_values= [logit, index, feature]
                Session_level_all[feature_lab_str]['SHAP_info']['_'.join(test_index.astype(str))].XTest=X_test
                # Session_level_all[feature_lab_str]['SHAP_info']['_'.join(test_index)].testIndex=test_index
            # shap.force_plot(explainer.expected_value[logit_number], shap_values[logit_number][inspect_sample,:], X_test.iloc[inspect_sample,:], matplotlib=True,show=False)
            
            
            # assert (result_bestmodel==result_bestmodel_fitted_again).all()
            assert (result_bestmodel==CVpred_fromFunction).all()
        
            
        assert (CVpredict_manual==CVpredict).all()
        Session_level_all[feature_lab_str]['y_pred']=CVpredict_manual
        Session_level_all[feature_lab_str]['y_true']=features.y


        
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
print()


#%%
# =============================================================================
'''

    Analysis part

'''
# =============================================================================
# =============================================================================

'''

    Part 1: Check incorrect to correct and correct to incorrect

'''

if args.Print_Analysis_grp_Manual_select == True:
    count=0
    for exp_str in FeatureLabelMatch_manual:
        for lab_str in label_choose:
            exp_lst_str='::'.join([exp_str,lab_str])
            if count < len(FeatureLabelMatch_manual)/2:
                print("proposed_expstr='{}'".format(exp_lst_str))
            else:
                print("baseline_expstr='{}'".format(exp_lst_str))
            count+=1
'''

    Part 1: Check incorrect to correct and correct to incorrect

'''


proposed_expstr='static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation-LOC_columns+DEP_columns+LOCDEP_Trend_D_cols+LOCDEP_Syncrony_cols::ADOS_C'
baseline_expstr='static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation-Phonation_Trend_K_cols::ADOS_C'
experiment_title='Regression'


proposed_featset_lst=proposed_expstr[re.search("-",proposed_expstr).end():re.search("::",proposed_expstr).start()].split("+")
baseline_featset_lst=baseline_expstr[re.search("-",baseline_expstr).end():re.search("::",baseline_expstr).start()].split("+")
Additional_featureSet=set(proposed_featset_lst) - set(baseline_featset_lst)
print("For Task", experiment_title, " additional feature sets are", Additional_featureSet)
if Session_level_all[proposed_expstr]['feattype'] == 'regression':
    SHAP_Inspect_logit=0
logit_number=SHAP_Inspect_logit



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

df_Y_pred=pd.DataFrame(Y_pred_lst,index=['proposed','baseline','y_true','name']).T
df_Y_pred_withName=pd.DataFrame(Y_pred_lst,index=['proposed','baseline','y_true','name']).T
df_Index2Name_mapping=df_Y_pred_withName['name']

# proposed的距離比baseline 的還要近
Improved=(df_Y_pred['y_true'] - df_Y_pred['proposed']).abs() < (df_Y_pred['y_true'] - df_Y_pred['baseline']).abs()
Degraded=(df_Y_pred['y_true'] - df_Y_pred['proposed']).abs() >= (df_Y_pred['y_true'] - df_Y_pred['baseline']).abs()


LargerVal= (df_Y_pred['proposed'] - df_Y_pred['baseline'])>0
LowerVal= (df_Y_pred['proposed'] - df_Y_pred['baseline'])<=0
ChangedPeople= LargerVal | LowerVal


Improved_indexes=list(df_Y_pred[Improved].index)
Degraded_indexes=list(df_Y_pred[Degraded].index)

LargerVal_indexes=list(df_Y_pred[LargerVal].index)
LowerVal_indexes=list(df_Y_pred[LowerVal].index)

proposed_ypred_array=df_Y_pred['proposed'].values
baseline_ypred_array=df_Y_pred['baseline'].values

# 這邊計算出變高的人誤差修正（增加/減少）了多少，變低的人誤差修正（增加/減少）了多少
# 3.675 -> 3.167
for insp_instances_bool_str in ['LargerVal','LowerVal','ChangedPeople']:
    insp_instances_bool=vars()[insp_instances_bool_str]
    delta_proposed_diff_squared=(df_Y_pred[insp_instances_bool]['proposed'] - df_Y_pred[insp_instances_bool]['y_true'])**2
    delta_proposed_diff_MSE=delta_proposed_diff_squared.sum()/len(df_Y_pred)
    delta_baseline_diff_squared=(df_Y_pred[insp_instances_bool]['baseline'] - df_Y_pred[insp_instances_bool]['y_true'])**2
    delta_baseline_diff_MSE=delta_baseline_diff_squared.sum()/len(df_Y_pred)
    assert len(delta_proposed_diff_squared) == len(delta_baseline_diff_squared)
    print(insp_instances_bool_str,": have changed MSE",delta_proposed_diff_MSE-delta_baseline_diff_MSE)


# 這邊秀出proposed和baseline的實際改變
sns.scatterplot(data=df_Y_pred)

quadrant1_indexes=intersection(Degraded_indexes, LargerVal_indexes)
quadrant2_indexes=intersection(Improved_indexes, LargerVal_indexes)
quadrant3_indexes=intersection(Degraded_indexes, LowerVal_indexes)
quadrant4_indexes=intersection(Improved_indexes, LowerVal_indexes)



assert len(Improved_indexes+Degraded_indexes) == len(LargerVal_indexes+LowerVal_indexes)


'''

    Part 2: Check the SHAP values based on indexes in part 1
    
    先紀錄，再執行分析和畫圖

'''


def Organize_Needed_SHAP_info(Selected_indexes, Session_level_all, proposed_expstr):
    Selected_info_dict=Dict()
    for tst_idx in Selected_indexes:
        for key, values in Session_level_all[proposed_expstr]['SHAP_info'].items():
            test_fold_idx=[int(k) for k in key.split("_")]
            for i,ii in enumerate(test_fold_idx): #ii is the index of the sample, i is the position of this sample in this test fold
                if tst_idx == ii:
                    Selected_info_dict[tst_idx]['XTest']=values['XTest'].iloc[i,:]
                    Selected_info_dict[tst_idx]['explainer_expected_value']=values['explainer_expected_value']
                    shap_values_array=values['shap_values'][i,:].reshape(1,-1)
                    df_shap_values=pd.DataFrame(shap_values_array,columns=Selected_info_dict[tst_idx]['XTest'].index)
                    Selected_info_dict[tst_idx]['shap_values']=df_shap_values
                    print("testing sample ", ii, "is in the ", i, "position of test fold", key)
                    assert (Selected_info_dict[tst_idx]['XTest'] == Session_level_all[proposed_expstr]['X'].iloc[tst_idx]).all()
                    # print("It's feature value captured is", Selected_info_dict[tst_idx]['XTest'])
                    # print("It's original X value is", Session_level_all[proposed_expstr]['X'].iloc[tst_idx])
                    # print("See if they match")
    return Selected_info_dict


selected_idxs=LargerVal_indexes+LowerVal_indexes
Baseline_changed_info_dict=Organize_Needed_SHAP_info(selected_idxs, Session_level_all, baseline_expstr)
Proposed_changed_info_dict=Organize_Needed_SHAP_info(selected_idxs, Session_level_all, proposed_expstr)
Baseline_totalPoeple_info_dict=Organize_Needed_SHAP_info(df_Y_pred.index, Session_level_all, baseline_expstr)
Proposed_totalPoeple_info_dict=Organize_Needed_SHAP_info(df_Y_pred.index, Session_level_all, proposed_expstr)

#%%
# 個體分析： 會存到SAHP_figures/{quadrant}/的資料夾，再開Jupyter去看
def Calculate_sum_of_SHAP_vals(df_values,FeatSel_module,FeatureSet_lst=['Phonation_Proximity_cols'],PprNmeMp=None):
    df_FeaSet_avg_Comparison=pd.DataFrame([],columns=FeatureSet_lst)
    for feat_set in FeatureSet_lst:
        feature_cols = getattr(FeatSel_module, feat_set)
        if PprNmeMp!=None:
            feature_cols=[ Swap2PaperName(name,PprNmeMp) for name in feature_cols]
        df_FeaSet_avg_Comparison[feat_set]=df_values.loc[feature_cols,:].sum()
    return df_FeaSet_avg_Comparison

SHAP_save_path_root="SAHP_figures/Regression/{quadrant}/"

shutil.rmtree(SHAP_save_path_root.format(quadrant=""), ignore_errors = True)


N=5
import scipy
# expected_value_lst=[]
UsePaperName_bool=True
Quadrant_FeatureImportance_dict={}
Quadrant_feature_AddedTopFive_dict={}
Quadrant_feature_AddedFeatureImportance_dict={}
Quadrant_Sumedfeature_AddedFeatureImportance_sorted_dict={}
# for Analysis_grp_str in ['Manual_inspect_idxs','quadrant1_indexes','quadrant2_indexes','quadrant3_indexes','quadrant4_indexes']:
for Analysis_grp_str in ['quadrant1_indexes','quadrant2_indexes','quadrant3_indexes','quadrant4_indexes']:
    Analysis_grp_indexes=vars()[Analysis_grp_str]
    df_shap_values_stacked=pd.DataFrame([])
    Xtest_dict={}
    expected_value_lst=[]
    for Inspect_samp in Analysis_grp_indexes:
        shap_info=Proposed_changed_info_dict[Inspect_samp]
        
        expected_value=shap_info['explainer_expected_value']
        shap_values=shap_info['shap_values'].values
        df_shap_values=shap_info['shap_values']
        df_shap_values.index=[Inspect_samp]
        
        df_shap_values_T=df_shap_values.T
        
        Xtest=shap_info['XTest']
        
        Xtest_dict[Inspect_samp]=Xtest
        df_shap_values_stacked=pd.concat([df_shap_values_stacked,df_shap_values],)
        expected_value_lst.append(expected_value)
        
        
        Xtest.index=[ Swap2PaperName(name,PprNmeMp) for name in Xtest.index]
        # shap.force_plot(expected_value, df_shap_values.values, Xtest.T, matplotlib=True,show=False)
        df_shap_avgvalues=Calculate_sum_of_SHAP_vals(df_shap_values.T,\
                                                    FeatSel_module=FeatSel,\
                                                    FeatureSet_lst=proposed_featset_lst,)
        df_avg_Xtest=Calculate_sum_of_SHAP_vals(pd.DataFrame(Xtest.values, index=Xtest.index),\
                                                FeatSel_module=FeatSel,\
                                                FeatureSet_lst=proposed_featset_lst,\
                                                PprNmeMp=PprNmeMp)
        
        p=shap.force_plot(expected_value, df_shap_avgvalues.values, df_avg_Xtest)
        # p=shap.force_plot(expected_value, df_shap_values.values, Xtest.T)
        
        
        
        SHAP_save_path=SHAP_save_path_root.format(quadrant=Analysis_grp_str)
        if not os.path.exists(SHAP_save_path):
            os.makedirs(SHAP_save_path)
            
        shap.save_html(SHAP_save_path+'{sample}.html'.format(sample=Inspect_samp), p)
        
        Lists_of_addedFeatures=[getattr(FeatSel,k)  for k in Additional_featureSet]
        Lists_of_addedFeatures_flatten=[e for ee in Lists_of_addedFeatures for e in ee]
        df_FeatureImportance_AddedFeatures=df_shap_values[Lists_of_addedFeatures_flatten].T
        df_FeatureImportance_AddedFeatures_absSorted=df_FeatureImportance_AddedFeatures.abs()[Inspect_samp].sort_values(ascending=False)
        
        
        
        df_addedFeatures_TopN=df_FeatureImportance_AddedFeatures.loc[df_FeatureImportance_AddedFeatures_absSorted.head(N).index]
        # Quadrant_feature_AddedTopFive_dict[Inspect_samp]=df_addedFeatures_TopN
        Quadrant_feature_AddedFeatureImportance_dict[Inspect_samp]=df_FeatureImportance_AddedFeatures[Inspect_samp].sort_values(ascending=False)
    df_XTest_stacked=pd.DataFrame.from_dict(Xtest_dict).T
    assert (df_XTest_stacked.index == df_shap_values_stacked.index).all()
    
    df_FeaSet_avg_Comparison_proposed=Calculate_sum_of_SHAP_vals(df_shap_values_stacked.T,\
                                                                 FeatSel_module=FeatSel,\
                                                                 FeatureSet_lst=proposed_featset_lst,)
                                                                 # PprNmeMp=PprNmeMp)
    df_FeaAvg=df_FeaSet_avg_Comparison_proposed.copy()
    df_XTest_stacked_proposed=Calculate_sum_of_SHAP_vals(df_XTest_stacked.T,\
                                                                 FeatSel_module=FeatSel,\
                                                                 FeatureSet_lst=proposed_featset_lst,
                                                                 PprNmeMp=PprNmeMp)
    Quadrant_Sumedfeature_AddedFeatureImportance_sorted_dict[Analysis_grp_str]=df_FeaAvg.sort_values(list(df_FeaAvg.columns),\
                                                                                                    ascending=[False]*len(df_FeaAvg.columns))
    # clustering
    df_tocluster=df_FeaAvg.copy()
    # df_tocluster=df_shap_values_stacked.copy()
    D = scipy.spatial.distance.pdist(df_tocluster.loc[:,:], 'sqeuclidean')
    Z = scipy.cluster.hierarchy.complete(D)
    plt.figure()
    dn = scipy.cluster.hierarchy.dendrogram(Z)
    plt.show()
    clustOrder = scipy.cluster.hierarchy.leaves_list(Z)
    df_tocluster.iloc[dn['leaves']].index
    # df_tocluster.iloc[clustOrder].index
    
    # p=shap.force_plot(np.mean(expected_value_lst),df_shap_values_stacked.values, df_XTest_stacked)
    p=shap.force_plot(np.mean(expected_value_lst),df_FeaSet_avg_Comparison_proposed.values, df_XTest_stacked_proposed)
    # shap_explanation=shap.Explanation(df_shap_values_stacked)
    
    SHAP_save_path="SAHP_figures/Regression/Stacked/"
    if not os.path.exists(SHAP_save_path):
        os.makedirs(SHAP_save_path)
    shap.save_html(SHAP_save_path+'{sample}.html'.format(sample=Analysis_grp_str), p)

#%%
# =============================================================================
# 測試heatmap hierachycal clustering
from scipy.spatial.distance import pdist
def hclust_order(X, metric="sqeuclidean"):
    """ A leaf ordering is under-defined, this picks the ordering that keeps nearby samples similar.
    """
    
    # compute a hierarchical clustering
    D = scipy.spatial.distance.pdist(X, metric)
    cluster_matrix = scipy.cluster.hierarchy.complete(D)
    
    # merge clusters, rotating them to make the end points match as best we can
    sets = [[i] for i in range(X.shape[0])]
    for i in range(cluster_matrix.shape[0]):
        s1 = sets[int(cluster_matrix[i,0])]
        s2 = sets[int(cluster_matrix[i,1])]
        
        # compute distances between the end points of the lists
        d_s1_s2 = pdist(np.vstack([X[s1[-1],:], X[s2[0],:]]), metric)[0]
        d_s2_s1 = pdist(np.vstack([X[s1[0],:], X[s2[-1],:]]), metric)[0]
        d_s1r_s2 = pdist(np.vstack([X[s1[0],:], X[s2[0],:]]), metric)[0]
        d_s1_s2r = pdist(np.vstack([X[s1[-1],:], X[s2[-1],:]]), metric)[0]

        # concatenete the lists in the way the minimizes the difference between
        # the samples at the junction
        best = min(d_s1_s2, d_s2_s1, d_s1r_s2, d_s1_s2r)
        if best == d_s1_s2:
            sets.append(s1 + s2)
        elif best == d_s2_s1:
            sets.append(s2 + s1)
        elif best == d_s1r_s2:
            sets.append(list(reversed(s1)) + s2)
        else:
            sets.append(s1 + list(reversed(s2)))
    
    return sets[-1]
# =============================================================================
import scipy
import matplotlib.gridspec as gridspec
xgb_shap=df_shap_values_stacked.values
D = scipy.spatial.distance.pdist(xgb_shap, 'sqeuclidean')
clustOrder = scipy.cluster.hierarchy.leaves_list(scipy.cluster.hierarchy.complete(D))

# col_inds = np.argsort(-np.abs(xgb_shap).mean(0))[:10]
col_inds = np.argsort(-np.abs(xgb_shap).mean(0))[:10]
xgb_shap_normed = xgb_shap.copy()
for i in col_inds:
    xgb_shap_normed[:,i] -= xgb_shap_normed[:,i].min()
    xgb_shap_normed[:,i] /= xgb_shap_normed[:,i].max()



gs = gridspec.GridSpec(2,1)
fig = plt.figure(figsize=(15,7))

ax = fig.add_subplot(gs[0])
# ax.plot(xgb_shap[clustOrder,:].sum(1))
ax.plot(xgb_shap[clustOrder,:].sum(1))
ax.set_ylabel(r'Label One', size =16)
ax.get_yaxis().set_label_coords(-0.1,0.5)
ax.axis('off')


ax = fig.add_subplot(gs[1])
# ax.imshow(xgb_shap_normed[clustOrder,:][:,:].T, aspect=400, cmap=shap.plots.colors.red_blue_transparent)
ax.imshow(xgb_shap_normed[clustOrder,:][:,col_inds].T, aspect=400, cmap=shap.plots.colors.red_blue_transparent)

ax.set_yticks(np.arange(len(col_inds)))
ax.set_yticklabels(df_XTest_stacked.columns[col_inds])
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.show()
#%%
def Prepare_data_for_summaryPlot_regression(SHAPval_info_dict, feature_columns=None,PprNmeMp=None):
    keys_bag=[]
    XTest_dict={}
    shap_values_0_bag=[]
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
    shap_values_0_array=np.vstack(shap_values_0_bag)
    shap_values=shap_values_0_array
    df_XTest=pd.DataFrame.from_dict(XTest_dict,orient='columns').T
    if PprNmeMp!=None:
        df_XTest.columns=[Swap2PaperName(k,PprNmeMp) for k in df_XTest.columns]
    return shap_values, df_XTest, keys_bag

# inspect_featuresets='Trend[LOCDEP]_d'  #Trend[LOCDEP]d + Proximity[phonation]
# inspect_featuresets='LOC_columns'  #LOC_columns + Syncrony[phonation]
inspect_featuresets_lst=['LOC_columns','DEP_columns','Trend[LOCDEP]_d','Trend[LOCDEP]_k','Syncrony[LOCDEP]']

featuresetsListTotal_lst=[]
for inspect_featuresets in inspect_featuresets_lst:
    featuresetsListTotal_lst+=FeatSel.CategoricalName2cols[inspect_featuresets]
shap_values, df_XTest, keys=Prepare_data_for_summaryPlot_regression(Proposed_totalPoeple_info_dict,\
                                                          feature_columns=featuresetsListTotal_lst,\
                                                          PprNmeMp=PprNmeMp)


def ReorganizeFeatures4SummaryPlot(shap_values_logit, df_XTest,FeatureSet_lst=None,Featurechoose_lst=None):
    # step1 convert shapvalue to df_shapvalues
    df_shap_values=pd.DataFrame(shap_values_logit,columns=df_XTest.columns)
    # step2 Categorize according to FeatureSet_lst
    df_Reorganized_shap_values=pd.DataFrame()
    df_Reorganized_XTest=pd.DataFrame()
    if FeatureSet_lst!=None:
        for FSL in FeatureSet_lst:
            FSL_papercolumns=[Swap2PaperName(k,PprNmeMp) for k in FeatSel.CategoricalName2cols[FSL]]
        
            df_Reorganized_shap_values=pd.concat([df_Reorganized_shap_values,df_shap_values[FSL_papercolumns]],axis=1)
            df_Reorganized_XTest=pd.concat([df_Reorganized_XTest,df_XTest[FSL_papercolumns]],axis=1)
    elif Featurechoose_lst!=None:
        FSL_papercolumns=[Swap2PaperName(k,PprNmeMp) for k in Featurechoose_lst]
        df_Reorganized_shap_values=df_shap_values[FSL_papercolumns]
        df_Reorganized_XTest=df_XTest[FSL_papercolumns]

    assert df_Reorganized_shap_values.shape == df_Reorganized_XTest.shape
    return df_Reorganized_shap_values.values, df_Reorganized_XTest

FeatureSet_lst=['Vowel_dispersion_inter__vowel_centralization','Vowel_dispersion_inter__vowel_dispersion',#LOC_columns
                'formant_dependency',#DEP_columns
                'Trend[Vowel_dispersion_inter__vowel_centralization]_d','Trend[Vowel_dispersion_inter__vowel_dispersion]_d','Trend[Vowel_dispersion_intra]_d','Trend[formant_dependency]_d',#Trend[LOCDEP]_d
                'Trend[Vowel_dispersion_inter__vowel_centralization]_k','Trend[Vowel_dispersion_inter__vowel_dispersion]_k','Trend[Vowel_dispersion_intra]_k','Trend[formant_dependency]_k',#Trend[LOCDEP]_k
                'Syncrony[Vowel_dispersion_inter__vowel_centralization]','Syncrony[Vowel_dispersion_inter__vowel_dispersion]','Syncrony[Vowel_dispersion_intra]','Syncrony[formant_dependency]',#Syncrony[LOCDEP]
                ]

FeatureChoose_lst=[
    'FCR2',
    '$Wilks$',
    'VSA2',
    '$BCC$',
    '$BCV$',
    '$Pillai$',
    '$Hotel$',
    '$Roys$',
    '$Det(W^{-1}B)$',
    '$Tr(W^{-1}B)$',
    'pear_12',
    'spear_12',
    'kendall_12',
    'dcorr_12',
    # 'Proximity[dcorr_12]',
    # 'Convergence[pear_12',
    # 'Convergence[dcorr_12]',
    ]

Reorganized_shap_values, df_Reorganized_XTest=ReorganizeFeatures4SummaryPlot(shap_values, df_XTest, Featurechoose_lst=FeatureChoose_lst)
shap.summary_plot(Reorganized_shap_values, df_Reorganized_XTest,show=False, max_display=len(df_XTest.columns),sort=False)
plt.title(experiment_title)
plt.show()
# shap_values, df_XTest, keys=Prepare_data_for_summaryPlot_regression(Proposed_changed_info_dict,\
#                                                          feature_columns=getattr(FeatSel, 'LOC_columns') + getattr(FeatSel, 'DEP_columns') + getattr(FeatSel, 'LOCDEP_Trend_D_cols'),\
#                                                          )

# shap.summary_plot(shap_values, df_XTest,feature_names=df_XTest.columns,show=False)
# plt.title(experiment_title)
# plt.show()
# shap_values, df_XTest, keys=Prepare_data_for_summaryPlot_regression(Proposed_totalPoeple_info_dict,\
#                                                          feature_columns=getattr(FeatSel, 'LOC_columns') + getattr(FeatSel, 'DEP_columns') + getattr(FeatSel, 'LOCDEP_Trend_D_cols'),\
#                                                          )

# shap.summary_plot(shap_values, df_XTest,feature_names=df_XTest.columns,show=False)
# plt.title(experiment_title)
# plt.show()




def Get_Inspected_SHAP_df(Info_dict,logits=[0,1]):
    Top_shap_values_collect=Dict()
    for logit_number in logits:
        Top_shap_values_collect[logit_number]=pd.DataFrame()
        
        for Inspect_samp in Info_dict.keys():
            shap_info=Info_dict[Inspect_samp]
            df_shap_values=shap_info['shap_values'].loc[[logit_number]].T
            df_shap_values.columns=[Inspect_samp]
            Top_shap_values_collect[logit_number]=pd.concat([Top_shap_values_collect[logit_number],df_shap_values],axis=1)

        Top_shap_values_collect[logit_number]['Average']=Top_shap_values_collect[logit_number].mean(axis=1)
        Top_shap_values_collect[logit_number]['abs_Average']=Top_shap_values_collect[logit_number].abs().mean(axis=1)
        Top_shap_values_collect[logit_number]=Top_shap_values_collect[logit_number].sort_values(by='Average')
    return Top_shap_values_collect


Baseline_shap_values=Get_Inspected_SHAP_df(Baseline_totalPoeple_info_dict,logits=[SHAP_Inspect_logit])[SHAP_Inspect_logit]
Proposed_shap_values=Get_Inspected_SHAP_df(Proposed_totalPoeple_info_dict,logits=[SHAP_Inspect_logit])[SHAP_Inspect_logit]
# =============================================================================
# 分析統計整體SHAP value的平均
#Baseline model    
baseline_featset_lst=baseline_expstr[re.search("-",baseline_expstr).end():re.search("::",baseline_expstr).start()].split("+")
proposed_featset_lst=proposed_expstr[re.search("-",proposed_expstr).end():re.search("::",proposed_expstr).start()].split("+")


df_FeaSet_avg_Comparison_baseline=Calculate_sum_of_SHAP_vals(Baseline_shap_values,FeatSel_module=FeatSel,FeatureSet_lst=baseline_featset_lst)
df_FeaSet_avg_Comparison_proposed=Calculate_sum_of_SHAP_vals(Proposed_shap_values,FeatSel_module=FeatSel,FeatureSet_lst=proposed_featset_lst)

LargerVal_indexes=list(df_Y_pred[Improved].index)
LowerVal_indexes=list(df_Y_pred[Degraded].index)
# =============================================================================
'''

    Check feature importance

'''
# =============================================================================


Proposed_changed_shap_values=Get_Inspected_SHAP_df(Proposed_changed_info_dict,logits=[SHAP_Inspect_logit]) [SHAP_Inspect_logit]
Proposed_All_shap_values=Get_Inspected_SHAP_df(Proposed_totalPoeple_info_dict,logits=[SHAP_Inspect_logit]) [SHAP_Inspect_logit]

Self_selected_columns=['LOC_columns',
 'DEP_columns',
 'Trend[LOCDEP]_d',
]

df_catagorical_featImportance=pd.DataFrame()
df_catagorical_featImportance.name='CategoricalFeatureImportance'
for shap_valdfs in ['Proposed_changed_shap_values','Proposed_All_shap_values']:
    
    for key in Self_selected_columns:
        #feature importance is defined as absolute average of SHAP values 
        #refer to https://christophm.github.io/interpretable-ml-book/shap.html#shap-feature-importance
        df_catagorical_featImportance.loc[key,shap_valdfs]=vars()[shap_valdfs].loc[FeatSel.CategoricalName2cols[key],'abs_Average'].round(2).sum()
print(df_catagorical_featImportance)

def column_index(df, query_cols):
    # column_index(df, ['peach', 'banana', 'apple'])
    cols = df.index.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]