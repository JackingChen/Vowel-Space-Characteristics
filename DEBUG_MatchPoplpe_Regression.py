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

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import recall_score, make_scorer

from articulation.HYPERPARAM import phonewoprosody, Label
import articulation.HYPERPARAM.FeatureSelect as FeatSel
import articulation.HYPERPARAM.PaperNameMapping as PprNmeMp

import articulation.articulation
from itertools import combinations
from tqdm import tqdm
import copy
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
def CCC_numpy(y_true, y_pred):
    '''Reference numpy implementation of Lin's Concordance correlation coefficient'''
    
    # covariance between y_true and y_pred
    s_xy = np.cov([y_true, y_pred])[0,1]
    # means
    x_m = np.mean(y_true)
    y_m = np.mean(y_pred)
    # variances
    s_x_sq = np.var(y_true)
    s_y_sq = np.var(y_pred)
    
    # condordance correlation coefficient
    ccc = (2.0*s_xy) / (s_x_sq + s_y_sq + (x_m-y_m)**2)
    
    return ccc
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
    parser.add_argument('--Mergefeatures', default=True,
                        help='')
    parser.add_argument('--knn_weights', default='uniform',
                            help='path of the base directory')
    parser.add_argument('--knn_neighbors', default=2,  type=int,
                            help='path of the base directory')
    parser.add_argument('--Reorder_type', default='DKIndividual',
                            help='[DKIndividual, DKcriteria]')
    parser.add_argument('--Normalize_way', default='proposed',
                            help='')
    parser.add_argument('--ADDUtt_feature', default=False,
                            help='[DKIndividual, DKcriteria]')
    parser.add_argument('--FeatureComb_mode', default='baselineFeats',
                            help='[Add_UttLvl_feature, feat_comb3, feat_comb5, feat_comb6,feat_comb7, baselineFeats,Comb_dynPhonation,Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation, Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation_Add_UttLvl_feature]')
    parser.add_argument('--exclude_people', default=['2015_12_07_02_003','2017_03_18_01_196_1'],
                        help='what kind of data you want to get')
    args = parser.parse_args(args=[])
    return args


args = get_args()
start_point=args.start_point
experiment=args.experiment
knn_weights=args.knn_weights
knn_neighbors=args.knn_neighbors
Reorder_type=args.Reorder_type
# Add_UttLvl_feature=args.Add_UttLvl_feature
# =============================================================================
columns=[
'intensity_mean_mean(A:,i:,u:)', 'meanF0_mean(A:,i:,u:)',
       'stdevF0_mean(A:,i:,u:)', 'hnr_mean(A:,i:,u:)',
       'localJitter_mean(A:,i:,u:)', 'localabsoluteJitter_mean(A:,i:,u:)',
       'rapJitter_mean(A:,i:,u:)', 'ddpJitter_mean(A:,i:,u:)',
       'localShimmer_mean(A:,i:,u:)', 'localdbShimmer_mean(A:,i:,u:)',
]



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


label_choose=['ADOS_C']


pearson_scorer = make_scorer(pearsonr, greater_is_better=False)

df_formant_statistics_CtxPhone_collect_dict=Dict()
# =============================================================================

# =============================================================================
'''

    Feature merging function
    
    Ths slice of code provide user to manually make functions to combine df_XXX_infos

'''
# =============================================================================
def MERGEFEATURES():
    # =============================================================================
    '''

        Feature merging function
        
        Ths slice of code provide user to manually make functions to combine df_XXX_infos

    '''
    # =============================================================================
    # dataset_role='ASD_DOCKID'
    for dataset_role in ['ASD_DOCKID']:
        if args.ADDUtt_feature==True:
            role=dataset_role.split("_")[0] #ASD or TD
        Merg_filepath={}
        Merg_filepath['static_feautre_LOC']='Features/artuculation_AUI/Vowels/Formants/{Normalize_way}/Formant_AUI_tVSAFCRFvals_KID_From{dataset_role}.pkl'.format(dataset_role=dataset_role,Normalize_way=args.Normalize_way)
        # Merg_filepath['static_feautre_phonation']='Features/artuculation_AUI/Vowels/Phonation/Phonation_meanvars_KID_From{dataset_role}.pkl'.format(dataset_role=dataset_role)
        Merg_filepath['dynamic_feature_LOC']='Features/artuculation_AUI/Interaction/Formants/{Normalize_way}/Syncrony_measure_of_variance_DKIndividual_{dataset_role}.pkl'.format(dataset_role=dataset_role,Normalize_way=args.Normalize_way)
        Merg_filepath['dynamic_feature_phonation']='Features/artuculation_AUI/Interaction/Phonation/Syncrony_measure_of_variance_phonation_{dataset_role}.pkl'.format(dataset_role=dataset_role)
        
        merge_out_path='Features/RegressionMerged_dfs/{Normalize_way}/{dataset_role}/'.format(
            knn_weights=knn_weights,
            knn_neighbors=knn_neighbors,
            Reorder_type=Reorder_type,
            dataset_role=dataset_role,
            Normalize_way=args.Normalize_way
            )
        if not os.path.exists(merge_out_path):
            os.makedirs(merge_out_path)
        
        df_infos_dict=Dict()
        for keys, paths in Merg_filepath.items():
            df_data=pickle.load(open(paths,"rb")).sort_index()
            #!!!!!!!!!!!!!!!!!  注意這邊我們人為移除一些資訊量太少的人
            for key_wrds in args.exclude_people:
                if df_data.index.str.contains(key_wrds).any():
                    df_data=df_data.drop(index=key_wrds)
            df_infos_dict[keys]=df_data
        
        Merged_df_dict=Dict()
        comb1 = list(combinations(list(Merg_filepath.keys()), 1))
        comb2 = list(combinations(list(Merg_filepath.keys()), 2))
        if args.ADDUtt_feature==True: # output only Utt features
            OutPklpath=merge_out_path+ "Utt_feature.pkl"
            #!!!!!!!!!!!!!!!!!  注意這邊我們人為移除一些資訊量太少的人
            df_data=Utt_featuresCombinded_dict[role].copy()
            for key_wrds in args.exclude_people:
                if df_data.index.str.contains(key_wrds).any():
                    df_data=df_data.drop(index=key_wrds)
            pickle.dump(df_data,open(OutPklpath,"wb"))    
            # pickle.dump(Utt_featuresCombinded_dict[role],open(OutPklpath,"wb"))
        for c in comb1:
            e1=c[0]

            if args.ADDUtt_feature==True: # Merge with Utt features
                Merged_df_dict[e1]=Merge_dfs(df_infos_dict[e1],Utt_featuresCombinded_dict[role])
                OutPklpath=merge_out_path+"Utt_feature+"+ e1 + ".pkl"
            else:
                Merged_df_dict[e1]=df_infos_dict[e1]
                OutPklpath=merge_out_path+ e1 + ".pkl"
            pickle.dump(Merged_df_dict[e1],open(OutPklpath,"wb"))
            # print(len(Merged_df_dict[e1]))
            # assert Merged_df_dict[e1].isna().any().any() !=True
        for c in comb2:
            e1, e2=c
            
            if args.ADDUtt_feature==True:
                Merged_df_dict['+'.join(c)]=Merge_dfs(df_infos_dict[e1],df_infos_dict[e2])
                Merged_df_dict['+'.join(c)]=Merge_dfs(Merged_df_dict['+'.join(c)],Utt_featuresCombinded_dict[role])
                OutPklpath=merge_out_path+"Utt_feature+"+'+'.join(c)+".pkl"
            else:
                Merged_df_dict['+'.join(c)]=Merge_dfs(df_infos_dict[e1],df_infos_dict[e2])
                OutPklpath=merge_out_path+'+'.join(c)+".pkl"
            pickle.dump(Merged_df_dict['+'.join(c)],open(OutPklpath,"wb"))
            # print(len(Merged_df_dict['+'.join(c)]))
        # Condition for : Columns_comb3 = All possible LOC feature combination + phonation_proximity_col
        c = ('static_feautre_LOC', 'dynamic_feature_LOC', 'dynamic_feature_phonation')
        e1, e2, e3=c
        
        
        if args.ADDUtt_feature==True:
            Merged_df_dict['Utt_features'+'+'+'+'.join(c)]=Merge_dfs(df_infos_dict[e1],df_infos_dict[e2])
            Merged_df_dict['Utt_features'+'+'+'+'.join(c)]=Merge_dfs(Merged_df_dict['Utt_features'+'+'+'+'.join(c)],df_infos_dict[e3])
            Merged_df_dict['Utt_features'+'+'+'+'.join(c)]=Merge_dfs(Merged_df_dict['Utt_features'+'+'+'+'.join(c)],Utt_featuresCombinded_dict[role])
            OutPklpath=merge_out_path+'Utt_features+'+'+'.join(c)+".pkl"
            pickle.dump(Merged_df_dict['Utt_features'+'+'+'+'.join(c)],open(OutPklpath,"wb"))
        else:
            Merged_df_dict['+'.join(c)]=Merge_dfs(df_infos_dict[e1],df_infos_dict[e2])
            Merged_df_dict['+'.join(c)]=Merge_dfs(Merged_df_dict['+'.join(c)],df_infos_dict[e3])
            OutPklpath=merge_out_path+'+'.join(c)+".pkl"
            pickle.dump(Merged_df_dict['+'.join(c)],open(OutPklpath,"wb"))
if args.Mergefeatures:
    MERGEFEATURES()

        


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
    Merge_feature_path='RegressionMerged_dfs/{Normalize_way}/{dataset_role}/'.format(
            knn_weights=knn_weights,
            knn_neighbors=knn_neighbors,
            Reorder_type=Reorder_type,
            dataset_role='ASD_DOCKID',
            Normalize_way=args.Normalize_way
            )
ChooseData_manual=['static_feautre_LOC','dynamic_feature_LOC','dynamic_feature_phonation']
# ChooseData_manual=['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation']
# ChooseData_manual=['Utt_features+static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation']
# ChooseData_manual=None

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

Session_level_all_new=copy.deepcopy(Session_level_all)
# =============================================================================
# 
# =============================================================================
# =============================================================================
# 
# =============================================================================# =============================================================================
# 
# =============================================================================
class ADOSdataset():
    def __init__(self,):
        self.featurepath='Features_BeforeTASLPreview120230607/'            
        self.N=2
        self.LabelType=Dict()
        self.LabelType['ADOS_S']='regression'
        self.LabelType['ADOS_C']='regression'
        self.LabelType['ADOS_D']='regression'
        self.LabelType['ADOS_cate']='classification'
        self.LabelType['ASDTD']='classification'
        self.Fractionfeatures_str='Features_BeforeTASLPreview120230607/artuculation_AUI/Vowels/Fraction/*.pkl'    
        self.FeatureCombs=Dict()

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

ChooseData_manual=['static_feautre_LOC','dynamic_feature_LOC','dynamic_feature_phonation']
# ChooseData_manual=['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation']
# ChooseData_manual=['Utt_features+static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation']
# ChooseData_manual=None

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
Session_level_all_old=copy.deepcopy(Session_level_all)
#%%
import pandas as pd

def compare_dicts(dict1, dict2, path=""):
    if set(dict1.keys()) != set(dict2.keys()):
        print(f"Key mismatch at path: {path}")
        return False

    for key in dict1:
        if "TD vs df_feature_ASD" in key:
            continue
        value1 = dict1[key]
        value2 = dict2[key]
        current_path = f"{path}.{key}" if path else key

        if isinstance(value1, dict) and isinstance(value2, dict):
            if not compare_dicts(value1, value2, path=current_path):
                return False
        elif isinstance(value1, (pd.DataFrame, pd.Series)) and isinstance(value2, (pd.DataFrame, pd.Series)):
            if not compare_data_objects(value1, value2, path=current_path):
                return False
        elif value1 != value2:
            print(f"Value mismatch at path: {current_path}")
            return False

    return True

def compare_data_objects(obj1, obj2, path, tolarence=1e-5):
    cond=((obj1 - obj2)<tolarence).all(axis=None)
    if isinstance(obj1, pd.DataFrame) and isinstance(obj2, pd.DataFrame):
        if not cond:
            print(f"DataFrame mismatch at path: {path}")
            return False
    elif isinstance(obj1, pd.Series) and isinstance(obj2, pd.Series):
        if not cond:
            print(f"Series mismatch at path: {path}")
            return False
    else:
        print(f"Value type mismatch at path: {path}")
        return False

    return True


compare_dicts(Session_level_all_old,Session_level_all_new)
# # len(Session_level_all_old)
# # len(Session_level_all_new)
# tolarence=1e-5
# diff12_df=Session_level_all_old['TD vs df_feature_lowMinimal_CSS >> LOCDEP_Proximity_cols::ASDTD'].X - Session_level_all_new['TD vs df_feature_lowMinimal_CSS >> LOCDEP_Proximity_cols::ASDTD'].X
# # diff21_df=Session_level_all_new['TD vs df_feature_lowMinimal_CSS >> LOCDEP_Proximity_cols::ASDTD'].X - Session_level_all_old['TD vs df_feature_lowMinimal_CSS >> LOCDEP_Proximity_cols::ASDTD'].X
# df_old=Session_level_all_old['TD vs df_feature_lowMinimal_CSS >> LOCDEP_Proximity_cols::ASDTD'].X
# df_new=Session_level_all_new['TD vs df_feature_lowMinimal_CSS >> LOCDEP_Proximity_cols::ASDTD'].X

