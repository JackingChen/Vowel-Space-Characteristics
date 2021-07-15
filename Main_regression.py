#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:56:45 2021

@author: jackchen
"""

import os, sys
import pandas as pd
import numpy as np

import glob
import pickle

from scipy.stats import spearmanr,pearsonr 
from sklearn.model_selection import LeaveOneOut

from addict import Dict
import functions
import argparse
from scipy.stats import zscore

from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
import torch

# path_app = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(path_app+"/utils")

from articulation.HYPERPARAM import phonewoprosody, Label
import articulation.articulation


def Assert_labelfeature(feat_name,lab_name):
    # =============================================================================
    #     To check if the label match with feature
    # =============================================================================
    for i,n in enumerate(feat_name):
        assert feat_name[i] == lab_name[i]


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
    args = parser.parse_args()
    return args

args = get_args()
start_point=args.start_point
experiment=args.experiment

# =============================================================================
# Feature
Session_level_all=Dict()
featuresOfInterest=[['MSB_f1(A:,i:,u:)', 'MSB_f2(A:,i:,u:)']]
# featuresOfInterest=[['u_num', 'a_num', 'i_num']]
# featuresOfInterest=[['u_num', 'a_num', 'i_num','a+b+c']]
# featuresOfInterest=[['F_vals_f1(u:,i:,A:)','F_vals_f2(u:,i:,A:)', 'F_val_mix(u:,i:,A:)'] ,
#         ['F_vals_f1(u:,i:)', 'F_vals_f2(u:,i:)', 'F_val_mix(u:,i:)'], 
#         ['F_vals_f1(u:,A:)','F_vals_f2(u:,A:)','F_val_mix(u:,A:)'],
#         ['F_vals_f1(i:,A:)','F_vals_f2(i:,A:)','F_val_mix(i:,A:)']]
# featuresOfInterest=[['F_vals_f1(u:,i:,A:)',
#  'F_vals_f2(u:,i:,A:)',
#  'F_val_mix(u:,i:,A:)',
#  'F_vals_f1(u:,i:)',
#  'F_vals_f2(u:,i:)',
#  'F_val_mix(u:,i:)',
#  'F_vals_f1(u:,A:)',
#  'F_vals_f2(u:,A:)',
#  'F_val_mix(u:,A:)',
#  'F_vals_f1(i:,A:)',
#  'F_vals_f2(i:,A:)',
#  'F_val_mix(i:,A:)']]
# featuresOfInterest=[['F1(a-i)_mean','F1(a-i)_25%', 'F1(a-i)_50%','F1(a-i)_max'], ['F1(a-u)_mean',  'F1(a-u)_25%',
#         'F1(a-u)_50%', 'F1(a-u)_max'], ['F2(i-u)_mean',
#         'F2(i-u)_50%','F2(i-u)_max']]
# featuresOfInterest=[['F1(a-i)_mean',
#  'F1(a-i)_min',
#  'F1(a-i)_25%',
#  'F1(a-i)_50%',
#  'F1(a-i)_75%',
#  'F1(a-i)_max',
#  'F1(a-u)_mean',
#  'F1(a-u)_min',
#  'F1(a-u)_25%',
#  'F1(a-u)_50%',
#  'F1(a-u)_75%',
#  'F1(a-u)_max',
#  'F2(i-u)_mean',
#  'F2(i-u)_min',
#  'F2(i-u)_25%',
#  'F2(i-u)_50%',
#  'F2(i-u)_75%',
#  'F2(i-u)_max']]
# =============================================================================

class ADOSdataset():
    def __init__(self,):
        self.featurepath='Features'            
    def Get_FormantAUI_feat(self,label_choose,pickle_path,featuresOfInterest=['MSB_f1','MSB_f2','MSB_mix']):
        self.featuresOfInterest=featuresOfInterest

        df_tmp=pickle.load(open(pickle_path,"rb"))
        arti=articulation.articulation.Articulation()
        df_tmp=arti.BasicFilter_byNum(df_tmp,N=1)
        
        # experiment!! remove soon
        df_tmp['a+b+c']=df_tmp[['u_num', 'a_num', 'i_num']].sum(axis=1)
        # print(df_tmp['a+b+c'])
        
        for people in df_tmp.index:
            lab=Label.label_raw[label_choose][Label.label_raw['name']==people]
            df_tmp.loc[people,'ADOS']=lab.values
            
            feature_array=df_tmp[featuresOfInterest].values
            y_array=df_tmp['ADOS'].values
            
        return feature_array, y_array


label_choose=['ADOS_C','Multi1','Multi2','Multi3','Multi4']
ados_ds=ADOSdataset()


CtxDepPhone_path='artuculation_AUI/CtxDepVowels'
Vowel_path='artuculation_AUI/Vowels'
# for feature_paths in [Vowel_path, CtxDepPhone_path]:
for feature_paths in [Vowel_path]:
    files = glob.glob(ados_ds.featurepath +'/'+ feature_paths+'/*.pkl')
    for file in files: #iterate over features
        feat_=os.path.basename(file).split(".")[0]
        for feat_col in featuresOfInterest:
            # if len(feat_col)==1:
            #     feat_col_ = list([feat_col]) # ex: ['MSB_f1']
            # else:
            feat_col_ = list(feat_col) # ex: ['MSB_f1']
            for lab_ in label_choose:
                X,y=ados_ds.Get_FormantAUI_feat(label_choose=lab_,pickle_path=file,featuresOfInterest=feat_col_)
                Item_name="{feat}::{lab}".format(feat='-'.join([feat_]+feat_col_),lab=lab_)
                Session_level_all[Item_name].X, \
                    Session_level_all[Item_name].y= X,y

        


# =============================================================================
# Model parameters
# =============================================================================
C_variable=np.array([0.1,0.25,0.5,1.0,5.0,10.0])
n_estimator=[2, 4, 8, 16, 32, 64]
Scoring=['neg_mean_absolute_error','neg_mean_squared_error']

'''

    Classifier

'''
Classifier={}
# Classifier['EN']={'model':ElasticNet(random_state=0),\
#                   'parameters':{'alpha':np.arange(0,1,0.25),\
#                                 'l1_ratio': np.arange(0,1,0.25)}} #Just a initial value will be changed by parameter tuning
                                                    # l1_ratio = 1 is the lasso penalty
Classifier['SVR']={'model':SVR(),\
                  'parameters':{'C':C_variable,\
                                }} #Just a initial value will be changed by parameter tuning
                                                   # l1_ratio = 1 is the lasso penalty

    
# Classifier['EN']={'model':ElasticNet(random_state=0),\
#               'parameters':{'alpha':[0.75],\
#                             'l1_ratio': [0.75]}} #Just a initial value will be changed by parameter tuning
                                                  # l1_ratio = 1 is the lasso penalty


loo=LeaveOneOut()

# =============================================================================
# Outputs
Best_predict_optimize={}

# df_best_result_r2=pd.DataFrame([],columns=Label.label_choose_EN,index=Session_level_all.keys())
# df_best_result_pear=pd.DataFrame([],columns=Label.label_choose_EN,index=Session_level_all.keys())
# df_best_result_spear=pd.DataFrame([],columns=Label.label_choose_EN,index=Session_level_all.keys())
# df_best_cross_score=pd.DataFrame([],columns=Label.label_full_EN.keys(),index=Session_level_all.keys())

df_best_result_r2=pd.DataFrame([])
df_best_result_pear=pd.DataFrame([])
df_best_result_spear=pd.DataFrame([])
df_best_cross_score=pd.DataFrame([])
# =============================================================================
Result_path="RESULTS/"
if not os.path.exists(Result_path):
    os.makedirs(Result_path)
final_result_file="ADOS.xlsx"


count=0
OutFeature_dict=Dict()
for clf_keys, clf in Classifier.items(): #Iterate among different classifiers 
    writer_clf = pd.ExcelWriter(Result_path+"/"+clf_keys+"_"+args.Feature_mode+"_"+final_result_file, engine = 'xlsxwriter')
    for feature_lab_str, features in Session_level_all.items():
        feature_keys, label_keys= feature_lab_str.split("::")
        Labels = Session_level_all.X[feature_keys]
        print("=====================Cross validation start==================")
        p_grid=clf['parameters']
        Gclf = GridSearchCV(estimator=clf['model'], param_grid=p_grid, scoring='neg_mean_squared_error', cv=loo, refit=True, n_jobs=-1)
        Score=cross_val_score(Gclf, features.X, features.y, cv=loo)            
        Gclf.fit(features.X,features.y)
        if clf_keys == "EN":
            print('The coefficient of best estimator is: ',Gclf.best_estimator_.coef_)
        
        print("The best score with scoring parameter: 'r2' is", Gclf.best_score_)
        print("The best parameters are :", Gclf.best_params_)
        best_parameters=Gclf.best_params_
        best_score=Gclf.best_score_
        best_parameters.update({'best_score':best_score})
        cv_results_info=Gclf.cv_results_

        CVpredict=cross_val_predict(Gclf, features.X, features.y, cv=loo)
        regression=r2_score(features.y,CVpredict )
        pearson_result, pearson_p=pearsonr(features.y,CVpredict )
        spear_result, spearman_p=spearmanr(features.y,CVpredict )
        print('Feature {0}, label {1} ,spear_result {2}, Cross_score {3}'.format(feature_keys, label_keys,spear_result,Score.mean()))
#
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
            
        df_best_result_r2.loc[feature_keys,label_keys]=regression
        df_best_result_pear.loc[feature_keys,label_keys]=pearson_result
        df_best_result_spear.loc[feature_keys,label_keys]='{0}/{1}'.format(np.round(spear_result,3),np.round(spearman_p,6))
        df_best_result_spear.loc[feature_keys,'de-zero_num']=len(features.X)
        df_best_cross_score.loc[feature_keys,label_keys]=Score.mean()

df_best_result_r2.to_excel(writer_clf,sheet_name="R2")
df_best_result_pear.to_excel(writer_clf,sheet_name="pear")
df_best_result_spear.to_excel(writer_clf,sheet_name="spear")
df_best_result_spear.to_csv(Result_path+"/"+clf_keys+"_"+args.Feature_mode+"_spearman.csv")
writer_clf.save()