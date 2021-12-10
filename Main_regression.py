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
from sklearn.metrics import recall_score
# path_app = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(path_app+"/utils")

from articulation.HYPERPARAM import phonewoprosody, Label
import articulation.articulation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


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
    parser.add_argument('--UseManualCtxFeat', default=True,
                        help='')
    parser.add_argument('--Plot', default=True,
                        help='')
    parser.add_argument('--selectModelScoring', default='neg_mean_squared_error',
                        help='')
    args = parser.parse_args()
    return args


args = get_args()
start_point=args.start_point
experiment=args.experiment

# =============================================================================
# Feature
Session_level_all=Dict()
columns=[
    'Divergence[within_covariance_norm(A:,i:,u:)]',
    'Divergence[within_variance_norm(A:,i:,u:)]',    
    'Divergence[between_covariance_norm(A:,i:,u:)]',    
    'Divergence[between_variance_norm(A:,i:,u:)]',    
    'Divergence[sam_wilks_lin_norm(A:,i:,u:)]',    
    'Divergence[pillai_lin_norm(A:,i:,u:)]',
    'Divergence[pillai_lin_norm(A:,i:,u:)]_var_p1',
    'Divergence[within_covariance_norm(A:,i:,u:)]_var_p1',
    'Divergence[within_variance_norm(A:,i:,u:)]_var_p1',
    'Divergence[between_covariance_norm(A:,i:,u:)]_var_p1',
    'Divergence[between_variance_norm(A:,i:,u:)]_var_p1',
    'Divergence[sam_wilks_lin_norm(A:,i:,u:)]_var_p1',
    'Divergence[pillai_lin_norm(A:,i:,u:)]_var_p2',
    'Divergence[within_covariance_norm(A:,i:,u:)]_var_p2',    
    'Divergence[within_variance_norm(A:,i:,u:)]_var_p2',    
    'Divergence[between_covariance_norm(A:,i:,u:)]_var_p2',    
    'Divergence[between_variance_norm(A:,i:,u:)]_var_p2',    
    'Divergence[sam_wilks_lin_norm(A:,i:,u:)]_var_p2',
    ]
# columns=['totalword']

# columns=[
#     'FCR+AUINum',
#     'VSA1+AUINum',
#     'FCR*AUINum',
#     'VSA1*AUINum',
#     'FCR',
#     'VSA1',
#     'between_covariance(A:,i:,u:)',
#     'between_variance(A:,i:,u:)',
#     'within_covariance(A:,i:,u:)',
#     'within_variance(A:,i:,u:)',
#     'sam_wilks_lin(A:,i:,u:)',
#     'pillai_lin(A:,i:,u:)',
#     'u_num+i_num+a_num',
#     ]

# columns=[
#     'FCR+AUINum',
#     'VSA1+AUINum',
#     'FCR*AUINum',
#     'VSA1*AUINum',
#     'FCR',
#     'VSA1',
#     'between_covariance(A:,i:,u:)',
#     'between_variance(A:,i:,u:)',
#     'within_covariance(A:,i:,u:)',
#     'within_variance(A:,i:,u:)',
#     'sam_wilks_lin(A:,i:,u:)',
#   'hotelling_lin(A:,i:,u:)',
#   'u_num+i_num+a_num',
    # 'pillai_lin_norm(A:,i:,u:)',
    # 'pillai_lin(A:,i:,u:)'
    # ]
# featuresOfInterest=[['u_num+i_num+a_num']]
# featuresOfInterest=[['between_covariance(A:,i:,u:)'],['between_variance(A:,i:,u:)'],['u_num+i_num+a_num']]
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
FeatureLabelMatch=[['TD_normal vs ASDSevere_agesexmatch','ASDTD'],
                    ['TD_normal vs ASDMild_agesexmatch','ASDTD'],
                    ]
df_formant_statistics_CtxPhone_collect_dict=Dict()


# =============================================================================

class ADOSdataset():
    def __init__(self,):
        self.featurepath='Features'            
        self.N=2
        self.LabelType=Dict()
        self.LabelType['ADOS_C']='regression'
        self.LabelType['ADOS_cate']='classification'
        self.LabelType['ASDTD']='classification'
        self.Fractionfeatures_str='Features/artuculation_AUI/Vowels/Fraction/*.pkl'    
        self.FeatureCombs=Dict()
        self.FeatureCombs['TD_normal vs ASDSevere_agesexmatch']=['df_formant_statistic_TD_normal', 'df_formant_statistic_agesexmatch_ASDSevere']
        self.FeatureCombs['TD_normal vs ASDMild_agesexmatch']=['df_formant_statistic_TD_normal', 'df_formant_statistic_agesexmatch_ASDMild']
        self.FeatureCombs['Notautism vs ASD']=['df_formant_statistic_77_Notautism', 'df_formant_statistic_77_ASD']
        self.FeatureCombs['ASD vs Autism']=['df_formant_statistic_77_ASD', 'df_formant_statistic_77_Autism']
        self.FeatureCombs['Notautism vs Autism']=['df_formant_statistic_77_Notautism', 'df_formant_statistic_77_Autism']
    
        self._FeatureBuild()
    def Get_FormantAUI_feat(self,label_choose,pickle_path,featuresOfInterest=['MSB_f1','MSB_f2','MSB_mix'],filterbyNum=True,**kwargs):
        self.featuresOfInterest=featuresOfInterest
        arti=articulation.articulation.Articulation()
        if not kwargs and len(pickle_path)>0:
            df_tmp=pickle.load(open(pickle_path,"rb"))
        elif len(kwargs)>0: # usage Get_FormantAUI_feat(...,key1=values1):
            for k, v in kwargs.items(): #there will be only one element
                df_tmp=kwargs[k]

        if filterbyNum:
            df_tmp=arti.BasicFilter_byNum(df_tmp,N=self.N)
        
        # experiment!! remove soon
        # df_tmp['a+b+c']=df_tmp[['u_num', 'a_num', 'i_num']].sum(axis=1)
        # print(df_tmp['a+b+c'])
        
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
            df_tmp=pickle.load(open(file,"rb"))
            Features[feat_name]=df_tmp
        for keys in self.FeatureCombs.keys():
            combF=[Features[k] for k in self.FeatureCombs[keys]]
            Features_comb[keys]=pd.concat(combF)
        
        self.Features_comb=Features_comb


ados_ds=ADOSdataset()


manual_sel_feat_path='Manual_sel_faetures_raw'
# manual_sel_feat_path='Manual_sel_features_raw'
if not os.path.exists(manual_sel_feat_path):
    raise FileNotFoundError


if args.UseManualCtxFeat:
    Manual_choosen_feature=[]
    with open(manual_sel_feat_path,'r') as f:
        content=f.read()
        for line in content.split("\n"):
            if len(line) > 0:
                Manual_choosen_feature.append(line[:re.search('__ADOS_C',line).end()])
            
def FilterFile_withinManualName(files,Manual_choosen_feature):
    files_manualChoosen=[f  for f in files if os.path.basename(f).split(".")[0]  in Manual_choosen_feature]
    return files_manualChoosen

ErrorFeat_bookeep=Dict()

Pseudo_CtxDepPhone_path='artuculation_AUI/Pseudo_CtxDepVowels'
CtxDepPhone_path='artuculation_AUI/CtxDepVowels/bkup0729'
Vowel_path='artuculation_AUI/Vowels'
Interactionfeat_path='artuculation_AUI/Interaction'
OtherFeat_path='Other/Static_BasicInfo'
# for feature_paths in [Vowel_path, CtxDepPhone_path, Pseudo_CtxDepPhone_path]:
for feature_paths in [Vowel_path]:
# for feature_paths in [Vowel_path, CtxDepPhone_path]:
    files = glob.glob(ados_ds.featurepath +'/'+ feature_paths+'/*.pkl')
    
    if args.UseManualCtxFeat and len(Manual_choosen_feature)>0 and feature_paths == CtxDepPhone_path:
        files = FilterFile_withinManualName(files, Manual_choosen_feature)
    
    # load features from file
    # for file in files: #iterate over features
    #     feat_=os.path.basename(file).split(".")[0]        
    #     for feat_col in featuresOfInterest:
    #         # if len(feat_col)==1:
    #         #     feat_col_ = list([feat_col]) # ex: ['MSB_f1']
    #         # else:
    #         feat_col_ = list(feat_col) # ex: ['MSB_f1']
    #         for lab_ in label_choose:
    #             X,y, featType=ados_ds.Get_FormantAUI_feat(label_choose=lab_,pickle_path=file,featuresOfInterest=feat_col_,filterbyNum=False)
    #             Item_name="{feat}::{lab}".format(feat='-'.join([feat_]+feat_col_),lab=lab_)
    #             Session_level_all[Item_name].X, \
    #                 Session_level_all[Item_name].y, \
    #                     Session_level_all[Item_name].feattype = X,y, featType
    
    #load features from fraction feats
    
    for feat_,lab_ in FeatureLabelMatch:
        # feat_=key
        for feat_col in featuresOfInterest:
            feat_col_ = list(feat_col) # ex: ['MSB_f1']
            
            # for lab_ in label_choose:
            X,y, featType=ados_ds.Get_FormantAUI_feat(label_choose=lab_,pickle_path='',featuresOfInterest=feat_col_,feat_=ados_ds.Features_comb[feat_])
            
            if np.isnan(X).any() or np.isnan(y).any():
                print("Feat: ",feat_col_,'Contains nan')
                ErrorFeat_bookeep['{0} {1} {2}'.format(feat_,lab_,feat_col_)].X=X
                ErrorFeat_bookeep['{0} {1} {2}'.format(feat_,lab_,feat_col_)].y=y
                continue
            
            Item_name="{feat}::{lab}".format(feat='-'.join([feat_]+feat_col_),lab=lab_)
            Session_level_all[Item_name].X, \
                Session_level_all[Item_name].y, \
                    Session_level_all[Item_name].feattype = X,y, featType



# =============================================================================
# Model parameters
# =============================================================================
C_variable=np.array([0.01, 0.1,0.5,1.0,10.0, 50.0, 100.0, 1000.0])
# C_variable=np.array([0.01, 0.1,0.5,1.0,10.0, 100.0, 1000.0, 10000.0])
n_estimator=[2, 4, 8, 16, 32, 64]
Scoring=['neg_mean_absolute_error','neg_mean_squared_error']

'''

    Classifier

'''
Classifier={}
Classifier['SVC']={'model':sklearn.svm.SVC(),\
                  'parameters':{'C':C_variable,\
                    'kernel': ['rbf'],\
                                }}

Classifier['LR']={'model':sklearn.linear_model.LogisticRegression(),\
                  'parameters':{'C':C_variable,\
                                }}
    
Classifier['DT']={'model':sklearn.tree.DecisionTreeClassifier(),\
                  'parameters':{'criterion':['gini','entropy'],
                                'splitter':['splitter','random'],\
                                }}
    



# Classifier['SVC']={'model':sklearn.svm.SVC(),\
#                   'parameters':{'C':[100.0],\
#                     'kernel': ['rbf'],\
#                                 }}

'''

    Regressor

'''
###############################################################################
# Classifier['EN']={'model':ElasticNet(random_state=0),\
#                   'parameters':{'alpha':np.arange(0,1,0.25),\
#                                 'l1_ratio': np.arange(0,1,0.25)}} #Just a initial value will be changed by parameter tuning
#                                                     # l1_ratio = 1 is the lasso penalty
# Classifier['SVR']={'model':sklearn.svm.SVR(),\
#                   'parameters':{'C':C_variable,\
#                     'kernel': ['rbf'],\
#                                 }}

# Classifier['LinR']={'model':sklearn.linear_model.LinearRegression(),\
#                   'parameters':{'fit_intercept':[True,False],\
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
        Labels = Session_level_all.X[feature_keys]
        print("=====================Cross validation start==================")
        p_grid=clf['parameters']
        Gclf = GridSearchCV(estimator=clf['model'], param_grid=p_grid, scoring=args.selectModelScoring, cv=loo, refit=True, n_jobs=-1)
        # Score=cross_val_score(Gclf, features.X, features.y, cv=loo) 
        CVpredict=cross_val_predict(Gclf, features.X, features.y, cv=loo)           
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