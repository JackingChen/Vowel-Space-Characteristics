#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:14:41 2020

@author: jackchen


This script is a inherited from Analyze_F1F2_tVSA_FCR.py
1. Data prepare area: 
    Gather raw data of the three critical monophthongs (F1 & F2) and save in: df_formant_statistic.
    
    1-1 Filtering area:
        Filter out the outliers by IQR method (defined in muti.FilterUttDictsByCriterion_map)
    
2. Feature calculating area
    a. We use articulation.calculate_features() method to calculate LOC features 
    
3. Evaluation area


Input:
    Phonation_utt_symb

Output:
    df_phonation_statistic_77

"""

import pickle
import argparse
from addict import Dict
import numpy as np
import pandas as pd
from articulation.HYPERPARAM import phonewoprosody, Label
import matplotlib.pyplot as plt
from itertools import combinations

from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from scipy import stats
from scipy.stats import spearmanr,pearsonr 
import statistics 
import os, glob, sys
import statsmodels.api as sm
from varname import nameof
from tqdm import tqdm
import re
from multiprocessing import Pool, current_process
# from articulation.articulation import Articulation
import articulation.Multiprocess as Multiprocess
from datetime import datetime as dt
import pathlib
from phonation.phonation import  Phonation
from scipy import special, stats
import warnings
from utils_jack  import  Formant_utt2people_reshape, Gather_info_certainphones, \
                         FilterUttDictsByCriterion, GetValuelimit_IQR, \
                         Get_aligned_sequences, WER, Get_Vowels_AUI
from metric import Evaluation_method     

def criterion_filter(df_formant_statistic,N=10,\
                     constrain_sex=-1, constrain_module=-1,constrain_agemax=-1,constrain_ADOScate=-1,constrain_agemin=-1,\
                     evictNamelst=[]):
    filter_bool=np.logical_and(df_formant_statistic['u_num']>N,df_formant_statistic['a_num']>N)
    # filter_bool=np.logical_and(df_formant_statistic['a_num']>N)
    filter_bool=np.logical_and(filter_bool,df_formant_statistic['i_num']>N)
    if constrain_sex != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['sex']==constrain_sex)
    if constrain_module != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['Module']==constrain_module)
    if constrain_agemax != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['age']<=constrain_agemax)
    if constrain_agemin != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['age']>=constrain_agemin)
    if constrain_ADOScate != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['ADOS_cate_C']==constrain_ADOScate)
        
    if len(evictNamelst)>0:
        for name in evictNamelst:
            filter_bool.loc[name]=False
    # get rid of nan values
    filter_bool=np.logical_and(filter_bool,~df_formant_statistic.isna().T.any())
    return df_formant_statistic[filter_bool]



def NameMatchAssertion(Formants_people_symb,name):
    ''' check the name in  Formants_people_symb matches the names in label'''
    for name in Formants_people_symb.keys():
        assert name in name

def find_group(ADOS_label,group):
    flag=-1
    for i,g in enumerate(group):
        if ADOS_label in g:
            flag=0
            return i
    if flag ==-1: # -1 if for conditions that groups are hisglow groups (which means we don't have middle class group) 
        return flag


def to_matrix(l, n): #Create a 2D list out of 1D list
    return [l[i:i+n] for i in range(0, len(l), n)]


    
def Process_IQRFiltering_Phonation_Multi(Formants_utt_symb, limit_people_rule,\
                               outpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',\
                               prefix='Formants_utt_symb',\
                               suffix='Phonation_utt_symb'):
    pool = Pool(int(os.cpu_count()))
    keys=[]
    interval=20
    for i in range(0,len(Formants_utt_symb.keys()),interval):
        # print(list(combs_tup.keys())[i:i+interval])
        keys.append(list(Formants_utt_symb.keys())[i:i+interval])
    flat_keys=[item for sublist in keys for item in sublist]
    assert len(flat_keys) == len(Formants_utt_symb.keys())
    muti=Multiprocess.Multi()
    final_results=pool.starmap(muti.FilterUttDictsByCriterion_map, [([Formants_utt_symb,Formants_utt_symb,file_block,limit_people_rule]) for file_block in tqdm(keys)])
    # final_results=pool.starmap(FilterUttDictsByCriterion_map, [([Formants_utt_symb,Formants_utt_symb,file_block,limit_people_rule]) for file_block in tqdm(keys)])
    
    Formants_utt_symb_limited=Dict()
    for load_file_tmp,_ in final_results:        
        for utt, df_utt in load_file_tmp.items():
            Formants_utt_symb_limited[utt]=df_utt
    
    pickle.dump(Formants_utt_symb_limited,open(outpath+"/[Analyzing]{0}_limited_{1}.pkl".format(prefix,suffix),"wb"))
    print('Formants_utt_symb saved to ',outpath+"/[Analyzing]{0}_limited_{1}.pkl".format(prefix,suffix))


# =============================================================================
def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice/articulation',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--inpklpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--outpklpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--reFilter', default=False,
                            help='')
    parser.add_argument('--correlation_type', default='spearmanr',
                            help='spearmanr|pearsonr')
    parser.add_argument('--label_choose_lst', default=['ADOS_C'],
                            help='path of the base directory')
    parser.add_argument('--Stat_med_str_VSA', default='mean',
                            help='path of the base directory')
    parser.add_argument('--dataset_role', default='KID_FromASD_DOCKID',
                            help='kid_TD| kid88')
    # parser.add_argument('--Inspect_features', default=['F1','F2'],
    #                         help='')
    parser.add_argument('--Inspect_features_phonations', default=['intensity_mean', 'meanF0', 'stdevF0', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer'],
                            help="['duration', 'intensity_mean', 'meanF0', 'stdevF0', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'aqpq5Shimmer', 'apq11Shimmer', 'ddaShimmer']")
    
    args = parser.parse_args()
    return args

args = get_args()
base_path=args.base_path

# =============================================================================
'''
    
    1. Data prepare area

'''
# =============================================================================
''' parse namespace '''
args = get_args()
base_path=args.base_path
pklpath=args.inpklpath
label_choose_lst=args.label_choose_lst # labels are too biased
role=args.dataset_role
outpklpath=args.inpklpath+"/Session_formants_people_vowel_feat/"
if not os.path.exists(outpklpath):
    os.makedirs(outpklpath)


Phonation_utt_symb=pickle.load(open(pklpath+"/Phonation_utt_symb_{role}.pkl".format(role=role),'rb'))
label_set=['ADOS_C','ADOS_S','ADOS_SC']


# =============================================================================
'''

    1-1. Filtering area
    
    Filter out data using by 1.5*IQR

'''
PhoneMapp_dict=phonewoprosody.PhoneMapp_dict
PhoneOfInterest=list(PhoneMapp_dict.keys())
# =============================================================================


''' Not doing the filtering '''
Phonation_people_information=Formant_utt2people_reshape(Phonation_utt_symb,Phonation_utt_symb,Align_OrinCmp=False)
AUI_info_phonation=Gather_info_certainphones(Phonation_people_information,PhoneMapp_dict,PhoneOfInterest)
limit_people_rule_Phonation=GetValuelimit_IQR(AUI_info_phonation,PhoneMapp_dict,args.Inspect_features_phonations)

''' multi processing start '''
prefix,suffix = 'Phonation_utt_symb', role
# date_now='{0}-{1}-{2} {3}'.format(dt.now().year,dt.now().month,dt.now().day,dt.now().hour)
date_now='{0}-{1}-{2}'.format(dt.now().year,dt.now().month,dt.now().day)
outpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles'
filepath=outpath+"/[Analyzing]{0}_limited_{1}.pkl".format(prefix,suffix)
if os.path.exists(filepath) and args.reFilter==False:
    fname = pathlib.Path(filepath)
    mtime = dt.fromtimestamp(fname.stat().st_mtime)
    # filemtime='{0}-{1}-{2} {3}'.format(mtime.year,mtime.month,mtime.day,mtime.hour)
    filemtime='{0}-{1}-{2}'.format(mtime.year,mtime.month,mtime.day)
    
    # If file last modify time is not now (precisions to the hours) than we create new one
    if filemtime != date_now:
        Process_IQRFiltering_Phonation_Multi(Phonation_utt_symb,limit_people_rule_Phonation,\
                                outpath=outpath,\
                                prefix=prefix,\
                                suffix=suffix) # the results will be output as pkl file at outpath+"/[Analyzing]Phonation_utt_symb_limited.pkl"
else:
    Process_IQRFiltering_Phonation_Multi(Phonation_utt_symb,limit_people_rule_Phonation,\
                                outpath=outpath,\
                                prefix=prefix,\
                                suffix=suffix)

Phonation_utt_symb_limited=pickle.load(open(filepath,"rb"))
''' multi processing end '''
if len(limit_people_rule_Phonation) >0:
    Phonation_utt_symb=Phonation_utt_symb_limited

Phonation_people_information=Formant_utt2people_reshape(Phonation_utt_symb,Phonation_utt_symb,Align_OrinCmp=False)
AUI_info_phonation=Gather_info_certainphones(Phonation_people_information,PhoneMapp_dict,PhoneOfInterest)



# =============================================================================        
''' 

    2. Feature calculating area


'''
# =============================================================================
Vowels_AUI_phonation=Get_Vowels_AUI(AUI_info_phonation, args.Inspect_features_phonations,VUIsource="From__Formant_people_information")

# Calculate phonation features
phonation=Phonation(Inspect_features=args.Inspect_features_phonations)
phonation._updateISSegmentFeature(True)
df_phonation_statistic=phonation.calculate_features(Vowels_AUI_phonation,Label,PhoneOfInterest,label_choose_lst=label_choose_lst)

for i in range(len(df_phonation_statistic)):
    name=df_phonation_statistic.iloc[i].name
    df_phonation_statistic.loc[name,'ADOS_cate_C']=Label.label_raw[Label.label_raw['name']==name]['ADOS_cate_C'].values
pickle.dump(df_phonation_statistic,open(outpklpath+"Phonation_meanvars_{}.pkl".format(role),"wb"))

sex=-1
module=-1
agemax=-1
agemin=-1
ADOScate=-1
N=0
df_phonation_statistic_77=criterion_filter(df_phonation_statistic,\
                                        constrain_sex=sex,constrain_module=module,N=N,constrain_agemax=agemax,constrain_agemin=agemin,constrain_ADOScate=ADOScate,\
                                        evictNamelst=[])

# =============================================================================
'''

    2. Evaluation area

    We still keep this area to get a peek of the correlation result.
    The evaluation function should be the same as the one in Statistical_tests.py
    
    The evaluation module is defined in Evaluation_method()

'''
# =============================================================================


''' Calculate correlations for Phonation fetures'''
columns=[
 'intensity_mean_mean(A:,i:,u:)', 'meanF0_mean(A:,i:,u:)',
       'stdevF0_mean(A:,i:,u:)', 'hnr_mean(A:,i:,u:)',
       'localJitter_mean(A:,i:,u:)', 'localabsoluteJitter_mean(A:,i:,u:)',
       'rapJitter_mean(A:,i:,u:)', 'localShimmer_mean(A:,i:,u:)',
       'localdbShimmer_mean(A:,i:,u:)', 'apq3Shimmer_mean(A:,i:,u:)',
       'intensity_mean_var(A:,i:,u:)', 'meanF0_var(A:,i:,u:)',
       'stdevF0_var(A:,i:,u:)', 'hnr_var(A:,i:,u:)',
       'localJitter_var(A:,i:,u:)', 'localabsoluteJitter_var(A:,i:,u:)',
       'rapJitter_var(A:,i:,u:)', 'localShimmer_var(A:,i:,u:)',
       'localdbShimmer_var(A:,i:,u:)', 'apq3Shimmer_var(A:,i:,u:)'
    ]

df_phonation_statistic_77['u_num+i_num+a_num']=df_phonation_statistic_77['u_num'] +\
                                            df_phonation_statistic_77['i_num'] +\
                                            df_phonation_statistic_77['a_num']

N=0
Eval_med=Evaluation_method()
Aaadf_spearmanr_table_NoLimit=Eval_med.Calculate_correlation(label_choose_lst,df_phonation_statistic_77,N,columns,constrain_sex=-1, constrain_module=-1)

def TBMEB1Preparation_SaveForClassifyData(dfFormantStatisticpath,\
                        df_phonation_statistic_77):
    '''
        
        We generate data for nested cross-valated analysis in Table.5 in TBME2021
        
        The data will be stored at Pickles/Session_formants_people_vowel_feat
    
    '''
    dfFormantStatisticFractionpath='Pickles/Session_formants_people_vowel_feat'
    if not os.path.exists(dfFormantStatisticFractionpath):
        os.makedirs(dfFormantStatisticFractionpath)
    pickle.dump(df_phonation_statistic_77,open(dfFormantStatisticFractionpath+'/df_phonation_statistic_77.pkl','wb'))

# TBMEB1Preparation_SaveForClassifyData(pklpath,df_phonation_statistic_77)

# =============================================================================
''' Not presented in TBME2021 '''
# localabsoluteJitter_mean(A:,i:,u:)	0.38122827488960553	0.00037688388116778577	0.3206747363851191	0.003120031808349423	0.13478357779228367	83.0
# localJitter_mean(A:,i:,u:)	0.305461627101333	0.004982941065908693	0.27766361798609374	0.011039131388925498	0.08211306249104067	83.0
# localabsoluteJitter_var(A:,i:,u:)	0.27595935584822306	0.011562545817772442	0.2917206509956145	0.00745363424619023	0.06474805455029153	83.0

# =============================================================================


# =============================================================================
# Workplace 
# =============================================================================
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def TBMEB1Preparation_LoadForFromOtherData(dfFormantStatisticpath):
    '''
        
        We generate data for nested cross-valated analysis in Table.5 in TBME2021
        
        The data will be stored at Pickles/Session_formants_people_vowel_feat
    
    '''
    dfFormantStatisticFractionpath=dfFormantStatisticpath+'/Session_formants_people_vowel_feat'
    if not os.path.exists(dfFormantStatisticFractionpath):
        raise FileExistsError('Directory not exist')
    df_phonation_statistic_77=pickle.load(open(dfFormantStatisticFractionpath+'/Formant_AUI_tVSAFCRFvals_KID_FromASD_DOCKID.pkl','rb'))
    return df_phonation_statistic_77

df_formant_statistic_77=TBMEB1Preparation_LoadForFromOtherData(pklpath)
df_formant_statistic_added=pd.concat([df_phonation_statistic_77,df_formant_statistic_77],axis=1)
df_formant_statistic_added=df_formant_statistic_added.loc[:,~df_formant_statistic_added.columns.duplicated()]



# Aaa_check=df_formant_statistic_added[feature_chos_lst_top]

sex=-1
module=-1
agemax=-1
agemin=-1
ADOScate=-1
N=0
df_formant_statistic_added=criterion_filter(df_formant_statistic_added,\
                                        constrain_sex=sex,constrain_module=module,N=N,constrain_agemax=agemax,constrain_agemin=agemin,constrain_ADOScate=ADOScate,\
                                        evictNamelst=[])


''' cross validation prediction '''

# feature_chos_lst_top=['localabsoluteJitter_mean(A:,i:,u:)','localJitter_mean(A:,i:,u:)',\
#                       'rapJitter_var(A:,i:,u:)']
feature_chos_lst_top=['between_covariance_norm(A:,i:,u:)','localabsoluteJitter_mean(A:,i:,u:)']
baseline_lst=['FCR2']


# C_variable=np.array([0.001,0.01, 0.1,0.5,1.0,10.0,50,100])
C_variable=np.array(np.arange(0.1,1.5,0.1))
Classifier={}
loo=LeaveOneOut()
# CV_settings=loo
CV_settings=10
pca = PCA(n_components=1)

# =============================================================================
Classifier['SVR']={'model':sklearn.svm.SVR(),\
                  'parameters':{'C':C_variable,\
                    # 'kernel': ['rbf','sigmoid'],\
                    'kernel': ['rbf'],\
                    'gamma': ['scale'],\
                    # 'gamma': ['auto'],\
                    # 'gamma': ['scale','auto'],\
                                }}
Classifier['EN']={'model':ElasticNet(random_state=0),\
                  'parameters':{'alpha':np.arange(0,1,0.25),\
                                'l1_ratio': np.arange(0,1,0.25),\
                                'max_iter':[2000]}} #Just a initial value will be changed by parameter tuning
                                                    # l1_ratio = 1 is the lasso penalty

Classifier['LinR']={'model':sklearn.linear_model.LinearRegression(),\
                  'parameters':{'fit_intercept':[True],\
                                }}

clf_keys='SVR'
clf=Classifier[clf_keys]
# comb2 = combinations(feature_chos_lst_top, 2)
# comb3 = combinations(feature_chos_lst_top, 3)
# comb4 = combinations(feature_chos_lst_top, 4)
# combinations_lsts=list(comb2) + list(comb3)+ list(comb4)
combinations_lsts=[feature_chos_lst_top]
lab_chos_lst=['ADOS_C']

RESULT_dict=Dict()
for feature_chos_tup in combinations_lsts:
    feature_chos_lst=list(feature_chos_tup)
    for feature_chooses in [feature_chos_lst]:
        pipe = Pipeline(steps=[("model", clf['model'])])
        # pipe = Pipeline(steps=[ ("pca", pca), ("model", clf['model'])])
        param_grid = {
        # "pca__n_components": [3],
        "model__C": C_variable,
        # "model__l1_ratio": np.arange(0,1,0.25),
        # "model__alpha": np.arange(0,1,0.25),
        # "model__max_iter": [2000],
        }
        features=Dict()
        # features.X=df_formant_statistic[feature_chooses]
        # features.y=df_formant_statistic[lab_chos_lst]
        
        features.X=df_formant_statistic_added[feature_chooses]
        features.y=df_formant_statistic_added[lab_chos_lst]
        
        Gclf = GridSearchCV(pipe, param_grid=param_grid, scoring='r2', cv=CV_settings, refit=True, n_jobs=-1)
        CVpredict=cross_val_predict(Gclf, features.X, features.y.values.ravel(), cv=CV_settings)  
        Gclf.fit(features.X,features.y)
        
        if clf_keys == "EN":
            print('The coefficient of best estimator is: ',Gclf.best_estimator_.coef_)
        
        print("The best score with scoring parameter: 'r2' is", Gclf.best_score_)
        print("The best parameters are :", Gclf.best_params_)
        
        # Score=cross_val_score(Gclf, features.X, features.y, cv=10)
        
        r2=r2_score(features.y,CVpredict )
        n,p=features.X.shape
        r2_adj=1-(1-r2)*(n-1)/(n-p-1)
        pearson_result, pearson_p=pearsonr(features.y.values.ravel(),CVpredict )
        spear_result, spearman_p=spearmanr(features.y.values.ravel(),CVpredict )
        
        
        feature_keys='+'.join(feature_chooses)
        print('Feature {0}, r2_result {1}, pearson_result {2} ,spear_result {3}'.format(feature_keys, r2_adj, pearson_result,spear_result))
        RESULT_dict[feature_keys]=[r2_adj,pearson_result,spear_result]


df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict,orient='index')
''' multiple regression model '''
# =============================================================================
# 
# =============================================================================

import statsmodels.api as sm
from itertools import combinations
feature_chos_lst=['between_covariance_norm(A:,i:,u:)',
'sam_wilks_lin_norm(A:,i:,u:)',
'hotelling_lin_norm(A:,i:,u:)',
'pillai_lin_norm(A:,i:,u:)']
baseline_lst=['FCR2']
comb2 = combinations(feature_chos_lst, 2)
print(list(comb2))

[('between_covariance_norm(A:,i:,u:)', 'sam_wilks_lin_norm(A:,i:,u:)'),
 ('between_covariance_norm(A:,i:,u:)', 'hotelling_lin_norm(A:,i:,u:)'),
 ('between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)'),
 ('sam_wilks_lin_norm(A:,i:,u:)', 'hotelling_lin_norm(A:,i:,u:)'),
 ('sam_wilks_lin_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)'),
 ('hotelling_lin_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)')]

['absAng_a','absAng_u','absAng_i', 'ang_ai','ang_iu','ang_ua']

# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)','Between_Within_Det_ratio_norm(A:,i:,u:)']]
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','dcorr_12']]
# X = df_formant_statistic[[ 'pillai_lin_norm(A:,i:,u:)','dcorr_12']]
# X = df_formant_statistic[['dcorr_12','dcov_12']]
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ai']]
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ua']] 
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ai','ang_ua']] 
# X = df_formant_statistic[['FCR2']] 
X = df_formant_statistic_added[['between_covariance_norm(A:,i:,u:)','localabsoluteJitter_mean(A:,i:,u:)','dcorr_12']]
y = df_formant_statistic_added[lab_chos_lst]
## fit a OLS model with intercept on TV and Radio
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
est.summary()

ypred = est.predict(X)
spear_result, spearman_p=spearmanr(y,ypred )
print('Manual test , spear_result {1}'.format(feature_keys,spear_result))