#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:14:41 2020

@author: jackchen

This script does the main expeeriments in Table 1 (Correlation between Level of clustering and ADOS_A) 
1. Data prepare area: 
    Gather raw data of the three critical monophthongs (F1 & F2) and save in: df_formant_statistic.
    
    1-1 Filtering area:
        Filter out the outliers by IQR method (defined in muti.FilterUttDictsByCriterion_map)
    
2. Feature calculating area
    a. We use articulation.calculate_features() method to calculate LOC features 
    
3. Evaluation area


Input:
    Formants_utt_symb

Output:
    df_formant_statistic


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
from articulation.articulation import Articulation
import articulation.Multiprocess as Multiprocess
from datetime import datetime as dt
import pathlib

from utils_jack  import  Formant_utt2people_reshape, Gather_info_certainphones, \
                         FilterUttDictsByCriterion, GetValuelimit_IQR, \
                         Get_aligned_sequences, WER, Get_Vowels_AUI
from metric import Evaluation_method 

# from sklearn.utils import (as_float_array, check_array, check_X_y, safe_sqr,
#                      safe_mask)
from scipy import special, stats
import warnings


def criterion_filter(df_formant_statistic,N=10,\
                     constrain_sex=-1, constrain_module=-1,constrain_agemax=-1,constrain_ADOScate=-1,constrain_agemin=-1,\
                     evictNamelst=[]):
    # filter by number of phones
    filter_bool=np.logical_and(df_formant_statistic['u_num']>N,df_formant_statistic['a_num']>N)
    filter_bool=np.logical_and(filter_bool,df_formant_statistic['i_num']>N)
    
    # filer by other biological information
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
    
    # filter the names given the name list
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




def Process_IQRFiltering_Multi(Formants_utt_symb, limit_people_rule, outpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles'):
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
    
    Formants_utt_symb_limited=Dict()
    for load_file_tmp,_ in final_results:        
        for utt, df_utt in load_file_tmp.items():
            Formants_utt_symb_limited[utt]=df_utt
    
    pickle.dump(Formants_utt_symb_limited,open(outpath+"/[Analyzing]Formants_utt_symb_limited.pkl","wb"))
    print('Formants_utt_symb saved to ',outpath+"/[Analyzing]Formants_utt_symb_limited.pkl")
    
'''
Calculating FCR
FCR=(F2u+F2a+F1i+F1u)/(F2i+F1a)
VSA1=ABS((F1i*(F2a –F2u)+F1a *(F2u–F2i)+F1u*(F2i–F2a))/2)
VSA2=sqrt(S*(S-EDiu)(S-EDia)(S-EDau))
LnVSA=sqrt(LnS*(LnS-LnEDiu)(LnS-LnEDia)(LnS-LnEDau))

where,
u=F12_val_dict['w']
a=F12_val_dict['A']
i=F12_val_dict['j']

EDiu=sqrt((F2u–F2i)^2+(F1u–F1i)^2)
EDia=sqrt((F2a–F2i)^2+(F1a–F1i)^2)
EDau=sqrt((F2u–F2a)^2+(F1u–F1a)^2)
S=(EDiu+EDia+EDau)/2

The above formulas are implemented inside articulation.calculate_features()
'''

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
    parser.add_argument('--Inspect', default=False,
                            help='path of the base directory')
    parser.add_argument('--reFilter', default=False,
                            help='')
    parser.add_argument('--correldf_formant_statistication_type', default='spearmanr',
                            help='spearmanr|pearsonr')
    parser.add_argument('--label_choose_lst', default=['ADOS_C'],
                            help='path of the base directory')
    parser.add_argument('--Stat_med_str_VSA', default='mean',
                            help='path of the base directory')
    parser.add_argument('--poolMed', default='middle',
                            help='path of the base directory')
    parser.add_argument('--poolWindowSize', default=3,
                            help='path of the base directory')
    parser.add_argument('--dataset_role', default='KID_FromASD_DOCKID',
                            help='kid_TD| kid88| DOC_FromASD_DOCKID | KID_FromASD_DOCKID')
    parser.add_argument('--Inspect_features', default=['F1','F2'],
                            help='')

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
INSPECT=args.Inspect
windowsize=args.poolWindowSize
label_choose_lst=args.label_choose_lst # labels are too biased
role=args.dataset_role
Stat_med_str=args.Stat_med_str_VSA
outpklpath=args.inpklpath+"/Session_formants_people_vowel_feat/"
if not os.path.exists(outpklpath):
    os.makedirs(outpklpath)


Formants_utt_symb=pickle.load(open(pklpath+"/Formants_utt_symb_by{0}_window{1}_{2}.pkl".format(args.poolMed,windowsize,role),'rb'))
print("Loading Formants_utt_symb from ", pklpath+"/Formants_utt_symb_by{0}_window{1}_{2}.pkl".format(args.poolMed,windowsize,role))



# =============================================================================
'''

    1-1. Filtering area
    
    Filter out data using by 1.5*IQR

'''
PhoneMapp_dict=phonewoprosody.PhoneMapp_dict
PhoneOfInterest=list(PhoneMapp_dict.keys())
# =============================================================================


''' Vowel AUI rule is using phonewoprosody '''
Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
AUI_info=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)
limit_people_rule=GetValuelimit_IQR(AUI_info,PhoneMapp_dict,args.Inspect_features)



''' multi processing start '''
date_now='{0}-{1}-{2} {3}'.format(dt.now().year,dt.now().month,dt.now().day,dt.now().hour)
outpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles'
filepath=outpath+"/[Analyzing]Formants_utt_symb_limited.pkl"
if os.path.exists(filepath) and args.reFilter==False:
    fname = pathlib.Path(filepath)
    mtime = dt.fromtimestamp(fname.stat().st_mtime)
    filemtime='{0}-{1}-{2} {3}'.format(mtime.year,mtime.month,mtime.day,mtime.hour)
    
    # If file last modify time is not now (precisions to the hours) than we create new one
    if filemtime != date_now:
        Process_IQRFiltering_Multi(Formants_utt_symb,limit_people_rule) # the results will be output as pkl file at outpath+"/[Analyzing]Formants_utt_symb_limited.pkl"
else:
    Process_IQRFiltering_Multi(Formants_utt_symb,limit_people_rule)
Formants_utt_symb_limited=pickle.load(open(filepath,"rb"))
''' multi processing end '''
if len(limit_people_rule) >0:
    Formants_utt_symb=Formants_utt_symb_limited



Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
AUI_info=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)

    
# =============================================================================        
''' 

    2. Feature calculating area


'''
#Continuous
'Multi1','Multi2','Multi3','Multi4','T_ADOS_C','T_ADOS_S','T_ADOS_SC'
'VIQ','VCI','ADOS_CSS'
'C1','D1','D2','D3','D4','D5','E1','E2','E3'
'BB1','BB2','BB3','BB4','BB5','BB6','BB7','BB8','BB9','BB10',
'AA1','AA2','AA3','AA4','AA5','AA6','AA7','AA8','AA9'
'ADOS_S','ADOS_C','C','D','C+S',
'fso_com','fst_beh','fso_awr','fsoc_emo','fsrs_t','ssoc','sstere','scom','scq'
'adircta','adirctb','adirctc'
'BRI','MI','GEC'

#Category
'ADOS_cate_C'
'dia_num'
# =============================================================================
Vowels_AUI=Get_Vowels_AUI(AUI_info, args.Inspect_features,VUIsource="From__Formant_people_information")



additional_columns=['ADOS_cate_C','T_ADOS_C','T_ADOS_S','T_ADOS_SC','ADOS_S','ADOS_C','C+S',\
                    'ADOS_CSS','AA1','AA2','AA3','AA4','AA5','AA6','AA7','AA8','AA9','fso_com','fst_beh','fso_awr','fsoc_emo','fsrs_t','ssoc','sstere','scom','scq',\
                    'adircta','adirctb','adirctc']
# additional_columns=['Multi1','Multi2','Multi3','Multi4','T_ADOS_C','T_ADOS_S','T_ADOS_SC',\
#                     'VIQ','VCI','ADOS_CSS',\
#                     'C1','D1','D2','D3','D4','D5','E1','E2','E3',\
#                     'BB1','BB2','BB3','BB4','BB5','BB6','BB7','BB8','BB9','BB10',\
#                     'AA1','AA2','AA3','AA4','AA5','AA6','AA7','AA8','AA9',\
#                     'ADOS_S','ADOS_C','C','D','C+S',\
#                     'ADOS_cate_C']


label_generate_choose_lst=['ADOS_C','dia_num'] + additional_columns



articulation=Articulation(Stat_med_str_VSA='mean')
# df_formant_statistic=articulation.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_generate_choose_lst)
# df_formant_statistic=articulation.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_generate_choose_lst,FILTER_overlap_thrld=0)
# df_formant_statistic=articulation.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_generate_choose_lst, FILTERING_method='Silhouette',FILTER_overlap_thrld=0)
df_formant_statistic=articulation.calculate_features(Vowels_AUI,Label,PhoneOfInterest=PhoneOfInterest,label_choose_lst=label_generate_choose_lst, FILTERING_method='KDE', KDE_THRESHOLD=40)

# For pseudo acoustic features generation
df_formant_statistic['u_num+i_num+a_num']=df_formant_statistic['u_num'] +\
                                            df_formant_statistic['i_num'] +\
                                            df_formant_statistic['a_num']

for i in range(len(df_formant_statistic)):
    name=df_formant_statistic.iloc[i].name
    df_formant_statistic.loc[name,'ADOS_cate_C']=Label.label_raw[Label.label_raw['name']==name]['ADOS_cate_C'].values
    ''' ADOS_cate_C, cate stands for category '''
    
# =============================================================================        
''' 

    2. Evaluation area

    We still keep this area to get a peek of the correlation result.
    The evaluation function should be the same as the one in Statistical_tests.py
    
    The evaluation module is defined in Evaluation_method()
    
'''
# =============================================================================

Eval_med=Evaluation_method()


sex=-1
module=-1
agemax=-1
agemin=-1
ADOScate=-1
N=2
df_formant_statistic_77=criterion_filter(df_formant_statistic,\
                                        constrain_sex=sex,constrain_module=module,N=N,constrain_agemax=agemax,constrain_agemin=agemin,constrain_ADOScate=ADOScate,\
                                        evictNamelst=[])

pickle.dump(df_formant_statistic,open(outpklpath+"Formant_AUI_tVSAFCRFvals_{}.pkl".format(role),"wb"))


''' Calculate correlations for Formant fetures'''
columns=list(set(df_formant_statistic.columns) - set(additional_columns)) # Exclude added labels
columns=list(set(columns) - set([co for co in columns if "_norm" not in co]))
columns= columns + [co for co in columns if "Between_Within" in co]
columns= columns + ['VSA1','FCR','ConvexHull', 'MeanVFD','VSA2','FCR2']
columns= columns + ['absAng_a','absAng_u','absAng_i', 'ang_ai','ang_iu','ang_ua']
columns= columns + ['dcov_12','dcorr_12','dvar_1', 'dvar_2','pear_12']
columns= columns + ['dcor_a','dcor_u','dcor_i','dist_tot']
columns= columns + ['pointDistsTotal','repulsive_force']
# columns=['total_variance_norm(A:,i:,u:)',
# 'total_covariance_norm(A:,i:,u:)',
# 'between_variance_norm(A:,i:,u:)',
# 'between_covariance_norm(A:,i:,u:)',
# 'VSA2',
# 'FCR2',
# 'MeanVFD',
# 'ConvexHull',
# 'within_variance_norm(A:,i:,u:)',
# 'within_covariance_norm(A:,i:,u:)',
# 'sam_wilks_lin_norm(A:,i:,u:)',
# 'roys_root_lin_norm(A:,i:,u:)',
# 'pillai_lin_norm(A:,i:,u:)',
# 'hotelling_lin_norm(A:,i:,u:)',
# 'dcov_12',
# 'dcorr_12',
# 'ang_ua',
# 'ang_iu',
# 'ang_ai',
# 'absAng_u',
# 'absAng_i',
# 'absAng_a',]






ManualCondition=Dict()
suffix='.xlsx'
condfiles=glob.glob('Inspect/condition/*'+suffix)
for file in condfiles:
    df_cond=pd.read_excel(file)
    name=os.path.basename(file).replace(suffix,"")
    ManualCondition[name]=df_cond['Unnamed: 0'][df_cond['50%']==True]

# label_correlation_choose_lst=label_generate_choose_lst
label_correlation_choose_lst=['ADOS_C','T_ADOS_C','AA1','AA2','AA3','AA4','AA5','AA6','AA7','AA8','AA9']


N=2
Eval_med=Evaluation_method()
Aaadf_spearmanr_table_NoLimit=Eval_med.Calculate_correlation(label_correlation_choose_lst,df_formant_statistic,N,columns,constrain_sex=-1, constrain_module=-1,feature_type='Session_formant')


# Just a little check for equal indexes
lab_chos_lst=['ADOS_C']

indexes_dict={}
for label_choose in lab_chos_lst:
    df_formant_statistic=Eval_med._Postprocess_InCalCorr( label_choose,df_formant_statistic,N) #Filter unwanted samples
    # print(len(df_formant_statistic))
    indexes_dict[label_choose]=df_formant_statistic.index
df_RESULT_list=pd.DataFrame.from_dict(indexes_dict)

# =============================================================================
'''

Parse through evaluation dictionary

'''





# =============================================================================
def Survey_nice_variable(df_result_table):
    def Criteria(df_Corr_val):
        pear_str='pearson_p'
        spear_str='spearman_p'
        
        criteria_bool=(df_Corr_val[pear_str]<=0.05) & (df_Corr_val[spear_str]<=0.05)
        return df_Corr_val[criteria_bool]
    
    
    for key in df_result_table.keys():
        df_Corr_val=df_result_table[key]
        df_Corr_val_criteria=Criteria(df_Corr_val)
        if len(df_Corr_val_criteria)>0:
            print('Predicting label', key)
            print("=========================")
            print(df_Corr_val_criteria)
            print("                         ")


Survey_nice_variable(Aaadf_spearmanr_table_NoLimit)
aaa=ccc

# =============================================================================
''' 
    Multi feature prediction 

'''
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

''' cross validation prediction '''
# feature_chos_lst=['between_covariance_norm(A:,i:,u:)',
# 'sam_wilks_lin_norm(A:,i:,u:)',
# 'hotelling_lin_norm(A:,i:,u:)',
# 'pillai_lin_norm(A:,i:,u:)']


feature_chos_lst_top=['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)',\
                      'ang_ai','ang_ua','ang_iu',\
                      'dcorr_12']
# feature_chos_lst=['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ai']
# feature_chos_lst=['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ua']
# feature_chos_lst=['between_covariance_norm(A:,i:,u:)']
# feature_chos_lst=['pillai_lin_norm(A:,i:,u:)']
# feature_chos_lst=['ang_ai']
# feature_chos_lst=['ang_ua']
# feature_chos_lst=['FCR2']
# feature_chos_lst=['FCR2','ang_ai']
# feature_chos_lst=['FCR2','ang_ua']
# feature_chos_lst=['FCR2','ang_ai','ang_ua']

baseline_lst=['FCR2']


C_variable=np.array([0.001,0.01, 0.1,0.5,1.0,10.0,50,100])
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

    
clf=Classifier['EN']
comb2 = combinations(feature_chos_lst_top, 2)
comb3 = combinations(feature_chos_lst_top, 3)
comb4 = combinations(feature_chos_lst_top, 4)
combinations_lsts=list(comb2) + list(comb3)+ list(comb4)
combinations_lsts=[feature_chos_lst_top]


RESULT_dict=Dict()
for feature_chos_tup in combinations_lsts:
    feature_chos_lst=list(feature_chos_tup)
    for feature_chooses in [feature_chos_lst,baseline_lst]:
        pipe = Pipeline(steps=[("model", clf['model'])])
        # pipe = Pipeline(steps=[ ("pca", pca), ("model", clf['model'])])
        param_grid = {
        # "pca__n_components": [3],
        # "model__C": C_variable,
        "model__l1_ratio": np.arange(0,1,0.25),
        "model__alpha": np.arange(0,1,0.25),
        }
        Gclf = GridSearchCV(pipe, param_grid=param_grid, scoring='neg_mean_squared_error', cv=CV_settings, refit=True, n_jobs=-1)
        
        features=Dict()
        features.X=df_formant_statistic[feature_chooses]
        features.y=df_formant_statistic[lab_chos_lst]
        
        
        
        # Score=cross_val_score(Gclf, features.X, features.y, cv=10)
        CVpredict=cross_val_predict(Gclf, features.X, features.y.values.ravel(), cv=CV_settings)  
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

# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)']]
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','dcorr_12']]
# X = df_formant_statistic[[ 'pillai_lin_norm(A:,i:,u:)','dcorr_12']]
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)','dcorr_12']]
X = df_formant_statistic[['dcorr_12','dcov_12']]
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ai']]
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ua']] 
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ai','ang_ua']] 
# X = df_formant_statistic[['FCR2']] 
y = df_formant_statistic[lab_chos_lst]
## fit a OLS model with intercept on TV and Radio
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
est.summary()

ypred = est.predict(X)
spear_result, spearman_p=spearmanr(y,ypred )
print('Manual test , spear_result {1}'.format(feature_keys,spear_result))


# =============================================================================
# 'between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)'
# r2_adj: 0.144   spear: 0.43





from scipy.stats import f_oneway
import itertools
''' Classification area '''
category_col=['ADOS_cate_C']
for cate in category_col:
    for col in columns:
        groups_compare=df_formant_statistic.groupby(cate)[col].apply(list)
        groups_compare_keys=df_formant_statistic.groupby(cate)[col].groups.keys()
        # f_oneway(groups_compare[0],groups_compare[1],groups_compare[2])
        result=f_oneway(*groups_compare)
        if result[-1] <= 0.05:
            print("anova result")
            print("Autisttic trait: ",cate, " Feature: ",col)
            print(result)
        
        
        
        # Two by two comparison
        # iteration_sel=list(itertools.combinations(groups_compare_keys, 2))
        # for key in iteration_sel:
        #     pair=[groups_compare[key[0]],groups_compare[key[1]]]
        #     # result=stats.ttest_ind(pair[0],pair[1])
        #     # if result[-1] <= 0.05:
        #     #     print("ttest result")
        #     #     print("Autisttic trait: ",cate, " Feature: ",col, "Between keys: ", key)
        #     #     print(result)
        #     result=stats.mannwhitneyu(pair[0],pair[1])
        #     if result[-1] <= 0.05:
        #         print("utest result")
        #         print("Autisttic trait: ",cate, " Feature: ",col, "Between keys: ", key)
        #         print(result)
            

# Plot
# import seaborn as sns
# sns.boxplot(x="dia_num", y="between_covariance_norm(A:,i:,u:)", data=df_formant_statistic)

# sns.boxplot(x="dia_num", y="sam_wilks_lin_norm(A:,i:,u:)", data=df_formant_statistic)

# sns.boxplot(x="dia_num", y="pillai_lin_norm(A:,i:,u:)", data=df_formant_statistic)





# =============================================================================
#  Pearson
#  between_covariance(A:,i:,u:)       -0.346864  ...         86.0
#  between_variance(A:,i:,u:)         -0.465486  ...         86.0




# =============================================================================



# =============================================================================
'''

    WorkSpace

'''
# =============================================================================
df_formant_statistic_toy=pd.DataFrame()

for people in SCATTER_matrixBookeep_dict.keys():
    Result_dict={}
    # Add basic information
    for label_choose in label_generate_choose_lst:
        Result_dict[label_choose]=Label.label_raw[label_choose][Label.label_raw['name']==people].values    
    Result_dict['u_num'], Result_dict['a_num'], Result_dict['i_num']=\
                len(Vowels_AUI[people]['u:']),len(Vowels_AUI[people]['A:']),len(Vowels_AUI[people]['i:'])
    # Matrix information
    Sctter_matrix_person=SCATTER_matrixBookeep_dict[people]
    for SCtr_max in Sctter_matrix_person.keys():
        prefix=SCtr_max+'_'
        Sctter_matrix=Sctter_matrix_person[SCtr_max]
        F1_var=Sctter_matrix[0][0]
        linear_coeff=Sctter_matrix[0][1]
        F2_var=Sctter_matrix[1][1]
        linear_coeff2=Sctter_matrix[1][0]
        
        Result_dict[prefix+'F1_var']=F1_var
        Result_dict[prefix+'linear_coeff']=linear_coeff
        Result_dict[prefix+'F2_var']=F2_var
        Result_dict[prefix+'linear_coeff2']=linear_coeff2
    # df_RESULT_list=pd.DataFrame.from_dict(Result_dict,orient='index').T
    df_RESULT_list=pd.DataFrame.from_dict(Result_dict)
    df_RESULT_list.index=[people]

    df_formant_statistic_toy=df_formant_statistic_toy.append(df_RESULT_list)

''' Calculate correlations for Formant fetures'''
columns=list(set(df_formant_statistic_toy.columns)  - set(additional_columns)) # Exclude added labels
columns=list(set(columns)  - set(['a_num','u_num','i_num']))

Aaad_Correlation_toy=Eval_med.Calculate_correlation(label_correlation_choose_lst,df_formant_statistic_toy,N,columns,constrain_sex=-1, constrain_module=-1,feature_type='Session_formant')
Survey_nice_variable(Aaad_Correlation_toy)



# =============================================================================

# Play code for convex hull area
# from scipy.spatial import ConvexHull, convex_hull_plot_2d
# import matplotlib.pyplot as plt

# for people in Vowels_AUI.keys(): #update 2021/05/27 fixed 
#     points=np.empty((0,2), float)
#     F12_raw_dict=Vowels_AUI[people]
#     for k, phone in F12_raw_dict.items():
#         data=phone[args.Inspect_features].copy()
#         # points=np.vstack((points,data.values))
#         points = np.append(points, data.values, axis=0)
#     hull = ConvexHull(points)
#     plt.plot(points[:,0], points[:,1], 'o')
#     for simplex in hull.simplices:

#         plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
# =====================
# Play code for Vowel vector dispersion
# import math
# Info_dict_total=Dict()
# for people in list(Vowels_AUI.keys()): #update 2021/05/27 fixed 
#     Info_dict=Dict()
    
#     points=np.empty((0,2), float)
#     F12_raw_dict=Vowels_AUI[people]
#     for k, phone in F12_raw_dict.items():
#         data=phone[args.Inspect_features].copy()
#         # points=np.vstack((points,data.values))
#         points = np.append(points, data.values, axis=0)
#     # Determine the f1 mean
#     F1m=np.mean(points[:,0])
#     # Determine the f2 mean
#     f2_criteria_bag=np.empty((0,2), float)
#     for V in points:
#         if V[0]<F1m:
#             f2_criteria_bag=np.append(f2_criteria_bag, V.reshape((1,-1)), axis=0)
#     F2m=np.mean(f2_criteria_bag[:,1])
    
#     center_coordinate=np.array([F1m,F2m])
#     Info_dict['center']=center_coordinate
#     Info_dict['vector'], Info_dict['scalar'], Info_dict['angle']=[], [], []
#     # Calculate the Vowel vector, angle and scale 
#     for i, V in enumerate(points):
#         VFDi=V-center_coordinate
#         Info_dict['vector'].append(VFDi)
#         Info_dict['scalar'].append(np.linalg.norm(VFDi))
        
#         # The math.atan2() method returns the arc tangent of y/x, and has take care the special conditions
#         omega=math.atan2(V[0]-center_coordinate[0], V[1]-center_coordinate[1])
    
#         Info_dict['angle'].append(omega)
#     Info_dict_total[people]=Info_dict

# count=0
# for people, Info_dict in Info_dict_total.items():
#     plt.figure(count)
#     plt.plot(Info_dict['center'][0], Info_dict['center'][1], 'o')
#     for vector in Info_dict['vector']:
#         # matplotlib.pyplot.arrow(x, y, dx, dy, **kwargs)
#         plt.arrow(Info_dict['center'][0],Info_dict['center'][1],vector[0],vector[1])
#     count+=1

# Play code for scatter plot with silhouette_samples score calculated
# import seaborn as sns
# import sklearn
# count=0
# for people in Vowels_AUI.keys():
#     plt.figure(count)
#     F12_raw_dict=Vowels_AUI[people]
#     df_vowel = pd.DataFrame()
#     for keys in F12_raw_dict.keys():
#         if len(df_vowel) == 0:
#             df_vowel=F12_raw_dict[keys]
#             df_vowel['vowel']=keys
#         else:
#             df_=F12_raw_dict[keys]
#             df_['vowel']=keys
#             df_vowel=df_vowel.append(df_)
    
#     X=df_vowel[args.Inspect_features]
#     labels=df_vowel['vowel']
#     sklearn.metrics.silhouette_samples(X, labels,  metric='euclidean')
#     sklearn.metrics.calinski_harabasz_score(X, labels)
#     sns.scatterplot(data=df_vowel, x="F1", y="F2", hue="vowel")
#     count+=1
# =====================
    
# Play code for KDE filtering
# count=0
# import seaborn as sns
# from sklearn.neighbors import KernelDensity
# from sklearn import preprocessing

# THRESHOLD=50
# scale_factor=100
# N=2
# # for people in list(Vowels_AUI.keys())[:3]:
# for people in Vowels_AUI.keys():
#     plt.figure(count)
#     F12_raw_dict=Vowels_AUI[people]
#     df_vowel = pd.DataFrame()
#     for keys in F12_raw_dict.keys():
#         if len(df_vowel) == 0:
#             df_vowel=F12_raw_dict[keys]
#             df_vowel['vowel']=keys
#         else:
#             df_=F12_raw_dict[keys]
#             df_['vowel']=keys
#             df_vowel=df_vowel.append(df_)
    
#     len_a=len(np.where(df_vowel['vowel']=='A:')[0])
#     len_u=len(np.where(df_vowel['vowel']=='u:')[0])
#     len_i=len(np.where(df_vowel['vowel']=='i:')[0])
    
    
#     if len_a<=N or len_u<=N or len_i<=N:
#         continue
    
#     def KDE_Filtering(df_vowel,THRESHOLD=10,scale_factor=100):
#         X=df_vowel[args.Inspect_features].values
#         labels=df_vowel['vowel']
        
#         df_vowel_calibrated=pd.DataFrame([])
#         for phone in set(labels):
            
#             df=df_vowel[df_vowel['vowel']==phone][args.Inspect_features]
#             data_array=df_vowel[df_vowel['vowel']==phone][args.Inspect_features].values

#             x=data_array[:,0]
#             y=data_array[:,1]
#             xmin = x.min()
#             xmax = x.max()        
#             ymin = y.min()
#             ymax = y.max()
            
#             image_num=1j
#             X, Y = np.mgrid[xmin:xmax:image_num*scale_factor, ymin:ymax:image_num*scale_factor]
            
#             positions = np.vstack([X.ravel(), Y.ravel()])
            
#             values = np.vstack([x, y])
            
#             kernel = stats.gaussian_kde(values)
                    
#             Z = np.reshape(kernel(positions).T, X.shape)
#             normalized_z = preprocessing.normalize(Z)
            
#             df['x_to_scale'] = (100*(x - np.min(x))/np.ptp(x)).astype(int) 
#             df['y_to_scale'] = (100*(y - np.min(y))/np.ptp(y)).astype(int) 
            
#             normalized_z=(100*(Z - np.min(Z.ravel()))/np.ptp(Z.ravel())).astype(int)
#             to_delete = zip(*np.where((normalized_z<THRESHOLD) == True))
            
#             # The indexes that are smaller than threshold
#             deletepoints_bool=df.apply(lambda x: (x['x_to_scale'], x['y_to_scale']), axis=1).isin(to_delete)
#             df_calibrated=df.loc[(deletepoints_bool==False).values]
#             df_deleted_after_calibrated=df.loc[(deletepoints_bool==True).values]
            
#             df_vowel_calibrated_tmp=df_calibrated.drop(columns=['x_to_scale','y_to_scale'])
#             df_vowel_calibrated_tmp['vowel']=phone
#             df_vowel_output=df_vowel_calibrated_tmp.copy()
#             df_vowel_calibrated=df_vowel_calibrated.append(df_vowel_output)
            
            
#             # Data prepare for plotting 
#             # df_calibrated_tocombine=df_calibrated.copy()
#             # df_calibrated_tocombine['cal']='calibrated'
#             # df_deleted_after_calibrated['cal']='deleted'
#             # df_calibratedcombined=df_calibrated_tocombine.append(df_deleted_after_calibrated)
            
#             # #Plotting code
#             # fig = plt.figure(figsize=(8,8))
#             # ax = fig.gca()
#             # ax.set_xlim(xmin, xmax)
#             # ax.set_ylim(ymin, ymax)
#             # # cfset = ax.contourf(X, Y, Z, cmap='coolwarm')
#             # # ax.imshow(Z, cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
#             # # cset = ax.contour(X, Y, Z, colors='k')
#             # cfset = ax.contourf(X, Y, normalized_z, cmap='coolwarm')
#             # ax.imshow(normalized_z, cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
#             # cset = ax.contour(X, Y, normalized_z, colors='k')
#             # ax.clabel(cset, inline=1, fontsize=10)
#             # ax.set_xlabel('X')
#             # ax.set_ylabel('Y')
#             # plt.title('2D Gaussian Kernel density estimation')
            
#             # sns.scatterplot(data=df_vowel[df_vowel['vowel']==phone], x="F1", y="F2")
#             # sns.scatterplot(data=df_calibratedcombined, x="F1", y="F2",hue='cal')
#         return df_vowel_calibrated
#     df_vowel_calibrated=KDE_Filtering(df_vowel,THRESHOLD=THRESHOLD,scale_factor=100)
#     sns.scatterplot(data=df_vowel_calibrated, x="F1", y="F2", hue="vowel")

#     count+=1

# Play code for relative angles
# import seaborn as sns
# import dcor
# import math
# import matplotlib.patheffects as pe
# from matplotlib.offsetbox import AnchoredText

# def cosAngle(a, b, c):
#     # angles between line segments (Python) from https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
#     cosine_angle = np.dot((b-a), (b-c)) / (np.linalg.norm((b-a)) * np.linalg.norm((b-c)))
#     angle = np.arccos(cosine_angle)
#     return np.degrees(angle)

count=0
for people in list(Vowels_AUI.keys())[:10]:
    plt.figure(count)
    F12_raw_dict=Vowels_AUI[people]
    df_vowel = pd.DataFrame()
    for keys in F12_raw_dict.keys():
        if len(df_vowel) == 0:
            df_vowel=F12_raw_dict[keys]
            df_vowel['vowel']=keys
        else:
            df_=F12_raw_dict[keys]
            df_['vowel']=keys
            df_vowel=df_vowel.append(df_)
    
    
    def Calculate_raelative_angles(df_vowel, additional_infos=False):
        
        a=df_vowel[df_vowel['vowel']=='A:'][args.Inspect_features]
        u=df_vowel[df_vowel['vowel']=='u:'][args.Inspect_features]
        i=df_vowel[df_vowel['vowel']=='i:'][args.Inspect_features]
        
        a_center=a.mean()
        u_center=u.mean()
        i_center=i.mean()
        total_center=df_vowel.mean()
        # gravity_center=(a_center*len(a) + u_center*len(u) + i_center*len(i)) / len(df_vowel)
        
        
        
        omega_a=np.degrees(math.atan2((a_center - total_center)[1], (a_center - total_center)[0]))
        omega_u=np.degrees(math.atan2((u_center - total_center)[1], (u_center - total_center)[0]))
        omega_i=np.degrees(math.atan2((i_center - total_center)[1], (i_center - total_center)[0]))
    
        ang_ai = cosAngle(a_center,total_center,i_center)
        ang_iu = cosAngle(i_center,total_center,u_center)
        ang_ua = cosAngle(u_center,total_center,a_center)
        
        absolute_ang=[omega_a, omega_u, omega_i]
        relative_ang=[ang_ai, ang_iu, ang_ua]
        addition_info=[total_center, a_center, u_center, i_center]
        
        if additional_infos != True:
            return absolute_ang, relative_ang
        else:
            return absolute_ang, relative_ang, addition_info
    
    
    absolute_ang, relative_ang, addition_info = Calculate_raelative_angles(df_vowel, additional_infos=True)
    [omega_a, omega_u, omega_i]=absolute_ang
    [ang_ai, ang_iu, ang_ua]=relative_ang
    [total_center, a_center, u_center, i_center]=addition_info
    print(sum([ang_ai,ang_iu,ang_ua]))
    
    # omega_a_deg=omega_a*(180/np.pi)
    
    center2plot=total_center
    plt.plot(center2plot[0],center2plot[1],'*',markersize=30)
    # dx,dy=(a_center - center2plot)
    # origin=center2plot
    # dest=center2plot+[dx,dy]
    
    
    
    plt.plot([center2plot[0],center2plot[0] + (a_center - center2plot)[0]],[center2plot[1],center2plot[1] + (a_center - center2plot)[1]],'ro-', color="green", linewidth=3,\
              path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])
    plt.plot([center2plot[0],center2plot[0] + (u_center - center2plot)[0]],[center2plot[1],center2plot[1] + (u_center - center2plot)[1]],'ro-', color="blue", linewidth=3,\
              path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])
    plt.plot([center2plot[0],center2plot[0] + (i_center - center2plot)[0]],[center2plot[1],center2plot[1] + (i_center - center2plot)[1]],'ro-', color="orange", linewidth=3,\
              path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])
    sns.scatterplot(data=df_vowel, x="F1", y="F2", hue="vowel")
    
    at = AnchoredText(
    "angles\na:{0}\nu:{1}\ni:{2}".format(np.round(omega_a,2),np.round(omega_u,2),np.round(omega_i,2)), prop=dict(size=15), frameon=True, loc='lower right')
    # at = AnchoredText(
    # "angles\nai{0}\niu{1}\nua{2}".format(np.round(ang_ai,2),np.round(ang_iu,2),np.round(ang_ua,2)), prop=dict(size=15), frameon=True, loc='lower right')
    plt.setp(at.patch, facecolor='white', alpha=0.5)
    plt.gca().add_artist(at)
    
    plt.show()
    count+=1

# Play code for Kmeans on ADOS labels
from sklearn.cluster import KMeans
from sklearn import manifold, datasets
base_dir='/homes/ssd1/jackchen/gop_prediction/'

label_path=base_dir+'ADOS_label20220309.xlsx'
label_raw=pd.read_excel(label_path)

# label_choose_list=['AA1','AA2','AA3','AA4','AA5','AA6','AA7','AA8','AA9']
# label_choose_list=['AA4','AA7','AA8','AA9']
label_choose_list=['ADOS_C']
X = label_raw[['ADOS_C']].values.reshape(-1,1)

kmeans_bag=Dict()
count=0
for i in range(4):
    plt.figure(count)
    X = label_raw[label_choose_list]
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    kmeans_bag[i]=y_kmeans
    df_kmeans_bag=pd.DataFrame.from_dict(kmeans_bag)
    
    
    All_TSNE=manifold.TSNE(n_jobs=-1).fit_transform(np.vstack([X,centers]))
    
    X_tsne=All_TSNE[:len(X),:]
    centers_tsne=All_TSNE[len(X):,:]
    
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(centers_tsne[:, 0], centers_tsne[:, 1], c='black', s=200, alpha=0.5);
    count+=1



# Play code for distance covariance
# import dcor
# import math
# from scipy.stats import spearmanr,pearsonr 
# count=0
# for people in list(Vowels_AUI.keys())[:]:
#     plt.figure(count)
#     F12_raw_dict=Vowels_AUI[people]
#     df_vowel = pd.DataFrame()
#     for keys in F12_raw_dict.keys():
#         if len(df_vowel) == 0:
#             df_vowel=F12_raw_dict[keys]
#             df_vowel['vowel']=keys
#         else:
#             df_=F12_raw_dict[keys]
#             df_['vowel']=keys
#             df_vowel=df_vowel.append(df_)
    
#     def Calculate_distanceCorr(df_vowel):
#         a=df_vowel[df_vowel['vowel']=='A:'][args.Inspect_features]
#         u=df_vowel[df_vowel['vowel']=='u:'][args.Inspect_features]
#         i=df_vowel[df_vowel['vowel']=='i:'][args.Inspect_features]
        
        
#         a_1, a_2=a['F1'], a['F2']
#         u_1, u_2=u['F1'], u['F2']
#         i_1, i_2=i['F1'], i['F2']
        
#         d_stats_a=dcor.distance_stats(a_1, a_2)
#         d_stats_u=dcor.distance_stats(u_1, u_2)
#         d_stats_i=dcor.distance_stats(i_1, i_2)
        
#         pear_a=pearsonr(a_1, a_2)[0]
#         pear_u=pearsonr(u_1, u_2)[0]
#         pear_i=pearsonr(i_1, i_2)[0]
        
#         def get_values(X_stats):
#             cov_xy, corr_xy, var_x, var_y=X_stats
#             return cov_xy, corr_xy, var_x, var_y
        
        
#         data_a = get_values(d_stats_a)
#         data_u = get_values(d_stats_u)
#         data_i = get_values(d_stats_i)
        
#         Cov_sum=[sum(x) for x in zip(data_a,data_u,data_i)]
#         Corr_aui=[data_a[1],data_u[1],data_i[1]]

#         pear_sum=sum([pear_a,pear_u,pear_i])
#         return Cov_sum, Corr_aui, pear_sum
#     ADOS_lab=df_formant_statistic[df_formant_statistic.index==people]['ADOS_C'].values[0]
#     [a,b,c,d], [dcor_a,dcor_u,dcor_i] ,pearsum=Calculate_distanceCorr(df_vowel)
#     sns.scatterplot(data=df_vowel, x="F1", y="F2", hue="vowel")
#     at = AnchoredText(
#     "dcorr:{0}\nADOS:{1}".format(np.round(b,2),np.round(ADOS_lab,2)), prop=dict(size=15), frameon=True, loc='lower right')
#     plt.setp(at.patch, facecolor='white', alpha=0.5)
#     plt.gca().add_artist(at)
    
#     plt.show()
#     count+=1


# Play code for scipy c distance
# import scipy
# count=0
# for people in list(Vowels_AUI.keys())[:10]:
#     # plt.figure(count)
#     F12_raw_dict=Vowels_AUI[people]
#     df_vowel = pd.DataFrame()
#     for keys in F12_raw_dict.keys():
#         if len(df_vowel) == 0:
#             df_vowel=F12_raw_dict[keys]
#             df_vowel['vowel']=keys
#         else:
#             df_=F12_raw_dict[keys]
#             df_['vowel']=keys
#             df_vowel=df_vowel.append(df_)
    
#     def Calculate_pointDistsTotal(df_vowel):
        # a=df_vowel[df_vowel['vowel']=='A:'][args.Inspect_features]
        # u=df_vowel[df_vowel['vowel']=='u:'][args.Inspect_features]
        # i=df_vowel[df_vowel['vowel']=='i:'][args.Inspect_features]
        
#         dist_au=scipy.spatial.distance.cdist(a,u)
#         dist_ai=scipy.spatial.distance.cdist(a,i)
#         dist_iu=scipy.spatial.distance.cdist(i,u)
#         mean_dist_au=np.mean(dist_au)
#         mean_dist_ai=np.mean(dist_ai)
#         mean_dist_iu=np.mean(dist_iu)
#         dist_total=mean_dist_au+mean_dist_ai+mean_dist_iu
#         return dist_total
#     dist_tot=Calculate_pointDistsTotal(df_vowel)


# play code for betweenClusters_distrib_dist from Li-Min Wang
# import scipy
# count=0
# for people in list(Vowels_AUI.keys())[:10]:
#     # plt.figure(count)
#     F12_raw_dict=Vowels_AUI[people]
#     df_vowel = pd.DataFrame()
#     for keys in F12_raw_dict.keys():
#         if len(df_vowel) == 0:
#             df_vowel=F12_raw_dict[keys]
#             df_vowel['vowel']=keys
#         else:
#             df_=F12_raw_dict[keys]
#             df_['vowel']=keys
#             df_vowel=df_vowel.append(df_)
            
#     a=df_vowel[df_vowel['vowel']=='A:'][args.Inspect_features]
#     u=df_vowel[df_vowel['vowel']=='u:'][args.Inspect_features]
#     i=df_vowel[df_vowel['vowel']=='i:'][args.Inspect_features]

#     def calculate_betweenClusters_distrib_dist(df_vowel):
#         def Within_matrices(df_vowel):
#             ''' Not used'''
#             class_feature_means = pd.DataFrame(columns=list(set(df_vowel['vowel'])))
#             n_samples = len(df_vowel)

#             for c, rows in df_vowel.groupby('vowel'):
#                 class_feature_means[c] = rows[args.Inspect_features].mean()
            
#             groups_num=len(class_feature_means.index)
#             # Within class scatter matrix 
#             within_class_scatter_matrix = np.zeros((groups_num,groups_num))
#             for c, rows in df_vowel.groupby('vowel'):
#                 rows = rows[args.Inspect_features]
#                 s = np.zeros((groups_num,groups_num))
                                
#                 for index, row in rows.iterrows():
#                     x, mc = row.values.reshape(groups_num,1), class_feature_means[c].values.reshape(groups_num,1)
#                     class_variance=(x - mc).dot((x - mc).T).astype(float)
#                     s += class_variance
                    
#                 within_class_scatter_matrix += s
#             within_class_scatter_matrix_norm = within_class_scatter_matrix / n_samples
        
#             return within_class_scatter_matrix_norm
#         def calculate_stdev(df_vowel_phone):
#             ''' Not used'''
#             u_scatter_mtrx_norm=Within_matrices(df_vowel_phone)
#             eigen_values, _ = np.linalg.eig(u_scatter_mtrx_norm)
#             stdev=np.sqrt(sum(eigen_values))
#             return stdev
        
#         def calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel', vowel1='A:',\
#                                                vowel2='u:', Inspect_features=['F1', 'F2']):
#             v1=df_vowel[df_vowel[vowelCol_name]==vowel1][args.Inspect_features]
#             v2=df_vowel[df_vowel[vowelCol_name]==vowel2][args.Inspect_features]
            
#             mean_dist_v1v2=np.sum(scipy.spatial.distance.cdist(v1,v2.mean().values.reshape(1,-1)))
#             mean_dist_v2=np.sum(scipy.spatial.distance.cdist(v2,v2.mean().values.reshape(1,-1)))
#             dristrib_dist_v1v2= mean_dist_v2/mean_dist_v1v2
            
#             # stdev_v2=calculate_stdev(df_vowel[df_vowel[vowelCol_name]==vowel1])
#             # dristrib_dist_v1v2= stdev_v2/mean_dist_v1v2
            
#             mean_dist_v2v1=np.mean(scipy.spatial.distance.cdist(v2,v1.mean().values.reshape(1,-1)))
#             mean_dist_v1=np.mean(scipy.spatial.distance.cdist(v1,v1.mean().values.reshape(1,-1)))
#             dristrib_dist_v2v1= mean_dist_v1/mean_dist_v2v1
#             # stdev_v1=calculate_stdev(df_vowel[df_vowel[vowelCol_name]==vowel2])
#             # dristrib_dist_v2v1= stdev_v1/mean_dist_v2v1
            
#             dristrib_dist=dristrib_dist_v1v2 * dristrib_dist_v2v1
#             return dristrib_dist
        
#         dristrib_dist_au= calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel', vowel1='A:',\
#                                            vowel2='u:', Inspect_features=args.Inspect_features)
#         dristrib_dist_ui= calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel', vowel1='u:',\
#                                            vowel2='i:', Inspect_features=args.Inspect_features)
#         dristrib_dist_ai= calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel', vowel1='A:',\
#                                            vowel2='i:', Inspect_features=args.Inspect_features)
#         return sum(dristrib_dist_au,dristrib_dist_ui,dristrib_dist_ai)
    
# play code for repulsive force
count=0
import scipy
for people in list(Vowels_AUI.keys())[:10]:
    # plt.figure(count)
    F12_raw_dict=Vowels_AUI[people]
    df_vowel = pd.DataFrame()
    for keys in F12_raw_dict.keys():
        if len(df_vowel) == 0:
            df_vowel=F12_raw_dict[keys]
            df_vowel['vowel']=keys
        else:
            df_=F12_raw_dict[keys]
            df_['vowel']=keys
            df_vowel=df_vowel.append(df_)
    
    
    def calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel'):
        repuls_forc_inst_bag=[]
        for index, row in df_vowel.iterrows():
            phone=row[vowelCol_name]
            formant_values=row[args.Inspect_features]
            other_phones=df_vowel[df_vowel[vowelCol_name]!=phone]
            other_phones_values=other_phones[args.Inspect_features]
            
            repuls_forc_inst=np.mean(1/scipy.spatial.distance.cdist(other_phones_values,formant_values.values.reshape(1,-1)))
            repuls_forc_inst_bag.append(repuls_forc_inst)
        assert len(repuls_forc_inst_bag) == len(df_vowel)
        return np.mean(repuls_forc_inst_bag)
    repulsive_force_norm=calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel')
    print(repulsive_force_norm)
    
