#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 17:38:27 2021

@author: jackchen
"""
import os, glob, sys
import pickle
import numpy as np
import argparse
import pandas as pd
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
import seaborn as sns
import matplotlib.pyplot as plt
from addict import Dict



def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--inpklpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--dfFormantStatisticpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--Inspect', default=False,
                            help='path of the base directory')
    parser.add_argument('--correlation_type', default='spearmanr',
                            help='spearmanr|pearsonr')
    parser.add_argument('--label_choose_lst', default=['ADOS_C'],
                            help='path of the base directory')
    parser.add_argument('--Stat_med_str_VSA', default='mean',
                            help='path of the base directory')
    parser.add_argument('--poolMed', default='middle',
                            help='path of the base directory')
    parser.add_argument('--poolWindowSize', default=3,
                            help='path of the base directory')
    parser.add_argument('--role', default='ASDTD',
                            help='path of the base directory')
    parser.add_argument('--Inspect_features', default=['F1','F2'],
                            help='')
    args = parser.parse_args()
    return args


args = get_args()
base_path=args.base_path
dfFormantStatisticpath=args.dfFormantStatisticpath

required_path_app = '/mnt/sdd/jackchen/egs/formosa/s6/local'  # for WER module imported in metric
sys.path.append(required_path_app)
from metric import Evaluation_method     
# =============================================================================
'''

    Data preparation

'''
# =============================================================================
'''  T-Test ASD vs TD''' 

df_formant_statistic77_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/Formant_AUI_tVSAFCRFvals_ASDkid.pkl'
df_formant_statistic_77=pickle.load(open(df_formant_statistic77_path,'rb'))
df_formant_statistic_ASDTD_path=dfFormantStatisticpath+'/Session_formants_people_vowel_feat/Formant_AUI_tVSAFCRFvals_ASDTD.pkl'
df_formant_statistic_TD=pickle.load(open(df_formant_statistic_ASDTD_path,'rb'))


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
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['ADOS_cate']==constrain_ADOScate)
        
    if len(evictNamelst)>0:
        for name in evictNamelst:
            filter_bool.loc[name]=False
        
    return df_formant_statistic[filter_bool]

N=1
ManualCondition=Dict()
suffix='.xlsx'
condfiles=glob.glob('articulation/Inspect/condition/*'+suffix)
for file in condfiles:
    df_cond=pd.read_excel(file)
    name=os.path.basename(file).replace(suffix,"")
    ManualCondition[name]=df_cond['Unnamed: 0'][df_cond['50%']==True]


sex=-1
module=-1
agemax=-1
agemin=-1
ADOScate=-1
for N in range(1,2,1):
    df_formant_statistic_77=criterion_filter(df_formant_statistic_77,\
                                            constrain_sex=sex,constrain_module=module,N=N,constrain_agemax=agemax,constrain_agemin=agemin,constrain_ADOScate=ADOScate,\
                                            evictNamelst=ManualCondition[ 'unreasonable_all'])
    df_formant_statistic_TD=criterion_filter(df_formant_statistic_TD,constrain_sex=sex,constrain_module=module,N=N,constrain_agemax=agemax)
    
    
    comb=[['df_formant_statistic_TD','df_formant_statistic_77'],]
    Parameters=['u_num', 'a_num', 'i_num', 'ADOS', 'age', 'FCR',
       'F2i_u', 'F1a_u', 'VSA1', 'F_vals_f1(A:,i:,u:)', 'F_vals_f2(A:,i:,u:)',
       'F_val_mix(A:,i:,u:)', 'MSB_f1(A:,i:,u:)', 'MSB_f2(A:,i:,u:)',
       'MSB_mix', 'F_vals_f1(i:,u:)', 'F_vals_f2(i:,u:)', 'F_val_mix(i:,u:)',
       'MSB_f1(i:,u:)', 'MSB_f2(i:,u:)', 'F_vals_f1(A:,u:)',
       'F_vals_f2(A:,u:)', 'F_val_mix(A:,u:)', 'MSB_f1(A:,u:)',
       'MSB_f2(A:,u:)', 'F_vals_f1(A:,i:)', 'F_vals_f2(A:,i:)',
       'F_val_mix(A:,i:)', 'MSB_f1(A:,i:)', 'MSB_f2(A:,i:)', 'F1(A:-i:)_mean',
       'F1(A:-i:)_min', 'F1(A:-i:)_25%', 'F1(A:-i:)_50%', 'F1(A:-i:)_75%',
       'F1(A:-i:)_max', 'F1(A:-u:)_mean', 'F1(A:-u:)_min', 'F1(A:-u:)_25%',
       'F1(A:-u:)_50%', 'F1(A:-u:)_75%', 'F1(A:-u:)_max', 'F2(i:-u:)_mean',
       'F2(i:-u:)_min', 'F2(i:-u:)_25%', 'F2(i:-u:)_50%', 'F2(i:-u:)_75%',
       'F2(i:-u:)_max']
    
    df_ttest_result=pd.DataFrame([],columns=['doc-kid','p-val'])
    for role_1,role_2  in comb:
        for parameter in Parameters:
            # test=stats.ttest_ind(vars()[role_1][parameter], vars()[role_2][parameter])
            test=stats.mannwhitneyu(vars()[role_1][parameter], vars()[role_2][parameter])
            # print(parameter, '{0} vs {1}'.format(role_1,role_2),test)
            # print(role_1+':',vars()[role_1][parameter].mean(),role_2+':',vars()[role_2][parameter].mean())
            df_ttest_result.loc[parameter,'doc-kid'] = vars()[role_1][parameter].mean() - vars()[role_2][parameter].mean()
            df_ttest_result.loc[parameter,'p-val'] = test[1]
    print(df_ttest_result.loc[['MSB_f1(A:,i:,u:)', 'MSB_f2(A:,i:,u:)']])
    df_formant_statistic_77_mean=df_formant_statistic_77.mean()
# inspect_cols=['u_num', 'a_num', 'i_num','MSB_f1(A:,i:,u:)', 'MSB_f2(A:,i:,u:)','ADOS']
inspect_cols=['MSB_f1(A:,i:,u:)', 'MSB_f2(A:,i:,u:)']
df_formant_statistic_77_inspect=df_formant_statistic_77[inspect_cols]
df_formant_statistic_TD_inspect=df_formant_statistic_TD[inspect_cols]

df_formant_statistic_77_Notautism_mean=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=0).mean()
df_formant_statistic_77_ASD_mean=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=1).mean()
df_formant_statistic_77_autism_mean=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=2).mean()
df_formant_statistic_TD_mean=df_formant_statistic_TD.mean()

filter_boy=df_formant_statistic_77['sex']==1
filter_girl=df_formant_statistic_77['sex']==2
print(df_formant_statistic_77_inspect[filter_boy].mean())
print(df_formant_statistic_77_inspect[filter_girl].mean())

filter_boy=df_formant_statistic_TD['sex']==1
filter_girl=df_formant_statistic_TD['sex']==2
print(df_formant_statistic_TD_inspect[filter_boy].mean())
print(df_formant_statistic_TD_inspect[filter_girl].mean())

# =============================================================================
'''

    Regression area

'''

columns=['F_vals_f1(A:,i:,u:)', 'F_vals_f2(A:,i:,u:)',
       'F_val_mix(A:,i:,u:)', 'MSB_f1(A:,i:,u:)', 'MSB_f2(A:,i:,u:)',
       'MSB_mix', 'F_vals_f1(A:,u:)', 'F_vals_f2(A:,u:)', 'F_val_mix(A:,u:)',
       'MSB_f1(A:,u:)', 'MSB_f2(A:,u:)', 'F_vals_f1(A:,i:)',
       'F_vals_f2(A:,i:)', 'F_val_mix(A:,i:)', 'MSB_f1(A:,i:)',
       'MSB_f2(A:,i:)', 'F_vals_f1(i:,u:)', 'F_vals_f2(i:,u:)',
       'F_val_mix(i:,u:)', 'MSB_f1(i:,u:)', 'MSB_f2(i:,u:)']
# =============================================================================
Eval_med=Evaluation_method()
label_choose_lst=['ADOS_C']
df_formant_statistic_all=df_formant_statistic_77.append(df_formant_statistic_TD)
tmpoutpath='Features/artuculation_AUI/Vowels/'
pickle.dump(df_formant_statistic_all,open(tmpoutpath+'Formant_AUI_tVSAFCRFvals_ASDkid+TD.pkl','wb'))

Aaadf_pearsonr_table_NoLimit=Eval_med.Calculate_correlation(label_choose_lst,df_formant_statistic_all,N,columns,constrain_sex=-1, constrain_module=-1,evictNamelst=ManualCondition[ 'unreasonable_all'])
Aaadf_pearsonr_table_NoLimit=Eval_med.Calculate_correlation(label_choose_lst,df_formant_statistic_all,N,columns,constrain_sex=-1, constrain_module=-1)

# =============================================================================
'''

    Plot area

''' 


# inspect_cols=['MSB_f1(A:,i:,u:)', 'MSB_f2(A:,i:,u:)']

# df_formant_statistic_77_Notautism=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=0)
# df_formant_statistic_77_ASD=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=1)
# df_formant_statistic_77_autism=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=2)


# df_formant_statistic_77_inspect=df_formant_statistic_77[inspect_cols]
# df_formant_statistic_TD_inspect=df_formant_statistic_TD[inspect_cols]

# df_formant_statistic_77[]
# =============================================================================
# Top_data_lst=['df_formant_statistic_77_Notautism', 'df_formant_statistic_77_ASD', \
#               'df_formant_statistic_77_autism', 'df_formant_statistic_TD']
Top_data_lst=['df_formant_statistic_77','df_formant_statistic_TD']
# Top_data_lst=['M3','M4']
import warnings
warnings.filterwarnings("ignore")
for columns in inspect_cols:
    
    # plt.figure()
    fig, ax = plt.subplots()
    # data=[df_formant_statistic_77, df_formant_statistic_TD]
    data=[]
    for dstr in Top_data_lst:
        data.append(vars()[dstr])
    # data=[vars()[dstr] for dstr in Top_data_lst]
    # data=[df_formant_statistic_TD[columns]]
    for d in data:

        # ax = sns.distplot(d[columns], ax=ax, kde=False)
        ax = sns.distplot(d[columns], ax=ax, label=Top_data_lst)
        title='{0}'.format('Inspecting feature ' + columns)
        plt.title( title )
        
warnings.simplefilter('always')



# =============================================================================
'''

    ANOVA area

'''
# =============================================================================
# df_formant_statistic_77_tomerge=df_formant_statistic_77.copy()
# df_formant_statistic_77_tomerge['ASD']='ASD'
# df_formant_statistic_TD_tomerge=df_formant_statistic_TD.copy()
# df_formant_statistic_TD_tomerge['ASD']='TD'
# df_all=df_formant_statistic_77_tomerge.append(df_formant_statistic_TD_tomerge)
# import re
# punc=":,()"
# # df_allremaned=df_all.rename(columns=lambda s: s.replace("(","").replace(")",""))
# df_allremaned=df_all.rename(columns=lambda s: re.sub(u"[{}]+".format(punc),"",s))
# formula='MSB_f1(A:,i:,u:) + MSB_f2(A:,i:,u:) ~ ADOS'
# formula=re.sub(u"[{}]+".format(punc),"",formula)
# maov = MANOVA.from_formula(formula, data=df_allremaned)


# print(maov.mv_test())
