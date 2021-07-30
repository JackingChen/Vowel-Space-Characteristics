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
import itertools
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from itertools import combinations
from pylab import text
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

# Other features
df_dur_strlen_speed_ASD77_path='Features/Other/df_dur_strlen_speed_ASD.pkl'
df_dur_strlen_speed_TD_path='Features/Other/df_dur_strlen_speed_TD.pkl'
df_dur_strlen_speed_77=pickle.load(open(df_dur_strlen_speed_ASD77_path,'rb'))
df_dur_strlen_speed_TD=pickle.load(open(df_dur_strlen_speed_TD_path,'rb'))

df_formant_statistic_77=pd.concat([df_formant_statistic_77,df_dur_strlen_speed_77],axis=1)
df_formant_statistic_TD=pd.concat([df_formant_statistic_TD,df_dur_strlen_speed_TD],axis=1)

# sns.scatterplot(data=df_formant_statistic_77, x="name", y="BWratio(A:,i:,u:)", hue="ADOS_cate")

''' Load DF that describes cases not qualified (ex: one example might be defined in articulation ) '''
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
N=2


comb=[['df_formant_statistic_TD','df_formant_statistic_77'],]
Parameters=['FCR','totalword',
   'VSA1', 'F_vals_f1(A:,i:,u:)', 'F_vals_f2(A:,i:,u:)',
   'F_val_mix(A:,i:,u:)', 'MSB_f1(A:,i:,u:)', 'MSB_f2(A:,i:,u:)',
   'MSB_mix', 'BWratio(A:,i:,u:)_norm', 'BWratio(A:,i:,u:)', 'BV(A:,i:,u:)_l2', 'WV(A:,i:,u:)_l2',
   'F_vals_f1(i:,u:)', 'F_vals_f2(i:,u:)', 'F_val_mix(i:,u:)',
   'MSB_f1(i:,u:)', 'MSB_f2(i:,u:)', 'BWratio(i:,u:)', 'BV(i:,u:)_l2',
   'WV(i:,u:)_l2', 'F_vals_f1(A:,u:)', 'F_vals_f2(A:,u:)',
   'F_val_mix(A:,u:)', 'MSB_f1(A:,u:)', 'MSB_f2(A:,u:)', 'BWratio(A:,u:)',
   'BV(A:,u:)_l2', 'WV(A:,u:)_l2', 'F_vals_f1(A:,i:)', 'F_vals_f2(A:,i:)',
   'F_val_mix(A:,i:)', 'MSB_f1(A:,i:)', 'MSB_f2(A:,i:)', 'BWratio(A:,i:)',
   'BV(A:,i:)_l2', 'WV(A:,i:)_l2']

df_formant_statistic_77=criterion_filter(df_formant_statistic_77,\
                                        constrain_sex=sex,constrain_module=module,N=N,constrain_agemax=agemax,constrain_agemin=agemin,constrain_ADOScate=ADOScate,\
                                        evictNamelst=[])
df_formant_statistic_TD=criterion_filter(df_formant_statistic_TD,constrain_sex=sex,constrain_module=module,N=N,constrain_agemax=agemax)

df_ttest_result=pd.DataFrame([],columns=['doc-kid','p-val'])
for role_1,role_2  in comb:
    for parameter in Parameters:
        # test=stats.ttest_ind(vars()[role_1][parameter], vars()[role_2][parameter])
        test=stats.mannwhitneyu(vars()[role_1][parameter], vars()[role_2][parameter])
        # print(parameter, '{0} vs {1}'.format(role_1,role_2),test)
        # print(role_1+':',vars()[role_1][parameter].mean(),role_2+':',vars()[role_2][parameter].mean())
        df_ttest_result.loc[parameter,'doc-kid'] = vars()[role_1][parameter].mean() - vars()[role_2][parameter].mean()
        df_ttest_result.loc[parameter,'p-val'] = test[1]

  
# =============================================================================
'''

    Regression area

'''
Eval_med=Evaluation_method()
label_choose_lst=['ADOS_C']
df_formant_statistic_77['ASDTD']=1
df_formant_statistic_TD['ASDTD']=2
df_formant_statistic_all=df_formant_statistic_77.append(df_formant_statistic_TD)
df_formant_statistic_all=df_formant_statistic_all[~df_formant_statistic_all[['Module','sex']].isna().any(axis=1)]
# tmpoutpath='Features/artuculation_AUI/Vowels/'
# pickle.dump(df_formant_statistic_all,open(tmpoutpath+'Formant_AUI_tVSAFCRFvals_ASDkid+TD.pkl','wb'))

# =============================================================================



''' OLS report '''
'''
    Operation steps:
        Unfortunately stepwise remove regression requires heavily on manual operation
        we then set the standard operation rules 
        
        1. put all your variables, and run regression
        2. bookeep the previous formula and eliminate the most unsignificant variables
        3. back to 1. and repeat untill all variables are significant
        
    
'''
def Regression_Preprocess_setp(DV_str, IV_lst ,df,punc=":,()"):
    df_remaned=df.rename(columns=lambda s: re.sub(u"[{}]+".format(punc),"",s))
    DV=re.sub(u"[{}]+".format(punc),"",DV_str)
    IV_lst_renamed=[]
    for i,IV_str in enumerate(IV_lst):
        if IV_str in categorical_cols:
            IV_lst_renamed.append('C({})'.format(IV_str))
        else:
            IV_lst_renamed.append(re.sub(u"[{}]+".format(punc),"",IV_str))
    IV_lst_renamed_str = ' + '.join(IV_lst_renamed)
    formula = '{DV} ~ {IV}'.format(DV=DV,IV=IV_lst_renamed_str)
    
    return df_remaned, formula

Formula_bookeep_dict=Dict()


IV_lst=['FCR',
   'VSA1', 'F_vals_f1(A:,i:,u:)', 'F_vals_f2(A:,i:,u:)',
   'F_val_mix(A:,i:,u:)', 'MSB_f1(A:,i:,u:)', 'MSB_f2(A:,i:,u:)',
   'MSB_mix', 'BWratio(A:,i:,u:)', 'BV(A:,i:,u:)_l2', 'WV(A:,i:,u:)_l2',
   'F_vals_f1(i:,u:)', 'F_vals_f2(i:,u:)', 'F_val_mix(i:,u:)',
   'MSB_f1(i:,u:)', 'MSB_f2(i:,u:)', 'BWratio(i:,u:)', 'BV(i:,u:)_l2',
   'WV(i:,u:)_l2', 'F_vals_f1(A:,u:)', 'F_vals_f2(A:,u:)',
   'F_val_mix(A:,u:)', 'MSB_f1(A:,u:)', 'MSB_f2(A:,u:)', 'BWratio(A:,u:)',
   'BV(A:,u:)_l2', 'WV(A:,u:)_l2', 'F_vals_f1(A:,i:)', 'F_vals_f2(A:,i:)',
   'F_val_mix(A:,i:)', 'MSB_f1(A:,i:)', 'MSB_f2(A:,i:)', 'BWratio(A:,i:)',
   'BV(A:,i:)_l2', 'WV(A:,i:)_l2','sex','Module']
categorical_cols=['sex','Module','ADOS_cate','ASDTD']
DV_str='ADOS'

df_remaned, formula = Regression_Preprocess_setp(DV_str, IV_lst, df_formant_statistic_all)

significant_lvl=0.05
max_p=1
count=1
# for count in range(10):
while max_p > significant_lvl:
    res = smf.ols(formula=formula, data=df_remaned).fit()    
    
    max_p=max(res.pvalues  )
    if max_p < significant_lvl:
        print(res.summary())
        break
    remove_indexs=res.pvalues[res.pvalues==max_p].index
    print('remove Indexes ',remove_indexs,' with pvalue', max_p)
    
    Formula_bookeep_dict['step_'+str(count)].formula=formula
    Formula_bookeep_dict['step_'+str(count)].removed_indexes=remove_indexs.values.astype(str).tolist()
    Formula_bookeep_dict['step_'+str(count)].removedP=max_p
    Formula_bookeep_dict['step_'+str(count)].rsquared_adj=res.rsquared_adj
    
    header_str=formula[:re.search('~ ',formula).end()]
    formula_lst=formula[re.search('~ ',formula).end():].split(" + ")
    for rm_str in Formula_bookeep_dict['step_'+str(count)].removed_indexes:
        rm_str=re.sub("[\[].*?[\]]", "", rm_str)
        formula_lst.remove(rm_str)
    formula = header_str + ' + '.join(formula_lst)
    count+=1

# =============================================================================
''' foward selection '''
# =============================================================================

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

punc=":,()"
df_formant_statistic_all_remaned=df_formant_statistic_TD.rename(columns=lambda s: re.sub(u"[{}]+".format(punc),"",s))
df_formant_statistic_all_remaned['Module']=df_formant_statistic_all_remaned['Module'].astype(str)
df_formant_statistic_all_remaned.loc[df_formant_statistic_all_remaned['Module']==3,'Module']=1
df_formant_statistic_all_remaned.loc[df_formant_statistic_all_remaned['Module']==4,'Module']=2


formula='ADOS ~ '
formula+= ' BWratio(A:,i:,u:) '
# formula+= ' + BV(A:,i:,u:)_l2'
formula=re.sub(u"[{}]+".format(punc),"",formula)

# formula+=' + C(sex) '
# formula+=' + age '
# formula+=' + C(ASDTD) '
# formula+=' + C(Module) * C(sex)'
# formula+=' + C(Module) * C(ASDTD)'
# formula+=' + C(sex) * C(ASDTD)'
# formula+=' + C(sex) + C(Module) * C(ASDTD)'
res = smf.ols(formula=formula, data=df_formant_statistic_all_remaned).fit()
print(res.summary())




formula='BWratio(A:,i:,u:) ~'
formula=re.sub(u"[{}]+".format(punc),"",formula)
# formula+=' + ADOS '
# formula+=' + C(sex) '
# formula+=' + C(Module) '
formula+=' + age '
# formula+=' ADOS * age '
# formula+=' + C(ASDTD) '
# formula+=' + C(Module) * C(sex)'
# formula+=' + C(Module) * C(ASDTD)'
# formula+=' + C(sex) * C(ASDTD)'
# formula+=' + C(sex) + C(Module) * C(ASDTD)'
res_rev = smf.ols(formula=formula, data=df_formant_statistic_all_remaned).fit()
print(res_rev.summary())



''' small, stepwise forward regression '''

comb_lsts=[' + C(sex) ', ' + age ', ]
df_result=pd.DataFrame([],columns=['R2_adj'])
for i in range(0,len(comb_lsts)+1):
    combs = combinations(comb_lsts, i)
    for additional_str in combs:
        formula='BWratio(A:,i:,u:) ~ ADOS'
        formula=re.sub(u"[{}]+".format(punc),"",formula)
        
        formula+= ''.join(additional_str)
        res = smf.ols(formula=formula, data=df_formant_statistic_all_remaned).fit()
        variables = formula[re.search('~ ',formula).end():]
        print(variables, res.rsquared_adj)
        df_result.loc[variables]=res.rsquared_adj
        

# =============================================================================
'''

    ANOVA area

'''
# =============================================================================
IV_list=['sex','Module','ASDTD']
# IV_lst=['ASDTD']
# comcinations=list(itertools.product(*Effect_comb))
ways=2
combination = combinations(IV_list, ways)
for comb in combination:
    IV_lst = list(comb)
    DV_str='BWratio(A:,i:,u:)'
    df_remaned, formula = Regression_Preprocess_setp(DV_str, IV_lst, df_formant_statistic_all)
    punc=":,()"
    model = ols(formula, data=df_remaned).fit()
    anova = sm.stats.anova_lm(model, typ=ways)
    print(anova)

# =============================================================================
''' Single Variable correlation ''' 
inspect_cols=['BWratio(A:,i:,u:)_norm','BV(A:,i:,u:)_l2', 'WV(A:,i:,u:)_l2']
Mapping_dict={'girl':2,'boy':1,'M3':3,'M4':4,'None':-1, 'ASD':1,'TD':2}
# =============================================================================
Regess_result_dict=Dict()

gender_set=['None','boy','girl']
module_set=['None','M3','M4']
ASDTD_set=['None','ASD','TD']
Effect_comb=[gender_set,module_set,ASDTD_set]
comcinations=list(itertools.product(*Effect_comb))

for correlation_type in ['spearmanr','pearsonr']:
    df_collector_top=pd.DataFrame()
    for sex_str, module_str, ASDTD_str in comcinations:
        sex=Mapping_dict[sex_str]
        module=Mapping_dict[module_str]
        ASDTD=Mapping_dict[ASDTD_str]
        df_correlation=Eval_med.Calculate_correlation(label_choose_lst,df_formant_statistic_all,N,Parameters,\
                                                      constrain_sex=sex, constrain_module=module, constrain_ASDTD=ASDTD,\
                                                      correlation_type=correlation_type)
        if len(df_correlation)>0:
            df_correlation=df_correlation.loc[inspect_cols].round(3)
        
            df_collector=pd.DataFrame([],index=df_correlation.index)
            Namer=Dict()
            for col in df_correlation:
                Namer[col]=df_correlation[col].astype(str).values
            r_value_str=Namer[list(Namer.keys())[0]].astype(str)
            p_value_str=Namer[list(Namer.keys())[1]].astype(str)
            p_value_str=np.core.defchararray.add(["("]*len(r_value_str), p_value_str)
            p_value_str=np.core.defchararray.add(p_value_str,[")"]*len(r_value_str))
            corr_result_str=np.core.defchararray.add(r_value_str,p_value_str)
            df_collector['{sex}_{module}_{ASDTD}'.format(sex=sex_str,module=module_str,ASDTD=ASDTD_str).replace('None','')]\
                                                    =corr_result_str
            df_collector_top=pd.concat([df_collector_top,df_collector],axis=1)
    Regess_result_dict['/'.join(df_correlation.columns[:2])]=df_collector_top


# Aaadf_pearsonr_table_all=Eval_med.Calculate_correlation(label_choose_lst,df_formant_statistic_all,N,Parameters,constrain_sex=-1, constrain_module=-1)
Aaadf_pearsonr_table_ASD=Eval_med.Calculate_correlation(label_choose_lst,df_formant_statistic_77,N,Parameters,constrain_sex=-1, constrain_module=-1)
# Aaadf_pearsonr_table_TD=Eval_med.Calculate_correlation(label_choose_lst,df_formant_statistic_TD,N,Parameters,constrain_sex=-1, constrain_module=-1)

# Aaadf_pearsonr_table_ASDM3=Eval_med.Calculate_correlation(label_choose_lst,df_formant_statistic_77,N,Parameters,constrain_sex=-1, constrain_module=3)
# Aaadf_pearsonr_table_ASDM4=Eval_med.Calculate_correlation(label_choose_lst,df_formant_statistic_77,N,Parameters,constrain_sex=-1, constrain_module=4)
# Aaadf_pearsonr_table_ASDboy=Eval_med.Calculate_correlation(label_choose_lst,df_formant_statistic_77,N,Parameters,constrain_sex=1, constrain_module=-1)
# Aaadf_pearsonr_table_ASDgirl=Eval_med.Calculate_correlation(label_choose_lst,df_formant_statistic_77,N,Parameters,constrain_sex=2, constrain_module=-1)



# =============================================================================
'''

    Plot area

''' 
filter_boy=df_formant_statistic_77['sex']==1
filter_girl=df_formant_statistic_77['sex']==2
filter_M3=df_formant_statistic_77['Module']==3
filter_M4=df_formant_statistic_77['Module']==4
df_formant_statistic_77_boy=df_formant_statistic_77[filter_boy]
df_formant_statistic_77_girl=df_formant_statistic_77[filter_girl]
df_formant_statistic_77_M3boy=df_formant_statistic_77[filter_M3 & filter_boy]
df_formant_statistic_77_M4boy=df_formant_statistic_77[filter_M4 & filter_boy]
df_formant_statistic_77_M3girl=df_formant_statistic_77[filter_M3 & filter_girl]
df_formant_statistic_77_M4girl=df_formant_statistic_77[filter_M4 & filter_girl]
df_formant_statistic_77_M3=df_formant_statistic_77[filter_M3 ]
df_formant_statistic_77_M4=df_formant_statistic_77[filter_M4 ]

filter_boy=df_formant_statistic_TD['sex']==1
filter_girl=df_formant_statistic_TD['sex']==2
filter_M3=df_formant_statistic_TD['Module']==3
filter_M4=df_formant_statistic_TD['Module']==4

df_formant_statistic_TD_boy=df_formant_statistic_TD[filter_boy]
df_formant_statistic_TD_girl=df_formant_statistic_TD[filter_girl]
df_formant_statistic_TD_M3boy=df_formant_statistic_TD[filter_M3 & filter_boy]
df_formant_statistic_TD_M4boy=df_formant_statistic_TD[filter_M4 & filter_boy]
df_formant_statistic_TD_M3girl=df_formant_statistic_TD[filter_M3 & filter_girl]
df_formant_statistic_TD_M4girl=df_formant_statistic_TD[filter_M4 & filter_girl]
df_formant_statistic_TD_M3=df_formant_statistic_TD[filter_M3 ]
df_formant_statistic_TD_M4=df_formant_statistic_TD[filter_M4 ]


TopTop_data_lst=[]
TopTop_data_lst.append(['df_formant_statistic_77','df_formant_statistic_TD'])
TopTop_data_lst.append(['df_formant_statistic_77_M3','df_formant_statistic_TD_M3'])
TopTop_data_lst.append(['df_formant_statistic_77_M4','df_formant_statistic_TD_M4'])
TopTop_data_lst.append(['df_formant_statistic_77_boy','df_formant_statistic_TD_boy'])
TopTop_data_lst.append(['df_formant_statistic_77_girl','df_formant_statistic_TD_girl'])
TopTop_data_lst.append(['df_formant_statistic_77_M3boy','df_formant_statistic_TD_M3boy'])
TopTop_data_lst.append(['df_formant_statistic_77_M3girl','df_formant_statistic_TD_M3girl'])
TopTop_data_lst.append(['df_formant_statistic_77_M4boy','df_formant_statistic_TD_M4boy'])
TopTop_data_lst.append(['df_formant_statistic_77_M4girl','df_formant_statistic_TD_M4girl'])

# Top_data_lst=['df_formant_statistic_77','df_formant_statistic_TD']
# Top_data_lst=['df_dur_strlen_speed_77','df_dur_strlen_speed_TD']
# inspect_cols=['BWratio(A:,i:,u:)']
# inspect_cols=['speed']
for Top_data_lst in TopTop_data_lst:
    # inspect_cols=['BV(A:,i:,u:)_l2']
    # inspect_cols=['BWratio(A:,i:,u:)']
    inspect_cols=['age']
    
    # inspect_cols=['VSA1']
    # Top_data_lst=['M3','M4']
    import warnings
    warnings.filterwarnings("ignore")
    for columns in inspect_cols:
        fig, ax = plt.subplots()
        data=[]
        dataname=[]
        for dstr in Top_data_lst:
            dataname.append(dstr)
            data.append(vars()[dstr])
     
        for i,d in enumerate(data):
            # ax = sns.distplot(d[columns], ax=ax, kde=False)
            ax = sns.distplot(d[columns], ax=ax, label=Top_data_lst)
            title='{0}'.format('Inspecting feature ' + columns)
            plt.title( title )
        fig.legend(labels=dataname)    
        print('Testing Feature: ',columns)
        for tests in [stats.mannwhitneyu, stats.ttest_ind]:
            test_results=tests(vars()[Top_data_lst[0]][columns],vars()[Top_data_lst[1]][columns])
            print(test_results)
            p_val=test_results[1]
            if tests == stats.mannwhitneyu:
                mean_difference=vars()[Top_data_lst[0]][columns].median() - vars()[Top_data_lst[1]][columns].median()
            elif tests == stats.ttest_ind:
                mean_difference=vars()[Top_data_lst[0]][columns].mean() - vars()[Top_data_lst[1]][columns].mean()
        addtext='{0}/({1})'.format(np.round(mean_difference,3),np.round(p_val,3))
        text(0.9, 0.9, addtext, ha='center', va='center', transform=ax.transAxes)
        addtextvariable='{0} vs {1}'.format(Top_data_lst[0],Top_data_lst[1])
        text(0.9, 0.6, addtextvariable, ha='center', va='center', transform=ax.transAxes)
    warnings.simplefilter('always')
















# ''' Code backup '''




df_formant_statistic_77_mean=df_formant_statistic_77.mean()
# inspect_cols=['u_num', 'a_num', 'i_num','MSB_f1(A:,i:,u:)', 'MSB_f2(A:,i:,u:)','ADOS']

df_formant_statistic_77_inspect=df_formant_statistic_77[inspect_cols]
df_formant_statistic_TD_inspect=df_formant_statistic_TD[inspect_cols]

df_formant_statistic_77_Notautism_mean=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=0).mean()
df_formant_statistic_77_ASD_mean=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=1).mean()
df_formant_statistic_77_autism_mean=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=2).mean()
df_formant_statistic_TD_mean=df_formant_statistic_TD.mean()

filter_boy=df_formant_statistic_77['sex']==1
filter_girl=df_formant_statistic_77['sex']==2
filter_M3=df_formant_statistic_77['Module']==3
filter_M4=df_formant_statistic_77['Module']==4
# print(df_formant_statistic_77_inspect[filter_boy].mean())
# print(df_formant_statistic_77_inspect[filter_girl].mean())
print(df_formant_statistic_77['VSA1'][filter_M3 & filter_boy].mean())
print(df_formant_statistic_77['VSA1'][filter_M4 & filter_boy].mean())

print(df_formant_statistic_77['VSA1'][filter_M3 & filter_girl].mean())
print(df_formant_statistic_77['VSA1'][filter_M4 & filter_girl].mean())

# Top_data_lst=['df_formant_statistic_77','df_formant_statistic_TD']
df_formant_statistic_77_M3boy=df_formant_statistic_77[filter_M3 & filter_boy]
df_formant_statistic_77_M4boy=df_formant_statistic_77[filter_M4 & filter_boy]
df_formant_statistic_77_M3girl=df_formant_statistic_77[filter_M3 & filter_girl]
df_formant_statistic_77_M4girl=df_formant_statistic_77[filter_M4 & filter_girl]
df_formant_statistic_77_M3=df_formant_statistic_77[filter_M3 ]
df_formant_statistic_77_M4=df_formant_statistic_77[filter_M4 ]

df_formant_statistic_77_Notautism=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=0)
df_formant_statistic_77_ASD=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=1)
df_formant_statistic_77_autism=criterion_filter(df_formant_statistic_77,constrain_sex=-1,constrain_module=-1,N=N,constrain_agemax=-1,constrain_ADOScate=2)
# Top_data_lst=["df_formant_statistic_77_M3boy",\
#               "df_formant_statistic_77_M4boy"]
# Top_data_lst=["df_formant_statistic_77_M3girl",\
#               "df_formant_statistic_77_M4girl"]
# Top_data_lst=["df_formant_statistic_77_Notautism",\
#               "df_formant_statistic_77_ASD"]
Top_data_lst=["df_formant_statistic_77_autism",\
              "df_formant_statistic_77_ASD"]
# Top_data_lst=["df_formant_statistic_77_autism",\
#               "df_formant_statistic_77_Notautism"]
    

    

filter_boy=df_formant_statistic_TD['sex']==1
filter_girl=df_formant_statistic_TD['sex']==2
filter_M3=df_formant_statistic_TD['Module']==3
filter_M4=df_formant_statistic_TD['Module']==4
print(df_formant_statistic_TD_inspect[filter_boy].mean())
print(df_formant_statistic_TD_inspect[filter_girl].mean())

print(df_formant_statistic_TD['VSA1'][filter_M3 & filter_boy].mean())
print(df_formant_statistic_TD['VSA1'][filter_M4 & filter_boy].mean())

print(df_formant_statistic_TD['VSA1'][filter_M3 & filter_girl].mean())
print(df_formant_statistic_TD['VSA1'][filter_M4 & filter_girl].mean())

df_formant_statistic_TD_M3boy=df_formant_statistic_TD[filter_M3 & filter_boy]
df_formant_statistic_TD_M4boy=df_formant_statistic_TD[filter_M4 & filter_boy]
df_formant_statistic_TD_M3girl=df_formant_statistic_TD[filter_M3 & filter_girl]
df_formant_statistic_TD_M4girl=df_formant_statistic_TD[filter_M4 & filter_girl]
df_formant_statistic_TD_M3=df_formant_statistic_TD[filter_M3 ]
df_formant_statistic_TD_M4=df_formant_statistic_TD[filter_M4 ]
# Top_data_lst=["df_formant_statistic_TD_M3boy",\
#               "df_formant_statistic_TD_M4boy"]
# Top_data_lst=["df_formant_statistic_TD_M3girl",\
#               "df_formant_statistic_TD_M4girl"]
Top_data_lst=["df_formant_statistic_TD_M3",\
              "df_formant_statistic_TD_M4"]
# Top_data_lst=['M3','M4']

