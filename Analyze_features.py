#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:14:41 2020

@author: jackchen

This script does the main expeeriments in Table 1 (Correlation between Level of clustering and ADOS_A) 
1. Data prepare area: Gather raw data of the three critical monophthongs (F1 & F2) and save in: df_formant_statistic.
    * Note we hack into scipy f-classif function to decompose ssbn and sswn. We found that ssbn is the main factor of correlation 
2. Correlation area: Write a standard correlation function to calculate the correlations between all features ADOS_C 
3. t-test area: t-test between each groups 



4. Manual Ttest area: Not really important, just ignore it

"""

import pickle
import argparse
from addict import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

import statsmodels.api as sm

from scipy import stats
from scipy.stats import spearmanr,pearsonr 
import statistics 
import os, sys
import statsmodels.api as sm
from varname import nameof



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


from sklearn.utils import (as_float_array, check_array, check_X_y, safe_sqr,
                     safe_mask)
from scipy import special, stats
import warnings

def f_classif(X, y):
    """Compute the ANOVA F-value for the provided sample.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} shape = [n_samples, n_features]
        The set of regressors that will be tested sequentially.

    y : array of shape(n_samples)
        The data matrix.

    Returns
    -------
    F : array, shape = [n_features,]
        The set of F values.

    pval : array, shape = [n_features,]
        The set of p-values.

    See also
    --------
    chi2: Chi-squared stats of non-negative features for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    """
    # X, y = check_X_y(X, y, ['csr', 'csc', 'coo'])
    args = [X[safe_mask(X, y == k)] for k in np.unique(y)]
    n_classes = len(args)
    args = [as_float_array(a) for a in args]
    n_samples_per_class = np.array([a.shape[0] for a in args])
    n_samples = np.sum(n_samples_per_class)
    ss_alldata = sum(safe_sqr(a).sum(axis=0) for a in args) #sum of square of all data [21001079.59736017]
    sums_args = [np.asarray(a.sum(axis=0)) for a in args] #sum of data in each group [3204.23910828, 7896.25971663, 5231.79595847]
    square_of_sums_alldata = sum(sums_args) ** 2  # square of summed data [2.66743853e+08]
    square_of_sums_args = [s ** 2 for s in sums_args]
    sstot = ss_alldata - square_of_sums_alldata / float(n_samples)
    ssbn = 0.
    for k, _ in enumerate(args):
        ssbn += square_of_sums_args[k] / n_samples_per_class[k]
    ssbn -= square_of_sums_alldata / float(n_samples)
    sswn = sstot - ssbn
    dfbn = n_classes - 1
    dfwn = n_samples - n_classes
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    constant_features_idx = np.where(msw == 0.)[0]
    if (np.nonzero(msb)[0].size != msb.size and constant_features_idx.size):
        warnings.warn("Features %s are constant." % constant_features_idx,
                      UserWarning)
    f = msb / msw
    # flatten matrix to vector in sparse case
    f = np.asarray(f).ravel()
    prob = special.fdtrc(dfbn, dfwn, f)
    return f, prob, msb, msw, ssbn
            
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
'''

# =============================================================================

def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--pklpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--Inspect', default=False,
                            help='path of the base directory')
    parser.add_argument('--correlation_type', default='spearmanr',
                            help='spearmanr|pearsonr')
    parser.add_argument('--label_choose_lst', default=['ADOS_C'],
                            help='path of the base directory')
    parser.add_argument('--Stat_med_str_VSA', default='mean',
                            help='path of the base directory')
    parser.add_argument('--role', default='ASDTD',
                            help='path of the base directory')
    parser.add_argument('--outpklpath', default="articulation/Pickles/Session_formants_people_vowel_feat/",
                            help='path of the base directory')
    parser.add_argument('--F0WindowSize', default=1, type=int,
                            help='path of the base directory')
    args = parser.parse_args()
    return args

# =============================================================================
'''
    
    1. Data prepare area

'''
# =============================================================================


''' parse namespace '''
args = get_args()
base_path=args.base_path
pklpath=args.pklpath
INSPECT=args.Inspect
label_choose_lst=args.label_choose_lst # labels are too biased
Stat_med_str=args.Stat_med_str_VSA
outpklpath=args.outpklpath
windowsize=args.F0WindowSize


phonation_path=base_path + "/phonation"
articulation_path=base_path + "/articulation"

path_app = base_path
sys.path.append(path_app)
from articulation.HYPERPARAM import phonewoprosody, Label

''' Vowels sets '''
Vowels_single=['i_','E','axr','A_','u_','ax','O_']
Vowels_prosody_single=[phonewoprosody.Phoneme_sets[v]  for v in Vowels_single]
Vowels_prosody_single=[item for sublist in Vowels_prosody_single for item in sublist]

role=args.role
# Formants_utt_symb=pickle.load(open(pklpath+"/Formants_utt_symb_bymiddle.pkl","rb"))
#####################
''' Formants_people_symb[spkr_name][phone] = [F1, F2] record F1, F2's of each people'''
#####################
Formants_people_symb=pickle.load(open(articulation_path+"/Pickles/Formants_people_symb_bymiddle_window{0}_{1}.pkl".format(windowsize,role),"rb"))
#####################
''' Phonation_people_symb[spkr_name] -> DF.loc[phone]=phone_feats '''
#####################
Phonation_people_symb=pickle.load(open(phonation_path+"/features/Phonation_people_symb_ADOS_TD_WordLevel.pkl","rb"))

# Formants_people_symb=pickle.load(open(pklpath+"/Formants_people_symb_bymiddle.pkl".format(role),"rb"))

label_set=['ADOS_C','ADOS_S','ADOS_SC']
# =============================================================================

'''
    Inspect area
'''

# =============================================================================


# =============================================================================
''' grouping severity '''
ADOS_label=Label.label_raw['ADOS_C']
array=list(set(ADOS_label))
phoneme="A:"
groups=[np.split(array, idx)  for n_splits in range(3)  for idx in combinations(range(1, len(array)), n_splits) if len(np.split(array, idx))>1]
# =============================================================================

pid=list(sorted(Formants_people_symb.keys()))
pid_dict={}
for i,pi in enumerate(pid):
    pid_dict[pi]=i


######################################
''' iteration area '''

label_choose='ADOS_C'


group_std=[np.array([0, 1]),np.array([2,3,4,5]),np.array([6,7,8])]
group_clinical=[np.array([0, 1]),np.array([2,3]),np.array([4,5,6,7,8])]
group_uttphone=[np.array([0, 1]),np.array([2,3,4,5]),np.array([4,5,6,7,8])]
# groups=[]
# groups.append(group_std)
# groups.append(group_ADOSSCORE)
# groups.append(group_clinical)
#######################################


''' assert if the names don't matches' '''
NameMatchAssertion(Formants_people_symb,Label.label_raw['name'].values)


''' Vowel AUI rule is using phonewoprosody '''
# PhoneMapp_dict={'u':['w']+phonewoprosody.Phoneme_sets['u_'],\
#                 'i':['j']+phonewoprosody.Phoneme_sets['i_'],\
#                 'a':phonewoprosody.Phoneme_sets['A_']}

PhoneMapp_dict={'u':phonewoprosody.Phoneme_sets['u_'],\
                'i':phonewoprosody.Phoneme_sets['i_'],\
                'a':phonewoprosody.Phoneme_sets['A_']}
    

    
Vowels_AUI=Dict()
Vowels_AUI_sampNum=Dict()
# for people in Label.label_raw.sort_values(by='ADOS_C')['name']:
for people in Formants_people_symb.keys():    
    for phone, values in Formants_people_symb[people].items():
        if phone not in [e for _, phoneme in PhoneMapp_dict.items() for e in phoneme]: # update fixed 2021/05/27
            continue
        else:
            for p_key, p_val in PhoneMapp_dict.items():
                if phone in p_val:
                    Phone_represent=p_key
            if people not in Vowels_AUI.keys():
                if Phone_represent not in Vowels_AUI[people].keys():
                    Vowels_AUI[people][Phone_represent]=values
                    Vowels_AUI_sampNum[people][Phone_represent]=len(values)
                else:
                    Vowels_AUI[people][Phone_represent].extend(values)
                    Vowels_AUI_sampNum[people][Phone_represent]+=len(values)
            else:
                if Phone_represent not in Vowels_AUI[people].keys():
                    Vowels_AUI[people][Phone_represent]=values
                    Vowels_AUI_sampNum[people][Phone_represent]=len(values)
                else:
                    Vowels_AUI[people][Phone_represent].extend(values)
                    Vowels_AUI_sampNum[people][Phone_represent]+=len(values)

Vowels_AUI_mean=Dict()
for key1, values1 in Vowels_AUI.items():
    for key2, values2 in Vowels_AUI[key1].items():
        Vowels_AUI_mean[key1][key2]=np.mean(np.vstack(values2 ),axis=0)

Vowels_5As=Dict()
# for people in Label.label_raw.sort_values(by='ADOS_C')['name']:
for people in Formants_people_symb.keys():
    for phone, values in Formants_people_symb[people].items():
        if phone in phonewoprosody.Phoneme_sets['A_']:
            Vowels_5As[people][phone]=values # Take only the first character of phone, [A_]
# =============================================================================
Statistic_method={'mean':np.mean,'median':np.median,'mode':stats.mode}

# =============================================================================
''' 

    Put your data to analyze here

'''
# =============================================================================
df_formant_statistic=pd.DataFrame()
for people in Vowels_AUI_mean.keys(): #update 2021/05/27 fixed 
    RESULT_dict={}
    F12_raw_dict=Vowels_AUI[people]
    F12_val_dict={k:[] for k in ['u','a','i']}
    for k,v in F12_raw_dict.items():
        if Stat_med_str == 'mode':
            F12_val_dict[k]=Statistic_method[Stat_med_str](v,axis=0)[0].ravel()
        else:
            F12_val_dict[k]=Statistic_method[Stat_med_str](v,axis=0)
    
    RESULT_dict['u_num'], RESULT_dict['a_num'], RESULT_dict['i_num']=Vowels_AUI_sampNum[people]['u'],Vowels_AUI_sampNum[people]['a'],Vowels_AUI_sampNum[people]['i']
    
    RESULT_dict['ADOS']=Label.label_raw[label_choose][Label.label_raw['name']==people].values    
    RESULT_dict['sex']=Label.label_raw['sex'][Label.label_raw['name']==people].values[0]
    RESULT_dict['age']=Label.label_raw['age_year'][Label.label_raw['name']==people].values[0]
    RESULT_dict['Module']=Label.label_raw['Module'][Label.label_raw['name']==people].values[0]
    
    u=F12_val_dict['u']
    a=F12_val_dict['a']
    i=F12_val_dict['i']

    
    if len(u)==0 or len(a)==0 or len(i)==0:
        u_num= RESULT_dict['u_num'] if type(RESULT_dict['u_num'])==int else 0
        i_num= RESULT_dict['i_num'] if type(RESULT_dict['i_num'])==int else 0
        a_num= RESULT_dict['a_num'] if type(RESULT_dict['a_num'])==int else 0
        df_RESULT_list=pd.DataFrame(np.zeros([1,len(df_formant_statistic.columns)]),columns=df_formant_statistic.columns)
        df_RESULT_list.index=[people]
        df_RESULT_list['FCR']=10
        df_RESULT_list['ADOS']=RESULT_dict['ADOS'][0]
        df_RESULT_list[['u_num','a_num','i_num']]=[u_num, i_num, a_num]
        df_formant_statistic=df_formant_statistic.append(df_RESULT_list)
        continue
    
    numerator=u[1] + a[1] + i[0] + u[0]
    demominator=i[1] + a[0]
    RESULT_dict['FCR']=np.float(numerator/demominator)
    RESULT_dict['F2i_u']= u[1]/i[1]
    RESULT_dict['F1a_u']= u[0]/a[0]
    # assert FCR <=2
    
    RESULT_dict['VSA1']=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
    
    EDiu=np.sqrt((u[1]-i[1])**2+(u[0]-i[0])**2)
    EDia=np.sqrt((a[1]-i[1])**2+(a[0]-i[0])**2)
    EDau=np.sqrt((u[1]-a[1])**2+(u[0]-a[0])**2)
    S=(EDiu+EDia+EDau)/2
    # =============================================================================
    ''' F-value, Valid Formant measure '''
    
    # =============================================================================
    # Get data
    F12_raw_dict=Vowels_AUI[people]
    u=F12_raw_dict['u']
    a=F12_raw_dict['a']
    i=F12_raw_dict['i']
    df_vowel = pd.DataFrame(np.vstack([u,a,i]),columns=['F1','F2'])
    df_vowel['vowel'] = np.hstack([np.repeat('u',len(u)),np.repeat('a',len(a)),np.repeat('i',len(i))])
    df_vowel['target']=pd.Categorical(df_vowel['vowel'])
    df_vowel['target']=df_vowel['target'].cat.codes
    # F-test
    print("utt number of group u = {0}, utt number of group i = {1}, utt number of group A = {2}".format(\
        len(u),len(a),len(i)))
    F_vals=f_classif(df_vowel[['F1','F2']].values,df_vowel['target'].values)[0]
    RESULT_dict['F_vals_f1']=F_vals[0]
    RESULT_dict['F_vals_f2']=F_vals[1]
    RESULT_dict['F_val_mix']=RESULT_dict['F_vals_f1'] + RESULT_dict['F_vals_f2']
    
    msb=f_classif(df_vowel[['F1','F2']].values,df_vowel['target'].values)[2]
    msw=f_classif(df_vowel[['F1','F2']].values,df_vowel['target'].values)[3]
    ssbn=f_classif(df_vowel[['F1','F2']].values,df_vowel['target'].values)[4]
    
    
    
    RESULT_dict['MSB_f1']=msb[0]
    RESULT_dict['MSB_f2']=msb[1]
    MSB_f1 , MSB_f2 = RESULT_dict['MSB_f1'], RESULT_dict['MSB_f2']
    RESULT_dict['MSB_mix']=MSB_f1 + MSB_f2
    RESULT_dict['MSW_f1']=msw[0]
    RESULT_dict['MSW_f2']=msw[1]
    MSW_f1 , MSW_f2 = RESULT_dict['MSW_f1'], RESULT_dict['MSW_f2']
    RESULT_dict['MSW_mix']=MSW_f1 + MSW_f2


    df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict)
    df_RESULT_list.index=[people]
    df_formant_statistic=df_formant_statistic.append(df_RESULT_list)


if not os.path.exists(outpklpath):
    os.makedirs(outpklpath)
pickle.dump(df_formant_statistic,open(outpklpath+"Formant_AUI_tVSAFCRFvals_{}.pkl".format(role),"wb"))


df_formant_statistic_bool=(df_formant_statistic['u_num']!=0) & (df_formant_statistic['a_num']!=0) & (df_formant_statistic['i_num']!=0)
df_formant_statistic=df_formant_statistic[df_formant_statistic_bool]

''' ADD ADOS category '''
df_formant_statistic['ADOS_cate']=np.array([0]*len(df_formant_statistic))
df_formant_statistic['ADOS_cate'][df_formant_statistic['ADOS']<2]=0
df_formant_statistic['ADOS_cate'][(df_formant_statistic['ADOS']<3) & (df_formant_statistic['ADOS']>=2)]=1
df_formant_statistic['ADOS_cate'][df_formant_statistic['ADOS']>=3]=2

# =============================================================================
'''

    2. Correlation area

'''
# =============================================================================

''' Calculate correlations '''
columns=['FCR','VSA1','F_vals_f1', 'F_vals_f2', 'F_val_mix',\
          'MSB_f1','MSB_f2','MSB_mix','MSW_f1','MSW_f2','SSBN_f1','SSBN_f2']

# columns=['FCR','VSA1','F_vals_f1', 'F_vals_f2', 'F_val_mix','MSB_f1','MSB_f2',\
#          'dau1','dai1','diu1','daudai1','daudiu1','daidiu1','daidiudau1',\
#          'dau2','dai2','diu2','daudai2','daudiu2','daidiu2','daidiudau2',\
#          'F2i_u','F1a_u']
def Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns, corr_label='ADOS', constrain_sex=-1, constrain_module=-1, constrain_assessment=-1):
    '''
        constrain_sex: 1 for boy, 2 for girl
        constrain_module: 3 for M3, 4 for M4
    '''
    df_pearsonr_table=pd.DataFrame([],columns=[args.correlation_type,'{}_pvalue'.format(args.correlation_type[:5]),'de-zero_num'])
    for lab_choose in label_choose_lst:
        filter_bool=np.logical_and(df_formant_statistic['u_num']>N,df_formant_statistic['a_num']>N)
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['i_num']>N)
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['ADOS'].isna()!=True)
        if constrain_sex != -1:
            filter_bool=np.logical_and(filter_bool,df_formant_statistic['sex']==constrain_sex)
        if constrain_module != -1:
            filter_bool=np.logical_and(filter_bool,df_formant_statistic['Module']==constrain_module)
        if constrain_assessment != -1:
            filter_normal=df_formant_statistic['ADOS']<2
            filter_ASD=(df_formant_statistic['ADOS']<3) & (df_formant_statistic['ADOS']>=2)
            filter_autism=df_formant_statistic['ADOS']>=3
            
            if constrain_assessment == 0:
                filter_bool=np.logical_and(filter_bool,filter_normal)
            elif constrain_assessment == 1:
                filter_bool=np.logical_and(filter_bool,filter_ASD)
            elif constrain_assessment == 2:
                filter_bool=np.logical_and(filter_bool,filter_autism)
            
            
        df_formant_qualified=df_formant_statistic[filter_bool]
        for col in columns:
            spear,spear_p=spearmanr(df_formant_qualified[col],df_formant_qualified[corr_label])
            pear,pear_p=pearsonr(df_formant_qualified[col],df_formant_qualified[corr_label])

            if args.correlation_type == 'pearsonr':
                df_pearsonr_table.loc[col]=[pear,pear_p,len(df_formant_qualified[col])]
                # pear,pear_p=pearsonr(df_denan["{}_LPP_{}".format(ps,ps)],df_formant_qualified['ADOS'])
                # df_pearsonr_table_GOP.loc[ps]=[pear,pear_p,len(df_denan)]
            elif args.correlation_type == 'spearmanr':
                df_pearsonr_table.loc[col]=[spear,spear_p,len(df_formant_qualified[col])]
        print("Setting N={0}, the correlation metric is: ".format(N))
        print("Using evaluation metric: {}".format(args.correlation_type))
        print(df_pearsonr_table)
    return df_pearsonr_table


# for N in range(10):
#     df_pearsonr_table=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=-1, constrain_module=-1)
N=5
# Aaadf_pearsonr_table_NoLimit=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=-1, constrain_module=-1)
# Aaadf_pearsonr_table_NoLimitWithADOScat=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,corr_label='ADOS_cate',constrain_sex=-1, constrain_module=-1)
# Aaadf_pearsonr_table_normal=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_assessment=0)
# Aaadf_pearsonr_table_ASD=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_assessment=1)
# Aaadf_pearsonr_table_autism=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_assessment=2)

# Aaadf_pearsonr_table_boy_M3=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=1, constrain_module=3)
# Aaadf_pearsonr_table_girl_M3=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=2, constrain_module=3)
# Aaadf_pearsonr_table_boy_M4=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=1, constrain_module=4)
# Aaadf_pearsonr_table_girl_M4=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=2, constrain_module=4)
# Aaadf_pearsonr_table_M3=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=-1, constrain_module=3)
# Aaadf_pearsonr_table_M4=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=-1, constrain_module=4)
# Aaadf_pearsonr_table_boy=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=1, constrain_module=-1)
# Aaadf_pearsonr_table_girl=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=2, constrain_module=-1)
# Aaadf_pearsonr_table_N7=Calculate_correlation(df_formant_statistic,7,columns)
filter_boy=df_formant_statistic['sex']==1
filter_girl=df_formant_statistic['sex']==2
filter_M3=df_formant_statistic['Module']==3
filter_M4=df_formant_statistic['Module']==4

filter_boy_M3 = filter_boy & filter_M3
filter_boy_M4 = filter_boy & filter_M4
filter_girl_M3 = filter_girl & filter_M3
filter_girl_M4 = filter_girl & filter_M4

# =============================================================================
'''

    3. t-test area

''' 
# =============================================================================


from itertools import combinations
Filter_list=['filter_boy','filter_girl','filter_M3','filter_M4','filter_boy_M3',\
             'filter_boy_M4','filter_boy_M4','filter_girl_M3','filter_girl_M4']
feature_list=['ADOS','F_vals_f1','F_vals_f2','F_val_mix','VSA1','FCR','MSB_f1','MSB_f2']
comb=combinations(Filter_list,2)
Filter_pair_list=[[x1,x2] for x1, x2 in comb]


df_TTest_results=pd.DataFrame([],columns=feature_list)
for filt1, filt2 in Filter_pair_list:
    for feature in feature_list:
        sign=np.sign((df_formant_statistic[vars()[filt1]][feature].mean() - df_formant_statistic[vars()[filt2]][feature].mean()))
        df_TTest_results.loc[','.join([filt1,filt2]),feature] = np.round(sign* float(stats.ttest_ind(df_formant_statistic[vars()[filt1]][feature], df_formant_statistic[vars()[filt2]][feature])[1]),3)
        # print(','.join([filt1,filt2]),feature,stats.ttest_ind(df_formant_statistic[vars()[filt1]][feature], df_formant_statistic[vars()[filt2]][feature])[1])
df_TTest_results=df_TTest_results.astype(float)


# =============================================================================
'''

    4. Manual Ttest area

'''
def criterion_filter(df_formant_statistic,N=10,constrain_sex=-1, constrain_module=-1):
    filter_bool=np.logical_and(df_formant_statistic['u_num']>N,df_formant_statistic['a_num']>N)
    filter_bool=np.logical_and(filter_bool,df_formant_statistic['i_num']>N)
    if constrain_sex != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['sex']==constrain_sex)
    if constrain_module != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['Module']==constrain_module)
    return df_formant_statistic[filter_bool]
# =============================================================================
# df_formant_statistic_doc=pickle.load(open(outpklpath+"Formant_AUI_tVSAFCRFvals_ASDdoc.pkl","rb"))
# df_formant_statistic_kid=pickle.load(open(outpklpath+"Formant_AUI_tVSAFCRFvals_ASDkid.pkl","rb"))
# df_formant_statistic77_path=outpklpath+'Formant_AUI_tVSAFCRFvals_ASDkid.pkl'
# df_formant_statistic_77=pickle.load(open(df_formant_statistic77_path,'rb'))
# df_formant_statistic_ASDTD_path=outpklpath+'Formant_AUI_tVSAFCRFvals_ASDTD.pkl'
# df_formant_statistic_TD=pickle.load(open(df_formant_statistic_ASDTD_path,'rb'))



# sex=-1
# df_formant_statistic_77=criterion_filter(df_formant_statistic_77,constrain_sex=sex)
# df_formant_statistic_TD=criterion_filter(df_formant_statistic_TD,constrain_sex=sex)


# comb=[['df_formant_statistic_TD','df_formant_statistic_77'],]
# Parameters=['F_vals_f1','F_vals_f2','F_val_mix','MSB_f1','MSB_f2','MSB_mix','MSW_f1','MSW_f2','MSW_mix']

# df_ttest_result=pd.DataFrame([],columns=['doc-kid','p-val'])
# for role_1,role_2  in comb:
#     for parameter in Parameters:
#         test=stats.ttest_ind(vars()[role_1][parameter], vars()[role_2][parameter])
#         print(parameter, '{0} vs {1}'.format(role_1,role_2),test)
#         print(role_1+':',vars()[role_1][parameter].mean(),role_2+':',vars()[role_2][parameter].mean())
#         df_ttest_result.loc[parameter,'doc-kid'] = vars()[role_1][parameter].mean() - vars()[role_2][parameter].mean()
#         df_ttest_result.loc[parameter,'p-val'] = test[1]
        
# =============================================================================
'''

    Regression area

'''
# =============================================================================

df_formant_statistic77_path=outpklpath+'Formant_AUI_tVSAFCRFvals_ASDkid.pkl'
df_formant_statistic_77=pickle.load(open(df_formant_statistic77_path,'rb'))
df_formant_statistic_ASDTD_path=outpklpath+'Formant_AUI_tVSAFCRFvals_ASDTD.pkl'
df_formant_statistic_TD=pickle.load(open(df_formant_statistic_ASDTD_path,'rb'))

sex=-1
df_formant_statistic_77=criterion_filter(df_formant_statistic_77,constrain_sex=sex)
df_formant_statistic_TD=criterion_filter(df_formant_statistic_TD,constrain_sex=sex)

df_formant_statistic_77['group']=np.array(['ASD']*len(df_formant_statistic_77))
# df_formant_statistic_77['group']=np.array(['ASD']*22 + ['TD']*24)
df_formant_statistic_77['group']=df_formant_statistic_77['group'].astype(str)
df_formant_statistic_TD['group']=np.array(['TD']*len(df_formant_statistic_TD)).astype(str)
df_formant_statistic_77['group']=df_formant_statistic_77['group'].astype(str)



df_total=df_formant_statistic_77.append(df_formant_statistic_TD)

df_total['target']=pd.Categorical(df_total['group'])
df_total['target']=df_total['target'].cat.codes

feature_columns=['F_vals_f1', 'F_vals_f2',
       'F_val_mix', 'MSB_f1', 'MSB_f2', 'MSB_mix', 'MSW_f1', 'MSW_f2',
       'MSW_mix']

from patsy import dmatrices

y, X = dmatrices('C(target) ~ F_val_mix + MSB_mix + MSW_mix', df_total, return_type = 'dataframe')

X_train_with_constant=sm.add_constant(df_total[feature_columns])
sm_model_all_predictors = sm.Logit(np.array(df_total['target']), X_train_with_constant.astype(float)).fit()

sm_model_all_predictors.summary()



























'''

Plotting area

'''
# phoneme_color_map={'a':'tab:blue','u':'tab:orange','i':'tab:green',\
#                    'A:':'tab:blue','A:1':'tab:orange','A:2':'tab:green','A:3':'tab:red','A:4':'tab:purple','A:5':'tab:gray'}
# # =============================================================================

# Plotout_path="Plots/"

# if not os.path.exists(Plotout_path):
#     os.makedirs(Plotout_path)



# def plot(Vowels_AUI,outname=Plotout_path+'{0}_ADOS{1}'):
#     for people in Vowels_AUI.keys():
#         if people not in df_formant_statistic.index:
#             continue
#         formant_info=df_formant_statistic.loc[people]
#         ASDlab=Label.label_raw[label_choose][Label.label_raw['name']==people].values
#         fig, ax = plt.subplots()
#         for phone, values in Vowels_AUI[people].items():
#             x,y=np.vstack(values)[:,0],np.vstack(values)[:,1]
    
#             area=np.repeat(1,len(x))
#             cms=np.repeat(phoneme_color_map[phone],len(x))
#             plt.scatter(x, y, s=area, c=cms,  label=phone)
#             plt.title('{0}_ADOS{1}'.format(pid_dict[people],ASDlab[0]))
            
#         additional_info1="FCR={FCR}, VSA1={VSA1}, VSA2={VSA2}, LnVSA={LnVSA}".format(\
#             FCR=formant_info['FCR'],VSA1=formant_info['VSA1'],VSA2=formant_info['VSA2'],LnVSA=formant_info['LnVSA'])
#         additional_info2="F_vals_f1={F_vals_f1}, F_vals_f2={F_vals_f2}, F_val_mix={F_val_mix}".format(\
#             F_vals_f1=formant_info['F_vals_f1'],F_vals_f2=formant_info['F_vals_f2'],F_val_mix=formant_info['F_val_mix'])

#         plt.ylim(0, 5000)
#         plt.xlim(0, 5000)
#         ax.legend()
#         plt.figtext(0,0,additional_info1)
#         plt.figtext(0,-0.1,additional_info2)
#         plt.savefig(outname.format(pid_dict[people],ASDlab[0]),dpi=300, bbox_inches = "tight")
#         plt.show()


# plot(Vowels_AUI,Plotout_path+'{0}_ADOS{1}')
# # plot(Vowels_5As,Plotout_path+'{0}_ADOS{1}')

# if INSPECT:
#     phone="ax5"
    
#     label_ADOSC=Label.label_raw[['name','ADOS_C']].set_index("name")
#     Num_spokenphone=Vowels_AUI_sampNum[people][phone]
    
#     df_formant_statistic[phone]=pd.concat([df_formant_statistic[phone],Num_spokenphone],axis=1)

#     # =============================================================================
#     '''
    
#     Plot x = desired value, y= ADOS score. We want to sample critical samples for further inspection
    
#     '''
#     # num_spoken_phone=pd.DataFrame.from_dict(Vowels_AUI_sampNum,orient='index',columns=['num_spoken_phone'])
#     N=10
#     # =============================================================================
#     fig, ax = plt.subplots()
#     feature_str='F_val_mix'
#     df_formant_statistic_col=df_formant_statistic.astype(float)[feature_str]
#     label_ADOSC=Label.label_raw[['name','ADOS_C']].set_index("name")
#     df_formant_statistic_qualified=pd.concat([df_formant_statistic_col,label_ADOSC],axis=1)
#     # df_formant_statistic_col=pd.concat([df_formant_statistic_col,num_spoken_phone],axis=1)
#     # df_formant_statistic_qualified=df_formant_statistic_col[df_formant_statistic_col['num_spoken_phone']>N]
    
#     area=np.repeat(10,len(df_formant_statistic_qualified))
#     cms=range(len(df_formant_statistic_qualified))
    
    
#     x, y=df_formant_statistic_qualified.iloc[:,0], df_formant_statistic_qualified.iloc[:,1]
#     plt.scatter(x,y, c=cms, s=area)
#     for xi, yi, pidi in zip(x,y,df_formant_statistic_qualified.index):
#         ax.annotate(str(pid_dict[pidi]), xy=(xi,yi))
#     plt.title(feature_str)
#     plt.xlabel("feature")
#     plt.ylabel("ADOS")
#     ax.legend()
#     plt.savefig(Plotout_path+'Acorrelation{0}.png'.format(feature_str),dpi=300, bbox_inches = "tight")
#     plt.show()
# df_pid=pd.DataFrame.from_dict(pid_dict,orient='index')
# df_pid.columns=['name_id']
# df_pid[df_pid['name_id']==69].index    




