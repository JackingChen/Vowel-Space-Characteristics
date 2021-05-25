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
from HYPERPARAM import phonewoprosody, Label
import matplotlib.pyplot as plt
from itertools import combinations

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from scipy import stats
from scipy.stats import spearmanr,pearsonr 
import statistics 
import os
import statsmodels.api as sm


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
Stat_med_str='mean'
# =============================================================================

def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice/articulation',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--filepath', default='/mnt/sdd/jackchen/egs/formosa/s6/Segmented_ADOS_emotion',
                        help='path of the base directory')
    parser.add_argument('--trnpath', default='/mnt/sdd/jackchen/egs/formosa/s6/Audacity',
                        help='path of the base directory')
    parser.add_argument('--pklpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--Inspect', default=False,
                            help='path of the base directory')
    parser.add_argument('--correlation_type', default='pearsonr',
                            help='path of the base directory')
    parser.add_argument('--label_choose_lst', default=['ADOS_C'],
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
filepath=args.filepath
trnpath=args.trnpath
pklpath=args.pklpath
INSPECT=args.Inspect
label_choose_lst=args.label_choose_lst # labels are too biased

''' Vowels sets '''
Vowels_single=['i_','E','axr','A_','u_','ax','O_']
Vowels_prosody_single=[phonewoprosody.Phoneme_sets[v]  for v in Vowels_single]
Vowels_prosody_single=[item for sublist in Vowels_prosody_single for item in sublist]



Formants_utt_symb=pickle.load(open(pklpath+"/Formants_utt_symb_bymiddle.pkl","rb"))
Formants_people_symb=pickle.load(open(pklpath+"/Formants_people_symb_bymiddle.pkl","rb"))

label_set=['ADOS_C','ADOS_S','ADOS_SC']



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
PhoneMapp_dict={'u':['w']+phonewoprosody.Phoneme_sets['u_'],\
                'i':['j']+phonewoprosody.Phoneme_sets['i_'],\
                'a':phonewoprosody.Phoneme_sets['A_']}
Vowels_AUI=Dict()
Vowels_AUI_sampNum=Dict()
for people in Label.label_raw.sort_values(by='ADOS_C')['name']:
    for phone, values in Formants_people_symb[people].items():
        if phone not in [e  for phoneme in ['w','j','A_','u_','i_'] for e in phonewoprosody.Phoneme_sets[phoneme]]:
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
for people in Label.label_raw.sort_values(by='ADOS_C')['name']:
    for phone, values in Formants_people_symb[people].items():
        if phone in phonewoprosody.Phoneme_sets['A_']:
            Vowels_5As[people][phone]=values # Take only the first character of phone, [A_]
# =============================================================================
Statistic_method={'mean':np.mean,'median':np.median,'mode':stats.mode}

df_formant_statistic=pd.DataFrame([],columns=['FCR','VSA1','VSA2','LnVSA','ADOS','u_num','a_num','i_num',\
                                              'F_vals_f1', 'F_vals_f2', 'F_val_mix', 'criterion_score',\
                                              'sex', 'age_year', 'Module',\
                                              'MSB_f1','MSB_f2','MSW_f1','MSW_f2','SSBN_f1','SSBN_f2',\
                                              'dau1','dai1','diu1','daudai1','daudiu1','daidiu1','daidiudau1',\
                                              'dau2','dai2','diu2','daudai2','daudiu2','daidiu2','daidiudau2',\
                                              'F2i_u','F1a_u'])
for people in Vowels_AUI_mean.keys():
    F12_raw_dict=Vowels_AUI[people]
    F12_val_dict={k:[] for k in ['u','a','i']}
    for k,v in F12_raw_dict.items():
        if Stat_med_str == 'mode':
            F12_val_dict[k]=Statistic_method[Stat_med_str](v,axis=0)[0].ravel()
        else:
            F12_val_dict[k]=Statistic_method[Stat_med_str](v,axis=0)
    
    u_num, a_num, i_num=Vowels_AUI_sampNum[people]['u'],Vowels_AUI_sampNum[people]['a'],Vowels_AUI_sampNum[people]['i']
    
    ASDlab=Label.label_raw[label_choose][Label.label_raw['name']==people].values    
    sex=Label.label_raw['sex'][Label.label_raw['name']==people].values[0]
    age=Label.label_raw['age_year'][Label.label_raw['name']==people].values[0]
    module=Label.label_raw['Module'][Label.label_raw['name']==people].values[0]
    
    u=F12_val_dict['u']
    a=F12_val_dict['a']
    i=F12_val_dict['i']

    
    if len(u)==0 or len(a)==0 or len(i)==0:
        df_formant_statistic.loc[people]=[10, 0,\
                                      0, 0, ASDlab[0],\
                                      len(u), len(a), len(i),\
                                      0, 0, 0,\
                                      0,0,0,0,\
                                      0,0,0,0,0,0,\
                                      0,0,0,0,0,0,0,\
                                      0,0,0,0,0,0,0,\
                                      0,0]
        
        continue
    
    numerator=u[1] + a[1] + i[0] + u[0]
    demominator=i[1] + a[0]
    FCR=np.float(numerator/demominator)
    F2i_u= u[1]/i[1]
    F1a_u= u[0]/a[0]
    # assert FCR <=2
    
    VSA1=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
    
    LnVSA=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
    
    EDiu=np.sqrt((u[1]-i[1])**2+(u[0]-i[0])**2)
    EDia=np.sqrt((a[1]-i[1])**2+(a[0]-i[0])**2)
    EDau=np.sqrt((u[1]-a[1])**2+(u[0]-a[0])**2)
    S=(EDiu+EDia+EDau)/2
    VSA2=np.sqrt(S*(S-EDiu)*(S-EDia)*(S-EDau))
    
    LnVSA=np.sqrt(np.log(S)*(np.log(S)-np.log(EDiu))*(np.log(S)-np.log(EDia))*(np.log(S)-np.log(EDau)))
    
    ''' a u i distance '''
    dau1 = np.abs(a[0] - u[0])
    dai1 = np.abs(a[0] - i[0])
    diu1 = np.abs(i[0] - u[0])
    daudai1 = dau1 + dai1
    daudiu1 = dau1 + diu1
    daidiu1 = dai1 + diu1
    daidiudau1 = dai1 + diu1+ dau1
    
    dau2 = np.abs(a[1] - u[1])
    dai2 = np.abs(a[1] - i[1])
    diu2 = np.abs(i[1] - u[1])
    daudai2 = dau2 + dai2
    daudiu2 = dau2 + diu2
    daidiu2 = dai2 + diu2
    daidiudau2 = dai2 + diu2+ dau2
    
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
    F_vals_f1=F_vals[0]
    F_vals_f2=F_vals[1]
    F_val_mix=F_vals_f1 + F_vals_f2
    
    msb=f_classif(df_vowel[['F1','F2']].values,df_vowel['target'].values)[2]
    msw=f_classif(df_vowel[['F1','F2']].values,df_vowel['target'].values)[3]
    ssbn=f_classif(df_vowel[['F1','F2']].values,df_vowel['target'].values)[4]
    
    MSB_f1=msb[0]
    MSB_f2=msb[1]
    MSW_f1=msw[0]
    MSW_f2=msw[1]
    SSBN_f1=ssbn[0]
    SSBN_f2=ssbn[1]
    # =============================================================================
    # criterion
    # F1u < F1a
    # F2u < F2a
    # F2u < F2i
    # F1i < F1a
    # F2a < F2i
    # =============================================================================
    u_mean=F12_val_dict['u']
    a_mean=F12_val_dict['a']
    i_mean=F12_val_dict['i']
    
    F1u, F2u=u_mean[0], u_mean[1]
    F1a, F2a=a_mean[0], a_mean[1]
    F1i, F2i=i_mean[0], i_mean[1]
    
    filt1 = [1 if F1u < F1a else 0]
    filt2 = [1 if F2u < F2a else 0]
    filt3 = [1 if F2u < F2i else 0]
    filt4 = [1 if F1i < F1a else 0]
    filt5 = [1 if F2a < F2i else 0]
    criterion_score=np.sum([filt1,filt2,filt3,filt4,filt5])
    
    
    df_formant_statistic.loc[people]=[np.round(FCR,3), np.round(VSA1,3),\
                                      np.round(VSA2,3), np.round(LnVSA,3), ASDlab[0],\
                                      u_num, a_num, i_num,\
                                      np.round(F_vals_f1,3), np.round(F_vals_f2,3), np.round(F_val_mix,3),\
                                      np.round(criterion_score,3),sex,age,module,\
                                      MSB_f1, MSB_f2, MSW_f1, MSW_f2,SSBN_f1,SSBN_f2,\
                                      dau1,dai1,diu1,daudai1,daudiu1,daidiu1,daidiudau1,\
                                      dau2,dai2,diu2,daudai2,daudiu2,daidiu2,daidiudau2,\
                                      F2i_u, F1a_u]

outpklpath="Pickles/Session_formants_people_vowel_feat/"
if not os.path.exists(outpklpath):
    os.makedirs(outpklpath)
pickle.dump(df_formant_statistic,open(outpklpath+"Formant_AUI_tVSAFCRFvals.pkl","wb"))


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
# columns=['FCR','VSA1','VSA2','LnVSA','F_vals_f1', 'F_vals_f2', 'F_val_mix', 'criterion_score',\
#          'MSB_f1','MSB_f2','MSW_f1','MSW_f2','SSBN_f1','SSBN_f2']

columns=['FCR','VSA1','F_vals_f1', 'F_vals_f2', 'F_val_mix','MSB_f1','MSB_f2',\
         'dau1','dai1','diu1','daudai1','daudiu1','daidiu1','daidiudau1',\
         'dau2','dai2','diu2','daudai2','daudiu2','daidiu2','daidiudau2',\
         'F2i_u','F1a_u']
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
#     df_pearsonr_table=Calculate_correlation(df_formant_statistic,N,columns)
N=5
Aaadf_pearsonr_table_NoLimit=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=-1, constrain_module=-1)
Aaadf_pearsonr_table_NoLimitWithADOScat=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,corr_label='ADOS_cate',constrain_sex=-1, constrain_module=-1)
Aaadf_pearsonr_table_normal=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_assessment=0)
Aaadf_pearsonr_table_ASD=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_assessment=1)
Aaadf_pearsonr_table_autism=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_assessment=2)

Aaadf_pearsonr_table_boy_M3=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=1, constrain_module=3)
Aaadf_pearsonr_table_girl_M3=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=2, constrain_module=3)
Aaadf_pearsonr_table_boy_M4=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=1, constrain_module=4)
Aaadf_pearsonr_table_girl_M4=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=2, constrain_module=4)
Aaadf_pearsonr_table_M3=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=-1, constrain_module=3)
Aaadf_pearsonr_table_M4=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=-1, constrain_module=4)
Aaadf_pearsonr_table_boy=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=1, constrain_module=-1)
Aaadf_pearsonr_table_girl=Calculate_correlation(label_choose_lst,df_formant_statistic,N,columns,constrain_sex=2, constrain_module=-1)
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
aaa=ccc


'''

Plotting area

'''
phoneme_color_map={'a':'tab:blue','u':'tab:orange','i':'tab:green',\
                   'A:':'tab:blue','A:1':'tab:orange','A:2':'tab:green','A:3':'tab:red','A:4':'tab:purple','A:5':'tab:gray'}
# =============================================================================

Plotout_path="Plots/"

if not os.path.exists(Plotout_path):
    os.makedirs(Plotout_path)



def plot(Vowels_AUI,outname=Plotout_path+'{0}_ADOS{1}'):
    for people in Vowels_AUI.keys():
        if people not in df_formant_statistic.index:
            continue
        formant_info=df_formant_statistic.loc[people]
        ASDlab=Label.label_raw[label_choose][Label.label_raw['name']==people].values
        fig, ax = plt.subplots()
        for phone, values in Vowels_AUI[people].items():
            x,y=np.vstack(values)[:,0],np.vstack(values)[:,1]
    
            area=np.repeat(1,len(x))
            cms=np.repeat(phoneme_color_map[phone],len(x))
            plt.scatter(x, y, s=area, c=cms,  label=phone)
            plt.title('{0}_ADOS{1}'.format(pid_dict[people],ASDlab[0]))
            
        additional_info1="FCR={FCR}, VSA1={VSA1}, VSA2={VSA2}, LnVSA={LnVSA}".format(\
            FCR=formant_info['FCR'],VSA1=formant_info['VSA1'],VSA2=formant_info['VSA2'],LnVSA=formant_info['LnVSA'])
        additional_info2="F_vals_f1={F_vals_f1}, F_vals_f2={F_vals_f2}, F_val_mix={F_val_mix}".format(\
            F_vals_f1=formant_info['F_vals_f1'],F_vals_f2=formant_info['F_vals_f2'],F_val_mix=formant_info['F_val_mix'])

        plt.ylim(0, 5000)
        plt.xlim(0, 5000)
        ax.legend()
        plt.figtext(0,0,additional_info1)
        plt.figtext(0,-0.1,additional_info2)
        plt.savefig(outname.format(pid_dict[people],ASDlab[0]),dpi=300, bbox_inches = "tight")
        plt.show()


plot(Vowels_AUI,Plotout_path+'{0}_ADOS{1}')
# plot(Vowels_5As,Plotout_path+'{0}_ADOS{1}')

if INSPECT:
    phone="ax5"
    
    label_ADOSC=Label.label_raw[['name','ADOS_C']].set_index("name")
    Num_spokenphone=Vowels_AUI_sampNum[people][phone]
    
    df_formant_statistic[phone]=pd.concat([df_formant_statistic[phone],Num_spokenphone],axis=1)

    # =============================================================================
    '''
    
    Plot x = desired value, y= ADOS score. We want to sample critical samples for further inspection
    
    '''
    # num_spoken_phone=pd.DataFrame.from_dict(Vowels_AUI_sampNum,orient='index',columns=['num_spoken_phone'])
    N=10
    # =============================================================================
    fig, ax = plt.subplots()
    feature_str='F_val_mix'
    df_formant_statistic_col=df_formant_statistic.astype(float)[feature_str]
    label_ADOSC=Label.label_raw[['name','ADOS_C']].set_index("name")
    df_formant_statistic_qualified=pd.concat([df_formant_statistic_col,label_ADOSC],axis=1)
    # df_formant_statistic_col=pd.concat([df_formant_statistic_col,num_spoken_phone],axis=1)
    # df_formant_statistic_qualified=df_formant_statistic_col[df_formant_statistic_col['num_spoken_phone']>N]
    
    area=np.repeat(10,len(df_formant_statistic_qualified))
    cms=range(len(df_formant_statistic_qualified))
    
    
    x, y=df_formant_statistic_qualified.iloc[:,0], df_formant_statistic_qualified.iloc[:,1]
    plt.scatter(x,y, c=cms, s=area)
    for xi, yi, pidi in zip(x,y,df_formant_statistic_qualified.index):
        ax.annotate(str(pid_dict[pidi]), xy=(xi,yi))
    plt.title(feature_str)
    plt.xlabel("feature")
    plt.ylabel("ADOS")
    ax.legend()
    plt.savefig(Plotout_path+'Acorrelation{0}.png'.format(feature_str),dpi=300, bbox_inches = "tight")
    plt.show()
df_pid=pd.DataFrame.from_dict(pid_dict,orient='index')
df_pid.columns=['name_id']
df_pid[df_pid['name_id']==69].index    





# =============================================================================
'''

    4. Manual Ttest area

'''
# =============================================================================

print("F_vals_f1: M3 vs M4",stats.ttest_ind(df_formant_statistic[filter_M3]['F_vals_f1'], df_formant_statistic[filter_M4]['F_vals_f1']))
print("F_vals_f1: M3 vs M4",df_formant_statistic[filter_M3]['F_vals_f1'].mean(), df_formant_statistic[filter_M4]['F_vals_f1'].mean())
print("F_vals_f1: boy vs girl",stats.ttest_ind(df_formant_statistic[filter_boy]['F_vals_f1'], df_formant_statistic[filter_girl]['F_vals_f1']))
print("F_vals_f1: boy vs girl",df_formant_statistic[filter_boy]['F_vals_f1'].mean(), df_formant_statistic[filter_girl]['F_vals_f1'].mean())
print("F_vals_f1: boy M3 vs boy M4",stats.ttest_ind(df_formant_statistic[filter_boy_M3]['F_vals_f1'], df_formant_statistic[filter_boy_M4]['F_vals_f1']))
print("F_vals_f1: boy M3 vs boy M4",df_formant_statistic[filter_boy_M3]['F_vals_f1'].mean(), df_formant_statistic[filter_boy_M4]['F_vals_f1'].mean())
print("F_vals_f1: girl M3 vs girl M4",stats.ttest_ind(df_formant_statistic[filter_girl_M3]['F_vals_f1'], df_formant_statistic[filter_girl_M4]['F_vals_f1']))
print("F_vals_f1: girl M3 vs girl M4",df_formant_statistic[filter_girl_M3]['F_vals_f1'].mean(), df_formant_statistic[filter_girl_M4]['F_vals_f1'].mean())
print("F_vals_f1: girl M3 vs boy M3",stats.ttest_ind(df_formant_statistic[filter_girl_M3]['F_vals_f1'], df_formant_statistic[filter_boy_M3]['F_vals_f1']))
print("F_vals_f1: girl M3 vs boy M3",df_formant_statistic[filter_girl_M3]['F_vals_f1'].mean(), df_formant_statistic[filter_boy_M3]['F_vals_f1'].mean())
print("F_vals_f1: girl M4 vs boy M4",stats.ttest_ind(df_formant_statistic[filter_girl_M4]['F_vals_f1'], df_formant_statistic[filter_boy_M4]['F_vals_f1']))
print("F_vals_f1: girl M4 vs boy M4",df_formant_statistic[filter_girl_M4]['F_vals_f1'].mean(), df_formant_statistic[filter_boy_M4]['F_vals_f1'].mean())


print("F_vals_f2: M3 vs M4",stats.ttest_ind(df_formant_statistic[filter_M3]['F_vals_f2'], df_formant_statistic[filter_M4]['F_vals_f2']))
print("F_vals_f2: M3 vs M4",df_formant_statistic[filter_M3]['F_vals_f2'].mean(), df_formant_statistic[filter_M4]['F_vals_f2'].mean())
print("F_vals_f2: boy vs girl",stats.ttest_ind(df_formant_statistic[filter_boy]['F_vals_f2'], df_formant_statistic[filter_girl]['F_vals_f2']))
print("F_vals_f2: boy vs girl",df_formant_statistic[filter_boy]['F_vals_f2'].mean(), df_formant_statistic[filter_girl]['F_vals_f2'].mean())
print("F_vals_f2: boy M3 vs boy M4",stats.ttest_ind(df_formant_statistic[filter_boy_M3]['F_vals_f2'], df_formant_statistic[filter_boy_M4]['F_vals_f2']))
print("F_vals_f2: boy M3 vs boy M4",df_formant_statistic[filter_boy_M3]['F_vals_f2'].mean(), df_formant_statistic[filter_boy_M4]['F_vals_f2'].mean())
print("F_vals_f2: girl M3 vs girl M4",stats.ttest_ind(df_formant_statistic[filter_girl_M3]['F_vals_f2'], df_formant_statistic[filter_girl_M4]['F_vals_f2']))
print("F_vals_f2: girl M3 vs girl M4",df_formant_statistic[filter_girl_M3]['F_vals_f2'].mean(), df_formant_statistic[filter_girl_M4]['F_vals_f2'].mean())

print("F_val_mix: M3 vs M4",stats.ttest_ind(df_formant_statistic[filter_M3]['F_val_mix'], df_formant_statistic[filter_M4]['F_val_mix']))
print("F_val_mix: M3 vs M4",df_formant_statistic[filter_M3]['F_val_mix'].mean(), df_formant_statistic[filter_M4]['F_val_mix'].mean())
print("F_val_mix: boy vs girl",stats.ttest_ind(df_formant_statistic[filter_boy]['F_val_mix'], df_formant_statistic[filter_girl]['F_val_mix']))
print("F_val_mix: boy vs girl",df_formant_statistic[filter_boy]['F_val_mix'].mean(), df_formant_statistic[filter_girl]['F_val_mix'].mean())
print("F_val_mix: boy M3 vs boy M4",stats.ttest_ind(df_formant_statistic[filter_boy_M3]['F_val_mix'], df_formant_statistic[filter_boy_M4]['F_val_mix']))
print("F_val_mix: boy M3 vs boy M4",df_formant_statistic[filter_boy_M3]['F_val_mix'].mean(), df_formant_statistic[filter_boy_M4]['F_val_mix'].mean())
print("F_val_mix: girl M3 vs girl M4",stats.ttest_ind(df_formant_statistic[filter_girl_M3]['F_val_mix'], df_formant_statistic[filter_girl_M4]['F_val_mix']))
print("F_val_mix: girl M3 vs girl M4",df_formant_statistic[filter_girl_M3]['F_val_mix'].mean(), df_formant_statistic[filter_girl_M4]['F_val_mix'].mean())

print("VSA1: M3 vs M4",stats.ttest_ind(df_formant_statistic[filter_M3]['VSA1'], df_formant_statistic[filter_M4]['VSA1']))
print("VSA1: M3 vs M4",df_formant_statistic[filter_M3]['VSA1'].mean(), df_formant_statistic[filter_M4]['VSA1'].mean())
print("VSA1: boy vs girl",stats.ttest_ind(df_formant_statistic[filter_boy]['VSA1'], df_formant_statistic[filter_girl]['VSA1']))
print("VSA1: boy vs girl",df_formant_statistic[filter_boy]['VSA1'].mean(), df_formant_statistic[filter_girl]['VSA1'].mean())
print("VSA1: boy M3 vs boy M4",stats.ttest_ind(df_formant_statistic[filter_boy_M3]['VSA1'], df_formant_statistic[filter_boy_M4]['VSA1']))
print("VSA1: boy M3 vs boy M4",df_formant_statistic[filter_boy_M3]['VSA1'].mean(), df_formant_statistic[filter_boy_M4]['VSA1'].mean())
print("VSA1: girl M3 vs girl M4",stats.ttest_ind(df_formant_statistic[filter_girl_M3]['VSA1'], df_formant_statistic[filter_girl_M4]['VSA1']))
print("VSA1: girl M3 vs girl M4",df_formant_statistic[filter_girl_M3]['VSA1'].mean(), df_formant_statistic[filter_girl_M4]['VSA1'].mean())

print("FCR: M3 vs M4",stats.ttest_ind(df_formant_statistic[filter_M3]['FCR'], df_formant_statistic[filter_M4]['FCR']))
print("FCR: M3 vs M4",df_formant_statistic[filter_M3]['FCR'].mean(), df_formant_statistic[filter_M4]['FCR'].mean())
print("FCR: boy vs girl",stats.ttest_ind(df_formant_statistic[filter_boy]['FCR'], df_formant_statistic[filter_girl]['FCR']))
print("FCR: boy vs girl",df_formant_statistic[filter_boy]['FCR'].mean(), df_formant_statistic[filter_girl]['FCR'].mean())
print("FCR: boy M3 vs boy M4",stats.ttest_ind(df_formant_statistic[filter_boy_M3]['FCR'], df_formant_statistic[filter_boy_M4]['FCR']))
print("FCR: boy M3 vs boy M4",df_formant_statistic[filter_boy_M3]['FCR'].mean(), df_formant_statistic[filter_boy_M4]['FCR'].mean())
print("FCR: girl M3 vs girl M4",stats.ttest_ind(df_formant_statistic[filter_girl_M3]['FCR'], df_formant_statistic[filter_girl_M4]['FCR']))
print("FCR: girl M3 vs girl M4",df_formant_statistic[filter_girl_M3]['FCR'].mean(), df_formant_statistic[filter_girl_M4]['FCR'].mean())

print("MSB_f1: M3 vs M4",stats.ttest_ind(df_formant_statistic[filter_M3]['MSB_f1'], df_formant_statistic[filter_M4]['MSB_f1']))
print("MSB_f1: M3 vs M4",df_formant_statistic[filter_M3]['MSB_f1'].mean(), df_formant_statistic[filter_M4]['MSB_f1'].mean())
print("MSB_f1: boy vs girl",stats.ttest_ind(df_formant_statistic[filter_boy]['MSB_f1'], df_formant_statistic[filter_girl]['MSB_f1']))
print("MSB_f1: boy vs girl",df_formant_statistic[filter_boy]['MSB_f1'].mean(), df_formant_statistic[filter_girl]['MSB_f1'].mean())
print("MSB_f1: boy M3 vs boy M4",stats.ttest_ind(df_formant_statistic[filter_boy_M3]['MSB_f1'], df_formant_statistic[filter_boy_M4]['MSB_f1']))
print("MSB_f1: boy M3 vs boy M4",df_formant_statistic[filter_boy_M3]['MSB_f1'].mean(), df_formant_statistic[filter_boy_M4]['MSB_f1'].mean())
print("MSB_f1: girl M3 vs girl M4",stats.ttest_ind(df_formant_statistic[filter_girl_M3]['MSB_f1'], df_formant_statistic[filter_girl_M4]['MSB_f1']))
print("MSB_f1: girl M3 vs girl M4",df_formant_statistic[filter_girl_M3]['MSB_f1'].mean(), df_formant_statistic[filter_girl_M4]['MSB_f1'].mean())

print("MSB_f2: M3 vs M4",stats.ttest_ind(df_formant_statistic[filter_M3]['MSB_f2'], df_formant_statistic[filter_M4]['MSB_f2']))
print("MSB_f2: M3 vs M4",df_formant_statistic[filter_M3]['MSB_f2'].mean(), df_formant_statistic[filter_M4]['MSB_f2'].mean())
print("MSB_f2: boy vs girl",stats.ttest_ind(df_formant_statistic[filter_boy]['MSB_f2'], df_formant_statistic[filter_girl]['MSB_f2']))
print("MSB_f2: boy vs girl",df_formant_statistic[filter_boy]['MSB_f2'].mean(), df_formant_statistic[filter_girl]['MSB_f2'].mean())
print("MSB_f2: boy M3 vs boy M4",stats.ttest_ind(df_formant_statistic[filter_boy_M3]['MSB_f2'], df_formant_statistic[filter_boy_M4]['MSB_f2']))
print("MSB_f2: boy M3 vs boy M4",df_formant_statistic[filter_boy_M3]['MSB_f2'].mean(), df_formant_statistic[filter_boy_M4]['MSB_f2'].mean())
print("MSB_f2: girl M3 vs girl M4",stats.ttest_ind(df_formant_statistic[filter_girl_M3]['MSB_f2'], df_formant_statistic[filter_girl_M4]['MSB_f2']))
print("MSB_f2: girl M3 vs girl M4",df_formant_statistic[filter_girl_M3]['MSB_f2'].mean(), df_formant_statistic[filter_girl_M4]['MSB_f2'].mean())
