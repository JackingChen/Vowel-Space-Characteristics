#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:14:41 2020

@author: jackchen

This script is to generate data for IS2021: Using Measures of Vowel Space for Automated ASD Severity Assessment
But haven't used either'

1. Data prepare area: Gather raw data of the three critical monophthongs (F1 & F2) and save in: df_formant_statistic.
    First stage:  is to Get the basic information for latter group spliting
    Second stage:  it to use a bookkeeping dictionary to gather information within groups
    * Note we hack into scipy f-classif function to decompose ssbn and sswn. We found that ssbn is the main factor of correlation 

2. Plot area: 
    plot joint plot (scatter plot and distribution) with Vowel space triangle and some feature values
    each person has one figure

Seaborn plot code: 
    plot the joint plot (scatter plot and distribution) with seaborn code
    However, seaborn code is fixed and do not allow to add Vowel space triangle plots
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
import os
import statsmodels.api as sm
from sklearn.feature_selection import f_classif


import numpy as np
import matplotlib.pyplot as plt

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
        Vowels_AUI_mean[key1][key2]=np.mean(np.vstack(values2),axis=0)
# =============================================================================
'''

    First stage is to Get the basic information for latter group spliting

'''
# =============================================================================

''' Get the basic information (ADOS_score, module, sex) '''
df_formant_basicinfos=pd.DataFrame([],columns=['ADOS','sex', 'age_year', 'Module'])
for people in Vowels_AUI.keys():
    F12_val_dict=Vowels_AUI[people]
    ASDlab=Label.label_raw[label_choose][Label.label_raw['name']==people].values
    sex=Label.label_raw['sex'][Label.label_raw['name']==people].values[0]
    age=Label.label_raw['age_year'][Label.label_raw['name']==people].values[0]
    module=Label.label_raw['Module'][Label.label_raw['name']==people].values[0]
    
    u=F12_val_dict['u']
    a=F12_val_dict['a']
    i=F12_val_dict['i']
    
    if len(u)==0 or len(a)==0 or len(i)==0:
        continue
    df_formant_basicinfos.loc[people]=[ASDlab[0],sex,age,module]
    
df_formant_basicinfos=df_formant_basicinfos.astype(int)

# =============================================================================
'''

    Second stage it to use a bookkeeping dictionary to gather information within groups

    We save this because we will need to calculate or plot the sample points
'''
Group_VSA_dict=Dict()
Keys_group=Dict()
critical_phones=['a','u','i']
# =============================================================================
filter_autism=df_formant_basicinfos['ADOS']>=3
filter_ASD=(df_formant_basicinfos['ADOS']<3) & (df_formant_basicinfos['ADOS']>=2)
filter_normal=df_formant_basicinfos['ADOS']<2
filter_boy=df_formant_basicinfos['sex']==1
filter_girl=df_formant_basicinfos['sex']==2
filter_M3=df_formant_basicinfos['Module']==3
filter_M4=df_formant_basicinfos['Module']==4
filter_boy_autism=filter_boy & filter_autism
filter_boy_ASD=filter_boy & filter_ASD
filter_boy_normal=filter_boy & filter_normal
filter_girl_autism=filter_girl & filter_autism
filter_girl_ASD=filter_girl & filter_ASD
filter_girl_normal=filter_girl & filter_normal
filter_M3_autism=filter_M3 & filter_autism
filter_M3_ASD=filter_M3 & filter_ASD
filter_M3_normal=filter_M3 & filter_normal
filter_M4_autism=filter_M4 & filter_autism
filter_M4_ASD=filter_M4 & filter_ASD
filter_M4_normal=filter_M4 & filter_normal
filter_boy_M3_autism=filter_boy & filter_autism & filter_M3
filter_boy_M3_ASD=filter_boy & filter_ASD & filter_M3
filter_boy_M3_normal=filter_boy & filter_normal & filter_M3
filter_girl_M3_autism=filter_girl & filter_autism & filter_M3
filter_girl_M3_ASD=filter_girl & filter_ASD & filter_M3
filter_girl_M3_normal=filter_girl & filter_normal & filter_M3
filter_boy_M4_autism=filter_boy & filter_autism & filter_M4
filter_boy_M4_ASD=filter_boy & filter_ASD & filter_M4
filter_boy_M4_normal=filter_boy & filter_normal & filter_M4
filter_girl_M4_autism=filter_girl & filter_autism & filter_M4
filter_girl_M4_ASD=filter_girl & filter_ASD & filter_M4
filter_girl_M4_normal=filter_girl & filter_normal & filter_M4

Keys_group['filter_autism']=df_formant_basicinfos[filter_autism].index
Keys_group['filter_ASD']=df_formant_basicinfos[filter_ASD].index
Keys_group['filter_normal']=df_formant_basicinfos[filter_normal].index

Keys_group['filter_boy_autism']=df_formant_basicinfos[filter_boy_autism].index
Keys_group['filter_boy_ASD']=df_formant_basicinfos[filter_boy_ASD].index
Keys_group['filter_boy_normal']=df_formant_basicinfos[filter_boy_normal].index

Keys_group['filter_girl_autism']=df_formant_basicinfos[filter_girl_autism].index
Keys_group['filter_girl_ASD']=df_formant_basicinfos[filter_girl_ASD].index
Keys_group['filter_girl_normal']=df_formant_basicinfos[filter_girl_normal].index

Keys_group['filter_M3_autism']=df_formant_basicinfos[filter_M3_autism].index
Keys_group['filter_M3_ASD']=df_formant_basicinfos[filter_M3_ASD].index
Keys_group['filter_M3_normal']=df_formant_basicinfos[filter_M3_normal].index

Keys_group['filter_M4_autism']=df_formant_basicinfos[filter_M4_autism].index
Keys_group['filter_M4_ASD']=df_formant_basicinfos[filter_M4_ASD].index
Keys_group['filter_M4_normal']=df_formant_basicinfos[filter_M4_normal].index

Keys_group['filter_boy_M3_autism']=df_formant_basicinfos[filter_boy_M3_autism].index
Keys_group['filter_boy_M3_ASD']=df_formant_basicinfos[filter_boy_M3_ASD].index
Keys_group['filter_boy_M3_normal']=df_formant_basicinfos[filter_boy_M3_normal].index

Keys_group['filter_boy_M4_autism']=df_formant_basicinfos[filter_boy_M4_autism].index
Keys_group['filter_boy_M4_ASD']=df_formant_basicinfos[filter_boy_M4_ASD].index
Keys_group['filter_boy_M4_normal']=df_formant_basicinfos[filter_boy_M4_normal].index

Keys_group['filter_girl_M3_autism']=df_formant_basicinfos[filter_girl_M3_autism].index
Keys_group['filter_girl_M3_ASD']=df_formant_basicinfos[filter_girl_M3_ASD].index
Keys_group['filter_girl_M3_normal']=df_formant_basicinfos[filter_girl_M3_normal].index

Keys_group['filter_girl_M4_autism']=df_formant_basicinfos[filter_girl_M4_autism].index
Keys_group['filter_girl_M4_ASD']=df_formant_basicinfos[filter_girl_M4_ASD].index
Keys_group['filter_girl_M4_normal']=df_formant_basicinfos[filter_girl_M4_normal].index




''' Gather all points within group '''
for groupname, Keys_ADOSgroups in Keys_group.items():
    if groupname not in Group_VSA_dict.keys():
        for p in critical_phones:
            Group_VSA_dict[groupname][p]=[]
        
            
    for key in Keys_ADOSgroups:
        F12_val_dict=Vowels_AUI[key]
        for p in critical_phones:
            Group_VSA_dict[groupname][p].extend(F12_val_dict[p])
# =============================================================================
# =============================================================================
# =============================================================================    


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
import statistics 

df_formant_statistic=pd.DataFrame([],columns=['FCR','VSA1','VSA2','LnVSA','ADOS','u_num','a_num','i_num',\
                                              'F_vals_f1', 'F_vals_f2', 'F_val_mix', 'criterion_score',\
                                              'sex', 'age_year', 'Module',\
                                              'MSB_f1','MSB_f2','MSW_f1','MSW_f2','SSBN_f1','SSBN_f2'])
for people in Vowels_AUI_mean.keys():
    F12_val_dict=Vowels_AUI_mean[people]
    u_num, a_num, i_num=Vowels_AUI_sampNum[people]['u'],Vowels_AUI_sampNum[people]['a'],Vowels_AUI_sampNum[people]['i']
    
    u=F12_val_dict['u']
    a=F12_val_dict['a']
    i=F12_val_dict['i']
    ASDlab=Label.label_raw[label_choose][Label.label_raw['name']==people].values
    sex=Label.label_raw['sex'][Label.label_raw['name']==people].values[0]
    age=Label.label_raw['age_year'][Label.label_raw['name']==people].values[0]
    module=Label.label_raw['Module'][Label.label_raw['name']==people].values[0]
    if len(u)==0 or len(a)==0 or len(i)==0:
        df_formant_statistic.loc[people]=[10, 0,\
                                      0, 0, ASDlab[0],\
                                      len(u), len(a), len(i),\
                                      0, 0, 0,\
                                      0,0,0,0,\
                                      0,0,0,0,0,0]
        
        continue
    
    numerator=u[1] + a[1] + i[0] + u[0]
    demominator=i[1] + a[0]
    FCR=np.float(numerator/demominator)
    # assert FCR <=2
    
    VSA1=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
    
    LnVSA=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
    
    EDiu=np.sqrt((u[1]-i[1])**2+(u[0]-i[0])**2)
    EDia=np.sqrt((a[1]-i[1])**2+(a[0]-i[0])**2)
    EDau=np.sqrt((u[1]-a[1])**2+(u[0]-a[0])**2)
    S=(EDiu+EDia+EDau)/2
    VSA2=np.sqrt(S*(S-EDiu)*(S-EDia)*(S-EDau))
    
    LnVSA=np.sqrt(np.log(S)*(np.log(S)-np.log(EDiu))*(np.log(S)-np.log(EDia))*(np.log(S)-np.log(EDau)))
    
    
    
    
    
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
                                      MSB_f1, MSB_f2, MSW_f1, MSW_f2,SSBN_f1,SSBN_f2]
            

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================            
# =============================================================================
'''

    2. Plot area: 
        plot joint plot (scatter plot and distribution) with Vowel space triangle and some feature values
        each person has one figure

'''
import seaborn as sns
import pandas as pd

pd.set_option("display.max_rows", 1000)    #設定最大能顯示1000rows
pd.set_option("display.max_columns", 1000) #設定最大能顯示1000columns

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  

MARKERS=['o', 'x', '+', 'v', '^', '<', '>', 's', 'd']
Phone2Markers={}
for i, k in enumerate(Group_VSA_dict['filter_autism'].keys()):
    Phone2Markers[k]=MARKERS[i]
# =============================================================================


count=0
Group_singlepeople_dict=Dict()
df_Alldata_gathered=pd.DataFrame([])
for group_str in ['filter_autism','filter_ASD','filter_normal']:
    for keys in Keys_group[group_str]:
        row=count%7
        column=int(count/7)
        people_aui_dict=Vowels_AUI[keys]
        df_singlepeople_data=pd.DataFrame([])
        for p, v in people_aui_dict.items():
            Grp_singlepeople_array=np.array(v)
            df_grp_singlepeople_array = pd.DataFrame(Grp_singlepeople_array, columns=['F1','F2'])
            df_grp_singlepeople_array['phone']=np.array([p]*len(df_grp_singlepeople_array))
            df_grp_singlepeople_array['group']=np.array([group_str]*len(df_grp_singlepeople_array))
            
            
            df_grp_singlepeople_array['row']=np.array([row]*len(df_grp_singlepeople_array))
            df_grp_singlepeople_array['column']=np.array([column]*len(df_grp_singlepeople_array))
            df_singlepeople_data=df_singlepeople_data.append(df_grp_singlepeople_array)
            df_Alldata_gathered=df_Alldata_gathered.append(df_grp_singlepeople_array)
            
            
        Group_singlepeople_dict[keys]=df_singlepeople_data
        count+=1

key2idx_dict={}
for i, key in enumerate(sorted(Vowels_AUI.keys())):
    key2idx_dict[key]=i
# aaa=ccc
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.stats import gaussian_kde

# =============================================================================
matplot_colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
# =============================================================================


def jointPlot(data,fig,ax, color='darkred', scatter_pts_size=2,kde=False,**kwargs):
    ymin, ymax=data[:,1].min(),data[:,1].max()
    xmin, xmax=data[:,0].min(),data[:,0].max()
    #Define grid for subplots
    

    #Create scatter plot
    # fig = plt.figure(facecolor='white')
    # ax = plt.subplot(gs[1, 0],frameon = False)
    cax = ax.scatter(data[:,0], data[:,1], color=color, alpha=.6,s=scatter_pts_size)
    ax.grid(True)
    
    #Create Y-marginal (right)
    axr = plt.subplot(gs[1, 1], sharey=ax, frameon = False,xticks = [] ) #xlim=(0, 1), ylim = (ymin, ymax) xticks=[], yticks=[]
    
    axr.hist(data[:,1], color = color, orientation = 'horizontal', density=True, histtype='stepfilled', alpha=0.0)

    #Create X-marginal (top)
    axt = plt.subplot(gs[0,0], sharex=ax,frameon = False,yticks = [])# xticks = [], , ) #xlim = (xmin, xmax), ylim=(0, 1)
    axt.hist(data[:,0], color = color, density=True, histtype='stepfilled', alpha=0.0)
    
    
    try:
        ax.set_title(kwargs['title'])
        ax.set_xlabel(kwargs['xlabel'])
        ax.set_ylabel(kwargs['ylabel'])
    except:
        pass
    
    #Bring the marginals closer to the scatter plot
    fig.tight_layout(pad = 1)

    if kde:
        kdex=gaussian_kde(data[:,0])
        kdey=gaussian_kde(data[:,1])
        x= np.linspace(0,3000,100)
        y= np.linspace(0,3000,100)
        dx=kdex(x)
        dy=kdey(y)
        axr.plot(dy,y,color=color)
        axt.plot(x,dx,color=color)


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


for group_str in ['filter_autism','filter_ASD','filter_normal']:
    count=0
    for people in Keys_group[group_str]:
        fig = plt.figure(facecolor='white', figsize=(8, 6), dpi=80)
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios = [1, 4])
        ax = plt.subplot(gs[1, 0],frameon = False)
        row=count%7
        column=int(count/7)
        
        
        value1 = int(np.round(NormalizeData(df_formant_statistic['MSB_f1']).loc[people],2)*100)
        value2 = int(np.round(NormalizeData(df_formant_statistic['MSB_f2']).loc[people],2)*100)
        value3 = int(np.round(NormalizeData(df_formant_statistic['F_vals_f1']).loc[people],2)*100)
        value4 = int(np.round(NormalizeData(df_formant_statistic['F_vals_f2']).loc[people],2)*100)
        value5 = int(np.round(NormalizeData(df_formant_statistic['FCR']).loc[people],2)*100)
        value6 = int(np.round(NormalizeData(df_formant_statistic['VSA1']).loc[people],2)*100)
        
        kwargs = {"title": '{0}_{1}'.format(key2idx_dict[people],people),"xlabel": "F1","ylabel":"F2"}
        
        
        df_group_singlepeople_data=Group_singlepeople_dict[people]
        Group_singlepeople_a_array=df_group_singlepeople_data[df_group_singlepeople_data['phone']=='a'][['F1','F2']].values
        Group_singlepeople_u_array=df_group_singlepeople_data[df_group_singlepeople_data['phone']=='u'][['F1','F2']].values
        Group_singlepeople_i_array=df_group_singlepeople_data[df_group_singlepeople_data['phone']=='i'][['F1','F2']].values
        jointPlot(Group_singlepeople_a_array, fig,ax,color=matplot_colors[0],kde=True,**kwargs)
        jointPlot(Group_singlepeople_u_array, fig,ax,color=matplot_colors[1],kde=True,**kwargs)
        jointPlot(Group_singlepeople_i_array, fig,ax,color=matplot_colors[2],kde=True,**kwargs)
        
        VowelSpaceArea=np.vstack([Group_singlepeople_a_array.mean(axis=0),Group_singlepeople_u_array.mean(axis=0),Group_singlepeople_i_array.mean(axis=0)])
        ax.fill(VowelSpaceArea[:,0],VowelSpaceArea[:,1])
        info_str="""f1MSB:{0}\nf2MSB:{1}\nF_vals_f1:{2}\nF_vals_f2:{3}\nFCR:{4}\nVSA:{5}\n{6}""".format(value1,value2,\
                    value3,value4,\
                    value5,value6,\
                    group_str)
        ax.text(x=1.0, y=1.1,s=info_str, ha='center', va='center', transform=ax.transAxes)
        outpath="Plot/{0}".format(group_str)
        if not os.path.exists(outpath):
            os.makedirs(outpath) 
        fig.savefig(outpath+"/"+"{0}_{1}ADOS{2}".format(key2idx_dict[people],people,int(df_formant_statistic['ADOS'].loc[people]))+".png")

df_formant_basicinfos['MSB_f1_norm']=NormalizeData(df_formant_statistic['MSB_f1'])
df_formant_basicinfos['MSB_f2_norm']=NormalizeData(df_formant_statistic['MSB_f2'])
df_formant_basicinfos['F_vals_f1_norm']=NormalizeData(df_formant_statistic['F_vals_f1'])
df_formant_basicinfos['F_vals_f2_norm']=NormalizeData(df_formant_statistic['F_vals_f2'])
df_formant_basicinfos['VSA']=NormalizeData(df_formant_statistic['VSA1'])
df_formant_basicinfos['ADOS_C']=df_formant_statistic['ADOS']
df_formant_basicinfos['idxs']=[key2idx_dict[key] for key in df_formant_basicinfos.index]









aaa=ccc    
# =============================================================================
'''

    Seaborn plot code

'''
# =============================================================================

Group_data_dict=Dict()
df_Alldata_gathered=pd.DataFrame([])
for group_str in ['filter_autism','filter_ASD','filter_normal']:
    df_group_data=pd.DataFrame([])
    for keys, values in Group_VSA_dict[group_str].items():
        Grp_samples_array=np.array(values)
        df_grp_samples_array = pd.DataFrame(Grp_samples_array, columns=['F1','F2'])
        df_grp_samples_array['phone']=np.array([keys]*len(df_grp_samples_array))
        df_grp_samples_array['group']=np.array([group_str]*len(df_grp_samples_array))
        df_Alldata_gathered=df_Alldata_gathered.append(df_grp_samples_array)
        df_group_data=df_group_data.append(df_grp_samples_array)
    Group_data_dict[group_str]=df_group_data

for group_str in ['filter_autism','filter_ASD','filter_normal']:
    count=0
    for people in Keys_group[group_str]:
        fig, ax = plt.subplots()
        row=count%7
        column=int(count/7)
        
        
        value1 = int(np.round(NormalizeData(df_formant_statistic['MSB_f1']).loc[people],2)*100)
        value2 = int(np.round(NormalizeData(df_formant_statistic['MSB_f2']).loc[people],2)*100)
        value3 = int(np.round(NormalizeData(df_formant_statistic['F_vals_f1']).loc[people],2)*100)
        value4 = int(np.round(NormalizeData(df_formant_statistic['F_vals_f2']).loc[people],2)*100)
        value5 = int(np.round(NormalizeData(df_formant_statistic['FCR']).loc[people],2)*100)
        value6 = int(np.round(NormalizeData(df_formant_statistic['VSA1']).loc[people],2)*100)
        sns.jointplot(data=Group_singlepeople_dict[people], x='F1',y='F2',hue='phone')
        
        
        
        # sns.jointplot(data=Group_singlepeople_dict['2016_04_27_02_161_1'], x='F1',y='F2',hue='phone', kind="kde", ax = g.ax_joint)
        
        
        # axs[row,column].set_title( "MSBf1{0}MSBf2{1}_{2}".format(value1,value2,group_str))
        # plt.text(15, -0.01, "MSBf1{0}MSBf2{1}_{2}".format(value1,value2,group_str))
        info_str="""f1MSB:{0}\nf2MSB:{1}\nF_vals_f1:{2}\nF_vals_f2:{3}\nFCR:{4}\nVSA:{5}\n{6}""".format(value1,value2,\
                    value3,value4,\
                    value5,value6,\
                    group_str)
        plt.title( '{0}_{1}'.format(key2idx_dict[people],people) )
        plt.text(x=0, y=0,s=info_str)
        
        outpath="Plot/{0}".format(group_str)
        if not os.path.exists(outpath):
            os.makedirs(outpath)    
        count+=1
        plt.savefig(outpath+"/"+"{0}_{1}ADOS{2}".format(key2idx_dict[people],people,int(df_formant_statistic['ADOS'].loc[people]))+".png")
    # fig.savefig(outpath+"/"+people+".png")






#結合兩種圖及雙變數資料(joinplot)
for group_str in ['filter_autism','filter_ASD','filter_normal']:
    sns.jointplot(data=Group_data_dict[group_str], x='F1',y='F2',hue='phone')
    plt.title(group_str)
