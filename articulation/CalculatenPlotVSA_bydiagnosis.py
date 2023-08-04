#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:14:41 2020

@author: jackchen

This script is to generate Plots for IS2021: Using Measures of Vowel Space for Automated ASD Severity Assessment
But was not used ><

The output of this script is to generate 1. box plots. 2.VSA plot 3. ttest results among three groups {normal, ASD, autism}

Parameter explain: Vowels_AUI: raw sample points of each person
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

for groupname, Keys_ADOSgroups in Keys_group.items():
    if groupname not in Group_VSA_dict.keys():
        for p in critical_phones:
            Group_VSA_dict[groupname][p]=[]
    for key in Keys_ADOSgroups:
        F12_val_dict=Vowels_AUI[key]
        for p in critical_phones:
            Group_VSA_dict[groupname][p].extend(F12_val_dict[p])

# =============================================================================
'''

    Third get the pooling data from the bookeeping dictionary

    *** Not helping our colclusion

'''
# =============================================================================
df_formant_statistic=pd.DataFrame([],columns=['FCR','VSA1','LnVSA',\
                                              'F_vals_f1', 'F_vals_f2', 'F_val_mix'])
for group, values in Group_VSA_dict.items():
    F12_val_dict=values

    u, u_num=F12_val_dict['u'],len(F12_val_dict['u'])
    a, a_num=F12_val_dict['a'],len(F12_val_dict['a'])
    i, i_num=F12_val_dict['i'],len(F12_val_dict['i'])
    
    u_mean=np.mean(u,axis=0)
    a_mean=np.mean(a,axis=0)
    i_mean=np.mean(i,axis=0)
    
    
    numerator=u_mean[1] + a_mean[1] + i_mean[0] + u_mean[0]
    demominator=i_mean[1] + a_mean[0]
    FCR=np.float(numerator/demominator)
    # assert FCR <=2
    
    VSA1=np.abs((i_mean[0]*(a_mean[1]-u_mean[1]) + a_mean[0]*(u_mean[1]-i_mean[1]) + u_mean[0]*(i_mean[1]-a_mean[1]) )/2)
    
    LnVSA=np.abs((i_mean[0]*(a_mean[1]-u_mean[1]) + a_mean[0]*(u_mean[1]-i_mean[1]) + u_mean[0]*(i_mean[1]-a_mean[1]) )/2)
    
    EDiu=np.sqrt((u_mean[1]-i_mean[1])**2+(u_mean[0]-i_mean[0])**2)
    EDia=np.sqrt((a_mean[1]-i_mean[1])**2+(a_mean[0]-i_mean[0])**2)
    EDau=np.sqrt((u_mean[1]-a_mean[1])**2+(u_mean[0]-a_mean[0])**2)
    S=(EDiu+EDia+EDau)/2
    
    LnVSA=np.sqrt(np.log(S)*(np.log(S)-np.log(EDiu))*(np.log(S)-np.log(EDia))*(np.log(S)-np.log(EDau)))
    

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
    
    df_formant_statistic.loc[group]=[np.round(FCR,3), np.round(VSA1,3),\
                                      np.round(LnVSA,3), \
                                      np.round(F_vals_f1,3), np.round(F_vals_f2,3), np.round(F_val_mix,3),\
                                      ]

# =============================================================================
'''

    Forth: average between person and take mean of the values within groups 


'''

# =============================================================================
df_formant_statistic=pd.DataFrame([],columns=['FCR','LnVSA',\
                                              'f-F1', 'f-F2', 'f-(F1+F2)',\
                                              'sex', 'age_year', 'Module',\
                                              'F12_u', 'F12_a', 'F12_i','ADOS','VSA'])
for people in Vowels_AUI_mean.keys():
    F12_val_dict=Vowels_AUI_mean[people]
    u_num, a_num, i_num=Vowels_AUI_sampNum[people]['u'],Vowels_AUI_sampNum[people]['a'],Vowels_AUI_sampNum[people]['i']
    
    u=F12_val_dict['u']
    a=F12_val_dict['a']
    i=F12_val_dict['i']
    
    # F1_u, F2_u, F1_a, F2_a, F1_i, F2_i= u[0], u[1], a[0], a[1], i[0], i[1]
    F12_u, F12_a, F12_i= u, a, i
    
    ASDlab=Label.label_raw[label_choose][Label.label_raw['name']==people].values
    sex=Label.label_raw['sex'][Label.label_raw['name']==people].values[0]
    age=Label.label_raw['age_year'][Label.label_raw['name']==people].values[0]
    module=Label.label_raw['Module'][Label.label_raw['name']==people].values[0]
    if len(u)==0 or len(a)==0 or len(i)==0:
        df_formant_statistic.loc[people]=[10, 0,\
                                      0, 0,\
                                      0,0,0,0,\
                                      0,0,0,\
                                      ASDlab[0],0]
        
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

    # =============================================================================
    # criterion
    # F1u < F1a
    # F2u < F2a
    # F2u < F2i
    # F1i < F1a
    # F2a < F2i
    # =============================================================================

    
    
    df_formant_statistic.loc[people]=[np.round(FCR,3),\
                                      np.round(LnVSA,3),\
                                      np.round(F_vals_f1,3), np.round(F_vals_f2,3), np.round(F_val_mix,3),\
                                      sex,age,module,\
                                      F12_u,F12_a,F12_i,\
                                      ASDlab[0], np.round(VSA1,3)]



from scipy import stats



'''
 Grouping area
'''
Df_group=pd.DataFrame([],columns=df_formant_statistic.columns)

filter_autism=df_formant_statistic['ADOS']>=3
filter_ASD=(df_formant_statistic['ADOS']<3) & (df_formant_statistic['ADOS']>=2)
filter_AutismnASD=filter_autism | filter_ASD
filter_normal=df_formant_statistic['ADOS']<2
filter_boy=df_formant_statistic['sex']==1
filter_girl=df_formant_statistic['sex']==2
filter_M3=df_formant_statistic['Module']==3
filter_M4=df_formant_statistic['Module']==4
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


Df_group.loc['Autism']=df_formant_statistic[filter_autism].mean()
Df_group.loc['ASD']=df_formant_statistic[filter_ASD].mean()
Df_group.loc['Not Autistic']=df_formant_statistic[filter_normal].mean()


Df_group.loc['filter_boy_autism']=df_formant_statistic[filter_boy_autism].mean()
Df_group.loc['filter_boy_ASD']=df_formant_statistic[filter_boy_ASD].mean()
Df_group.loc['filter_boy_normal']=df_formant_statistic[filter_boy_normal].mean()

Df_group.loc['filter_girl_autism']=df_formant_statistic[filter_girl_autism].mean()
Df_group.loc['filter_girl_ASD']=df_formant_statistic[filter_girl_ASD].mean()
Df_group.loc['filter_girl_normal']=df_formant_statistic[filter_girl_normal].mean()

Df_group.loc['filter_M3_autism']=df_formant_statistic[filter_M3_autism].mean()
Df_group.loc['filter_M3_ASD']=df_formant_statistic[filter_M3_ASD].mean()
Df_group.loc['filter_M3_normal']=df_formant_statistic[filter_M3_normal].mean()

Df_group.loc['filter_M4_autism']=df_formant_statistic[filter_M4_autism].mean()
Df_group.loc['filter_M4_ASD']=df_formant_statistic[filter_M4_ASD].mean()
Df_group.loc['filter_M4_normal']=df_formant_statistic[filter_M4_normal].mean()

Df_group.loc['filter_boy_M3_autism']=df_formant_statistic[filter_boy_M3_autism].mean()
Df_group.loc['filter_boy_M3_ASD']=df_formant_statistic[filter_boy_M3_ASD].mean()
Df_group.loc['filter_boy_M3_normal']=df_formant_statistic[filter_boy_M3_normal].mean()

Df_group.loc['filter_boy_M4_autism']=df_formant_statistic[filter_boy_M4_autism].mean()
Df_group.loc['filter_boy_M4_ASD']=df_formant_statistic[filter_boy_M4_ASD].mean()
Df_group.loc['filter_boy_M4_normal']=df_formant_statistic[filter_boy_M4_normal].mean()

Df_group.loc['filter_girl_M3_autism']=df_formant_statistic[filter_girl_M3_autism].mean()
Df_group.loc['filter_girl_M3_ASD']=df_formant_statistic[filter_girl_M3_ASD].mean()
Df_group.loc['filter_girl_M3_normal']=df_formant_statistic[filter_girl_M3_normal].mean()

Df_group.loc['filter_girl_M4_autism']=df_formant_statistic[filter_girl_M4_autism].mean()
Df_group.loc['filter_girl_M4_ASD']=df_formant_statistic[filter_girl_M4_ASD].mean()
Df_group.loc['filter_girl_M4_normal']=df_formant_statistic[filter_girl_M4_normal].mean()
Df_group.loc['ASD']=df_formant_statistic[filter_AutismnASD].mean()




DF_groupdata_dict=Dict()

DF_groupdata_dict['Df_Nolimit']=Df_group[:3]
DF_groupdata_dict['Df_boy']=Df_group[3:6]
DF_groupdata_dict['Df_girl']=Df_group[6:9]
DF_groupdata_dict['Df_AutnNoaut']=pd.concat([Df_group.loc['Not Autistic'],Df_group.loc['ASD']],axis=1).T

DF_groupdata_dict['Df_Nolimit'].to_excel("No_limit.xlsx")
# =============================================================================
'''
  Plot area
'''
key='Df_Nolimit'
df_data=DF_groupdata_dict[key]
dash_pattern=['solid', 'dashed', 'dotted', 'dotdash', 'dashdot']
# =============================================================================
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Magma, Inferno, Plasma, Viridis, Category20 
from bokeh.palettes import Category20 as palette
from bokeh.models import CustomJS, ColumnDataSource, Slider, LabelSet, Legend, LegendItem, Range1d
output_file(key+".html")

p = figure(title='(A)',\
           plot_width=800, plot_height=800,\
           )

p.xaxis[0].axis_label = 'F1 (Hz)'
p.yaxis[0].axis_label = 'F2 (Hz)'



x=[]
y=[]
for i in range(len(df_data)):
    print(df_data.index[i], )
    F12_u=df_data.iloc[i]['F12_u']
    F12_a=df_data.iloc[i]['F12_a']
    F12_i=df_data.iloc[i]['F12_i']
    print(F12_u)
    print(F12_a)
    print(F12_i)
    
    x.append([F12_u[0],F12_a[0],F12_i[0],F12_u[0]])
    y.append([F12_u[1],F12_a[1],F12_i[1],F12_u[1]])

data = {'xs':x,\
        'ys':y,\
        'names':['/u/','/a/','/i/'],\
        'color':Category20[3],\
        'labels':df_data.index,\
        'line_dash':dash_pattern[:len(df_data)],\
        'line_alpha':[1.0,0.7,0.7],\
        'line_width':[10,20,20],\
       }
    
labeldata = {'xs':[item for sublist in x for item in sublist],\
        'ys':[item for sublist in y for item in sublist],\
        'names':['/u/','/a/','/i/','/u/']*3,\
       }
source = ColumnDataSource(data)
labelsource = ColumnDataSource(labeldata)



r=p.multi_line('xs', 'ys', color='color',\
               line_width='line_width',line_alpha='line_alpha',source=source )

# 
legend = Legend(items=[\
    LegendItem(label=l, renderers=[r], index=i) for i,l in enumerate(df_data.index)],\
                glyph_height=40,glyph_width=40,\
                location=[0,400])

labels = LabelSet(x='xs', y='ys', text='names', level='glyph',\
              x_offset=5, y_offset=5, source=labelsource, render_mode='canvas',\
              text_font_size="20pt")

p.add_layout(labels)
p.add_layout(legend)
p.legend.location = "top_left"
p.legend.label_text_font_size = '20pt'
p.title.align = "center"
p.title.text_font_size = "40px"
p.xaxis.axis_label_text_font_size = "20pt"
p.yaxis.axis_label_text_font_size = "20pt"
p.xaxis.major_label_text_font_size = "15pt"
p.yaxis.major_label_text_font_size = "15pt"
show(p)


'''
 Box plot
'''
import copy
df_formant_statistic_cutoffgroup=copy.deepcopy(df_formant_statistic[['FCR','LnVSA','f-F1','f-F2','f-(F1+F2)']])
df_formant_statistic_cutoffgroup['group']=np.nan
df_formant_statistic_cutoffgroup['group'][df_formant_statistic['ADOS']>=3]='autism'
df_formant_statistic_cutoffgroup['group'][(df_formant_statistic['ADOS']<3) & (df_formant_statistic['ADOS']>=2)]='ASD'
df_formant_statistic_cutoffgroup['group'][(df_formant_statistic['ADOS']<2)]='Not autistic'




import numpy as np
import matplotlib.pyplot as plt

# get_columns=['FCR','LnVSA']
get_columns=['f-F1','f-F2','f-(F1+F2)']
groups=df_formant_statistic_cutoffgroup.groupby('group')[get_columns]

# width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars
autism = groups.get_group('autism').mean()
 
# Choose the height of the cyan bars
ASD = groups.get_group('ASD').mean()

NotAutism = groups.get_group('Not autistic').mean()
 
# Choose the height of the error bars (bars1)
yer_autism = groups.get_group('autism').std()
 
# Choose the height of the error bars (bars2)
yer_ASD = groups.get_group('ASD').std()

yer_NotAutism = groups.get_group('Not autistic').std()
 
# The x position of bars
r1 = np.arange(len(autism))
r2 = [x + barWidth for x in r1]
r3 = [x + 2*barWidth for x in r1]
 
# Create blue bars
plt.bar(r1, autism, width = barWidth, color = 'blue', edgecolor = 'black', yerr=yer_autism, capsize=7, label='Autism')
 
# Create cyan bars
plt.bar(r2, ASD, width = barWidth, color = 'orange', edgecolor = 'black', yerr=yer_ASD, capsize=7, label='ASD')

plt.bar(r3, NotAutism, width = barWidth, color = 'yellow', edgecolor = 'black', yerr=yer_NotAutism, capsize=7, label='Not Autistic')

# Create blue bars without errors, because the standard deviation is too large
# plt.bar(r1, autism, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='autism')
 
# # Create cyan bars
# plt.bar(r2, ASD, width = barWidth, color = 'orange', edgecolor = 'black', capsize=7, label='ASD')

# plt.bar(r3, NotAutism, width = barWidth, color = 'yellow', edgecolor = 'black', capsize=7, label='Not Autism')
 

 
r4 = r1.tolist() + r2 + r3
Samples = pd.concat([autism,ASD,NotAutism])

'''  normal vs ASD:*, ASD vs autism:**, autism vs normal:***    '''




''' Manual setting  (No need to automatically derive)'''
if get_columns==['FCR','LnVSA']:
    label = ['']*len(Samples)
    plt.title('(B)')
elif get_columns==['f-F1','f-F2','f-(F1+F2)']:
    label = ['*','','',\
             '','', '',\
             '*','','']
    plt.title('(C)')
for i in range(len(r4)):
    plt.text(x = r4[i]-0.05 , y = Samples[i]+0.1, s = label[i], size = 20)

# general layout
plt.xticks([r + barWidth for r in range(len(ASD))], get_columns)

# plt.ylabel('height')
plt.legend()

# Show graphic
plt.show()


# =============================================================================
'''

    T-test area

''' 
from itertools import combinations 
# =============================================================================
comb = combinations(['filter_normal', 'filter_ASD', 'filter_autism'], 2) 


for FILTER_1,FILTER_2  in comb:
    for parameter in ['FCR','VSA','f-F1','f-F2','f-(F1+F2)']:
        test=stats.ttest_ind(df_formant_statistic[vars()[FILTER_1]][parameter], df_formant_statistic[vars()[FILTER_2]][parameter])
        print(parameter, '{0} vs {1}'.format(FILTER_1,FILTER_2),test)
        
        
        
FVal_test_f1=stats.ttest_ind(df_formant_statistic[filter_normal]['f-F1'], df_formant_statistic[filter_ASD]['f-F1'])
FVal_test_f2=stats.ttest_ind(df_formant_statistic[filter_normal]['f-F2'], df_formant_statistic[filter_ASD]['f-F2'])
FVal_test_f_mix=stats.ttest_ind(df_formant_statistic[filter_normal]['f-(F1+F2)'], df_formant_statistic[filter_ASD]['f-(F1+F2)'])
FVal_test_FCR=stats.ttest_ind(df_formant_statistic[filter_normal]['FCR'], df_formant_statistic[filter_ASD]['FCR'])
FVal_test_VSA=stats.ttest_ind(df_formant_statistic[filter_normal]['VSA'], df_formant_statistic[filter_ASD]['VSA'])
