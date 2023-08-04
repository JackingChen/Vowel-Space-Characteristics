#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:14:41 2020

@author: jackchen

This the purpose of this script contains two parts, 
    1. Do ANOVA tests to find the best way of plotting obvious difference between high and low groups
    1-1 Inspect the F-values, pick the groups that have F-values over 100. This might have better chance for separating two groups
    1-2 manually copy and paste the founded groups from 1-1 to line 156 (you should also consider the utter numbers
    2. Plot the dataframe using section F9 start from around line 158

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

    args = parser.parse_args()
    return args


''' parse namespace '''
args = get_args()
base_path=args.base_path
filepath=args.filepath
trnpath=args.trnpath
pklpath=args.pklpath


''' Vowels sets '''
Vowels_single=['i_','E','axr','A_','u_','ax','O_']
Vowels_prosody_single=[phonewoprosody.Phoneme_sets[v]  for v in Vowels_single]
Vowels_prosody_single=[item for sublist in Vowels_prosody_single for item in sublist]



Formants_utt_symb=pickle.load(open(pklpath+"/Formants_utt_symb.pkl","rb"))
Formants_people_symb=pickle.load(open(pklpath+"/Formants_people_symb.pkl","rb"))

label_set=['ADOS_C','ADOS_S','ADOS_SC']



# =============================================================================
''' grouping severity '''
ADOS_label=Label.label_raw['ADOS_C']
array=list(set(ADOS_label))
phoneme="A:"
groups=[np.split(array, idx)  for n_splits in range(3)  for idx in combinations(range(1, len(array)), n_splits) if len(np.split(array, idx))>1]
# =============================================================================




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
Vowels_AUI=Dict()
for people in sorted(Formants_people_symb.keys()):
    ASDlab=Label.label_raw[label_choose][Label.label_raw['name']==people].values
    for phone, values in Formants_people_symb[people].items():
        if phone not in [e  for phoneme in ['w','j'] for e in phonewoprosody.Phoneme_sets[phoneme]] + ['A:','A:1']:
            continue
        Vowels_AUI[phone[0]][people]=values # Take only the first character of phone, [A, w, j]
        if phone[0] in Vowels_AUI.keys(): 
            if people in Vowels_AUI[phone[0]].keys():
                Vowels_AUI[phone[0]][people].extend(values)
            else:
                Vowels_AUI[phone[0]][people]=values
        else:
            Vowels_AUI[phone[0]][people]=values

''''############################################################################'''

three_grps=[g for g in groups if len(g)>2]
highlow_grps=[[sublist[0],sublist[-1]]  for sublist in three_grps ]
groups=highlow_grps
''' testing area '''
for group in groups:
    print(" processing under assumption of group", group)
    df_data_all=pd.DataFrame([],columns=['F1','F2','ADOS'])
    for people in Formants_people_symb.keys():
        for phone, values in Formants_people_symb[people].items():
            if phone not in phonewoprosody.Phoneme_sets[phoneme]:
                continue
            F1F2_vals=np.vstack(Formants_people_symb[people][phone])
            ASDlab=Label.label_raw[label_choose][Label.label_raw['name']==people].values
            assert ASDlab in list(set(ADOS_label))
            ADOS_severity=find_group(ASDlab,group)
            if ADOS_severity == -1:
                continue
            ADOS_catagory_array=np.repeat(ADOS_severity,len(F1F2_vals)).reshape(-1,1).astype(int)
            
            df_data=pd.DataFrame(np.hstack((F1F2_vals,ADOS_catagory_array)),columns=['F1','F2','ADOS'])
            df_data_all=pd.concat([df_data_all,df_data])
    
    '''  MANOVA test '''
    maov = MANOVA.from_formula('F1 + F2   ~ ADOS', data=df_data_all)
    
    # print(maov.mv_test())
    
    '''  ANOVA test '''
    moore_lm = ols('ADOS ~ F1 + F2 ',data=df_data_all).fit()
    print("utt number of group 0 = {0}, utt number of group 1 = {1}".format(len(df_data_all[df_data_all['ADOS']==0]),len(df_data_all[df_data_all['ADOS']==1])))
    print(sm.stats.anova_lm(moore_lm, typ=2))
''''############################################################################'''

#Catagorize ADOS cases in three groups and do t-test
groups=[[np.array([0]), np.array([2, 3, 4, 5, 6, 7, 8])],\
        [np.array([0]), np.array([3, 4, 5, 6, 7, 8])]]
for group in groups:
    print(" processing under assumption of group", group)
    df_data_all=pd.DataFrame([],columns=['F1','F2','ADOS'])
    for people in Formants_people_symb.keys():
        for phone, values in Formants_people_symb[people].items():
            if phone not in phonewoprosody.Phoneme_sets[phoneme]:
                continue
            F1F2_vals=np.vstack(Formants_people_symb[people][phone])
            ASDlab=Label.label_raw[label_choose][Label.label_raw['name']==people].values
            ADOS_severity=find_group(ASDlab,group)
            ADOS_catagory_array=np.repeat(ADOS_severity,len(F1F2_vals)).reshape(-1,1).astype(int)
            
            df_data=pd.DataFrame(np.hstack((F1F2_vals,ADOS_catagory_array)),columns=['F1','F2','ADOS'])
            df_data_all=pd.concat([df_data_all,df_data])

    ''' t-test '''
    sets={}
    for i in range(len(set(df_data_all['ADOS']))):
        sets[i]=df_data_all[df_data_all['ADOS']==i]
    set0=df_data_all[df_data_all['ADOS']==0]
    set1=df_data_all[df_data_all['ADOS']==1]
    set2=df_data_all[df_data_all['ADOS']==2]
    print(stats.ttest_ind(set0, set1, equal_var=False))
    print(stats.ttest_ind(set1, set2, equal_var=False))
    print(stats.ttest_ind(set0, set2, equal_var=False))
    
    #Data Visualization
    
    hisglow_sets=pd.concat([set0,set1])
    
    x = hisglow_sets['F1'].values
    y = hisglow_sets['F2'].values
    
    colors = hisglow_sets['ADOS'].values
    area=np.repeat(1,len(x))
    
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.show()
    
    
    set0F1_array=set0['F1'].values
    set1F1_array=set1['F1'].values
    set2F1_array=set2['F1'].values
    
    set0F2_array=set0['F2'].values
    set1F2_array=set1['F2'].values
    set2F2_array=set2['F2'].values
    
   
    plt.hist(set0F1_array ,alpha=0.3, label='set0')
    plt.hist(set1F1_array ,alpha=0.5, label='set1')
    plt.hist(set2F1_array ,alpha=0.7, label='set2')
    
    plt.legend(loc='upper right')
    plt.show()
    
    plt.hist(set0F2_array ,alpha=0.3, label='set0')
    plt.hist(set1F2_array ,alpha=0.5, label='set1')
    plt.hist(set2F2_array ,alpha=0.7, label='set2')
    plt.legend(loc='upper right')
    plt.show()
    
    
    print("Mean value of  Set 0 = ")
    print(set0.mean())
    print("Mean value of  Set 1 = ")
    print(set1.mean())
    print("Mean value of Set 2 = ")
    print(set2.mean())
''''############################################################################'''
'''  Catagorize ADOS cases into three groups  '''
df_data_all=pd.DataFrame([],columns=['F1','F2','ADOS'])
group_ADOSSCORE=[np.array([i]) for i in range(ADOS_label.max()+1)]
for people in Formants_people_symb.keys():
    for phone, values in Formants_people_symb[people].items():
        if phone not in phonewoprosody.Phoneme_sets["A_"]:
            continue
        F1F2_vals=np.vstack(Formants_people_symb[people][phone])
        ASDlab=Label.label_raw[label_choose][Label.label_raw['name']==people].values
        ADOS_severity=find_group(ASDlab,group)
        ADOS_catagory_array=np.repeat(ADOS_severity,len(F1F2_vals)).reshape(-1,1).astype(int)
        
        df_data=pd.DataFrame(np.hstack((F1F2_vals,ADOS_catagory_array)),columns=['F1','F2','ADOS'])
        df_data_all=pd.concat([df_data_all,df_data])
        
sets={}
for i in range(len(set(df_data_all['ADOS']))):
    sets[i]=df_data_all[df_data_all['ADOS']==i].mean()
df_groupmean=pd.DataFrame.from_dict(sets).T[['F1','F2']]

# =============================================================================
'''
TBD ....
Statistical tests



'''
# =============================================================================



# =============================================================================

'''

Plot area


deprecated in the future
'''

# =============================================================================
# person_vowel_symb=Dict()
# X_lst=[]
# y_lst=[]
# for phone, values in Formants_people_symb[people].items():
#     if phone not in Vowels_prosody_single:
#         continue
#     person_vowel_symb[phone]=np.vstack(values)
    
#     X_lst.append(person_vowel_symb[phone].astype(int))
#     y_lst.append(np.repeat(phone, person_vowel_symb[phone].shape[0]))

# X=np.vstack(X_lst)
# y=[item for sublist in y_lst for item in sublist]


# symb2num={}
# for i,s in enumerate(set(y)):
#     symb2num[s]=i
# =============================================================================
'''

Transform to array X and y

'''
# =============================================================================

'''

單母音
一,i:  
ㄝ,E
ㄦ,axr
ㄚ,A:
ㄨ,u:
ㄜ,ax
ㄛ,O:

雙母音
ㄟ,ei
ㄞ,ai
ㄠ,aU
ㄡ,oU

帶鼻音母音
ㄣ,ax+n
ㄥ,ax+N
ㄢ,A:+n
ㄤ,A:+N

'''

i

