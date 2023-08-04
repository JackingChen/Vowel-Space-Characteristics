#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:14:41 2020

@author: jackchen

This the purpose of this script is to plot tVSA andFormant Centralization Ratio 

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
import os, glob

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
def lookup_Psets(s,Phoneme_sets):
    for p,v in Phoneme_sets.items():
        if s in v:
            return p
    return -1        
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
    args = parser.parse_args()
    return args


''' parse namespace '''
args = get_args()
base_path=args.base_path
filepath=args.filepath
trnpath=args.trnpath
pklpath=args.pklpath
INSPECT=args.Inspect

''' Vowels sets '''
Vowels_single=['i_','E','axr','A_','u_','ax','O_']
Vowels_prosody_single=[phonewoprosody.Phoneme_sets[v]  for v in Vowels_single]
Vowels_prosody_single=[item for sublist in Vowels_prosody_single for item in sublist]

func=["mean",'std','skew','skew_sign','skew_abs','kurtosis']
Feat=['F1','F2']
Static_cols=[]
for fun in func:
    for f in Feat:
        Static_cols.append(f+"+"+fun)

label_set=['ADOS_C','ADOS_S','ADOS_SC']
label_choose='ADOS_C'
'''

Get the number of phone spoken by each people

'''
Formants_people_symb=pickle.load(open(pklpath+"/Formants_people_symb.pkl","rb"))
Vowels_sampNum=Dict()

Vowels_phonewoprosody_sampNum=Dict()
Vowels_phonewoprosody01_sampNum=Dict()
for people in Label.label_raw.sort_values(by='ADOS_C')['name']:
    for phone, values in Formants_people_symb[people].items():
        if phone == "SIL":
            continue
        for a in range(1): # Fill up the Vowels_phonewoprosody_sampNum dictionary but making the code neat
            ph_wopros=lookup_Psets(phone,phonewoprosody.Phoneme_sets)
            assert ph_wopros != -1
            Vowels_sampNum[phone][people]=len(values)
            if ph_wopros not in Vowels_phonewoprosody_sampNum.keys():
                Vowels_phonewoprosody_sampNum[ph_wopros][people]=len(values)
            else:
                Vowels_phonewoprosody_sampNum[ph_wopros][people]+=len(values)
        for a in range(1): # Fill up the Vowels_phonewoprosody01_sampNum dictionary but making the code neat
            ph_wopros01=lookup_Psets(phone,phonewoprosody.Phoneme01_sets)
            if ph_wopros01 != -1:
                if ph_wopros01 not in Vowels_phonewoprosody01_sampNum.keys():
                    Vowels_phonewoprosody01_sampNum[ph_wopros01][people]=len(values)
                else:
                    Vowels_phonewoprosody01_sampNum[ph_wopros01][people]+=len(values)
        

def get_phoneset_Rawform(Formants_people_dict , LABEL, phoneset):
    Vowels_bag=Dict()
    for people in LABEL.sort_values(by=label_choose)['name']:
        for phone, values in Formants_people_dict[people].items():
            if phone not in phoneset:
                continue
            Vowels_bag[people][phone]=values 
    return Vowels_bag

P_set=['A:1', 'A:4', 'A:5', 'ax4', 'ax5', 'i:2', 'i:3', 'O:1', 'O:3', 'O:4', 'u:2','E2']
Vowels_phonenprosody=get_phoneset_Rawform(Formants_people_symb, Label.label_raw, P_set)

# =============================================================================





pickleout_path   =pklpath+'/Session_formants_people_vowel/*'
files=glob.glob(pickleout_path)

'''

for phone Session_formants_people_vowel

'''
formants_people_vowel=Dict()
formants_df_people_vowel=Dict()
for file in files:
    name=file.split("/")[-1].replace(".pkl","")
    formants_people_vowel[name]=pickle.load(open(file,"rb"))
    formants_df_people_vowel[name]=pd.DataFrame.from_dict(formants_people_vowel[name],orient='index')
    formants_df_people_vowel[name].columns=Static_cols

pickle.dump(formants_people_vowel,open("formants_people_vowel.pkl","wb"))

def Correlation_formants_people_vowel(formants_people_vowel,Label,lab_name,Vowels_sampNum,N=5):
    df_pearsonr_table=pd.DataFrame([],columns=['pearsonr','pear_pvalue','de-zero_num'])
    Qualified_indicator=Dict()
    for phone in formants_people_vowel.keys():
        num_spoken_phone=pd.DataFrame.from_dict(Vowels_sampNum[phone],orient='index')
        ADOS_label=Label.label_raw[lab_name]
        ADOS_module=Label.label_raw['Module']
        NameMatchAssertion(formants_people_vowel[phone],Label.label_raw['name'].values)
        df_formants_people_vowel=pd.DataFrame.from_dict(formants_people_vowel[phone]).T
        df_formants_people_vowel.columns=Static_cols
        df_formants_people_vowel['ADOS']=ADOS_label.values
        df_formants_people_vowel['module']=ADOS_module.values
        # print(phone,num_spoken_phone)
        df_formants_people_vowel['num_spoken_phone']=num_spoken_phone
        df_formants_people_vowel_qualified=df_formants_people_vowel[df_formants_people_vowel['num_spoken_phone']>N]
        for feat in df_formants_people_vowel.columns:
            if feat in ['num_spoken_phone','ADOS' ,'F1+kurtosis','F2+kurtosis',\
                        'F1+skew_sign','F2+skew_sign','F1+skew_abs','F2+skew_abs']:
                continue
            if len(df_formants_people_vowel_qualified[feat]) < 10:
                continue
            formant_feat=df_formants_people_vowel_qualified[feat]
            ADOS_score=df_formants_people_vowel_qualified['ADOS']
            ADOS_mod=df_formants_people_vowel_qualified['module']
            if lab_name == "A10":
                pear_M4,pear_M4_p=pearsonr(formant_feat[ADOS_mod==4],ADOS_score[ADOS_mod==4])
                pear_M3,pear_M3_p=np.nan,np.nan
                pear,pear_p=np.nan,np.nan
            else:
                pear,pear_p=pearsonr(formant_feat,ADOS_score)
                pear_M4,pear_M4_p=pearsonr(formant_feat[ADOS_mod==4],ADOS_score[ADOS_mod==4])
                pear_M3,pear_M3_p=pearsonr(formant_feat[ADOS_mod==3],ADOS_score[ADOS_mod==3])

            df_pearsonr_table.loc[phone + '_' + feat]=[pear,pear_p,len(df_formants_people_vowel_qualified)]

            # df_pearsonr_table.loc['M3({})'.format(phone + '_' + feat)]=[pear_M3,pear_M3_p,len(formant_feat[ADOS_mod==3])]
            # df_pearsonr_table.loc['M4({})'.format(phone + '_' + feat)]=[pear_M4,pear_M4_p,len(formant_feat[ADOS_mod==4])]
            
        Qualified_indicator[phone]=df_formants_people_vowel['num_spoken_phone']
    return df_pearsonr_table, Qualified_indicator

df_pearsonr_table_phonenprosody_dict=Dict()
df_pearsonr_table_phonewoprosody_dict=Dict()
choosed_std=['ax4_F1+std','A:5_F2+std','A:5_F1+std']
choosed_std_M3=['M3({})'.format(s)  for s in choosed_std]
choosed_std_M4=['M4({})'.format(s)  for s in choosed_std]
choosed_skew=['A:4_F2+skew','ax5_F1+skew']
choosed_skew_M3=['M3({})'.format(s)  for s in choosed_skew]
choosed_skew_M4=['M4({})'.format(s)  for s in choosed_skew]
choosed_mean=['ax4_F1+mean','ax4_F2+mean']
choosed_mean_M3=['M3({})'.format(s)  for s in choosed_mean]
choosed_mean_M4=['M4({})'.format(s)  for s in choosed_mean]
chosen_features=choosed_std+choosed_skew+choosed_mean
chosen_features=chosen_features+choosed_std_M3+choosed_skew_M3+choosed_mean_M3
chosen_features=chosen_features+choosed_std_M4+choosed_skew_M4+choosed_mean_M4
df_choosen_feature=pd.DataFrame([],index=chosen_features,columns=['AA1','AA2', 'AA3', 'AA4','AA5','AA6','AA7', 'AA8', 'AA9','A10'])
df_choosen_feature_p=pd.DataFrame([],index=chosen_features,columns=['AA1','AA2', 'AA3', 'AA4','AA5','AA6','AA7', 'AA8', 'AA9','A10'])
df_choosen_feature_num=pd.DataFrame([],index=chosen_features,columns=['AA1','AA2', 'AA3', 'AA4','AA5','AA6','AA7', 'AA8', 'AA9','A10'])
# for label_choose in ['AA1','AA2', 'AA3', 'AA4','AA5','AA6','AA7', 'AA8', 'AA9','A10','ADOS_C']:
for label_choose in ['ADOS_C']:
    
    df_pearsonr_table, Table_phonenprosody_qualified=Correlation_formants_people_vowel(formants_people_vowel,Label,label_choose,Vowels_sampNum,N=5)
    Aaa_df_pearsonr_table_phonenprosody_qualified=df_pearsonr_table[df_pearsonr_table['de-zero_num']>30]
    df_pearsonr_table_phonenprosody_dict[label_choose]=Aaa_df_pearsonr_table_phonenprosody_qualified
    
    df_choosen_feature[label_choose]=Aaa_df_pearsonr_table_phonenprosody_qualified.loc[chosen_features]['pearsonr']
    df_choosen_feature_p[label_choose]=Aaa_df_pearsonr_table_phonenprosody_qualified.loc[chosen_features]['pear_pvalue']
    df_choosen_feature_num[label_choose]=Aaa_df_pearsonr_table_phonenprosody_qualified.loc[chosen_features]['de-zero_num']
    ###################################################################################
    
    pickleout_path   =pklpath+'/Session_formants_people_vowelwoprosody/*'
    files=glob.glob(pickleout_path)
    '''
    
    for phone Session_formants_people_vowelwoprosody
    
    '''
    formants_people_vowelwoprosody=Dict()
    for file in files:
        name=file.split("/")[-1].replace(".pkl","")
        formants_people_vowelwoprosody[name]=pickle.load(open(file,"rb"))
    
    df_pearsonr_table, Table_phonewoprosody_qualified=Correlation_formants_people_vowel(formants_people_vowelwoprosody,Label,label_choose,Vowels_phonewoprosody_sampNum,N=5)
    Aaa_df_pearsonr_table_phonewoprosody_qualified=df_pearsonr_table[df_pearsonr_table['de-zero_num']>10]
    df_pearsonr_table_phonewoprosody_dict[label_choose]=Aaa_df_pearsonr_table_phonewoprosody_qualified

###################################################################################

pickleout_path   =pklpath+'/Session_formants_people_vowelwoprosody01/*'
files=glob.glob(pickleout_path)
'''

for phone Session_formants_people_vowelwoprosody with 
phones taking only first tone
e.g. A: A:1
'''
formants_people_vowelwoprosody01=Dict()
for file in files:
    name=file.split("/")[-1].replace(".pkl","")
    formants_people_vowelwoprosody01[name]=pickle.load(open(file,"rb"))

df_pearsonr_table, Table_phonenprosody01_qualified=Correlation_formants_people_vowel(formants_people_vowelwoprosody01,Label,'ADOS_C',Vowels_phonewoprosody01_sampNum,N=5)
Aaa_df_pearsonr_table_phonewoprosody01_qualified=df_pearsonr_table[df_pearsonr_table['de-zero_num']>30]

# =============================================================================
'''

Plotting area

'''

# pid=list(sorted(Formants_people_symb.keys()))
# pid_dict={}
# for i,pi in enumerate(pid):
#     pid_dict[pi]=i

# phoneme_color_map={'A':'tab:blue','w':'tab:orange','j':'tab:green'}
# # =============================================================================

# Plotout_path="Plots/"

# if not os.path.exists(Plotout_path):
#     os.makedirs(Plotout_path)
# if not os.path.exists('Hist'):
#     os.makedirs('Hist')

# phone='E2'
# for people in Vowels_phonenprosody.keys():
#     df_formant_statistic=formants_df_people_vowel[phone].astype(float)
#     formant_info=df_formant_statistic.loc[people]
#     ASDlab=Label.label_raw[label_choose][Label.label_raw['name']==people].values
    
    
#     values=Vowels_phonenprosody[people][phone]
#     if len(values) == 0:
#         continue
    
#     x,y=np.vstack(values)[:,0],np.vstack(values)[:,1]
#     # =============================================================================
#     ''' Debug figure '''
#     ''' plot histogram '''
#     # F1F2Mapdict={'F1':x,'F2':y}
#     # inspect_dim='F2'
#     # #''' plot histogram '''
#     # figure = plt.figure()
#     # plt.hist(F1F2Mapdict[inspect_dim])
#     # plt.title('{0}_ADOS{1}'.format(pid_dict[people],ASDlab[0]))
#     # additional_infoF2="mean={mean}, std={std}, skew={skew}, kurtosis={kurtosis}".format(\
#     #     mean=formant_info[inspect_dim+'+mean'],std=formant_info[inspect_dim+'+std'],\
#     #         skew=formant_info[inspect_dim+'+skew'],kurtosis=formant_info[inspect_dim+'+kurtosis'])
#     # plt.legend()
#     # plt.xlim(0, 5000)
#     # plt.figtext(0,-0.05,additional_infoF2)
#     # plt.savefig("Hist/"+'{0}_ADOS{1}_{2}'.format(pid_dict[people],ASDlab[0],inspect_dim),dpi=300, bbox_inches = "tight")
#     # =============================================================================

#     fig, ax = plt.subplots()
#     area=np.repeat(10,len(x))
#     # cms=np.repeat(phoneme_color_map[phone],len(x))
    
    
#     plt.scatter(x, y, s=area,  label=phone)
#     plt.title('{0}_ADOS{1}'.format(pid_dict[people],ASDlab[0]))
        
#     additional_infoF1="mean={mean}, std={std}, skew={skew}, kurtosis={kurtosis}".format(\
#         mean=formant_info['F1+mean'],std=formant_info['F1+std'],skew=formant_info['F1+skew'],kurtosis=formant_info['F1+kurtosis'])
#     additional_infoF2="mean={mean}, std={std}, skew={skew}, kurtosis={kurtosis}".format(\
#         mean=formant_info['F2+mean'],std=formant_info['F2+std'],skew=formant_info['F2+skew'],kurtosis=formant_info['F2+kurtosis'])
#     plt.ylim(0, 5000)
#     plt.xlim(0, 5000)
#     ax.legend()
#     plt.figtext(0,0.02,additional_infoF1)
#     plt.figtext(0,-0.05,additional_infoF2)
#     plt.xlabel("F1")
#     plt.ylabel("F2")
#     plt.savefig(Plotout_path+'{0}_ADOS{1}'.format(pid_dict[people],ASDlab[0]),dpi=300, bbox_inches = "tight")
#     plt.show()


# =============================================================================
'''

Inspect area

'''
# 
# Vowels_sampNum: number of spoken phones 
# Formants_people_symb: raw F1 F2 points
# formants_df_people_vowel: formant features
# N=5

#In this area please use F9 to for execution
# =============================================================================
# if INSPECT:
#     phone="ax5"
    
#     label_ADOSC=Label.label_raw[['name','ADOS_C']].set_index("name")
#     Num_spokenphone=Table_phonenprosody_qualified[phone]
    
#     formants_df_people_vowel[phone]=pd.concat([formants_df_people_vowel[phone],label_ADOSC],axis=1)
#     formants_df_people_vowel[phone]=pd.concat([formants_df_people_vowel[phone],Num_spokenphone],axis=1)
    
#     # people= '2015_12_06_01_045_1'
#     # people= '2017_01_16_02_024_1'
#     people= '2017_12_23_01_407'
#     v=Formants_people_symb[people][phone]
    
#     people2= '2016_08_27_01_044_1'
#     v2=Formants_people_symb[people2][phone]
    
#     # =============================================================================
#     '''

#     Plot x = desired value, y= ADOS score. We want to sample critical samples for further inspection
    
#     '''
#     num_spoken_phone=pd.DataFrame.from_dict(Vowels_sampNum[phone],orient='index',columns=['num_spoken_phone'])
#     N=10
#     # =============================================================================
#     fig, ax = plt.subplots()
#     feature_str='E2_F2+skew'
#     phone=feature_str.split("_")[0]
#     Inspect_column='_'.join(feature_str.split("_")[1:])
#     df_formant_statistic=formants_df_people_vowel[phone].astype(float)[Inspect_column]
#     label_ADOSC=Label.label_raw[['name','ADOS_C']].set_index("name")
#     df_formant_statistic=pd.concat([df_formant_statistic,label_ADOSC],axis=1)
#     df_formant_statistic=pd.concat([df_formant_statistic,num_spoken_phone],axis=1)
#     df_formant_statistic_qualified=df_formant_statistic[df_formant_statistic['num_spoken_phone']>N]
    
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
# df_pid[df_pid['name_id']==34].index
