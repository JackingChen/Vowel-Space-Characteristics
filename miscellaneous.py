#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 21:23:48 2021

@author: jackchen
"""

import os, glob
import pickle
import re
import articulation.articulation
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from addict import Dict
from utils_jack  import  Formant_utt2people_reshape, Gather_info_certainphones
from articulation.HYPERPARAM import phonewoprosody, Label


# =============================================================================
'''

    Dig Result file
    
    find the Ctx Dep phones that fits several criteria:
        better than a+b+c
        effect higher enough
        significant

'''
# =============================================================================
label_col=['ADOS_C']
Results_excel_path='RESULTS/'
filepath=Results_excel_path + '{}.xlsx'.format('EN_Customized_feature_ADOS')
sheetnames=['R2_adj','pear','spear']

Sel_feature_dict=Dict()
for sheet_n in sheetnames:
    Regession_info=pd.read_excel(filepath,sheet_n).convert_dtypes()
    Regession_info_abc=Regession_info[Regession_info['Unnamed: 0'].astype(str).str.contains('a\+b\+c')]
    Regession_info_feat=Regession_info[~Regession_info['Unnamed: 0'].astype(str).str.contains('a\+b\+c')]
    df_sel_feature=pd.DataFrame()
    for label in label_col:
        for idx_feat,idx_abc in zip(Regession_info_feat.index,Regession_info_abc.index):
            feat_name='-'.join(Regession_info_feat.loc[idx_feat,'Unnamed: 0'].split('-')[:-1])
            abc_name='-'.join(Regession_info_abc.loc[idx_abc,'Unnamed: 0'].split('-')[:-1])
            assert feat_name == abc_name
            result_raw=Regession_info_feat.loc[idx_feat,label]
            result_abc_raw=Regession_info_abc.loc[idx_abc,label]
            
            if sheet_n == 'spear':
                r=np.float(result_raw.split("/")[0])
                p=np.float(result_raw.split("/")[1])
                
                abc_r=np.float(result_abc_raw.split("/")[0])
                abc_p=np.float(result_abc_raw.split("/")[1])
            else:
                r=np.float(result_raw)
                abc_r=np.float(result_abc_raw)
            
            
            if r > abc_r:
                df_sel_feature=df_sel_feature.append(Regession_info_feat.loc[idx_feat])
    Sel_feature_dict[sheet_n]=df_sel_feature


featurepath_base='Features/artuculation_AUI'
Feature_Vowelpath=featurepath_base+'/Vowels/'
Feature_CtxVowelpath=featurepath_base+'/CtxDepVowels/bkup0729/'

manual_sel_feat_path='Manual_sel_faetures_raw'
Manual_choosen_feature=[]
with open(manual_sel_feat_path,'r') as f:
    content=f.read()
    for line in content.split("\n"):
        if len(line) > 0:
            Manual_choosen_feature.append(line[:re.search('__ADOS_C',line).end()])

N=2
df_formant_statistics_ASDkid=pickle.load(open(Feature_Vowelpath + 'Formant_AUI_tVSAFCRFvals_ASDkid.pkl',"rb"))
arti=articulation.articulation.Articulation()
df_formant_statistics_ASDkid=arti.BasicFilter_byNum(df_formant_statistics_ASDkid,N=N)


''' Read Ctx Dep feature '''
df_formant_statistics_CtxPhone_collect_dict=Dict()
for CtxDepFeat in Manual_choosen_feature:
    filepath=Feature_CtxVowelpath+'{}.pkl'.format(CtxDepFeat)   
    df_formant_statistics_CtxPhone=pickle.load(open(filepath ,"rb"))
    df_formant_statistics_CtxPhone=arti.BasicFilter_byNum(df_formant_statistics_CtxPhone,N=N)

    dfout=df_formant_statistics_ASDkid.copy()
    dfout=dfout.loc[np.array(df_formant_statistics_CtxPhone.index)]
    
    df_formant_statistics_CtxPhone_collect_dict[CtxDepFeat]=df_formant_statistics_CtxPhone
    
    outpath = featurepath_base + '/Pseudo_CtxDepVowels/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    pickle.dump(dfout,open(outpath + 'PeopleMatch_{}.pkl'.format(CtxDepFeat)   ,"wb"))


aaa=ccc

# =============================================================================
''' 

    Scatter distrib plot area

'''
# =============================================================================
# =============================================================================
# Plot boxplots
# PeopleOfInterest=['2016_06_27_02_017_1', '2016_07_30_01_148', '2016_08_26_01_168_1',
#        '2016_09_24_01_174_1'] # Manual choose
AUI_feature_path='articulation/Pickles'
# =============================================================================

# =============================================================================
''' Get regular vowewl '''
PhoneMapp_dict=phonewoprosody.PhoneMapp_dict
PhoneOfInterest=list(PhoneMapp_dict.keys())
# =============================================================================
outpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles'
filepath=outpath+"/[Analyzing]Formants_utt_symb_limited.pkl"
Formants_utt_symb=pickle.load(open(filepath,"rb"))



 
Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
Vowels_AUI_TotalPhones=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)

''' Get regular vowel end'''

# This part is totally manual


feat_type_lst=['LeftDepVowel_AUI','CtxDepVowel_AUI','LeftDepVowel_AUI']
comb_strs_lst=['plosiveaspirated-A__affricateaspirated-u__SIL-i__ADOS_C',\
           'alveolpalatal-A+alveolar__alveolar-u+Vowel__All-i+All__ADOS_C',\
           'labiodental-A__Vowel-u__alveolar-i__ADOS_C']

# ['CtxDepVowel_AUI', 'LeftDepVowel_AUI', 'RightDepVowel_AUI']

    
    
articulatn=articulation.articulation.Articulation()
for CtxPhone_types in ['Manner_simp1']:
    Feature_dicts=pickle.load(open(AUI_feature_path+"/AUI_ContextDepPhonesMerge_{0}_uwij.pkl".format(CtxPhone_types),"rb"))
    for feat_type, phone_str in zip(feat_type_lst,comb_strs_lst):
        print(phone_str)
        CtxPhone={}
        CtxPhone['A:'], CtxPhone['u:'], CtxPhone['i:'], label=phone_str.split("__") # dtype: strings
        
        comb=[CtxPhone['A:'], CtxPhone['u:'], CtxPhone['i:'], label]
        CtxDepVowel_AUI_dict=Feature_dicts[feat_type]
        
        
        # Eval_med=Evaluation_method()
        # =============================================================================
        '''
        
            ANOVA F value genertor
            
            input: Vowel_AUI
            output df_f_val
        '''
        
        ''' Collect the people that have phones of A: u: i:, and Set_unionPoeple_AUI is a list of people containing the three phones of inspect'''
        Main_dict=Dict()
        Main_dict['A:'], Main_dict['u:'], Main_dict['i:']= CtxDepVowel_AUI_dict[comb[0]], CtxDepVowel_AUI_dict[comb[1]], CtxDepVowel_AUI_dict[comb[2]]
        # Take union of a_dict, u_dict, i_dict
        
        
        Set_unionPoeple_AUI=set(list(Main_dict['A:'].keys())+list(Main_dict['u:'].keys())+list(Main_dict['i:'].keys()))
        Set_unionPoeple_AUI=Set_unionPoeple_AUI.intersection(list(Main_dict['A:'].keys()))
        Set_unionPoeple_AUI=Set_unionPoeple_AUI.intersection(list(Main_dict['u:'].keys()))
        Set_unionPoeple_AUI=Set_unionPoeple_AUI.intersection(list(Main_dict['i:'].keys()))
        # Num_people=len(Set_unionPoeple_AUI)
        
        #The input to articulation.calculate_features has to be Vowels_AUI form
        Vowels_AUI = Dict()
        for people in Set_unionPoeple_AUI:
            for phone in Main_dict.keys():
                Vowels_AUI[people][phone]=Main_dict[phone][people]
                
                
                
        ''' Plot '''
        PeopleOfInterest=Vowels_AUI.keys()
        Name2num=Dict()
        for i,k in enumerate(sorted(Vowels_AUI.keys())):
            Name2num[k]=i
        
        # =============================================================================
        # Joint plot those people of data
        # =============================================================================
        plot_outpath='Plot/CtxPhoneAUI/'+phone_str + '/'
        if not os.path.exists(plot_outpath):
            os.makedirs(plot_outpath)
        print(plot_outpath)
        
        
        
        for people in PeopleOfInterest:
            fig, axes = plt.subplots(1, 2)
            for ax in axes:
                ax.set_xlim(0,2000) 
                ax.set_ylim(0,3000) 
            # df_samples for Ctx Dep Phone
            df_samples_AUI=pd.DataFrame()
            for symb in Vowels_AUI[people].keys():
                df_tmp=Vowels_AUI[people][symb]
                df_tmp['phone']=symb
                df_samples_AUI=df_samples_AUI.append(df_tmp)
            df_samples_AUI=df_samples_AUI.sort_values('phone')
            df_samples_AUI[['F1','F2']]=df_samples_AUI[['F1','F2']].astype(float)
            
            
            sns.scatterplot(ax=axes[0], data=df_samples_AUI, x='F1',y='F2',hue='phone')
            # for line in range(0,AUI_dict[people][symb].shape[0]):
            #     plt.text(AUI_dict[people][symb].F1.iloc[line]+0.2, AUI_dict[people][symb].F1.iloc[line], AUI_dict[people][symb].utt.iloc[line], horizontalalignment='left', size='medium', color='black', weight='semibold')
        
            
            
            
            info_str="""CtxDepPhone: {0}""".format(phone_str)
            title='{0}_{1}'.format(people, symb)
            axes[0].set_title( title )
            axes[0].text(x=0., y=0.,s=info_str, horizontalalignment='left', size='medium', color='black', weight='semibold')

            
            
            
            # df_samples for Total Phones
            df_samples_AUI=pd.DataFrame()
            for symb in Vowels_AUI_TotalPhones[people].keys():
                df_tmp=Vowels_AUI_TotalPhones[people][symb]
                df_tmp['phone']=symb
                df_samples_AUI=df_samples_AUI.append(df_tmp)
            df_samples_AUI=df_samples_AUI.sort_values('phone')
            df_samples_AUI[['F1','F2']]=df_samples_AUI[['F1','F2']].astype(float)
            
            sns.scatterplot(ax=axes[1], data=df_samples_AUI, x='F1',y='F2',hue='phone')
            # for line in range(0,AUI_dict[people][symb].shape[0]):
            #     plt.text(AUI_dict[people][symb].F1.iloc[line]+0.2, AUI_dict[people][symb].F1.iloc[line], AUI_dict[people][symb].utt.iloc[line], horizontalalignment='left', size='medium', color='black', weight='semibold')
        
            
            
            if not os.path.exists(plot_outpath):
                os.makedirs(plot_outpath)
            info_str="""CtxDepPhone: {0}""".format(phone_str)
            title='{0}_{1}'.format(people, symb)
            axes[1].set_title( title )
            # axes[1].text(x=0., y=0.,s=info_str, horizontalalignment='left', size='medium', color='black', weight='semibold')
            fig.savefig (plot_outpath+"/{0}.png".format(title))
            axes[0].clear()
            axes[1].clear()
            