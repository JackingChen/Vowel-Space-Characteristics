#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:45:57 2022

這個腳本用來畫TASLP paper的Fig.4 用來舉一個Vowel Space的例子

@author: jackchen
"""
import os
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import special, stats
from addict import Dict
from Syncrony import Syncrony
import seaborn as sns
from pylab import text
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing
from matplotlib.offsetbox import AnchoredText
import pandas as pd
from articulation.HYPERPARAM import phonewoprosody, Label
from utils_jack  import  Formant_utt2people_reshape, Gather_info_certainphones, \
                         FilterUttDictsByCriterion, GetValuelimit_IQR, \
                         Get_aligned_sequences, WER, Get_Vowels_AUI
from datetime import datetime as dt
import pathlib
from multiprocessing import Pool, current_process
import articulation.Multiprocess as Multiprocess
import articulation.HYPERPARAM.PaperNameMapping as PprNmeMp
from tqdm import tqdm
import seaborn as sns
from articulation.HYPERPARAM.PlotFigureVars import *


def Swap2PaperName(feature_rawname,PprNmeMp):
    if feature_rawname in PprNmeMp.Paper_name_map.keys():
        featurename_paper=PprNmeMp.Paper_name_map[feature_rawname]
        feature_keys=featurename_paper
    else: 
        feature_keys=feature_rawname
    return feature_keys

def Process_IQRFiltering_Multi(Formants_utt_symb, limit_people_rule,\
                               outpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',\
                               prefix='Formants_utt_symb',\
                               suffix='KID_FromASD_DOCKID'):
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
    
    pickle.dump(Formants_utt_symb_limited,open(outpath+"/[Analyzing]{0}_limited_{1}.pkl".format(prefix,suffix),"wb"))
    print('Formants_utt_symb saved to ',outpath+"/[Analyzing]{0}_limited_{1}.pkl".format(prefix,suffix))    
def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--knn_weights', default='uniform',
                            help='uniform distance')
    parser.add_argument('--knn_neighbors', default=2,  type=int,
                            help='path of the base directory')
    parser.add_argument('--Reorder_type', default='DKIndividual',
                            help='[DKIndividual, DKcriteria]')
    parser.add_argument('--Result_path', default='./Result_Interaction',
                            help='')
    parser.add_argument('--Inspect_roles', default=['D','K'],
                            help='')
    parser.add_argument('--dataset_role', default='ASD_DOCKID',
                            help='ASD_DOCKID, TD_DOCKID')
    parser.add_argument('--Person_segment_df_path', default="articulation/Pickles/Session_formants_people_vowel_feat/",
                            help='')
    parser.add_argument('--SyncronyFeatures_root_path', default='Features/artuculation_AUI/Interaction/Syncrony_Knnparameters/',
                            help='')
    parser.add_argument('--Formants_utt_symb_path', default='articulation/Pickles',
                            help='')
    parser.add_argument('--poolMed', default='middle',
                            help='path of the base directory')
    parser.add_argument('--poolWindowSize', default=3,
                            help='path of the base directory')
    parser.add_argument('--Inspect_features', default=['F1','F2'],
                            help='')
    parser.add_argument('--reFilter', default=False,
                            help='')
    parser.add_argument('--Normalize_way', default='None',
                            help='func1 func2 func3 func4 func7 proposed')
    args = parser.parse_args()
    return args

args = get_args()
syncrony=Syncrony()
PhoneMapp_dict=phonewoprosody.PhoneMapp_dict
PhoneOfInterest=list(PhoneMapp_dict.keys())


knn_weights=args.knn_weights
knn_neighbors=args.knn_neighbors
Reorder_type=args.Reorder_type
dataset_role=args.dataset_role
Person_segment_df_path=args.Person_segment_df_path
SyncronyFeatures_root_path=args.SyncronyFeatures_root_path

#%%
# =============================================================================
'''
    
    畫Vowel space的地方

'''
Formants_utt_symb=pickle.load(open(args.Formants_utt_symb_path+"/Formants_utt_symb_by{0}_window{1}_{2}.pkl".format(args.poolMed,args.poolWindowSize,args.dataset_role),'rb'))
print("Loading Formants_utt_symb from ", args.Formants_utt_symb_path+"/Formants_utt_symb_by{0}_window{1}_{2}.pkl".format(args.poolMed,args.poolWindowSize,args.dataset_role))


Formants_utt_symb,Formants_utt_symb_cmp,Align_OrinCmp= \
    Formants_utt_symb,Formants_utt_symb, False
    


# =============================================================================
'''

    第一步：IQR Filtering: 從Formants_utt_symb準備Vowel_AUI
'''
# =============================================================================

Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
AUI_info=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)
limit_people_rule=GetValuelimit_IQR(AUI_info,PhoneMapp_dict,args.Inspect_features)


keys=[]
interval=20
for i in range(0,len(Formants_utt_symb.keys()),interval):
    # print(list(combs_tup.keys())[i:i+interval])
    keys.append(list(Formants_utt_symb.keys())[i:i+interval])

''' multi processing start '''
prefix,suffix = 'Formants_utt_symb', args.dataset_role
# date_now='{0}-{1}-{2} {3}'.format(dt.now().year,dt.now().month,dt.now().day,dt.now().hour)
date_now='{0}-{1}-{2}'.format(dt.now().year,dt.now().month,dt.now().day)
outpath='articulation/Pickles'
filepath=outpath+"/[Analyzing]{0}_limited_{1}.pkl".format(prefix,suffix)
if os.path.exists(filepath) and args.reFilter==False:
    fname = pathlib.Path(filepath)
    mtime = dt.fromtimestamp(fname.stat().st_mtime)
    # filemtime='{0}-{1}-{2} {3}'.format(mtime.year,mtime.month,mtime.day,mtime.hour)
    filemtime='{0}-{1}-{2}'.format(mtime.year,mtime.month,mtime.day)
    
    # If file last modify time is not now (precisions to the hours) than we create new one
    if filemtime != date_now:
        Process_IQRFiltering_Multi(Formants_utt_symb,limit_people_rule,\
                               outpath=outpath,\
                               prefix=prefix,\
                               suffix=suffix) # the results will be output as pkl file at outpath+"/[Analyzing]Formants_utt_symb_limited.pkl"
else:
    Process_IQRFiltering_Multi(Formants_utt_symb,limit_people_rule,\
                               outpath=outpath,\
                               prefix=prefix,\
                               suffix=suffix)
Formants_utt_symb_limited=pickle.load(open(filepath,"rb"))
''' multi processing end '''
if len(limit_people_rule) >0:
    Formants_utt_symb=Formants_utt_symb_limited

Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
AUI_info=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)
Vowels_AUI=Get_Vowels_AUI(AUI_info, args.Inspect_features,VUIsource="From__Formant_people_information")

# Get BCC, FCR, ... utterance level features
dfFormantStatisticpath=os.getcwd()
df_formant_statistic77_path=dfFormantStatisticpath+'/Features/ClassificationMerged_dfs/{Normalize_way}/{dataset_role}/static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation.pkl'.format(
    knn_weights=knn_weights,
    knn_neighbors=knn_neighbors,
    Reorder_type=Reorder_type,
    dataset_role='ASD_DOCKID',
    Normalize_way=args.Normalize_way
    )
df_feature_ASD=pickle.load(open(df_formant_statistic77_path,'rb'))

# =============================================================================
'''

    第二步：KDE filtering: 準備 df_vowel_calibrated 或 df_vowel
'''
# =============================================================================

# Play code for KDE filtering
from articulation.articulation import Articulation
articulation=Articulation() #使用Articulation的KDE_Filtering
THRESHOLD=40
# scale_factor=100
N=2
RESULT_DICTIONARY=Dict()
df_simulate=pd.DataFrame()
df_vowel_calibrated_dict=Dict()
for people in Vowels_AUI.keys():
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
            # df_vowel=df_vowel.append(df_)
            df_vowel=pd.concat([df_vowel,df_], ignore_index=True)
    len_a=len(np.where(df_vowel['vowel']=='A:')[0])
    len_u=len(np.where(df_vowel['vowel']=='u:')[0])
    len_i=len(np.where(df_vowel['vowel']=='i:')[0])
    
    
    if len_a<=N or len_u<=N or len_i<=N:
        continue
    
    df_vowel_calibrated=articulation.KDE_Filtering(df_vowel,THRESHOLD=THRESHOLD,scale_factor=100)
    df_vowel_calibrated_dict[people]=df_vowel_calibrated

#%%
# =============================================================================
'''  
    第三步：Normalization
'''
# =============================================================================
from articulation.articulation import Normalizer
Normalize_way='func15'

normalizer=Normalizer()
Normalize_Functions={}
Normalize_Functions['func1']=normalizer.func1
Normalize_Functions['func2']=normalizer.func2
Normalize_Functions['func3']=normalizer.func3
Normalize_Functions['func7']=normalizer.func7
Normalize_Functions['func10']=normalizer.func10
Normalize_Functions['func13']=normalizer.func13
Normalize_Functions['func14']=normalizer.func14
Normalize_Functions['func15']=normalizer.func15
Normalize_Functions['func16']=normalizer.func16
Normalize_Functions['func17']=normalizer.func17


func_in=Normalize_Functions[Normalize_way]

df_vowel_calibrated_Normalized_dict=Dict()
for Pple in df_vowel_calibrated_dict.keys():
    df_vowel=df_vowel_calibrated_dict[Pple]
    df_vowel_norm=normalizer.apply_function(df=df_vowel,\
                                func=func_in,\
                                column=args.Inspect_features)
    df_vowel_calibrated_Normalized_dict[Pple]=df_vowel_norm

# =============================================================================
'''畫Vowel space'''
# =============================================================================
if args.Normalize_way!='None':
    df_vowel_ToInspect_dict=df_vowel_calibrated_Normalized_dict
else:
    df_vowel_ToInspect_dict=df_vowel_calibrated_dict
score_cols='ADOS_C'
# feature_name='between_covariance_norm(A:,i:,u:)'
feature_name='FCR2'
# People_VowelSpace_inspect=Vowels_AUI.keys()
ASD_samples_bool=~Label.label_raw['ADOS_cate_CSS'].isna().values

# People_VowelSpace_inspect=list(Label.label_raw.loc[ASD_samples_bool].sort_values(by=score_cols)['name'])
People_VowelSpace_inspect=[
    '2016_07_30_01_164',
    '2017_07_05_01_310_1',
    '2017_04_08_01_256_1',
    '2017_03_13_01_194_1'
    ]

Demonstration_people=Dict()

# 這是TASLP Fig.2 在show conversation-level feature 使用的人的紀錄
Demonstration_people['Demonstrate'].high='2017_12_23_01_407'
Demonstration_people['ADOS_{comm}'].high='2016_07_30_01_164'
Demonstration_people['ADOS_{comm}'].high='2017_07_05_01_310_1'



score_df=Label.label_raw.loc[ASD_samples_bool].sort_values(by=score_cols)[[score_cols,'name']]
score_df=score_df.set_index('name')

feature_df=df_feature_ASD.copy()

# Moderate ASD vs TD 時因為DEP feature 被判斷成TD的個案
# People_VowelSpace_inspect=[
# '2021_01_25_5833_1(醫生鏡頭模糊)_emotion',
# ]


# 這是TASLP Fig.6 Experiment2 舉的例子在舉例不同嚴重程度的ASD Vowel space 不一樣
FileName_dict={
    '2016_07_30_01_164':'images/VowelSpace_example-high1.png',\
    '2017_07_05_01_310_1':'images/VowelSpace_example-low1.png',
    '2017_04_08_01_256_1':'images/VowelSpace_example-high2.png',\
    '2017_03_13_01_194_1':'images/VowelSpace_example-low2.png',
    } 

count=0
for Pple in People_VowelSpace_inspect:
    # try:
    fig, ax = plt.subplots()
    # plt.figure(count)
    score_PprNme=feature_papername=Swap2PaperName(score_cols,PprNmeMp)
    score=score_df.loc[Pple,score_cols]
    # Title_str="{}: {} = {}".format(Pple, score_cols,score)
    Title_str="{} = {}".format(score_PprNme,score)
    sns.scatterplot(data=df_vowel_ToInspect_dict[Pple], x="F1", y="F2", hue="vowel").set(title=Title_str)
    
    
    feature_value=feature_df.loc[Pple,feature_name]
    feature_papername=Swap2PaperName(feature_name,PprNmeMp)
    info_arr=["{}: {}".format(feature_papername,np.round(feature_value,3))]
    # info_arr=["{}: {}".format(idx,v) for idx, v in zip(score.index.values,np.round(score.values,3))]
    addtext='\n'.join(info_arr)
    x0, xmax = plt.xlim()
    y0, ymax = plt.ylim()
    data_width = xmax - x0
    data_height = ymax - y0
    # text(x0/0.1 + data_width * 0.004, -data_height * 0.002, addtext, ha='center', va='center')
    # text(0, -0.1,addtext, ha='center', va='center', transform=ax.transAxes)
    
    plt.show()
    fig.set_size_inches((2.5,2.5))
    fig.savefig(fname=FileName_dict[Pple],bbox_inches='tight',dpi=300)
    fig.clf()
    
    
    count+=1
    # except: 
    #     pass