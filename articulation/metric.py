#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 22:45:20 2021

@author: jackchen
"""
from addict import Dict
import re 
import pandas as pd
import numpy as np
from utils_wer.wer import  wer as WER
from utils_jack  import  Get_aligned_sequences
def CalculateAliDist(Phone_utt_human,Phone_utt_alignment,feature_sel=['F1','F2']):
    '''

    Calculate alignment score

    We have several alternitives to present this score, either by delete 
    unreasonable data with F1 F2 for data cleansing or simply calculate the 
    difference
    

    '''
    r=Phone_utt_human['text'].values.astype(str)
    h=Phone_utt_alignment['text'].values.astype(str)
    
    error_info, WER_value=WER(r,h)
    utt_human_ali, utt_hype_ali=Get_aligned_sequences(ref=Phone_utt_human, hype=Phone_utt_alignment,error_info=error_info)
    
    df_AlignDist=pd.DataFrame([],columns=['d','human','hype'])
    for j in range(len(utt_human_ali)):
        human, hype=utt_human_ali.iloc[j], utt_hype_ali.iloc[j]
        dist_st=human['start'] - hype['start']
        dist_ed=human['end'] - hype['end']
        Main_dist= np.abs((dist_st + dist_ed)/2 )  #A value to represent the distance ("Mean" operation used in this code)
        df_AlignDist.loc[j]=[Main_dist,human['text'],hype['text']]
    
    CorrectAliDict=df_AlignDist[df_AlignDist['human'] == df_AlignDist['hype']] # df_AlignDist will contain both correctly matched \
                                                                               # string and non-correct matched string. This script \
                                                                               # only compute alignment score by correctly matched string
    df_feature_dist=pd.concat([utt_human_ali[feature_sel].subtract(utt_hype_ali[feature_sel]),utt_human_ali['text']],axis=1)
    df_feature_dist=df_feature_dist[df_AlignDist['human'] == df_AlignDist['hype']] # only compute feature distance score by correctly matched string
    return CorrectAliDict, error_info, df_feature_dist

def EvaluateAlignments(Utt_dict_human,Utt_dict_alignment,phonesInspect):
    # =============================================================================
    '''
        Step 1    
    
        To Bookkeep the distance information in Global_info_dict
        
        Global_info_dict will be traced in the next step
    
    '''
    # =============================================================================
    error_bag=[]
    Global_info_dict=Dict()
    Global_featDist_info_dict=Dict()
    for keys, values in Utt_dict_human.items():
        utterance=keys
        people=keys[:keys.find(re.findall("_[K|D]",keys)[0])]
        Phone_utt_human=pd.DataFrame(np.hstack([Utt_dict_human[keys],np.array(Utt_dict_human[keys].index).reshape(-1,1)]),columns=Utt_dict_human[keys].columns.values.tolist()+['text']).sort_values(by='start')
        Phone_utt_alignment=pd.DataFrame(np.hstack([Utt_dict_alignment[keys],np.array(Utt_dict_alignment[keys].index).reshape(-1,1)]),columns=Utt_dict_alignment[keys].columns.values.tolist()+['text']).sort_values(by='start')
    
        CorrectAliDict, error_info, df_feature_dist=CalculateAliDist(Phone_utt_human,Phone_utt_alignment)
        
        error_bag.extend(error_info)
        Global_info_dict[people][utterance+"_Phone"]=CorrectAliDict
        Global_featDist_info_dict[people][utterance+"_Phone"]=df_feature_dist
    # =============================================================================
    '''
        Step 2    
    
        To obtain the phoneme recall by tracing the Bookkeep information from 
        previous stage:  Global_info_dict
    
    
        You can set criteria for df_values_filtered, or just let df_values_filtered to be df_values
    '''
    
    Tolerances={'10ms':0.01,'20ms':0.02, '30ms':0.03}
    # =============================================================================
    df_evaluation_metric=pd.DataFrame([],columns=Tolerances.keys())
    for spk in Global_info_dict.keys():
        Score_acc={k:0 for k,v in Tolerances.items()}
        Total_len=0
        for utt, df_values in Global_info_dict[spk].items():
            if phonesInspect != None:
                df_values_filtered=df_values[df_values['human'].str.contains(phonesInspect)]
            else:
                df_values_filtered=df_values
            
            Total_len+=len(df_values_filtered)
            # print(df_values_filtered)
            
            
            for k,v in Tolerances.items():
                Score_acc[k]+=len(df_values_filtered[df_values_filtered['d']<=v]) 
        # Total_len=sum([len(df_values_filtered) for utt, df_values_filtered in Global_info_dict[spk].items()])
        
        
        for k,v in Score_acc.items():
            df_evaluation_metric.loc[spk,k]=v/Total_len
    return df_evaluation_metric