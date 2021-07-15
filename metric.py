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
from scipy.stats import spearmanr,pearsonr 

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



class Evaluation_method:
    def __init__(self,):
        self.featurepath='Features'
        self.columns=['F_vals_f1(A:,u:,i:)', 'F_vals_f2(A:,u:,i:)',
       'F_val_mix(A:,u:,i:)', 'F_vals_f1(A:,u:)', 'F_vals_f2(A:,u:)',
       'F_val_mix(A:,u:)', 'F_vals_f1(A:,i:)', 'F_vals_f2(A:,i:)',
       'F_val_mix(A:,i:)', 'F_vals_f1(u:,i:)', 'F_vals_f2(u:,i:)',
       'F_val_mix(u:,i:)']
        self.label_choose_lst=['ADOS_C']
        self.N=2
    def _Postprocess_dfformantstatistic(self, df_formant_statistic):
        ''' Remove person that has unsufficient data '''
        df_formant_statistic_bool=(df_formant_statistic['u_num']!=0) & (df_formant_statistic['a_num']!=0) & (df_formant_statistic['i_num']!=0)
        df_formant_statistic=df_formant_statistic[df_formant_statistic_bool]
        
        ''' ADD ADOS category '''
        df_formant_statistic['ADOS_cate']=np.array([0]*len(df_formant_statistic))
        df_formant_statistic.loc[df_formant_statistic['ADOS']<2,'ADOS_cate']=0
        df_formant_statistic.loc[df_formant_statistic['ADOS']<2,'ADOS_cate']=0
        df_formant_statistic.loc[(df_formant_statistic['ADOS']<3) & (df_formant_statistic['ADOS']>=2),'ADOS_cate']=1
        df_formant_statistic.loc[df_formant_statistic['ADOS']>=3,'ADOS_cate']=2
        return df_formant_statistic 
    def _Postprocess_dfformantstatistic_N_notnanADOS(self, df_formant_statistic,N=1,evictNamelst=[]):
        ''' Remove person that has unsufficient data '''
        filter_bool=np.logical_and(df_formant_statistic['u_num']>N,df_formant_statistic['a_num']>N)
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['i_num']>N)
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['ADOS'].isna()!=True)
        if len(evictNamelst)>0:
            for name in evictNamelst:
                filter_bool.loc[name]=False
                
        df_formant_qualified=df_formant_statistic[filter_bool]
        return df_formant_qualified
    
    def Calculate_correlation(self, label_choose_lst,df_formant_statistic,N,columns,\
                          corr_label='ADOS', constrain_sex=-1, constrain_module=-1, constrain_assessment=-1,\
                          evictNamelst=[],correlation_type='spearmanr'):
        '''
            constrain_sex: 1 for boy, 2 for girl
            constrain_module: 3 for M3, 4 for M4
        '''
        df_pearsonr_table=pd.DataFrame([],columns=[correlation_type,'{}_pvalue'.format(correlation_type[:5]),'de-zero_num'])
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
            if len(evictNamelst)>0:
                for name in evictNamelst:
                    filter_bool.loc[name]=False
                
            df_formant_qualified=df_formant_statistic[filter_bool]
            for col in columns:
                if len(df_formant_qualified) >2:
                    spear,spear_p=spearmanr(df_formant_qualified[col],df_formant_qualified[corr_label])
                    pear,pear_p=pearsonr(df_formant_qualified[col],df_formant_qualified[corr_label])
        
                    if correlation_type == 'pearsonr':
                        df_pearsonr_table.loc[col]=[pear,pear_p,len(df_formant_qualified[col])]
                        # pear,pear_p=pearsonr(df_denan["{}_LPP_{}".format(ps,ps)],df_formant_qualified['ADOS'])
                        # df_pearsonr_table_GOP.loc[ps]=[pear,pear_p,len(df_denan)]
                    elif correlation_type == 'spearmanr':
                        df_pearsonr_table.loc[col]=[spear,spear_p,len(df_formant_qualified[col])]
            # print("Setting N={0}, the correlation metric is: ".format(N))
            # print("Using evaluation metric: {}".format(correlation_type))
        return df_pearsonr_table