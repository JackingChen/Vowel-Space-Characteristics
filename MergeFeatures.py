#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 20:48:53 2022

@author: jackchen


better make it a class, but not now

"""
import argparse
import os
from addict import Dict
import pickle
from itertools import combinations
import pandas as pd
def Merge_dfs(df_1, df_2):
    return pd.merge(df_1,df_2,left_index=True, right_index=True)

def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser()
    parser.add_argument('--MergeClassificationfeatures', default=True,
                        help='what kind of data you want to get')
    parser.add_argument('--MergeRegressionfeatures', default=True,
                        help='what kind of data you want to get')
    parser.add_argument('--ADDUtt_feature', default=True,
                        help='what kind of data you want to get')
    parser.add_argument('--exclude_people', default=['2015_12_07_02_003','2017_03_18_01_196_1'],
                        help='what kind of data you want to get')
    args = parser.parse_args()
    return args
args = get_args()

def TBMEB1Preparation_SaveForClassifyData(dfpath,\
                        df_SegLvl_features,suffix=''):
    '''
        
        We generate data for nested cross-valated analysis in Table.5 in TBME2021
        
        The data will be stored at Pickles/Session_formants_people_vowel_feat
    
    '''
    dfFormantStatisticFractionpath= dfpath + 'Session_formants_people_vowel_feat'
    if not os.path.exists(dfFormantStatisticFractionpath):
        os.makedirs(dfFormantStatisticFractionpath)
    pickle.dump(df_SegLvl_features,open(dfFormantStatisticFractionpath+'/df_SegLvl_features_{}.pkl'.format(suffix),'wb'))

# TBMEB1Preparation_SaveForClassifyData('Pickles/',df_phonation_kid,suffix=method)

def TBMEB1Preparation_LoadForFromOtherData(dfFormantStatisticpath,\
                                           prefix='Formant_AUI_tVSAFCRFvals',\
                                           suffix='KID_FromASD_DOCKID'):
    '''
        
        We generate data for nested cross-valated analysis in Table.5 in TBME2021
        
        The data will be stored at Pickles/Session_formants_people_vowel_feat
    
    '''
    dfFormantStatisticFractionpath=dfFormantStatisticpath+'/Session_formants_people_vowel_feat'
    if not os.path.exists(dfFormantStatisticFractionpath):
        raise FileExistsError('Directory not exist')
    df_phonation_statistic_77=pickle.load(open(dfFormantStatisticFractionpath+'/{prefix}_{suffix}.pkl'.format(\
                                                                         prefix=prefix,suffix=suffix),'rb'))
    return df_phonation_statistic_77

Utt_features_dict=Dict()
Utt_featuresCombinded_dict=Dict()
role='ASD'
for role in ['ASD','TD']:
    Utt_features_dict['df_disvoice_prosody_energy_{role}'.format(role=role)]=TBMEB1Preparation_LoadForFromOtherData('Pickles/',\
                                                prefix='df_SegLvl_features',\
                                                suffix='Disvoice_prosody_energy_kid_{role}'.format(role=role))
    Utt_features_dict['df_disvoice_phonation_{role}'.format(role=role)]=TBMEB1Preparation_LoadForFromOtherData('Pickles/',\
                                                                prefix='df_SegLvl_features',\
                                                                suffix='Disvoice_phonation_kid_{role}'.format(role=role))
    Utt_features_dict['df_disvoice_prosodyF0_{role}'.format(role=role)]=TBMEB1Preparation_LoadForFromOtherData('Pickles/',\
                                                                prefix='df_SegLvl_features',\
                                                                suffix='Disvoice_prosodyF0_kid_{role}'.format(role=role))  
    
        
    df_kid_ManualComb=pd.merge(Utt_features_dict['df_disvoice_phonation_{role}'.format(role=role)],\
                               Utt_features_dict['df_disvoice_prosody_energy_{role}'.format(role=role)],left_index=True, right_index=True)
    df_kid_ManualComb=pd.merge(df_kid_ManualComb,Utt_features_dict['df_disvoice_prosodyF0_{role}'.format(role=role)],left_index=True, right_index=True)
    # df_kid_ManualComb=pd.merge(df_kid_ManualComb,df_disvoice_phonation,left_index=True, right_index=True)
    # df_kid_ManualComb=df_kid_ManualComb.loc[:,~df_kid_ManualComb.columns.duplicated()]
    df_kid_ManualComb=df_kid_ManualComb.loc[:,~df_kid_ManualComb.columns.duplicated()].sort_index()
    Utt_featuresCombinded_dict[role]=df_kid_ManualComb



for knn_weights in ['uniform', 'distance']:
    for Reorder_type in ['DKIndividual','DKcriteria']:
        for knn_neighbors in [2, 3, 4, 5, 6]:
            if args.MergeClassificationfeatures:
                # dataset_role='ASD_DOCKID'
                for dataset_role in ['ASD_DOCKID','TD_DOCKID']:
                    if args.ADDUtt_feature==True:
                        role=dataset_role.split("_")[0] #ASD or TD
                    Merg_filepath={}
                    Merg_filepath['static_feautre_LOC']='Features/artuculation_AUI/Vowels/Formants/Formant_AUI_tVSAFCRFvals_KID_From{dataset_role}.pkl'.format(dataset_role=dataset_role)
                    Merg_filepath['static_feautre_phonation']='Features/artuculation_AUI/Vowels/Phonation/Phonation_meanvars_KID_From{dataset_role}.pkl'.format(dataset_role=dataset_role)
                    Merg_filepath['dynamic_feature_LOC']='Features/artuculation_AUI/Interaction/Syncrony_Knnparameters/Syncrony_measure_of_variance_{knn_weights}_{knn_neighbors}_{Reorder_type}_{dataset_role}.pkl'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,dataset_role=dataset_role,Reorder_type=Reorder_type)
                    Merg_filepath['dynamic_feature_phonation']='Features/artuculation_AUI/Interaction/Syncrony_Knnparameters/Syncrony_measure_of_variance_phonation_{knn_weights}_{knn_neighbors}_{Reorder_type}_{dataset_role}.pkl'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,dataset_role=dataset_role,Reorder_type=Reorder_type)
                    
                    if args.ADDUtt_feature==True:
                        merge_out_path='Features/ClassificationMerged_dfs/ADDed_UttFeat/{knn_weights}_{knn_neighbors}_{Reorder_type}/{dataset_role}/'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,dataset_role=dataset_role,Reorder_type=Reorder_type)
                    else:
                        merge_out_path='Features/ClassificationMerged_dfs/{knn_weights}_{knn_neighbors}_{Reorder_type}/{dataset_role}/'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,dataset_role=dataset_role,Reorder_type=Reorder_type)
                    if not os.path.exists(merge_out_path):
                        os.makedirs(merge_out_path)
                    
                    df_infos_dict=Dict()
                    for keys, paths in Merg_filepath.items():
                        df_data=pickle.load(open(paths,"rb")).sort_index()
                        #!!!!!!!!!!!!!!!!!  注意這邊我們人為移除一些資訊量太少的人
                        for key_wrds in args.exclude_people:
                            if df_data.index.str.contains(key_wrds).any():
                                df_data=df_data.drop(index=key_wrds)
                        df_infos_dict[keys]=df_data
                    
                    Merged_df_dict=Dict()
                    comb1 = list(combinations(list(Merg_filepath.keys()), 1))
                    comb2 = list(combinations(list(Merg_filepath.keys()), 2))
                    
                    if args.ADDUtt_feature==True: # output only Utt features
                        OutPklpath=merge_out_path+ "Utt_feature.pkl"
                        pickle.dump(Utt_featuresCombinded_dict[role],open(OutPklpath,"wb"))
                    
                    for c in comb1:
                        e1=c[0]
                        
                        if args.ADDUtt_feature==True: # Merge with Utt features
                            Merged_df_dict[e1]=Merge_dfs(df_infos_dict[e1],Utt_featuresCombinded_dict[role])
                            OutPklpath=merge_out_path+"Utt_feature+"+ e1 + ".pkl"
                        else:
                            Merged_df_dict[e1]=df_infos_dict[e1]
                            OutPklpath=merge_out_path+ e1 + ".pkl"
                        pickle.dump(Merged_df_dict[e1],open(OutPklpath,"wb"))
                        
                        
                    for c in comb2:
                        e1, e2=c
                        if args.ADDUtt_feature==True:
                            Merged_df_dict['+'.join(c)]=Merge_dfs(df_infos_dict[e1],df_infos_dict[e2])
                            Merged_df_dict['+'.join(c)]=Merge_dfs(Merged_df_dict['+'.join(c)],Utt_featuresCombinded_dict[role])
                            OutPklpath=merge_out_path+"Utt_feature+"+'+'.join(c)+".pkl"
                        else:
                            Merged_df_dict['+'.join(c)]=Merge_dfs(df_infos_dict[e1],df_infos_dict[e2])
                            OutPklpath=merge_out_path+'+'.join(c)+".pkl"
                        pickle.dump(Merged_df_dict['+'.join(c)],open(OutPklpath,"wb"))
                    
                    # Condition for : Columns_comb3 = All possible LOC feature combination + phonation_proximity_col
                    c = ('static_feautre_LOC', 'dynamic_feature_LOC', 'dynamic_feature_phonation')
                    e1, e2, e3=c
                    Merged_df_dict['+'.join(c)]=Merge_dfs(df_infos_dict[e1],df_infos_dict[e2])
                    Merged_df_dict['+'.join(c)]=Merge_dfs(Merged_df_dict['+'.join(c)],df_infos_dict[e3])
                    OutPklpath=merge_out_path+'+'.join(c)+".pkl"
                    pickle.dump(Merged_df_dict['+'.join(c)],open(OutPklpath,"wb"))
                    
            if args.MergeRegressionfeatures:
                for dataset_role in ['ASD_DOCKID']:
                    if args.ADDUtt_feature==True:
                        role=dataset_role.split("_")[0] #ASD or TD
                    Merg_filepath={}
                    Merg_filepath['static_feautre_LOC']='Features/artuculation_AUI/Vowels/Formants/Formant_AUI_tVSAFCRFvals_KID_From{dataset_role}.pkl'.format(dataset_role=dataset_role)
                    Merg_filepath['static_feautre_phonation']='Features/artuculation_AUI/Vowels/Phonation/Phonation_meanvars_KID_From{dataset_role}.pkl'.format(dataset_role=dataset_role)
                    Merg_filepath['dynamic_feature_LOC']='Features/artuculation_AUI/Interaction/Syncrony_Knnparameters/Syncrony_measure_of_variance_{knn_weights}_{knn_neighbors}_{Reorder_type}_{dataset_role}.pkl'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,dataset_role=dataset_role,Reorder_type=Reorder_type)
                    Merg_filepath['dynamic_feature_phonation']='Features/artuculation_AUI/Interaction/Syncrony_Knnparameters/Syncrony_measure_of_variance_phonation_{knn_weights}_{knn_neighbors}_{Reorder_type}_{dataset_role}.pkl'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,dataset_role=dataset_role,Reorder_type=Reorder_type)
                    
                    if args.ADDUtt_feature==True:
                        merge_out_path='Features/RegressionMerged_dfs/ADDed_UttFeat/{knn_weights}_{knn_neighbors}_{Reorder_type}/{dataset_role}/'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,dataset_role=dataset_role,Reorder_type=Reorder_type)
                    else:
                        merge_out_path='Features/RegressionMerged_dfs/{knn_weights}_{knn_neighbors}_{Reorder_type}/{dataset_role}/'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,dataset_role=dataset_role,Reorder_type=Reorder_type)
                    # merge_out_path='Features/RegressionMerged_dfs/{knn_weights}_{knn_neighbors}_{Reorder_type}/'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,Reorder_type=Reorder_type)
                    if not os.path.exists(merge_out_path):
                        os.makedirs(merge_out_path)
                    
                    df_infos_dict=Dict()
                    for keys, paths in Merg_filepath.items():
                        df_data=pickle.load(open(paths,"rb")).sort_index()
                        #!!!!!!!!!!!!!!!!!  注意這邊我們人為移除一些資訊量太少的人
                        for key_wrds in args.exclude_people:
                            if df_data.index.str.contains(key_wrds).any():
                                df_data=df_data.drop(index=key_wrds)
                        df_infos_dict[keys]=df_data
                    
                    Merged_df_dict=Dict()
                    comb1 = list(combinations(list(Merg_filepath.keys()), 1))
                    comb2 = list(combinations(list(Merg_filepath.keys()), 2))
                    if args.ADDUtt_feature==True: # output only Utt features
                        OutPklpath=merge_out_path+ "Utt_feature.pkl"
                        #!!!!!!!!!!!!!!!!!  注意這邊我們人為移除一些資訊量太少的人
                        df_data=Utt_featuresCombinded_dict[role].copy()
                        for key_wrds in args.exclude_people:
                            if df_data.index.str.contains(key_wrds).any():
                                df_data=df_data.drop(index=key_wrds)
                        pickle.dump(df_data,open(OutPklpath,"wb"))    
                        # pickle.dump(Utt_featuresCombinded_dict[role],open(OutPklpath,"wb"))
                    for c in comb1:
                        e1=c[0]

                        if args.ADDUtt_feature==True: # Merge with Utt features
                            Merged_df_dict[e1]=Merge_dfs(df_infos_dict[e1],Utt_featuresCombinded_dict[role])
                            OutPklpath=merge_out_path+"Utt_feature+"+ e1 + ".pkl"
                        else:
                            Merged_df_dict[e1]=df_infos_dict[e1]
                            OutPklpath=merge_out_path+ e1 + ".pkl"
                        pickle.dump(Merged_df_dict[e1],open(OutPklpath,"wb"))
                        # print(len(Merged_df_dict[e1]))
                        # assert Merged_df_dict[e1].isna().any().any() !=True
                    for c in comb2:
                        e1, e2=c
                        
                        if args.ADDUtt_feature==True:
                            Merged_df_dict['+'.join(c)]=Merge_dfs(df_infos_dict[e1],df_infos_dict[e2])
                            Merged_df_dict['+'.join(c)]=Merge_dfs(Merged_df_dict['+'.join(c)],Utt_featuresCombinded_dict[role])
                            OutPklpath=merge_out_path+"Utt_feature+"+'+'.join(c)+".pkl"
                        else:
                            Merged_df_dict['+'.join(c)]=Merge_dfs(df_infos_dict[e1],df_infos_dict[e2])
                            OutPklpath=merge_out_path+'+'.join(c)+".pkl"
                        pickle.dump(Merged_df_dict['+'.join(c)],open(OutPklpath,"wb"))
                        # print(len(Merged_df_dict['+'.join(c)]))
                    # Condition for : Columns_comb3 = All possible LOC feature combination + phonation_proximity_col
                    c = ('static_feautre_LOC', 'dynamic_feature_LOC', 'dynamic_feature_phonation')
                    e1, e2, e3=c
                    Merged_df_dict['+'.join(c)]=Merge_dfs(df_infos_dict[e1],df_infos_dict[e2])
                    Merged_df_dict['+'.join(c)]=Merge_dfs(Merged_df_dict['+'.join(c)],df_infos_dict[e3])
                    OutPklpath=merge_out_path+'+'.join(c)+".pkl"
                    pickle.dump(Merged_df_dict['Utt_feature'+'+'+'+'.join(c)],open(OutPklpath,"wb"))
