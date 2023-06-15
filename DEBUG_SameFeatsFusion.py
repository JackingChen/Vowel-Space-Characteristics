#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:06:32 2023

@author: jack
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 10:26:05 2023

@author: jack
"""

import os, sys
import pandas as pd
import numpy as np

import glob
import pickle

from addict import Dict
# import functions
import argparse
# import torch


from articulation.HYPERPARAM import phonewoprosody, Label
from articulation.HYPERPARAM.PeopleSelect import SellectP_define
import articulation.HYPERPARAM.FeatureSelect as FeatSel
import articulation.HYPERPARAM.PaperNameMapping as PprNmeMp
from itertools import combinations, dropwhile
import articulation.articulation
import copy
def Assert_labelfeature(feat_name,lab_name):
    # =============================================================================
    #     To check if the label match with feature
    # =============================================================================
    for i,n in enumerate(feat_name):
        assert feat_name[i] == lab_name[i]

def FilterFile_withinManualName(files,Manual_choosen_feature):
    files_manualChoosen=[f  for f in files if os.path.basename(f).split(".")[0]  in Manual_choosen_feature]
    return files_manualChoosen

def Merge_dfs(df_1, df_2):
    # print("before merge: df_1=", df_1)
    # print("before merge: df_2=", df_2)
    merged_df=pd.merge(df_1,df_2,left_index=True, right_index=True)
    # print("after merge: ",merged_df)
    return merged_df

def Add_label(df_formant_statistic,Label,label_choose='ADOS_cate_C'):
    for people in df_formant_statistic.index:
        bool_ind=Label.label_raw['name']==people
        df_formant_statistic.loc[people,label_choose]=Label.label_raw.loc[bool_ind,label_choose].values
    return df_formant_statistic

def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser()
    parser.add_argument('--Feature_mode', default='Customized_feature',
                        help='what kind of data you want to get')
    parser.add_argument('--preprocess', default=True,
                        help='')
    parser.add_argument('--start_point', default=-1,
                        help='In case that the program stop at certain point, we can resume the progress by setting this variable')
    parser.add_argument('--experiment', default='gop_exp_ADOShappyDAAIKidallDeceiptformosaCSRC',
                        help='If the mode is set to Session_phone_phf, you may need to determine the experiment used to generate the gop feature')
    parser.add_argument('--pseudo', default=False,
                        help='what kind of data you want to get')
    parser.add_argument('--suffix', default="",
                        help='what kind of data you want to get')
    parser.add_argument('--FS_method_str', default=None,
                        help='Feature selection')
    parser.add_argument('--UseManualCtxFeat', default=True,
                        help='')
    parser.add_argument('--Plot', default=True,
                        help='')
    parser.add_argument('--selectModelScoring', default='recall_macro',
                        help='[recall_macro,accuracy]')
    parser.add_argument('--Mergefeatures', default=True,
                        help='')
    parser.add_argument('--DEBUG', default=True,
                            help='')
    parser.add_argument('--knn_weights', default='uniform',
                            help='uniform distance')
    parser.add_argument('--knn_neighbors', default=2,  type=int,
                            help='path of the base directory')
    parser.add_argument('--Reorder_type', default='DKIndividual',
                            help='[DKIndividual, DKcriteria]')
    parser.add_argument('--Normalize_way', default='proposed',
                            help='')
    parser.add_argument('--FeatureComb_mode', default='Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation',
                            help='[Add_UttLvl_feature, feat_comb3, feat_comb5, feat_comb6,feat_comb7, baselineFeats,Comb_dynPhonation,Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation]')
    parser.add_argument('--ADDUtt_feature', default=False,
                            help='[DKIndividual, DKcriteria]')
    parser.add_argument('--exclude_people', default=['2015_12_07_02_003','2017_03_18_01_196_1'],
                        help='what kind of data you want to get')
    args = parser.parse_args()
    return args
args = get_args()
start_point=args.start_point
experiment=args.experiment
knn_weights=args.knn_weights
knn_neighbors=args.knn_neighbors
Reorder_type=args.Reorder_type

Session_level_all_cmps=Dict()
for Normalize_way in ['func1','func2','func3','func4','func7','proposed']:
    args.Normalize_way=Normalize_way
    Session_level_all=Dict()
    # Discriminative analysis Main
    columns=[
        'Divergence[within_covariance_norm(A:,i:,u:)]',
        'Divergence[within_variance_norm(A:,i:,u:)]',    
        'Divergence[between_covariance_norm(A:,i:,u:)]',    
        'Divergence[between_variance_norm(A:,i:,u:)]',    
        'Divergence[total_covariance_norm(A:,i:,u:)]', 
        'Divergence[total_variance_norm(A:,i:,u:)]',
        'Divergence[sam_wilks_lin_norm(A:,i:,u:)]',    
        'Divergence[pillai_lin_norm(A:,i:,u:)]',
        'Divergence[roys_root_lin_norm(A:,i:,u:)]',
        'Divergence[hotelling_lin_norm(A:,i:,u:)]',
        'Divergence[Between_Within_Det_ratio_norm(A:,i:,u:)]',
        'Divergence[Between_Within_Tr_ratio_norm(A:,i:,u:)]',
        'Divergence[within_covariance_norm(A:,i:,u:)]_var_p1',
        'Divergence[within_variance_norm(A:,i:,u:)]_var_p1',
        'Divergence[between_covariance_norm(A:,i:,u:)]_var_p1',
        'Divergence[between_variance_norm(A:,i:,u:)]_var_p1',
        'Divergence[total_covariance_norm(A:,i:,u:)]_var_p1', 
        'Divergence[total_variance_norm(A:,i:,u:)]_var_p1',
        'Divergence[sam_wilks_lin_norm(A:,i:,u:)]_var_p1',
        'Divergence[pillai_lin_norm(A:,i:,u:)]_var_p1',
        'Divergence[roys_root_lin_norm(A:,i:,u:)]_var_p1',
        'Divergence[hotelling_lin_norm(A:,i:,u:)]_var_p1',
        'Divergence[Between_Within_Det_ratio_norm(A:,i:,u:)]_var_p1',
        'Divergence[Between_Within_Tr_ratio_norm(A:,i:,u:)]_var_p1',
        'Divergence[within_covariance_norm(A:,i:,u:)]_var_p2',    
        'Divergence[within_variance_norm(A:,i:,u:)]_var_p2',    
        'Divergence[between_covariance_norm(A:,i:,u:)]_var_p2',    
        'Divergence[between_variance_norm(A:,i:,u:)]_var_p2',
        'Divergence[total_covariance_norm(A:,i:,u:)]_var_p2', 
        'Divergence[total_variance_norm(A:,i:,u:)]_var_p2',
        'Divergence[sam_wilks_lin_norm(A:,i:,u:)]_var_p2',
        'Divergence[pillai_lin_norm(A:,i:,u:)]_var_p2',
        'Divergence[roys_root_lin_norm(A:,i:,u:)]_var_p2',
        'Divergence[hotelling_lin_norm(A:,i:,u:)]_var_p2',
        'Divergence[Between_Within_Det_ratio_norm(A:,i:,u:)]_var_p2',
        'Divergence[Between_Within_Tr_ratio_norm(A:,i:,u:)]_var_p2',
        ]
    featuresOfInterest_manual=[ [col] for col in columns]
    # featuresOfInterest_manual=[ [col] + ['u_num+i_num+a_num'] for col in columns]
    label_choose=['ADOS_C']

    def MERGEFEATURES():
        # =============================================================================
        '''
    
            Feature merging function
            
            Ths slice of code provide user to manually make functions to combine df_XXX_infos
    
        '''
        # =============================================================================
        # dataset_role='ASD_DOCKID'
        for dataset_role in ['ASD_DOCKID','TD_DOCKID']:
            Merg_filepath={}
            Merg_filepath['static_feautre_LOC']='Features/artuculation_AUI/Vowels/Formants/{Normalize_way}/Formant_AUI_tVSAFCRFvals_KID_From{dataset_role}.pkl'.format(dataset_role=dataset_role,Normalize_way=args.Normalize_way)
            # Merg_filepath['static_feautre_phonation']='Features/artuculation_AUI/Vowels/Phonation/Phonation_meanvars_KID_From{dataset_role}.pkl'.format(dataset_role=dataset_role)
            Merg_filepath['dynamic_feature_LOC']='Features/artuculation_AUI/Interaction/Formants/{Normalize_way}/Syncrony_measure_of_variance_DKIndividual_{dataset_role}.pkl'.format(dataset_role=dataset_role,Normalize_way=args.Normalize_way)
            Merg_filepath['dynamic_feature_phonation']='Features/artuculation_AUI/Interaction/Phonation/Syncrony_measure_of_variance_phonation_{dataset_role}.pkl'.format(dataset_role=dataset_role)
            
            merge_out_path='Features/ClassificationMerged_dfs/{Normalize_way}/{dataset_role}/'.format(
                knn_weights=knn_weights,
                knn_neighbors=knn_neighbors,
                Reorder_type=Reorder_type,
                dataset_role=dataset_role,
                Normalize_way=args.Normalize_way
                )
            if not os.path.exists(merge_out_path):
                os.makedirs(merge_out_path)
            
            df_infos_dict=Dict()
            for keys, paths in Merg_filepath.items():
                df_infos_dict[keys]=pickle.load(open(paths,"rb")).sort_index()
            
            Merged_df_dict=Dict()
            comb1 = list(combinations(list(Merg_filepath.keys()), 1))
            comb2 = list(combinations(list(Merg_filepath.keys()), 2))
            for c in comb1:
                e1=c[0]
                Merged_df_dict[e1]=df_infos_dict[e1]
                OutPklpath=merge_out_path+ e1 + ".pkl"
                pickle.dump(Merged_df_dict[e1],open(OutPklpath,"wb"))
                
                
            for c in comb2:
                e1, e2=c
                Merged_df_dict['+'.join(c)]=Merge_dfs(df_infos_dict[e1],df_infos_dict[e2])
                
                OutPklpath=merge_out_path+'+'.join(c)+".pkl"
                pickle.dump(Merged_df_dict['+'.join(c)],open(OutPklpath,"wb"))
            # Condition for : Columns_comb3 = All possible LOC feature combination + phonation_proximity_col
            c = ('static_feautre_LOC', 'dynamic_feature_LOC', 'dynamic_feature_phonation')
            e1, e2, e3=c
            
            Merged_df_dict['+'.join(c)]=Merge_dfs(df_infos_dict[e1],df_infos_dict[e2])
            Merged_df_dict['+'.join(c)]=Merge_dfs(Merged_df_dict['+'.join(c)],df_infos_dict[e3])
            # Merged_df_dict['+'.join(c)]=Merge_dfs(Merged_df_dict['+'.join(c)],Utt_featuresCombinded_dict[role])
            OutPklpath=merge_out_path+'+'.join(c)+".pkl"
            pickle.dump(Merged_df_dict['+'.join(c)],open(OutPklpath,"wb"))
    if args.Mergefeatures:
        MERGEFEATURES()
    
    df_formant_statistics_CtxPhone_collect_dict=Dict()
    # =============================================================================
    
    class ADOSdataset():
        def __init__(self,knn_weights,knn_neighbors,Reorder_type,FeatureComb_mode):
            self.featurepath='Features'            
            self.N=2
            self.LabelType=Dict()
            self.LabelType['ADOS_C']='regression'
            self.LabelType['ADOS_cate_C']='classification'
            self.LabelType['ASDTD']='classification'
            self.Fractionfeatures_str='Features/artuculation_AUI/Vowels/Fraction/*.pkl'    
            self.FeatureComb_mode=FeatureComb_mode
            if self.FeatureComb_mode == 'Add_UttLvl_feature':
                self.File_root_path='Features/ClassificationMerged_dfs/{Normalize_way}/ADDed_UttFeat/{knn_weights}_{knn_neighbors}_{Reorder_type}/'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,Reorder_type=Reorder_type,Normalize_way=args.Normalize_way)
                self.Merge_feature_path=self.File_root_path+'{dataset_role}/*.pkl'.format(dataset_role='ASD_DOCKID')
            else:
                self.File_root_path='Features/ClassificationMerged_dfs/{Normalize_way}/'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,Reorder_type=Reorder_type,Normalize_way=args.Normalize_way)
                self.Merge_feature_path=self.File_root_path+'{dataset_role}/*.pkl'.format(dataset_role='ASD_DOCKID')
            
            self.Top_ModuleColumn_mapping_dict={}
            self.Top_ModuleColumn_mapping_dict['Add_UttLvl_feature']=FeatSel.Columns_comb2.copy()
            self.Top_ModuleColumn_mapping_dict['feat_comb']=FeatSel.Columns_comb.copy()
            self.Top_ModuleColumn_mapping_dict['feat_comb3']=FeatSel.Columns_comb3.copy()
            self.Top_ModuleColumn_mapping_dict['feat_comb5']=FeatSel.Columns_comb5.copy()
            self.Top_ModuleColumn_mapping_dict['feat_comb6']=FeatSel.Columns_comb6.copy()
            self.Top_ModuleColumn_mapping_dict['feat_comb7']=FeatSel.Columns_comb7.copy()
            self.Top_ModuleColumn_mapping_dict['Comb_dynPhonation']=FeatSel.Comb_dynPhonation.copy()
            self.Top_ModuleColumn_mapping_dict['Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation']=FeatSel.Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation.copy()
            
            self.Top_ModuleColumn_mapping_dict['baselineFeats']=FeatSel.Baseline_comb.copy()
            
            self.FeatureCombs_manual=Dict()
            self.FeatureCombs_manual['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_kid']=['df_formant_statistic_TD_normal_kid', 'df_formant_statistic_agesexmatch_ASDSevereGrp_kid']
            self.FeatureCombs_manual['TD_normal vs ASDMild_agesexmatch >> FeatLOC_kid']=['df_formant_statistic_TD_normal_kid', 'df_formant_statistic_agesexmatch_ASDMildGrp_kid']
            self.FeatureCombs_manual['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_doc']=['df_formant_statistic_TD_normal_doc', 'df_formant_statistic_agesexmatch_ASDSevereGrp_doc']
            self.FeatureCombs_manual['TD_normal vs ASDMild_agesexmatch >> FeatLOC_doc']=['df_formant_statistic_TD_normal_doc', 'df_formant_statistic_agesexmatch_ASDMildGrp_doc']
            self.FeatureCombs_manual['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_DKRatio']=['df_formant_statistic_TD_normalGrp_DKRatio', 'df_formant_statistic_agesexmatch_ASDSevereGrp_DKRatio']
            self.FeatureCombs_manual['TD_normal vs ASDMild_agesexmatch >> FeatLOC_DKRatio']=['df_formant_statistic_TD_normalGrp_DKRatio', 'df_formant_statistic_agesexmatch_ASDMildGrp_DKRatio']
            
            self.FeatureCombs_manual['TD_normal vs ASDSevere_agesexmatch >> FeatCoor']=['df_syncrony_statistic_TD_normalGrp', 'df_syncrony_statistic_agesexmatch_ASDSevereGrp']
            self.FeatureCombs_manual['TD_normal vs ASDMild_agesexmatch >> FeatCoor']=['df_syncrony_statistic_TD_normalGrp', 'df_syncrony_statistic_agesexmatch_ASDMildGrp']
            # self.FeatureCombs_manual['Notautism vs ASD']=['df_formant_statistic_77_Notautism', 'df_formant_statistic_77_ASD']
            # self.FeatureCombs_manual['ASD vs Autism']=['df_formant_statistic_77_ASD', 'df_formant_statistic_77_Autism']
            # self.FeatureCombs_manual['Notautism vs Autism']=['df_formant_statistic_77_Notautism', 'df_formant_statistic_77_Autism']
        
            # self._FeatureBuild_single()
            self._FeatureBuild_Module()
        def Get_FormantAUI_feat(self,label_choose,pickle_path,featuresOfInterest=['MSB_f1','MSB_f2','MSB_mix'],filterbyNum=True,**kwargs):
            arti=articulation.articulation.Articulation()
            #如果path有放的話字串的話，就使用path的字串，不然就使用「feat_」等於的東西，在function裡面會以kwargs的形式出現
            if not kwargs and len(pickle_path)>0:
                df_tmp=pickle.load(open(pickle_path,"rb")).sort_index()
            elif len(kwargs)>0: # usage Get_FormantAUI_feat(...,key1=values1):
                for k, v in kwargs.items(): #there will be only one element
                    df_tmp=kwargs[k].sort_index()
    
            if filterbyNum:
                df_tmp=arti.BasicFilter_byNum(df_tmp,N=self.N)
    
            if label_choose not in df_tmp.columns:
                for people in df_tmp.index:
                    lab=Label.label_raw[label_choose][Label.label_raw['name']==people]
                    df_tmp.loc[people,'ADOS']=lab.values
                df_y=df_tmp['ADOS'] #Still keep the form of dataframe
            else:
                df_y=df_tmp[label_choose] #Still keep the form of dataframe
            
            
            feature_array=df_tmp[featuresOfInterest]
            
                
            LabType=self.LabelType[label_choose]
            return feature_array, df_y, LabType
        def _FeatureBuild_single(self):
            Features=Dict()
            Features_comb=Dict()
            files = glob.glob(self.Fractionfeatures_str)
            for file in files:
                feat_name=os.path.basename(file).replace(".pkl","")
                df_tmp=pickle.load(open(file,"rb")).sort_index()
                Features[feat_name]=df_tmp
            for keys in self.FeatureCombs_manual.keys():
                combF=[Features[k] for k in self.FeatureCombs_manual[keys]]
                Features_comb[keys]=pd.concat(combF)
            
            self.Features_comb_single=Features_comb
        def _FeatureBuild_Module(self):
            Labels_add=['ASDTD']
            ModuledFeatureCombination=self.Top_ModuleColumn_mapping_dict[self.FeatureComb_mode]
            
            sellect_people_define=SellectP_define()
            #Loading features from ASD        
            Features_comb=Dict()
            IterateFilesFullPaths = glob.glob(self.Merge_feature_path)
            
    
            if self.FeatureComb_mode in ['feat_comb3','feat_comb5','feat_comb6','feat_comb7','Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation']:
                DfCombFilenames=['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation.pkl']
            elif self.FeatureComb_mode == 'Comb_dynPhonation':
                DfCombFilenames=['dynamic_feature_phonation.pkl']
            elif self.FeatureComb_mode == 'baselineFeats':
                 DfCombFilenames=['{}.pkl'.format(Dataname) for Dataname in ModuledFeatureCombination.keys()]
            else:
                DfCombFilenames=[os.path.basename(f) for f in IterateFilesFullPaths]
            File_ASD_paths=[self.File_root_path+"ASD_DOCKID/"+f for f in DfCombFilenames]
            File_TD_paths=[self.File_root_path+"TD_DOCKID/"+f for f in DfCombFilenames]
            
            
            df_Top_Check_length=pd.DataFrame()
            for file_ASD, file_TD in zip(File_ASD_paths,File_TD_paths):
                if not os.path.exists(file_ASD) or not os.path.exists(file_TD):
                    raise FileExistsError()
                
                assert os.path.basename(file_ASD) == os.path.basename(file_TD)
                filename=os.path.basename(file_ASD)
                k_FeatTypeLayer1=filename.replace(".pkl","")
                df_feature_ASD=pickle.load(open(file_ASD,"rb")).sort_index()
                df_feature_TD=pickle.load(open(file_TD,"rb")).sort_index()
                df_feature_ASD['ASDTD']=sellect_people_define.ASDTD_label['ASD']
                df_feature_TD['ASDTD']=sellect_people_define.ASDTD_label['TD']
                
                # ADD label
                df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='ADOS_cate_CSS')
                df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='age_year')
                df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='sex')
                # create different ASD cohort
                filter_Minimal_TCSS=df_feature_ASD['ADOS_cate_CSS']==0
                filter_low_TCSS=df_feature_ASD['ADOS_cate_CSS']==1
                filter_moderate_TCSS=df_feature_ASD['ADOS_cate_CSS']==2
                filter_high_TCSS=df_feature_ASD['ADOS_cate_CSS']==3
                
                df_feauture_ASDgrp_dict={}
                df_feauture_ASDgrp_dict['df_feature_ASD']=df_feature_ASD
                
                # df_feauture_ASDgrp_dict['df_feature_Minimal_CSS']=df_feature_ASD[filter_Minimal_TCSS]
                # df_feauture_ASDgrp_dict['df_feature_low_CSS']=df_feature_ASD[filter_low_TCSS]
                # df_feauture_ASDgrp_dict['df_feature_moderate_CSS']=df_feature_ASD[filter_moderate_TCSS]
                # df_feauture_ASDgrp_dict['df_feature_high_CSS']=df_feature_ASD[filter_high_TCSS]
                df_feauture_ASDgrp_dict['df_feature_lowMinimal_CSS']=df_feature_ASD[filter_low_TCSS | filter_Minimal_TCSS]
                df_feauture_ASDgrp_dict['df_feature_moderate_CSS']=df_feature_ASD[filter_moderate_TCSS]
                df_feauture_ASDgrp_dict['df_feature_high_CSS']=df_feature_ASD[filter_high_TCSS]
    
                #Check the length of each paired comparison, should be stored on the top of for loop
                Tmp_Numcmp_dict={}
                for key in df_feauture_ASDgrp_dict.keys():
                    Numcmp_str='ASD({0}) vs TD({1})'.format(len(df_feauture_ASDgrp_dict[key]),len(df_feature_TD))
                    Tmp_Numcmp_dict[key]=Numcmp_str
                    
                
                df_Tmp_Numcmp_list=pd.DataFrame.from_dict(Tmp_Numcmp_dict,orient='index')
                df_Tmp_Numcmp_list.columns=[k_FeatTypeLayer1]
    
                if len(df_Top_Check_length)==0:
                    df_Top_Check_length=df_Tmp_Numcmp_list
                else:
                    df_Top_Check_length=Merge_dfs(df_Top_Check_length,df_Tmp_Numcmp_list)
                # 手動執行到這邊，從for 上面
                for k_FeatTypeLayer2 in ModuledFeatureCombination[k_FeatTypeLayer1].keys():
                    colums_sel=ModuledFeatureCombination[k_FeatTypeLayer1][k_FeatTypeLayer2]
                    
    
                    # 1. Set ASD vs TD experiment
                    for k_ASDgrp in df_feauture_ASDgrp_dict.keys():
                        df_ASD_subgrp=df_feauture_ASDgrp_dict[k_ASDgrp].copy()[colums_sel+Labels_add]
                        df_TD_subgrp=df_feature_TD.copy()[colums_sel+Labels_add]
                        
                        experiment_str="{TD_name} vs {ASD_name} >> {feature_type}".format(TD_name='TD',ASD_name=k_ASDgrp,feature_type=k_FeatTypeLayer2)
                        Features_comb[experiment_str]=pd.concat([df_ASD_subgrp,df_TD_subgrp],axis=0)
                    # 2. Set ASDsevere vs ASDmild experiment
                    # experiment_str="{ASDsevere_name} vs {ASDmild_name} >> {feature_type}".format(ASDsevere_name='df_feature_moderatehigh_CSS',ASDmild_name='df_feature_lowMinimal_CSS',feature_type=k_FeatTypeLayer2)
                    # df_ASDsevere_subgrp=df_feauture_ASDgrp_dict['df_feature_moderatehigh_CSS'].copy()
                    # df_ASDmild_subgrp=df_feauture_ASDgrp_dict['df_feature_lowMinimal_CSS'].copy()
                    # df_ASDsevere_subgrp['ASDsevereMild']=sellect_people_define.ASDsevereMild_label['ASDsevere']
                    # df_ASDmild_subgrp['ASDsevereMild']=sellect_people_define.ASDsevereMild_label['ASDmild']
                    # Features_comb[experiment_str]=pd.concat([df_ASDsevere_subgrp,df_ASDmild_subgrp],axis=0)
            self.Features_comb_multi=Features_comb
    
    # =============================================================================
    '''
    
        Feature merging function
        
        Ths slice of code provide user to manually make functions to combine df_XXX_infos
    
    '''
    # =============================================================================
    
    ados_ds=ADOSdataset(knn_weights,knn_neighbors,Reorder_type,FeatureComb_mode=args.FeatureComb_mode)
    ErrorFeat_bookeep=Dict()
    
    
    FeatureLabelMatch_manual=[
    
        ['TD vs df_feature_lowMinimal_CSS >> LOC_columns+Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols', 'ASDTD'],
        ['TD vs df_feature_lowMinimal_CSS >> LOC_columns+Phonation_Trend_K_cols+Phonation_Syncrony_cols', 'ASDTD'],
        ['TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols', 'ASDTD'],
        ['TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_K_cols+Phonation_Syncrony_cols', 'ASDTD'],
    
    
        ['TD vs df_feature_moderate_CSS >> LOCDEP_Trend_D_cols+Phonation_Proximity_cols', 'ASDTD'],
        ['TD vs df_feature_moderate_CSS >> Phonation_Proximity_cols', 'ASDTD'],
        
    
        ['TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_K_cols+Phonation_Syncrony_cols', 'ASDTD'],
        ['TD vs df_feature_high_CSS >> DEP_columns+Phonation_Trend_D_cols+Phonation_Proximity_cols', 'ASDTD'],
        ['TD vs df_feature_high_CSS >> Phonation_Trend_D_cols+Phonation_Proximity_cols', 'ASDTD'],
        ['TD vs df_feature_high_CSS >> DEP_columns+Phonation_Proximity_cols', 'ASDTD'],
        ['TD vs df_feature_high_CSS >> Phonation_Proximity_cols', 'ASDTD'],
    
    
     ]
        
    # FeatSel 掌管該出現的columns
    # ados_ds.Features_comb_multi 掌管load進來的data
    
    
    Top_ModuleColumn_mapping_dict={}
    Top_ModuleColumn_mapping_dict['Add_UttLvl_feature']={ e2_str:FeatSel.Columns_comb2[e_str][e2_str] for e_str in FeatSel.Columns_comb2.keys() for e2_str in FeatSel.Columns_comb2[e_str].keys()}
    Top_ModuleColumn_mapping_dict['feat_comb3']=ModuleColumn_mapping={ e2_str:FeatSel.Columns_comb3[e_str][e2_str] for e_str in FeatSel.Columns_comb3.keys() for e2_str in FeatSel.Columns_comb3[e_str].keys()}
    Top_ModuleColumn_mapping_dict['feat_comb5']=ModuleColumn_mapping={ e2_str:FeatSel.Columns_comb5[e_str][e2_str] for e_str in FeatSel.Columns_comb5.keys() for e2_str in FeatSel.Columns_comb5[e_str].keys()}
    Top_ModuleColumn_mapping_dict['feat_comb6']=ModuleColumn_mapping={ e2_str:FeatSel.Columns_comb6[e_str][e2_str] for e_str in FeatSel.Columns_comb6.keys() for e2_str in FeatSel.Columns_comb6[e_str].keys()}
    Top_ModuleColumn_mapping_dict['feat_comb7']=ModuleColumn_mapping={ e2_str:FeatSel.Columns_comb7[e_str][e2_str] for e_str in FeatSel.Columns_comb7.keys() for e2_str in FeatSel.Columns_comb7[e_str].keys()}
    Top_ModuleColumn_mapping_dict['Comb_dynPhonation']=ModuleColumn_mapping={ e2_str:FeatSel.Comb_dynPhonation[e_str][e2_str] for e_str in FeatSel.Comb_dynPhonation.keys() for e2_str in FeatSel.Comb_dynPhonation[e_str].keys()}
    Top_ModuleColumn_mapping_dict['Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation']=ModuleColumn_mapping={ e2_str:FeatSel.Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation[e_str][e2_str] for e_str in FeatSel.Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation.keys() for e2_str in FeatSel.Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation[e_str].keys()}
    Top_ModuleColumn_mapping_dict['feat_comb']=ModuleColumn_mapping={ e2_str:FeatSel.Columns_comb[e_str][e2_str] for e_str in FeatSel.Columns_comb.keys() for e2_str in FeatSel.Columns_comb[e_str].keys()}
    
    
    ModuleColumn_mapping=Top_ModuleColumn_mapping_dict[args.FeatureComb_mode]
    # =============================================================================
    '''
    
        Here starts to load features to Session_level_all dict
    
    '''
    # =============================================================================
    for exp_str, lab_ in FeatureLabelMatch_manual:
        comparison_pair=exp_str.split(" >> ")[0]
        ModuleColumn_str=exp_str.split(" >> ")[-1]
        featuresOfInterest=[ModuleColumn_mapping[ModuleColumn_str]]
        # feat_=key
        for feat_col in featuresOfInterest:
            feat_col_ = list(feat_col) # ex: ['MSB_f1']
            if len(feat_col) > 144: # 144 is the limit of the filename
                key=feat_col_
            else:
                key=[feat_col_]
            # X,y, featType=ados_ds.Get_FormantAUI_feat(\
            #     label_choose=lab_,pickle_path='',featuresOfInterest=feat_col_,filterbyNum=False,\
            #     feat_=ados_ds.Features_comb_single[feat_])
            
            X,y, featType=ados_ds.Get_FormantAUI_feat(\
                label_choose=lab_,pickle_path='',featuresOfInterest=feat_col,filterbyNum=False,\
                feat_=ados_ds.Features_comb_multi[exp_str])
            
            if X.isnull().values.any() or y.isnull().values.any():
                print("Feat: ",key,'Contains nan')
                ErrorFeat_bookeep['{0} {1} {2}'.format(exp_str,lab_,key)].X=X
                ErrorFeat_bookeep['{0} {1} {2}'.format(exp_str,lab_,key)].y=y
                continue
            
            Item_name="{feat}::{lab}".format(feat=' >> '.join([comparison_pair,ModuleColumn_str]),lab=lab_)
            Session_level_all[Item_name].X, \
                Session_level_all[Item_name].y, \
                    Session_level_all[Item_name].feattype = X,y, featType
    
    
    # pickle.dump(Session_level_all,open("Session_level_all_class_new.pkl","wb"))
    print("\n\n\n\n")
    Session_level_all_cmps[Normalize_way]=copy.deepcopy(Session_level_all)
    
    Session_feature_out="Session_feature_out/"
    if not os.path.exists(Session_feature_out):
        os.makedirs(Session_feature_out)
    
    pickle.dump(Session_level_all,open(f"{Session_feature_out}/Session_level_all[{args.Normalize_way}]",'wb'))
    
#%%
import pandas as pd

def compare_dicts(dict1, dict2, path="", includekey=[]):
    if set(dict1.keys()) != set(dict2.keys()):
        print(f"Key mismatch at path: {path}")
        return False

    for key in dict1:
        if "TD vs df_feature_ASD" in key:
            continue
        if key not in includekey and len(includekey) != 0:
            continue
        value1 = dict1[key]
        value2 = dict2[key]
        current_path = f"{path}.{key}" if path else key

        if isinstance(value1, dict) and isinstance(value2, dict):
            if not compare_dicts(value1, value2, path=current_path):
                return False
        elif isinstance(value1, (pd.DataFrame, pd.Series)) and isinstance(value2, (pd.DataFrame, pd.Series)):
            if not compare_data_objects(value1, value2, path=current_path):
                return False
        elif value1 != value2:
            print(f"Value mismatch at path: {current_path}")
            return False

    return True

def compare_data_objects(obj1, obj2, path, tolarence=1e-5):
    cond=((obj1 - obj2)<tolarence).all(axis=None)
    if isinstance(obj1, pd.DataFrame) and isinstance(obj2, pd.DataFrame):
        if not cond:
            print(f"DataFrame mismatch at path: {path}")
            return False
    elif isinstance(obj1, pd.Series) and isinstance(obj2, pd.Series):
        if not cond:
            print(f"Series mismatch at path: {path}")
            return False
    else:
        print(f"Value type mismatch at path: {path}")
        return False

    return True

includekey=[
    # 'TD vs df_feature_lowMinimal_CSS >> LOC_columns+Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD',
    # 'TD vs df_feature_lowMinimal_CSS >> LOC_columns+Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD',
    'TD vs df_feature_moderate_CSS >> LOCDEP_Trend_D_cols+Phonation_Proximity_cols::ASDTD',
    'TD vs df_feature_high_CSS >> DEP_columns+Phonation_Trend_D_cols+Phonation_Proximity_cols::ASDTD',
    'TD vs df_feature_high_CSS >> DEP_columns+Phonation_Proximity_cols::ASDTD'
    ]

compare_dicts(Session_level_all_cmps['func1'],Session_level_all_cmps['func2'], includekey=includekey)
Inspect_key="TD vs df_feature_moderate_CSS >> LOCDEP_Trend_D_cols+Phonation_Proximity_cols::ASDTD"
diff12_df=Session_level_all_cmps['func1'][Inspect_key].X -\
  Session_level_all_cmps['func2'][Inspect_key].X