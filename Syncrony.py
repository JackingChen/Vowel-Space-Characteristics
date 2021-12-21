#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:55:32 2021

@author: jackchen
"""
import pandas as pd
import numpy as np

class Syncrony:
    def __init__(self, ):
        self.label_choose_lst=['ADOS_C']
        self.MinNumTimeSeries=3
    def calculate_features(self,df_person_segment_feature_dict, df_person_half_feature_dict,\
                           features,PhoneOfInterest_str,\
                           Inspect_roles, Label,\
                           MinNumTimeSeries=3, label_choose_lst=['ADOS_C']):
        # =============================================================================
        '''
        
            This function calculates the coordinative features in TBME2021
            
            Input: df_person_segment_feature_dict, df_person_half_feature_dict
            Output: df_syncrony_measurement
        
        '''         
        # =============================================================================
        
        
        p_1=Inspect_roles[0]
        p_2=Inspect_roles[1]
        df_syncrony_measurement=pd.DataFrame()
        for people in df_person_segment_feature_dict.keys():
            if len(df_person_segment_feature_dict[people][p_1])<MinNumTimeSeries or len(df_person_segment_feature_dict[people][p_2])<MinNumTimeSeries:
                continue
            
            RESULT_dict={}
            
            person_0=df_person_segment_feature_dict[people][p_1]
            person_1=df_person_segment_feature_dict[people][p_2]
            
            assert len(person_0) == len(person_1)
            
            person_0_half=df_person_half_feature_dict[people][p_1]
            person_1_half=df_person_half_feature_dict[people][p_2]
            
            timeSeries_len=len(person_0)
            RESULT_dict['timeSeries_len[{}]'.format(PhoneOfInterest_str)]=timeSeries_len
            for label_choose in label_choose_lst:
                RESULT_dict[label_choose]=Label.label_raw[Label.label_raw['name']==people][label_choose].values[0]
            
            for col in features:
                # Calculate the syncrony
                timeseries_0=person_0[col]
                timeseries_1=person_1[col]
                
                timeseries_0.name=p_1
                timeseries_1.name=p_2
                
                RESULT_dict['Average[{}]_p1'.format(col)]=np.mean(timeseries_0)
                RESULT_dict['Average[{}]_p2'.format(col)]=np.mean(timeseries_1)
                
                df_timeseries_pairs=pd.concat([timeseries_0,timeseries_1],axis=1)
                r=df_timeseries_pairs.corr().loc[p_1,p_2]
                
                RESULT_dict['Syncrony[{}]'.format(col)]=r # The syncrony feature
                
        
                # Calculate the distance
                pairwise_dist = (timeseries_0 - timeseries_1).abs()
                pairwise_split_dist = (person_0_half[col] - person_1_half[col]).abs()
                assert len(pairwise_split_dist) == 2
                # cmpFirst_dist = (timeseries_1 - timeseries_0.iloc[0]).abs()
                
                # Divergence features
                Time=range(len(timeseries_0))
                r_pairwise_dist=np.corrcoef(Time, pairwise_dist)[0,1]
                # r_cmpFirst_dist=np.corrcoef(Time, cmpFirst_dist)[0,1]
                Dist_split_divergence=pairwise_split_dist[-1] - pairwise_split_dist[0]
                
                r_Variance_divergence_p1=np.corrcoef(Time, timeseries_0)[0,1]
                r_Variance_divergence_p2=np.corrcoef(Time, timeseries_1)[0,1]
                
                # Variance features
                var_p1=np.var(timeseries_0)
                var_p2=np.var(timeseries_1)
                var_pairwise_dist=np.var((timeseries_0 - timeseries_1))
                
                RESULT_dict['Divergence[{}]'.format(col)]=r_pairwise_dist
                assert not np.isnan(r_pairwise_dist)
                # RESULT_dict['Divergence[{}]_cmp1st'.format(col)]=r_cmpFirst_dist
                RESULT_dict['Divergence[{}]_split'.format(col)]=Dist_split_divergence
                RESULT_dict['Divergence[{}]_var_p1'.format(col)]=r_Variance_divergence_p1
                RESULT_dict['Divergence[{}]_var_p2'.format(col)]=r_Variance_divergence_p2
                
                # Average features  (is used in Discriminative analysis )
                RESULT_dict['Average[{}]'.format(col)]=np.mean(pairwise_dist)
                RESULT_dict['Average[{}]_split'.format(col)]=np.mean(pairwise_split_dist)
                # RESULT_dict['Variance[{}]_p1'.format(col)]=var_p1
                # RESULT_dict['Variance[{}]_p2'.format(col)]=var_p2
                # RESULT_dict['Variance[{}]_distp1p2'.format(col)]=var_pairwise_dist
                
            # Turn dict to dataframes
            df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict,orient='index').T
            df_RESULT_list.index=[people]
            df_syncrony_measurement=df_syncrony_measurement.append(df_RESULT_list)
            
        self.df_syncrony_measurement=df_syncrony_measurement
        contain_nan_bool=self._checknan()
 
        return df_syncrony_measurement
    def _checknan(self):
        lst=[]
        for col in self.df_syncrony_measurement.columns:
            if self.df_syncrony_measurement[col].isnull().values.any():
                lst.append(col)
        if len(lst) > 0:
            # print("There are nans containing in df_syncrony_measurement")
            # print(lst)
            return True
        else:
            return False
    def _dropnan(self):
        lst=[]
        for col in self.df_syncrony_measurement.columns:
            if self.df_syncrony_measurement[col].isnull().values.any():
                lst.append(col)
        if len(lst) > 0:
            print('dropped columns: ', lst)
            self.df_syncrony_measurement=self.df_syncrony_measurement.drop(columns=lst)
        return self.df_syncrony_measurement