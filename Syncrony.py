#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:55:32 2021

@author: jackchen
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import special, stats
from sklearn import neighbors
from addict import Dict
from scipy.stats import spearmanr,pearsonr 
import warnings


from numpy.testing import assert_equal, assert_allclose, assert_warns

from scipy import special
from scipy import linalg
class ConstantInputWarning(RuntimeWarning):

    def __init__(self, msg=None):
        if msg is None:
            msg = ("An input array is constant; the correlation coefficent "
                   "is not defined.")
        self.args = (msg,)


class NearConstantInputWarning(RuntimeWarning):

    def __init__(self, msg=None):
        if msg is None:
            msg = ("An input array is nearly constant; the computed "
                   "correlation coefficent may be inaccurate.")
        self.args = (msg,)
class Syncrony:
    def __init__(self, ):
        self.label_choose_lst=['ADOS_C']
        self.MinNumTimeSeries=3
        self.st_col_str='IPU_st'  #It is related to global info
        self.ed_col_str='IPU_ed'  #It is related to global info
        self.Colormap_role_dict=Dict()
        self.Colormap_role_dict['D']='orange'
        self.Colormap_role_dict['K']='blue'
    def KNNFitting(self,df_person_segment_feature_DKIndividual_dict,\
                   col_choose,Inspect_roles,\
                   knn_weights='uniform',knn_neighbors=2,MinNumTimeSeries=3,\
                   st_col_str='IPU_st', ed_col_str='IPU_ed', aggressive_mode=False):
        p_1=Inspect_roles[0]
        p_2=Inspect_roles[1]
        
        functionDK_people=Dict()
        for people in df_person_segment_feature_DKIndividual_dict.keys():
            if not aggressive_mode:
                #Aggressive mode means that we don't want to skip any participant, if there is a value error, then make sure previous procedure doesn't generate unavailable data
                if len(df_person_segment_feature_DKIndividual_dict[people][p_1])<MinNumTimeSeries or len(df_person_segment_feature_DKIndividual_dict[people][p_2])<MinNumTimeSeries:
                    continue
            df_person_segment_feature_role_dict=df_person_segment_feature_DKIndividual_dict[people]  
            try:
                Totalendtime=min([df_person_segment_feature_role_dict[role][ed_col_str].values[-1]  for role in Inspect_roles])
            except:
                print("The people causing error happens at people: ",people)
                print("The problem file is ",df_person_segment_feature_role_dict)
                raise KeyError
            T = np.linspace(0, Totalendtime, int(Totalendtime))[:, np.newaxis]
            
            functionDK={}
            for role_choose in Inspect_roles:
                df_dynVals=df_person_segment_feature_role_dict[role_choose][col_choose]
                if not np.isnan(np.abs(stats.zscore(df_dynVals))).all(): 
                    # remove outlier that is falls over 3 times of the std
                    df_dynVals_deleteOutlier=df_dynVals[(np.abs(stats.zscore(df_dynVals)) < 3)]
                else:
                    #Situation for padding when the time series is too short
                    df_dynVals_deleteOutlier=df_dynVals
                df_stidx=df_person_segment_feature_role_dict[role_choose][st_col_str]
                df_edidx=df_person_segment_feature_role_dict[role_choose][ed_col_str]
                
                
                Mid_positions=[]
                for x_1 , x_2, y in zip(df_stidx.values ,df_edidx.values,df_dynVals_deleteOutlier.values):            
                    start_time=x_1
                    end_time=x_2
                    mid_time=(start_time+end_time)/2
                    Mid_positions.append(mid_time)    
                
                if aggressive_mode:
                    #Aggressive mode means that we don't want to skip any participant
                    knn_neighbors = min (knn_neighbors,len(df_dynVals_deleteOutlier))
                knn = neighbors.KNeighborsRegressor(knn_neighbors, weights=knn_weights)
                X, y=np.array(Mid_positions).reshape(-1,1), df_dynVals_deleteOutlier
                try:
                    y_ = knn.fit(X, y.values).predict(T)
                except ValueError:
                    print("Problem people happen at ", people, role_choose)
                    print("df_dynVals", df_dynVals)
                    print("==================================================")
                    print("df_dynVals_deleteOutlier", df_dynVals_deleteOutlier)
                    raise ValueError
                functionDK[role_choose]=y_
            functionDK['T']=T
            functionDK_people[people]=functionDK
        return functionDK_people
    
    
    
    
    
    def calculate_features_continuous_modulized(self,df_person_segment_feature_DKIndividual_dict,features,PhoneOfInterest_str,\
                               Inspect_roles, Label,\
                               knn_weights="uniform",knn_neighbors=2,\
                               MinNumTimeSeries=3, label_choose_lst=['ADOS_C'],\
                               Knn_aggressive_mode=True):
        df_basic_additional_info=self._Add_additional_info(df_person_segment_feature_DKIndividual_dict,Label,label_choose_lst,\
                                                     Inspect_roles, MinNumTimeSeries=MinNumTimeSeries,PhoneOfInterest_str=PhoneOfInterest_str)
        df_syncrony_measurement_merge=pd.DataFrame()
        for col in features:
            Col_continuous_function_DK=self.KNNFitting(df_person_segment_feature_DKIndividual_dict,\
                       col, Inspect_roles,\
                       knn_weights=knn_weights,knn_neighbors=knn_neighbors,MinNumTimeSeries=MinNumTimeSeries,\
                       st_col_str='IPU_st', ed_col_str='IPU_ed', aggressive_mode=Knn_aggressive_mode)
                
            df_syncrony_measurement_col=self._calculate_features_col(Col_continuous_function_DK,col)
            df_syncrony_measurement_merge=pd.concat([df_syncrony_measurement_merge,df_syncrony_measurement_col],axis=1)
        df_syncrony_measurement=pd.concat([df_basic_additional_info,df_syncrony_measurement_merge],axis=1)
        return df_syncrony_measurement
    
    
    def calculate_features_continuous(self,df_person_segment_feature_DKIndividual_dict,features,PhoneOfInterest_str,\
                           Inspect_roles, Label,\
                           knn_weights="uniform",knn_neighbors=2,\
                           MinNumTimeSeries=3, label_choose_lst=['ADOS_C'],plot=False):
        p_1=Inspect_roles[0]
        p_2=Inspect_roles[1]
        df_syncrony_measurement=pd.DataFrame()
        for people in df_person_segment_feature_DKIndividual_dict.keys():
            if len(df_person_segment_feature_DKIndividual_dict[people][p_1])<MinNumTimeSeries or len(df_person_segment_feature_DKIndividual_dict[people][p_2])<MinNumTimeSeries:
                continue
            df_person_segment_feature_role_dict=df_person_segment_feature_DKIndividual_dict[people]
            
            RESULT_dict={}
            
            
            # kNN fitting
            Totalendtime=min([df_person_segment_feature_role_dict[role][self.ed_col_str].values[-1]  for role in Inspect_roles])
            Mintimeserieslen=min([len(df_person_segment_feature_role_dict[role])  for role in Inspect_roles])
            T = np.linspace(0, Totalendtime, int(Totalendtime))[:, np.newaxis]
            
            RESULT_dict['timeSeries_len[{}]'.format(PhoneOfInterest_str)]=Mintimeserieslen
            for label_choose in label_choose_lst:
                RESULT_dict[label_choose]=Label.label_raw[Label.label_raw['name']==people][label_choose].values[0]
    
            for col in features:
                functionDK={}
                if plot==True:
                    fig, ax = plt.subplots()
                for role_choose in Inspect_roles:
                    df_dynVals=df_person_segment_feature_role_dict[role_choose][col]
                    # remove outlier that is falls over 3 times of the std
                    df_dynVals_deleteOutlier=df_dynVals[(np.abs(stats.zscore(df_dynVals)) < 3)]
                    df_stidx=df_person_segment_feature_role_dict[role_choose][self.st_col_str]
                    df_edidx=df_person_segment_feature_role_dict[role_choose][self.ed_col_str]
                    
                    
                    Mid_positions=[]
                    for x_1 , x_2, y in zip(df_stidx.values ,df_edidx.values,df_dynVals_deleteOutlier.values):
                        if plot==True:
                            ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,0.5,color=self.Colormap_role_dict[role_choose]))
                        
                        start_time=x_1
                        end_time=x_2
                        mid_time=(start_time+end_time)/2
                        Mid_positions.append(mid_time)    
                    if plot==True:
                        recWidth=df_dynVals_deleteOutlier.min()
                        # add an totally overlapped rectangle but it will show the label
                        ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,recWidth,color=self.Colormap_role_dict[role_choose],label=role_choose))
                
                    
                    knn = neighbors.KNeighborsRegressor(knn_neighbors, weights=knn_weights)
                    X, y=np.array(Mid_positions).reshape(-1,1), df_dynVals_deleteOutlier
                    try:
                        y_ = knn.fit(X, y.values).predict(T)
                    except ValueError:
                        print("Problem people happen at ", people, role_choose)
                        print("df_dynVals", df_dynVals)
                        print("==================================================")
                        print("df_dynVals_deleteOutlier", df_dynVals_deleteOutlier)
                        raise ValueError
                    functionDK[role_choose]=y_
                    
                    if plot==True:
                        plt.plot(y_,color=self.Colormap_role_dict[role_choose],alpha=0.5)
                if plot==True:
                    ax.autoscale()
                    plt.title(col)
                    plt.legend()
                    plt.show()
                    fig.clf()
                
                proximity=-np.abs(np.mean(functionDK['D'] - functionDK['K']))
                D_t=-np.abs(functionDK['D']-functionDK['K'])
        
                time=T.reshape(-1)
                Convergence=pearsonr(D_t,time)[0]
                Trend_D=pearsonr(functionDK['D'],time)[0]
                Trend_K=pearsonr(functionDK['K'],time)[0]
                delta=[-15, -10, -5, 0, 5, 10, 15]        
                syncron_lst=[]
                for d in delta:
                    if d < 0: #ex_ d=-15
                        f_d_shifted=functionDK['D'][-d:]
                        f_k_shifted=functionDK['K'][:d]
                    elif d > 0: #ex_ d=15
                        f_d_shifted=functionDK['D'][:-d]
                        f_k_shifted=functionDK['K'][d:]
                    else: #d=0
                        f_d_shifted=functionDK['D']
                        f_k_shifted=functionDK['K']
                    syncron_candidate=pearsonr(f_d_shifted,f_k_shifted)[0]
                    
                    syncron_lst.append(syncron_candidate)
                syncrony=syncron_lst[np.argmax(np.abs(syncron_lst))]
                
                RESULT_dict['Proximity[{}]'.format(col)]=proximity
                RESULT_dict['Trend[{}]_d'.format(col)]=Trend_D
                RESULT_dict['Trend[{}]_k'.format(col)]=Trend_K
                RESULT_dict['Convergence[{}]'.format(col)]=Convergence
                RESULT_dict['Syncrony[{}]'.format(col)]=syncrony
            # Turn dict to dataframes
            df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict,orient='index').T
            df_RESULT_list.index=[people]
            df_syncrony_measurement=df_syncrony_measurement.append(df_RESULT_list)
        self.df_syncrony_measurement=df_syncrony_measurement
        contain_nan_bool=self._checknan()
 
        return df_syncrony_measurement
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
                if np.isnan(r_pairwise_dist):
                    print("Reproduce NaN ", col)
                    print("Original Val ", pairwise_dist)
                    print("Occurs at people ", people)
                    
                
                
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
    def _Add_additional_info(self,df_person_segment_feature_DKIndividual_dict,Label,label_choose_lst,\
                        Inspect_roles,\
                        MinNumTimeSeries=3,PhoneOfInterest_str=' '):
        p_1=Inspect_roles[0]
        p_2=Inspect_roles[1]
        df_basic_additional_info=pd.DataFrame()
        for people in df_person_segment_feature_DKIndividual_dict.keys():
            if len(df_person_segment_feature_DKIndividual_dict[people][p_1])<MinNumTimeSeries or len(df_person_segment_feature_DKIndividual_dict[people][p_2])<MinNumTimeSeries:
                continue
            
            df_person_segment_feature_role_dict=df_person_segment_feature_DKIndividual_dict[people]
            RESULT_dict={}
            Mintimeserieslen=min([len(df_person_segment_feature_role_dict[role])  for role in Inspect_roles])
            RESULT_dict['timeSeries_len[{}]'.format(PhoneOfInterest_str)]=Mintimeserieslen
            for label_choose in label_choose_lst:
                RESULT_dict[label_choose]=Label.label_raw[Label.label_raw['name']==people][label_choose].values[0]
            
            
            df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict,orient='index').T
            df_RESULT_list.index=[people]
            df_basic_additional_info=df_basic_additional_info.append(df_RESULT_list)
        return df_basic_additional_info
    def _calculate_features_col(self,functionDK_people,col):
        df_syncrony_measurement=pd.DataFrame()
        for people in functionDK_people.keys():
            functionDK=functionDK_people[people]
            T=functionDK['T']
            RESULT_dict={}    
            
            proximity=-np.abs(np.mean(functionDK['D'] - functionDK['K']))
            D_t=-np.abs(functionDK['D']-functionDK['K'])
        
            time=T.reshape(-1)
            Convergence= 0 if np.isnan(pearsonr(D_t,time)[0]) else pearsonr(D_t,time)[0]
            Trend_D= 0 if np.isnan(pearsonr(functionDK['D'],time)[0]) else pearsonr(functionDK['D'],time)[0]
            Trend_K= 0 if np.isnan(pearsonr(functionDK['K'],time)[0]) else pearsonr(functionDK['K'],time)[0]

            delta=[-15, -10, -5, 0, 5, 10, 15]        
            syncron_lst=[]
            for d in delta:
                if d < 0: #ex_ d=-15
                    f_d_shifted=functionDK['D'][-d:]
                    f_k_shifted=functionDK['K'][:d]
                elif d > 0: #ex_ d=15
                    f_d_shifted=functionDK['D'][:-d]
                    f_k_shifted=functionDK['K'][d:]
                else: #d=0
                    f_d_shifted=functionDK['D']
                    f_k_shifted=functionDK['K']
                syncron_candidate=pearsonr(f_d_shifted,f_k_shifted)[0]
                
                syncron_lst.append(syncron_candidate)
            syncrony= 0 if np.isnan(syncron_lst[np.argmax(np.abs(syncron_lst))]) else syncron_lst[np.argmax(np.abs(syncron_lst))] 

            
            RESULT_dict['Proximity[{}]'.format(col)]=proximity
            RESULT_dict['Trend[{}]_d'.format(col)]=Trend_D
            RESULT_dict['Trend[{}]_k'.format(col)]=Trend_K
            RESULT_dict['Convergence[{}]'.format(col)]=Convergence
            RESULT_dict['Syncrony[{}]'.format(col)]=syncrony
            
            assert np.isnan(np.array([proximity,Trend_D,Trend_K,Convergence,syncrony])).any() != True
            
            df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict,orient='index').T
            df_RESULT_list.index=[people]
            df_syncrony_measurement=df_syncrony_measurement.append(df_RESULT_list)
        return df_syncrony_measurement
    def xpearsonr(x, y):
        # x and y should have same length.
        x = np.asarray(x)
        y = np.asarray(y)
    
        # If an input is constant, the correlation coefficient is not defined.
        if x.ptp() == 0 or y.ptp() == 0:
            warnings.warn(ConstantInputWarning())
            return np.nan, np.nan
    
        if len(x) == len(y) == 2:
            return np.float64(np.sign(x[1] - x[0])*np.sign(y[1] - y[0])), 1.0
    
        xmean = x.mean()
        ymean = y.mean()
    
        xm = x - xmean
        ym = y - ymean
    
        sigmax = linalg.norm(xm)
        sigmay = linalg.norm(ym)
    
        # A warning is generated if the coefficient of variation of either
        # x or y is less than constant_threshold.
        constant_threshold = 1.0e-8
        if sigmax < constant_threshold*xmean or sigmay < constant_threshold*ymean:
            warnings.warn(NearConstantInputWarning())
    
        r = np.dot(xm/sigmax, ym/sigmay)
    
        # Presumably, if abs(r) > 1, then it is only some small artifact of
        # floating point arithmetic.
        r = max(min(r, 1.0), -1.0)
    
        df = len(x) - 2
        if abs(r) == 1.0:
            prob = 0.0
        else:
            t_squared = r**2 * (df / ((1.0 - r) * (1.0 + r)))
            prob = special.betainc(0.5*df, 0.5, df / (df + t_squared))
    
        return r, prob