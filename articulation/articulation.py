
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa
"""

from scipy.io.wavfile import read
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.mlab as mlab
import math
import pysptk
try:
    from .articulation_functions import extractTrans, V_UV
except: 
    from articulation_functions import extractTrans, V_UV
import uuid
import pandas as pd
import torch
from tqdm import tqdm
path_app = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path_app+'/../')
import praat.praat_functions as praat_functions
from script_mananger import script_manager
from scipy import stats
from utils_jack import dynamic2statict_artic, save_dict_kaldimat, get_dict, f_classif
import re
from itertools import combinations
class Extract_F1F2:

    """Extract the articulation features from an audio file
    
    >>> articulation=Articulation()
    >>> file_audio="../audios/001_ddk1_PCGITA.wav"
    >>> features1=articulation.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
    >>> features2=articulation.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
    >>> features3=articulation.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
    >>> articulation.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test")
    
    >>> path_audio="../audios/"
    >>> features1=articulation.extract_features_path(path_audio, static=True, plots=False, fmt="npy")
    >>> features2=articulation.extract_features_path(path_audio, static=True, plots=False, fmt="csv")
    >>> features3=articulation.extract_features_path(path_audio, static=False, plots=True, fmt="torch")
    >>> articulation.extract_features_path(path_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test.ark")
    """
    
    def __init__(self,maxf0=400, minf0=75):
        self.pitch_method="praat"
        self.sizeframe=0.04
        self.step=0.02
        self.nB=22
        self.nMFCC=12
        self.minf0=75
        self.maxf0=400
        self.voice_bias=-0.2
        self.len_thr_miliseconds=270.0
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        
    def extract_features_file(self,audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        fs, data_audio=read(audio)
        #normalize raw wavform
        data_audio=data_audio-np.mean(data_audio)
        data_audio=data_audio/float(np.max(np.abs(data_audio)))
        size_frameS=self.sizeframe*float(fs)
        size_stepS=self.step*float(fs)
        overlap=size_stepS/size_frameS
    
        if self.pitch_method == 'praat':
            name_audio=audio.split('/')
            temp_uuid='articulation'+name_audio[-1][0:-4]
            if not os.path.exists(self.PATH+'/../tempfiles/'):
                os.makedirs(self.PATH+'/../tempfiles/')
            temp_filename_vuv=self.PATH+'/../tempfiles/tempVUV'+temp_uuid+'.txt'
            temp_filename_f0=self.PATH+'/../tempfiles/tempF0'+temp_uuid+'.txt'
            praat_functions.praat_vuv(audio, temp_filename_f0, temp_filename_vuv, time_stepF0=self.step, minf0=self.minf0, maxf0=self.maxf0)
            F0,_=praat_functions.decodeF0(temp_filename_f0,len(data_audio)/float(fs),self.step)
            segmentsFull,segmentsOn,segmentsOff=praat_functions.read_textgrid_trans(temp_filename_vuv,data_audio,fs,self.sizeframe)
            os.remove(temp_filename_vuv)
            os.remove(temp_filename_f0)
        elif self.pitch_method == 'rapt':
            data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
            F0=pysptk.sptk.rapt(data_audiof, fs, int(size_stepS), min=self.minf0, max=self.maxf0, voice_bias=self.voice_bias, otype='f0')

            segmentsOn=V_UV(F0, data_audio, fs, 'onset')
            segmentsOff=V_UV(F0, data_audio, fs, 'offset')
            
            
        name_audio=audio.split('/')
        temp_uuid='artic'+name_audio[-1][0:-4]
        if not os.path.exists(self.PATH+'/../tempfiles/'):
            os.makedirs(self.PATH+'/../tempfiles/')
        temp_filename=self.PATH+'/../tempfiles/tempFormants'+temp_uuid+'.txt'
        praat_functions.praat_formants(audio, temp_filename,self.sizeframe,self.step)
        [F1, F2]=praat_functions.decodeFormants(temp_filename)

        os.remove(temp_filename)
        
        return [F1, F2]

class Articulation:
    def __init__(self, Stat_med_str_VSA='mean', Inspect_features=['F1','F2']):
        
        self.Stat_med_str_VSA=Stat_med_str_VSA
        self.Inspect_features=Inspect_features
        self.N=3
    def calculate_features(self,Vowels_AUI,Label,PhoneOfInterest,label_choose_lst=['ADOS_C']):
        # =============================================================================
        # Code calculate vowel features
        Statistic_method={'mean':np.mean,'median':np.median,'mode':stats.mode}
        # =============================================================================
        df_formant_statistic=pd.DataFrame()
        for people in Vowels_AUI.keys(): #update 2021/05/27 fixed 
            RESULT_dict={}
            F12_raw_dict=Vowels_AUI[people]
            F12_val_dict={k:[] for k in PhoneOfInterest}
            for k,v in F12_raw_dict.items():
                if self.Stat_med_str_VSA == 'mode':
                    F12_val_dict[k]=Statistic_method[self.Stat_med_str_VSA](v,axis=0)[0].ravel()
                else:
                    F12_val_dict[k]=Statistic_method[self.Stat_med_str_VSA](v,axis=0)
            RESULT_dict['u_num'], RESULT_dict['a_num'], RESULT_dict['i_num']=\
                len(Vowels_AUI[people]['u:']),len(Vowels_AUI[people]['A:']),len(Vowels_AUI[people]['i:'])
            
            for label_choose in label_choose_lst:
                RESULT_dict[label_choose]=Label.label_raw[label_choose][Label.label_raw['name']==people].values    
            RESULT_dict['sex']=Label.label_raw['sex'][Label.label_raw['name']==people].values[0]
            RESULT_dict['age']=Label.label_raw['age_year'][Label.label_raw['name']==people].values[0]
            RESULT_dict['Module']=Label.label_raw['Module'][Label.label_raw['name']==people].values[0]
            
            u=F12_val_dict['u:']
            a=F12_val_dict['A:']
            i=F12_val_dict['i:']
        
            
            if RESULT_dict['u_num']<self.N or RESULT_dict['a_num']<self.N or RESULT_dict['i_num']<self.N:
                u_num= RESULT_dict['u_num'] if type(RESULT_dict['u_num'])==int else 0
                i_num= RESULT_dict['i_num'] if type(RESULT_dict['i_num'])==int else 0
                a_num= RESULT_dict['a_num'] if type(RESULT_dict['a_num'])==int else 0
                df_RESULT_list=pd.DataFrame(np.zeros([1,len(df_formant_statistic.columns)]),columns=df_formant_statistic.columns)
                df_RESULT_list.index=[people]
                df_RESULT_list.loc[people,'u_num']=u_num
                df_RESULT_list.loc[people,'a_num']=a_num
                df_RESULT_list.loc[people,'i_num']=i_num
                df_RESULT_list['FCR']=10
                for label_choose in label_choose_lst:
                    df_RESULT_list[label_choose]=RESULT_dict[label_choose][0]
                df_formant_statistic=df_formant_statistic.append(df_RESULT_list)
                continue
            
            numerator=u[1] + a[1] + i[0] + u[0]
            demominator=i[1] + a[0]
            RESULT_dict['FCR']=np.float(numerator/demominator)
            RESULT_dict['FCR+AUINum']=RESULT_dict['FCR'] + (RESULT_dict['u_num']+ RESULT_dict['a_num']+ RESULT_dict['i_num'])
            RESULT_dict['FCR*AUINum']=RESULT_dict['FCR'] * (RESULT_dict['u_num']+ RESULT_dict['a_num']+ RESULT_dict['i_num'])
            
            RESULT_dict['VSA1']=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
            RESULT_dict['VSA1+AUINum']=RESULT_dict['VSA1'] + (RESULT_dict['u_num']+ RESULT_dict['a_num']+ RESULT_dict['i_num'])
            RESULT_dict['VSA1*AUINum']=RESULT_dict['VSA1'] * (RESULT_dict['u_num']+ RESULT_dict['a_num']+ RESULT_dict['i_num'])
            # =============================================================================
            ''' F-statistics, between class variance Valid Formant measure '''
            
            # =============================================================================
            # Get data
            F12_raw_dict=Vowels_AUI[people]
            u=F12_raw_dict['u:']
            a=F12_raw_dict['A:']
            i=F12_raw_dict['i:']
            def Get_DfVowels(F12_raw_dict,Inspect_features=['F1','F2']):
                df_vowel = pd.DataFrame()
                for keys in F12_raw_dict.keys():
                    if len(df_vowel) == 0:
                        df_vowel=F12_raw_dict[keys]
                        df_vowel['vowel']=keys
                    else:
                        df_=F12_raw_dict[keys]
                        df_['vowel']=keys
                        df_vowel=df_vowel.append(df_)
                df_vowel['target']=pd.Categorical(df_vowel['vowel'])
                df_vowel['target']=df_vowel['target'].cat.codes
                return df_vowel
            
            def SortbyEigenValue(eigen_values,eigen_vectors):
                # pairs = [(eigen_value),array(eigen_vector)], e.g (1.4485527363308637, array([-0.70778505, -0.70642786]))
                # the first value is the highest eigen pair
                pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
                pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
                return pairs
            
            def AngleToXaxis(var_vector, unit_vector_x):
                unit_vector_var = var_vector / np.linalg.norm(var_vector)
                dot_product = np.dot(unit_vector_x, unit_vector_var)
                angle = np.arccos(dot_product)
                return angle
            
            def LDA_scatter_matrix(df_vowel):
                '''    Calculate class variance by LDA '''  

                class_feature_means = pd.DataFrame(columns=list(set(df_vowel['vowel'])))
                n_classes = len(class_feature_means.columns)
                n_samples = len(df_vowel)
                dfbn = n_classes - 1
                dfwn = n_samples - n_classes
                for c, rows in df_vowel.groupby('vowel'):
                    class_feature_means[c] = rows[self.Inspect_features].mean()
                
                groups_num=len(class_feature_means.index)
                # Total_within_variance_l2=0.0
                within_class_scatter_matrix = np.zeros((groups_num,groups_num))
                for c, rows in df_vowel.groupby('vowel'):
                    rows = rows[self.Inspect_features]
                    s = np.zeros((groups_num,groups_num))
                    
                    for index, row in rows.iterrows():
                        x, mc = row.values.reshape(groups_num,1), class_feature_means[c].values.reshape(groups_num,1)
                        class_variance=(x - mc).dot((x - mc).T).astype(float)
                        s += class_variance
                        
                    within_class_scatter_matrix += s
                    
                within_class_scatter_matrix_norm = within_class_scatter_matrix / n_samples
                    
                
                feature_means = df_vowel.mean()
                between_class_scatter_matrix = np.zeros((groups_num,groups_num))
                for c in class_feature_means:    
                    n = len(df_vowel.loc[df_vowel['vowel'] == c].index)
                    
                    mc, m = class_feature_means[c].values.reshape(groups_num,1), feature_means[self.Inspect_features].values.reshape(groups_num,1)
                    
                    between_class_variance = n * (mc - m).dot((mc - m).T)
                    
                    between_class_scatter_matrix += between_class_variance
                    
                between_class_scatter_matrix_norm = between_class_scatter_matrix / n_samples
                Total_scatter_matrix=within_class_scatter_matrix + between_class_scatter_matrix
                Total_scatter_matrix_norm=Total_scatter_matrix / n_samples
                
                linear_discriminant_norm=np.linalg.inv(within_class_scatter_matrix ).dot(between_class_scatter_matrix )
                linear_discriminant=np.linalg.inv(within_class_scatter_matrix_norm ).dot(between_class_scatter_matrix )
                eigen_values_lin, eigen_vectors_lin = np.linalg.eig(linear_discriminant)
                eigen_values_lin_norm, eigen_vectors_lin_norm = np.linalg.eig(linear_discriminant_norm)
                eigen_values_B, eigen_vectors_B = np.linalg.eig(between_class_scatter_matrix)
                eigen_values_B_norm, eigen_vectors_B_norm = np.linalg.eig(between_class_scatter_matrix_norm)
                eigen_values_W, eigen_vectors_W = np.linalg.eig(within_class_scatter_matrix)
                eigen_values_W_norm, eigen_vectors_W_norm = np.linalg.eig(within_class_scatter_matrix_norm)
                eigen_values_T, eigen_vectors_T = np.linalg.eig(Total_scatter_matrix)
                eigen_values_T_norm, eigen_vectors_T_norm = np.linalg.eig(Total_scatter_matrix_norm)
                
                Eigenpair_lin=SortbyEigenValue(eigen_values_lin, eigen_vectors_lin)
                # Eigenpair_B=SortbyEigenValue(eigen_values_B, eigen_vectors_B)
                Eigenpair_B_norm=SortbyEigenValue(eigen_values_B_norm, eigen_vectors_B_norm)
                Eigenpair_W=SortbyEigenValue(eigen_values_W, eigen_vectors_W)
                
                def Covariance_representations(eigen_values):
                    sam_wilks=1
                    pillai=0
                    hotelling=0
                    for eigen_v in eigen_values:
                        wild_element=1.0/np.float(1+eigen_v)
                        sam_wilks*=wild_element
                        pillai+=wild_element
                        hotelling+=eigen_v
                    roys_root=np.max(eigen_values)
                    return sam_wilks, pillai, hotelling, roys_root
                Covariances={}
                Covariances['sam_wilks_lin'], Covariances['pillai_lin'], Covariances['hotelling_lin'], Covariances['roys_root_lin'] = Covariance_representations(eigen_values_lin)
                Covariances['sam_wilks_lin_norm'], Covariances['pillai_lin_norm'], Covariances['hotelling_lin_norm'], Covariances['roys_root_lin_norm'] = Covariance_representations(eigen_values_lin_norm)
                # Covariances['sam_wilks_B'], Covariances['pillai_B'], Covariances['hotelling_B'], Covariances['roys_root_B'] = Covariance_representations(eigen_values_B)
                # Covariances['sam_wilks_Bnorm'], Covariances['pillai_Bnorm'], Covariances['hotelling_Bnorm'], Covariances['roys_root_Bnorm'] = Covariance_representations(eigen_values_B_norm)
                # Covariances['sam_wilks_W'], Covariances['pillai_W'], Covariances['hotelling_W'], Covariances['roys_root_W'] = Covariance_representations(eigen_values_W)

                # between_covariance = np.linalg.det(between_class_scatter_matrix_norm)
                # between_variance = np.trace(between_class_scatter_matrix_norm)
                # within_covariance = np.linalg.det(within_class_scatter_matrix)
                # within_variance = np.trace(within_class_scatter_matrix)
                
                Multi_Variances={}
                Multi_Variances['between_covariance_norm'] = np.prod(eigen_values_B_norm)# product of every element
                Multi_Variances['between_variance_norm'] = np.sum(eigen_values_B_norm)
                Multi_Variances['between_covariance'] = np.prod(eigen_values_B)# product of every element
                Multi_Variances['between_variance'] = np.sum(eigen_values_B)
                Multi_Variances['within_covariance_norm'] = np.prod(eigen_values_W_norm)
                Multi_Variances['within_variance_norm'] = np.sum(eigen_values_W_norm)
                Multi_Variances['within_covariance'] = np.prod(eigen_values_W)
                Multi_Variances['within_variance'] = np.sum(eigen_values_W)
                Multi_Variances['total_covariance_norm'] = np.prod(eigen_values_T_norm)
                Multi_Variances['total_variance_norm'] = np.sum(eigen_values_T_norm)
                Multi_Variances['total_covariance'] = np.prod(eigen_values_T)
                Multi_Variances['total_variance'] = np.sum(eigen_values_T)
                
                
                # Variances in certain projection
                # Variances in f1 , f2, Major, Minor, Ratio:major/minor, and the angles of Major and Minor vectors
                Single_Variances={}
                between_variance_f, within_variance_f, linear_disc_f={}, {}, {}
                between_variance_f_norm, within_variance_f_norm, linear_disc_f_norm={}, {}, {}
                for i in range(between_class_scatter_matrix_norm.shape[0]):
                    between_variance_f[i+1]=between_class_scatter_matrix[i,i]
                    within_variance_f[i+1]=within_class_scatter_matrix[i,i]
                    linear_disc_f[i+1]=between_variance_f[i+1]/within_variance_f[i+1]
                    between_variance_f_norm[i+1]=between_class_scatter_matrix_norm[i,i]
                    within_variance_f_norm[i+1]=within_class_scatter_matrix_norm[i,i]
                    linear_disc_f_norm[i+1]=between_variance_f_norm[i+1]/within_variance_f_norm[i+1]
                for keys  in between_variance_f.keys():
                    Single_Variances['between_variance_f'+str(keys)]=between_variance_f[keys]
                    Single_Variances['within_variance_f'+str(keys)]=within_variance_f[keys]
                    Single_Variances['linear_disc_f'+str(keys)]=linear_disc_f[keys]
                    Single_Variances['between_variance_f'+str(keys)+'_norm']=between_variance_f_norm[keys]
                    Single_Variances['within_variance_f'+str(keys)+'_norm']=within_variance_f_norm[keys]
                    Single_Variances['linear_disc_f_norm'+str(keys)]=linear_disc_f_norm[keys]
                # Single_Variances['Major_variance_lin']=Eigenpair_lin[0][0]
                # Single_Variances['Major_variance_B_norm']=Eigenpair_B_norm[0][0]
                # Single_Variances['Major_variance_W']=Eigenpair_W[0][0]
                # Single_Variances['Minor_variance_lin']=Eigenpair_lin[1][0]
                # Single_Variances['Minor_variance_B_norm']=Eigenpair_B_norm[1][0]
                # Single_Variances['Minor_variance_W']=Eigenpair_W[1][0]
                # Single_Variances['Ratio_mjr_mnr_lin']=Single_Variances['Major_variance_lin']/Single_Variances['Minor_variance_lin']
                # Single_Variances['Ratio_mjr_mnr_B']=Single_Variances['Major_variance_B_norm']/Single_Variances['Minor_variance_B_norm']
                # Single_Variances['Ratio_mjr_mnr_W']=Single_Variances['Major_variance_W']/Single_Variances['Minor_variance_W']
                
                # Major_vector_lin=Eigenpair_lin[0][1]
                # Major_vector_B_norm=Eigenpair_B_norm[0][1]
                # Major_vector_W=Eigenpair_W[0][1]
                # Minor_vector_lin=Eigenpair_lin[1][1]
                # Minor_vector_B_norm=Eigenpair_B_norm[1][1]
                # Minor_vector_W=Eigenpair_W[1][1]
                # xaxis = [1, 0]
                # unit_vector_x = xaxis / np.linalg.norm(xaxis)
                
                
                # Angles={}
                # for var_vector in ['Major_vector_lin', 'Major_vector_B_norm', 'Major_vector_W',\
                #                    'Minor_vector_lin', 'Minor_vector_B_norm', 'Minor_vector_W']:
                #     angle=AngleToXaxis(vars()[var_vector], unit_vector_x)
                #     Angles[var_vector]=angle
                    
                
                
                # between_covariance = np.prod(eigen_values_B) / n_samples # product of every element
                # between_variance = np.sum(eigen_values_B) / n_samples 
                # within_covariance = np.prod(eigen_values_W) / n_samples 
                # within_variance = np.sum(eigen_values_W) / n_samples 
                
                
                
                
 
                # return Covariances, Multi_Variances, Single_Variances, Angles
                return Covariances, Multi_Variances, Single_Variances
                
            
            
            def Store_FeatVals(RESULT_dict,df_vowel,Inspect_features=['F1','F2'], cluster_str='u:,i:,A:'):
                # F_vals, _, msb, _, ssbn=f_classif(df_vowel[Inspect_features].values,df_vowel['target'].values)
                Covariances, Multi_Variances, Single_Variances,\
                    =LDA_scatter_matrix(df_vowel[Inspect_features+['vowel']])

                # RESULT_dict['between_covariance({0})'.format(cluster_str)]=between_covariance
                # RESULT_dict['between_variance({0})'.format(cluster_str)]=between_variance
                # RESULT_dict['between_covariance_norm({0})'.format(cluster_str)]=between_covariance_norm
                # RESULT_dict['between_variance_norm({0})'.format(cluster_str)]=between_variance_norm
                # RESULT_dict['within_covariance({0})'.format(cluster_str)]=within_covariance
                # RESULT_dict['within_variance({0})'.format(cluster_str)]=within_variance
                # RESULT_dict['linear_discriminant_covariance({0})'.format(cluster_str)]=linear_discriminant_covariance
                
                for keys, values in Single_Variances.items():
                    RESULT_dict[keys+'({0})'.format(cluster_str)]=values
                for keys, values in Multi_Variances.items():
                    RESULT_dict[keys+'({0})'.format(cluster_str)]=values
                for keys, values in Covariances.items():
                    RESULT_dict[keys+'({0})'.format(cluster_str)]=values
                return RESULT_dict
            
            
            
            df_vowel=Get_DfVowels(F12_raw_dict,self.Inspect_features)
            cluster_str=','.join(sorted(F12_raw_dict.keys()))
            # print(cluster_str)
            RESULT_dict=Store_FeatVals(RESULT_dict,df_vowel,self.Inspect_features, cluster_str=cluster_str)    
            
            
            # Get partial msb 
            # comb2 = combinations(F12_raw_dict.keys(), 2)
            comb3 = combinations(F12_raw_dict.keys(), 3)
            # cluster_vars=list(comb2) + list(comb3)
            cluster_vars=list(comb3)
            for cluster_lsts in cluster_vars:
                F12_tmp={cluster:F12_raw_dict[cluster] for cluster in cluster_lsts}
                df_vowel=Get_DfVowels(F12_tmp,self.Inspect_features)
                cluster_str=','.join(sorted(F12_tmp.keys()))
                RESULT_dict=Store_FeatVals(RESULT_dict,df_vowel,self.Inspect_features, cluster_str=cluster_str)  
            
            

            
            ''' End of feature calculation '''
            # =============================================================================
            df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict)
            df_RESULT_list.index=[people]
            df_formant_statistic=df_formant_statistic.append(df_RESULT_list)
        return df_formant_statistic
    def BasicFilter_byNum(self,df_formant_statistic,N=1):
        filter_bool=np.logical_and(df_formant_statistic['u_num']>N,df_formant_statistic['a_num']>N)
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['i_num']>N)
        df_formant_statistic_limited=df_formant_statistic[filter_bool]
        return df_formant_statistic_limited
    
    def calculate_features_20210730(self,Vowels_AUI,Label,PhoneOfInterest):
        # =============================================================================
        # Code calculate vowel features
        Statistic_method={'mean':np.mean,'median':np.median,'mode':stats.mode}
        label_choose='ADOS_C'
        # =============================================================================
        df_formant_statistic=pd.DataFrame()
        for people in Vowels_AUI.keys(): #update 2021/05/27 fixed 
            RESULT_dict={}
            F12_raw_dict=Vowels_AUI[people]
            F12_val_dict={k:[] for k in PhoneOfInterest}
            for k,v in F12_raw_dict.items():
                if self.Stat_med_str_VSA == 'mode':
                    F12_val_dict[k]=Statistic_method[self.Stat_med_str_VSA](v,axis=0)[0].ravel()
                else:
                    F12_val_dict[k]=Statistic_method[self.Stat_med_str_VSA](v,axis=0)
            RESULT_dict['u_num'], RESULT_dict['a_num'], RESULT_dict['i_num']=\
                len(Vowels_AUI[people]['u:']),len(Vowels_AUI[people]['A:']),len(Vowels_AUI[people]['i:'])
            
            RESULT_dict['ADOS']=Label.label_raw[label_choose][Label.label_raw['name']==people].values    
            RESULT_dict['sex']=Label.label_raw['sex'][Label.label_raw['name']==people].values[0]
            RESULT_dict['age']=Label.label_raw['age_year'][Label.label_raw['name']==people].values[0]
            RESULT_dict['Module']=Label.label_raw['Module'][Label.label_raw['name']==people].values[0]
            
            u=F12_val_dict['u:']
            a=F12_val_dict['A:']
            i=F12_val_dict['i:']
        
            
            if RESULT_dict['u_num']<self.N or RESULT_dict['a_num']<self.N or RESULT_dict['i_num']<self.N:
                u_num= RESULT_dict['u_num'] if type(RESULT_dict['u_num'])==int else 0
                i_num= RESULT_dict['i_num'] if type(RESULT_dict['i_num'])==int else 0
                a_num= RESULT_dict['a_num'] if type(RESULT_dict['a_num'])==int else 0
                df_RESULT_list=pd.DataFrame(np.zeros([1,len(df_formant_statistic.columns)]),columns=df_formant_statistic.columns)
                df_RESULT_list.index=[people]
                df_RESULT_list.loc[people,'u_num']=u_num
                df_RESULT_list.loc[people,'a_num']=a_num
                df_RESULT_list.loc[people,'i_num']=i_num
                df_RESULT_list['FCR']=10
                df_RESULT_list['ADOS']=RESULT_dict['ADOS'][0]
                df_formant_statistic=df_formant_statistic.append(df_RESULT_list)
                continue
            
            numerator=u[1] + a[1] + i[0] + u[0]
            demominator=i[1] + a[0]
            RESULT_dict['FCR']=np.float(numerator/demominator)
            # RESULT_dict['F2i_u']= u[1]/i[1]
            # RESULT_dict['F1a_u']= u[0]/a[0]
            # assert FCR <=2
            
            RESULT_dict['VSA1']=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
            # =============================================================================
            ''' F-statistics, between class variance Valid Formant measure '''
            
            # =============================================================================
            # Get data
            F12_raw_dict=Vowels_AUI[people]
            u=F12_raw_dict['u:']
            a=F12_raw_dict['A:']
            i=F12_raw_dict['i:']
            def Get_DfVowels(F12_raw_dict,Inspect_features=['F1','F2']):
                df_vowel = pd.DataFrame()
                for keys in F12_raw_dict.keys():
                    if len(df_vowel) == 0:
                        df_vowel=F12_raw_dict[keys]
                        df_vowel['vowel']=keys
                    else:
                        df_=F12_raw_dict[keys]
                        df_['vowel']=keys
                        df_vowel=df_vowel.append(df_)
                df_vowel['target']=pd.Categorical(df_vowel['vowel'])
                df_vowel['target']=df_vowel['target'].cat.codes
                return df_vowel
            
            def LDA_scatter_matrix(df_vowel):
                '''    Calculate class variance by LDA '''  

                class_feature_means = pd.DataFrame(columns=list(set(df_vowel['vowel'])))
                n_classes = len(class_feature_means.columns)
                n_samples = len(df_vowel)
                dfbn = n_classes - 1
                dfwn = n_samples - n_classes
                for c, rows in df_vowel.groupby('vowel'):
                    class_feature_means[c] = rows[self.Inspect_features].mean()
                
                groups_num=len(class_feature_means.index)
                # Total_within_variance_l2=0.0
                within_class_scatter_matrix = np.zeros((groups_num,groups_num))
                for c, rows in df_vowel.groupby('vowel'):
                    rows = rows[self.Inspect_features]
                    s = np.zeros((groups_num,groups_num))
                    # WCV= 0.0
                    for index, row in rows.iterrows():
                        x, mc = row.values.reshape(groups_num,1), class_feature_means[c].values.reshape(groups_num,1)
                        class_variance=(x - mc).dot((x - mc).T).astype(float)
                        s += class_variance
                        # WCV+=np.linalg.norm(class_variance.diagonal(),2)
                    within_class_scatter_matrix += s
                    # Total_within_variance_l2 +=WCV
                    
                    
                
                feature_means = df_vowel.mean()
                # Total_between_variance_l2=0.0
                between_class_scatter_matrix = np.zeros((groups_num,groups_num))
                for c in class_feature_means:    
                    n = len(df_vowel.loc[df_vowel['vowel'] == c].index)
                    
                    mc, m = class_feature_means[c].values.reshape(groups_num,1), feature_means[self.Inspect_features].values.reshape(groups_num,1)
                    
                    between_class_variance = n * (mc - m).dot((mc - m).T)
                    
                    between_class_scatter_matrix += between_class_variance
                    # Total_between_variance_l2+=np.linalg.norm(between_class_variance.diagonal(),2)
                
                # eigen_values, eigen_vectors = np.linalg.eig()
                linear_discriminant=np.linalg.inv(within_class_scatter_matrix / dfwn).dot(between_class_scatter_matrix / dfbn)
                # linear_discriminant=np.linalg.inv(within_class_scatter_matrix ).dot(between_class_scatter_matrix / dfbn)
                B_W_varianceRatio_l2=np.trace(linear_discriminant)
                B_W_varianceRatio_l2_norm= B_W_varianceRatio_l2 / n_samples
                Total_between_variance_l2=np.trace(between_class_scatter_matrix) / dfbn
                Total_between_variance_l2_norm = Total_between_variance_l2 / n_samples
                Total_within_variance_l2=np.trace(within_class_scatter_matrix) /dfwn
                # B_W_varianceRatio_l2=Total_between_variance_l2/Total_within_variance_l2
                # return B_W_varianceRatio, B_W_varianceRatio_l2, Total_between_variance_l2, Total_within_variance_l2
                return B_W_varianceRatio_l2_norm, B_W_varianceRatio_l2, Total_between_variance_l2, Total_between_variance_l2_norm, Total_within_variance_l2
                
            
            
            def Store_FeatVals(RESULT_dict,df_vowel,Inspect_features=['F1','F2'], cluster_str='u:,i:,A:'):
                F_vals, _, msb, _, ssbn=f_classif(df_vowel[Inspect_features].values,df_vowel['target'].values)
                B_W_v_n,B_W_v,bvl2, bvl2_n, wvl2=LDA_scatter_matrix(df_vowel[Inspect_features+['vowel']])
                RESULT_dict['F_vals_f1({0})'.format(cluster_str)], RESULT_dict['F_vals_f2({0})'.format(cluster_str)]=F_vals
                RESULT_dict['F_val_mix({0})'.format(cluster_str)]=RESULT_dict['F_vals_f1({0})'.format(cluster_str)] + RESULT_dict['F_vals_f2({0})'.format(cluster_str)]
                RESULT_dict['MSB_f1({0})'.format(cluster_str)],RESULT_dict['MSB_f2({0})'.format(cluster_str)]=msb
                RESULT_dict['MSB_mix']=RESULT_dict['MSB_f1({0})'.format(cluster_str)]+ RESULT_dict['MSB_f2({0})'.format(cluster_str)]
                RESULT_dict['BWratio({0})'.format(cluster_str)]=B_W_v
                RESULT_dict['BWratio({0})_norm'.format(cluster_str)]=B_W_v_n
                RESULT_dict['BV({0})_l2'.format(cluster_str)], RESULT_dict['BV({0})_l2_norm'.format(cluster_str)], RESULT_dict['WV({0})_l2'.format(cluster_str)]=bvl2, bvl2_n, wvl2
                return RESULT_dict
            
            
            
            df_vowel=Get_DfVowels(F12_raw_dict,self.Inspect_features)
            cluster_str=','.join(sorted(F12_raw_dict.keys()))
            # print(cluster_str)
            RESULT_dict=Store_FeatVals(RESULT_dict,df_vowel,self.Inspect_features, cluster_str=cluster_str)    
            
            
            # Get partial msb 
            comb2 = combinations(F12_raw_dict.keys(), 2)
            comb3 = combinations(F12_raw_dict.keys(), 3)
            cluster_vars=list(comb2) + list(comb3)
            for cluster_lsts in cluster_vars:
                F12_tmp={cluster:F12_raw_dict[cluster] for cluster in cluster_lsts}
                df_vowel=Get_DfVowels(F12_tmp,self.Inspect_features)
                cluster_str=','.join(sorted(F12_tmp.keys()))
                RESULT_dict=Store_FeatVals(RESULT_dict,df_vowel,self.Inspect_features, cluster_str=cluster_str)  
            
            
                
            # =============================================================================
            ''' distributional difference 
                F1(a-i)
                F1(a-u)
                F2(i-u)
                
                Note!! the feat names have to be in the form FX(X-X)
            '''             
            # indexs=['mean', 'min', '25%', '50%', '75%', 'max']
            # featnames=["F1(A:-i:)", "F1(A:-u:)", "F2(i:-u:)"]
            # # =============================================================================
            
            # for feat in featnames:
            #     f_=feat[:re.search("[(.*)]", feat).start()]
            #     col_names=[feat+"_"+ind for ind in indexs]
                
            #     formula=re.search("\((.*)\)", feat).group(1)
            #     minus = F12_raw_dict[formula.split("-")[0]][self.Inspect_features].astype(float) #a a i
            #     minuend = F12_raw_dict[formula.split("-")[1]][self.Inspect_features].astype(float) #i u u
                
            #     # Distributional subtraction
            #     # print(minus.describe(), minuend.describe())
                
            #     df_diff=(minus.describe() - minuend.describe()).loc[indexs]
            #     for ind in range(len(col_names)): 
            #         RESULT_dict[col_names[ind]]=df_diff.loc[col_names[ind].split("_")[-1],f_]
                
                
            
            ''' End of feature calculation '''
            # =============================================================================
            df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict)
            df_RESULT_list.index=[people]
            df_formant_statistic=df_formant_statistic.append(df_RESULT_list)
        return df_formant_statistic