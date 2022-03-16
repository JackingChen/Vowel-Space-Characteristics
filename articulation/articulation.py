
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
# from script_mananger import script_manager
from scipy import stats
from utils_jack import dynamic2statict_artic, save_dict_kaldimat, get_dict, f_classif
import re
from itertools import combinations
from addict import Dict
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import sklearn
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing
from scipy.stats import pearsonr 
import scipy
def cosAngle(a, b, c):
    # angles between line segments (Python) from https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
    cosine_angle = np.dot((b-a), (b-c)) / (np.linalg.norm((b-a)) * np.linalg.norm((b-c)))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

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
    def calculate_features(self,Vowels_AUI,Label,PhoneOfInterest,label_choose_lst=['ADOS_C'],\
                           RETURN_scatter_matrix=False, FILTER_overlap_thrld=None,KDE_THRESHOLD=None,\
                           FILTERING_method=None):
        # =============================================================================
        # Code Level of clustering features
        '''
            Input: Vowels_AUI (can be generated from Get_Vowels_AUI(AUI_info)) See usage from Analyze_F1F2_tVSA_FCR.py
            Output: df_formant_statistic
        
        '''
        Statistic_method={'mean':np.mean,'median':np.median,'mode':stats.mode}
        # =============================================================================
        SCATTER_matrixBookeep_dict=Dict()
        df_formant_statistic=pd.DataFrame()
        for people in Vowels_AUI.keys(): #update 2021/05/27 fixed 
            RESULT_dict={}
            F12_raw_dict=Vowels_AUI[people]
            F12_val_dict={k:[] for k in PhoneOfInterest}
            for k,v in F12_raw_dict.items():
                if self.Stat_med_str_VSA == 'mode':
                    F12_val_dict[k]=Statistic_method[self.Stat_med_str_VSA](v,axis=0)[0].ravel()
                else:
                    F12_val_dict[k]=Statistic_method[self.Stat_med_str_VSA](v[self.Inspect_features],axis=0)
            RESULT_dict['u_num'], RESULT_dict['a_num'], RESULT_dict['i_num']=\
                len(Vowels_AUI[people]['u:']),len(Vowels_AUI[people]['A:']),len(Vowels_AUI[people]['i:'])
            
            for label_choose in label_choose_lst:
                RESULT_dict[label_choose]=Label.label_raw[label_choose][Label.label_raw['name']==people].values    
            RESULT_dict['sex']=Label.label_raw['sex'][Label.label_raw['name']==people].values[0]
            RESULT_dict['age']=Label.label_raw['age_year'][Label.label_raw['name']==people].values[0]
            RESULT_dict['Module']=Label.label_raw['Module'][Label.label_raw['name']==people].values[0]
            
            u=F12_val_dict['u:'] # Get the averaged middle point formant values of the phone
            a=F12_val_dict['A:'] # Get the averaged middle point formant values of the phone
            i=F12_val_dict['i:'] # Get the averaged middle point formant values of the phone
        
            # Handle the case when the sample is not enough to calculate LOC
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
            # RESULT_dict['FCR+AUINum']=RESULT_dict['FCR'] + (RESULT_dict['u_num']+ RESULT_dict['a_num']+ RESULT_dict['i_num'])
            # RESULT_dict['FCR*AUINum']=RESULT_dict['FCR'] * (RESULT_dict['u_num']+ RESULT_dict['a_num']+ RESULT_dict['i_num'])
            
            RESULT_dict['VSA1']=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
            # RESULT_dict['VSA1+AUINum']=RESULT_dict['VSA1'] + (RESULT_dict['u_num']+ RESULT_dict['a_num']+ RESULT_dict['i_num'])
            # RESULT_dict['VSA1*AUINum']=RESULT_dict['VSA1'] * (RESULT_dict['u_num']+ RESULT_dict['a_num']+ RESULT_dict['i_num'])
            # =============================================================================
            ''' F-statistics, between class variance Valid Formant measure '''
            
            # =============================================================================
            F12_raw_dict=Vowels_AUI[people]
            # numerator2=F12_raw_dict['u:']['F2'].sum() + F12_raw_dict['A:']['F2'].sum() + F12_raw_dict['i:']['F1'].sum() + F12_raw_dict['u:']['F1'].sum()
            # demominator2=F12_raw_dict['i:']['F2'].sum() + F12_raw_dict['A:']['F1'].sum()
            # RESULT_dict['FCR2']=np.float(numerator2/demominator2)
            # u=F12_raw_dict['u:'] # Get the list of F1 F2 of a certain phone
            # a=F12_raw_dict['A:'] # Get the list of F1 F2 of a certain phone
            # i=F12_raw_dict['i:'] # Get the list of F1 F2 of a certain phone
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
            
            def Calculate_distanceCorr(df_vowel):
                import dcor
                a=df_vowel[df_vowel['vowel']=='A:'][self.Inspect_features]
                u=df_vowel[df_vowel['vowel']=='u:'][self.Inspect_features]
                i=df_vowel[df_vowel['vowel']=='i:'][self.Inspect_features]
                
                
                a_1, a_2=a['F1'], a['F2']
                u_1, u_2=u['F1'], u['F2']
                i_1, i_2=i['F1'], i['F2']
                
                d_stats_a=dcor.distance_stats(a_1, a_2)
                d_stats_u=dcor.distance_stats(u_1, u_2)
                d_stats_i=dcor.distance_stats(i_1, i_2)
                
                pear_a=np.abs(pearsonr(a_1, a_2)[0])
                pear_u=np.abs(pearsonr(u_1, u_2)[0])
                pear_i=np.abs(pearsonr(i_1, i_2)[0])
                
                
                def get_values(X_stats):
                    cov_xy, corr_xy, var_x, var_y=X_stats
                    return cov_xy, corr_xy, var_x, var_y
                
                
                data_a = get_values(d_stats_a)
                data_u = get_values(d_stats_u)
                data_i = get_values(d_stats_i)
                
                Corr_aui=[data_a[1],data_u[1],data_i[1]]
                Cov_sum=[sum(x) for x in zip(data_a,data_u,data_i)]
                pear_sum=sum([pear_a,pear_u,pear_i])
                return Cov_sum, Corr_aui, pear_sum   #Including [cov_xy, corr_xy, var_x, var_y]
            
            def LDA_scatter_matrices(df_vowel):
                class_feature_means = pd.DataFrame(columns=list(set(df_vowel['vowel'])))
                n_classes = len(class_feature_means.columns)
                n_samples = len(df_vowel)
                dfbn = n_classes - 1
                dfwn = n_samples - n_classes
                for c, rows in df_vowel.groupby('vowel'):
                    class_feature_means[c] = rows[self.Inspect_features].mean()
                
                groups_num=len(class_feature_means.index)
                # Within class scatter matrix 
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
                    
                # Between class scatter matrix 
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
                
                # Calculate eigen values
                linear_discriminant_norm=np.linalg.inv(within_class_scatter_matrix ).dot(between_class_scatter_matrix )
                linear_discriminant=np.linalg.inv(within_class_scatter_matrix_norm ).dot(between_class_scatter_matrix )
                return within_class_scatter_matrix, between_class_scatter_matrix, linear_discriminant,Total_scatter_matrix,\
                        within_class_scatter_matrix_norm, between_class_scatter_matrix_norm, linear_discriminant_norm, Total_scatter_matrix_norm
            def LDA_LevelOfClustering_feats(df_vowel):
                '''    Calculate class variance by LDA. Vowel space features are in this function 
                
                        Suffix "_norm"" represents normalized matrix or scalar
                '''  
                within_class_scatter_matrix, between_class_scatter_matrix, linear_discriminant,Total_scatter_matrix,\
                        within_class_scatter_matrix_norm, between_class_scatter_matrix_norm, linear_discriminant_norm, Total_scatter_matrix_norm = LDA_scatter_matrices(df_vowel)
                
                eigen_values_lin, eigen_vectors_lin = np.linalg.eig(linear_discriminant)
                eigen_values_lin_norm, eigen_vectors_lin_norm = np.linalg.eig(linear_discriminant_norm)
                eigen_values_B, eigen_vectors_B = np.linalg.eig(between_class_scatter_matrix)
                eigen_values_B_norm, eigen_vectors_B_norm = np.linalg.eig(between_class_scatter_matrix_norm)
                eigen_values_W, eigen_vectors_W = np.linalg.eig(within_class_scatter_matrix)
                eigen_values_W_norm, eigen_vectors_W_norm = np.linalg.eig(within_class_scatter_matrix_norm)
                eigen_values_T, eigen_vectors_T = np.linalg.eig(Total_scatter_matrix)
                eigen_values_T_norm, eigen_vectors_T_norm = np.linalg.eig(Total_scatter_matrix_norm)
                
                Eigenpair_lin_norm=SortbyEigenValue(eigen_values_lin_norm, eigen_vectors_lin_norm)
                # Eigenpair_B=SortbyEigenValue(eigen_values_B, eigen_vectors_B)
                Eigenpair_B_norm=SortbyEigenValue(eigen_values_B_norm, eigen_vectors_B_norm)
                Eigenpair_W_norm=SortbyEigenValue(eigen_values_W_norm, eigen_vectors_W_norm)
                
                def Covariance_representations(eigen_values):
                    sam_wilks=1
                    pillai=0
                    hotelling=0
                    for eigen_v in eigen_values:
                        wild_element=1.0/np.float(1+eigen_v)
                        sam_wilks*=wild_element
                        pillai+=wild_element * eigen_v
                        hotelling+=eigen_v
                    roys_root=np.max(eigen_values)
                    return sam_wilks, pillai, hotelling, roys_root
                Covariances={}
                Covariances['sam_wilks_lin'], Covariances['pillai_lin'], Covariances['hotelling_lin'], Covariances['roys_root_lin'] = Covariance_representations(eigen_values_lin)
                Covariances['sam_wilks_lin_norm'], Covariances['pillai_lin_norm'], Covariances['hotelling_lin_norm'], Covariances['roys_root_lin_norm'] = Covariance_representations(eigen_values_lin_norm)
                # Covariances['sam_wilks_B'], Covariances['pillai_B'], Covariances['hotelling_B'], Covariances['roys_root_B'] = Covariance_representations(eigen_values_B)
                # Covariances['sam_wilks_Bnorm'], Covariances['pillai_Bnorm'], Covariances['hotelling_Bnorm'], Covariances['roys_root_Bnorm'] = Covariance_representations(eigen_values_B_norm)
                # Covariances['sam_wilks_W'], Covariances['pillai_W'], Covariances['hotelling_W'], Covariances['roys_root_W'] = Covariance_representations(eigen_values_W)

                
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
                Covariances['Between_Within_Det_ratio'] = Multi_Variances['between_covariance_norm'] / Multi_Variances['within_covariance_norm']
                Covariances['Between_Within_Tr_ratio'] = Multi_Variances['between_variance_norm'] / Multi_Variances['within_variance_norm']
                
                
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
                
                Major_vector_lin_norm=Eigenpair_lin_norm[0][1]
                Major_vector_B_norm=Eigenpair_B_norm[0][1]
                Major_vector_W_norm=Eigenpair_W_norm[0][1]
                Minor_vector_lin_norm=Eigenpair_lin_norm[1][1]
                Minor_vector_B_norm=Eigenpair_B_norm[1][1]
                Minor_vector_W_norm=Eigenpair_W_norm[1][1]
                xaxis = [1, 0]
                unit_vector_x = xaxis / np.linalg.norm(xaxis)
                
                
                Angles={}
                for var_vector in ['Major_vector_lin_norm', 'Major_vector_B_norm', 'Major_vector_W_norm',\
                                    'Minor_vector_lin_norm', 'Minor_vector_B_norm', 'Minor_vector_W_norm']:
                    angle=AngleToXaxis(vars()[var_vector], unit_vector_x)
                    Angles[var_vector]=angle

                return Covariances, Multi_Variances, Single_Variances, Angles
                # return Covariances, Multi_Variances, Single_Variances, 
                
            
            
            def Store_FeatVals(RESULT_dict,df_vowel,Inspect_features=['F1','F2'], cluster_str='u:,i:,A:'):
                Covariances, Multi_Variances, Single_Variances,Angles,\
                    =LDA_LevelOfClustering_feats(df_vowel[Inspect_features+['vowel']])

                # RESULT_dict['between_covariance({0})'.format(cluster_str)]=between_covariance
                # RESULT_dict['between_variance({0})'.format(cluster_str)]=between_variance
                # RESULT_dict['between_covariance_norm({0})'.format(cluster_str)]=between_covariance_norm
                # RESULT_dict['between_variance_norm({0})'.format(cluster_str)]=between_variance_norm
                # RESULT_dict['within_covariance({0})'.format(cluster_str)]=within_covariance
                # RESULT_dict['within_variance({0})'.format(cluster_str)]=within_variance
                # RESULT_dict['linear_discriminant_covariance({0})'.format(cluster_str)]=linear_discriminant_covariance
                
                # for keys, values in Single_Variances.items():
                #     RESULT_dict[keys+'({0})'.format(cluster_str)]=values
                for keys, values in Multi_Variances.items():
                    RESULT_dict[keys+'({0})'.format(cluster_str)]=values
                for keys, values in Covariances.items():
                    RESULT_dict[keys+'({0})'.format(cluster_str)]=values
                for keys, values in Angles.items():
                    RESULT_dict[keys+'({0})'.format(cluster_str)]=values
                return RESULT_dict
            
            def CalculateConvexHull(df_vowel):
                hull = ConvexHull(df_vowel[self.Inspect_features].values)
                convexhull_area=hull.area
                return convexhull_area
            
            
            
            
            def CalculateVowelFormantDispersion(df_vowel):
                Info_dict=Dict()
                points=df_vowel[self.Inspect_features].values.copy()
                F1m=np.mean(points[:,0])
                # Determine the f2 mean
                f2_criteria_bag=np.empty((0,2), float)
                for V in points:
                    if V[0]<F1m:
                        f2_criteria_bag=np.append(f2_criteria_bag, V.reshape((1,-1)), axis=0)
                F2m=np.mean(f2_criteria_bag[:,1])
                
                center_coordinate=np.array([F1m,F2m])
                Info_dict['center']=center_coordinate
                Info_dict['vector'], Info_dict['scalar'], Info_dict['angle']=[], [], []
                # Calculate the Vowel vector, angle and scale 
                for i, V in enumerate(points):
                    VFDi=V-center_coordinate
                    Info_dict['vector'].append(VFDi)
                    Info_dict['scalar'].append(np.linalg.norm(VFDi))
                    
                    # The math.atan2() method returns the arc tangent of y/x, and has take care the special conditions
                    omega=math.atan2(V[0]-center_coordinate[0], V[1]-center_coordinate[1])
                    Info_dict['angle'].append(omega)
                return Info_dict  # contains center, vector, scalar, angle
            
            def Calculate_relative_angles(df_vowel, additional_infos=False):
                a=df_vowel[df_vowel['vowel']=='A:'][self.Inspect_features]
                u=df_vowel[df_vowel['vowel']=='u:'][self.Inspect_features]
                i=df_vowel[df_vowel['vowel']=='i:'][self.Inspect_features]
                
                a_center=a.mean()
                u_center=u.mean()
                i_center=i.mean()
                total_center=df_vowel.mean()
                # gravity_center=(a_center*len(a) + u_center*len(u) + i_center*len(i)) / len(df_vowel)
                
                
                
                omega_a=np.degrees(math.atan2((a_center - total_center)[1], (a_center - total_center)[0]))
                omega_u=np.degrees(math.atan2((u_center - total_center)[1], (u_center - total_center)[0]))
                omega_i=np.degrees(math.atan2((i_center - total_center)[1], (i_center - total_center)[0]))
            
                ang_ai = cosAngle(a_center,total_center,i_center)
                ang_iu = cosAngle(i_center,total_center,u_center)
                ang_ua = cosAngle(u_center,total_center,a_center)
                
                absolute_ang=[omega_a, omega_u, omega_i]
                relative_ang=[ang_ai, ang_iu, ang_ua]
                addition_info=[total_center, a_center, u_center, i_center]
                
                if additional_infos != True:
                    return absolute_ang, relative_ang
                else:
                    return absolute_ang, relative_ang, addition_info
            
            def Calculate_pointDistsTotal(df_vowel):
                a=df_vowel[df_vowel['vowel']=='A:'][self.Inspect_features]
                u=df_vowel[df_vowel['vowel']=='u:'][self.Inspect_features]
                i=df_vowel[df_vowel['vowel']=='i:'][self.Inspect_features]
                
                dist_au=scipy.spatial.distance.cdist(a,u)
                dist_ai=scipy.spatial.distance.cdist(a,i)
                dist_iu=scipy.spatial.distance.cdist(i,u)
                mean_dist_au=np.mean(dist_au)
                mean_dist_ai=np.mean(dist_ai)
                mean_dist_iu=np.mean(dist_iu)
                dist_total=mean_dist_au+mean_dist_ai+mean_dist_iu
                return dist_total
            def calculate_betweenClusters_distrib_dist(df_vowel):                
                def calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel', vowel1='A:',\
                                               vowel2='u:', Inspect_features=['F1', 'F2']):
                    v1=df_vowel[df_vowel[vowelCol_name]==vowel1][self.Inspect_features]
                    v2=df_vowel[df_vowel[vowelCol_name]==vowel2][self.Inspect_features]
                    
                    mean_dist_v1v2=np.sum(scipy.spatial.distance.cdist(v1,v2.mean().values.reshape(1,-1)))
                    mean_dist_v2=np.sum(scipy.spatial.distance.cdist(v2,v2.mean().values.reshape(1,-1)))
                    dristrib_dist_v1v2= mean_dist_v2/mean_dist_v1v2
                    
                    # stdev_v2=calculate_stdev(df_vowel[df_vowel[vowelCol_name]==vowel1])
                    # dristrib_dist_v1v2= stdev_v2/mean_dist_v1v2
                    
                    mean_dist_v2v1=np.mean(scipy.spatial.distance.cdist(v2,v1.mean().values.reshape(1,-1)))
                    mean_dist_v1=np.mean(scipy.spatial.distance.cdist(v1,v1.mean().values.reshape(1,-1)))
                    dristrib_dist_v2v1= mean_dist_v1/mean_dist_v2v1
                    # stdev_v1=calculate_stdev(df_vowel[df_vowel[vowelCol_name]==vowel2])
                    # dristrib_dist_v2v1= stdev_v1/mean_dist_v2v1
                    
                    dristrib_dist=dristrib_dist_v1v2 * dristrib_dist_v2v1
                    return dristrib_dist
                
                
                
                dristrib_dist_au= calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel', vowel1='A:',\
                                                   vowel2='u:', Inspect_features=self.Inspect_features)
                dristrib_dist_ui= calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel', vowel1='u:',\
                                                   vowel2='i:', Inspect_features=self.Inspect_features)
                dristrib_dist_ai= calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel', vowel1='A:',\
                                                   vowel2='i:', Inspect_features=self.Inspect_features)
                return sum(dristrib_dist_au,dristrib_dist_ui,dristrib_dist_ai)
            def Calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel'):
 
                repuls_forc_inst_bag=[]
                for index, row in df_vowel.iterrows():
                    phone=row[vowelCol_name]
                    formant_values=row[self.Inspect_features]
                    other_phones=df_vowel[df_vowel[vowelCol_name]!=phone]
                    other_phones_values=other_phones[self.Inspect_features]
                    
                    repuls_forc_inst=np.mean(1/scipy.spatial.distance.cdist(other_phones_values,formant_values.values.reshape(1,-1)))
                    repuls_forc_inst_bag.append(repuls_forc_inst)
                assert len(repuls_forc_inst_bag) == len(df_vowel)
                return np.mean(repuls_forc_inst_bag)
            
            df_vowel=Get_DfVowels(F12_raw_dict,self.Inspect_features)
            if FILTERING_method=='Silhouette':
                if FILTER_overlap_thrld==None:
                    raise BaseException("FILTER_overlap_thrld should not be None, \
                                        0 is a default choice")
                df_vowel=self.Silhouette_filtering(df_vowel,FILTER_overlap_thrld)
            elif FILTERING_method=='KDE':
                if KDE_THRESHOLD==None:
                    raise BaseException("KDE_THRESHOLD should not be None, \
                                        10 - 50 are better choices")
                df_vowel=self.KDE_Filtering(df_vowel,THRESHOLD=KDE_THRESHOLD,scale_factor=100)
            
            # =============================================================================
            F12_val_dict2={k:[] for k in PhoneOfInterest}
            for k in PhoneOfInterest:
                v=df_vowel[df_vowel['vowel']==k]
                if self.Stat_med_str_VSA == 'mode':
                    F12_val_dict2[k]=Statistic_method[self.Stat_med_str_VSA](v,axis=0)[0].ravel()
                else:
                    F12_val_dict2[k]=Statistic_method[self.Stat_med_str_VSA](v[self.Inspect_features],axis=0)
            u=F12_val_dict2['u:'] # Get the averaged middle point formant values of the phone
            a=F12_val_dict2['A:'] # Get the averaged middle point formant values of the phone
            i=F12_val_dict2['i:'] # Get the averaged middle point formant values of the phone
            numerator=u[1] + a[1] + i[0] + u[0]
            demominator=i[1] + a[0]
            RESULT_dict['FCR2']=np.float(numerator/demominator)
            RESULT_dict['VSA2']=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
            # =============================================================================
            
            
            cluster_str=','.join(sorted(F12_raw_dict.keys()))
            #Level of clustering features using three vowel clusters
            RESULT_dict=Store_FeatVals(RESULT_dict,df_vowel,self.Inspect_features, cluster_str=cluster_str)    
            #Baseline feature ConvexHulls
            RESULT_dict['ConvexHull']=CalculateConvexHull(df_vowel)
            VFD_info_dict = CalculateVowelFormantDispersion(df_vowel)
            RESULT_dict['MeanVFD']=np.mean(VFD_info_dict['scalar'])
            RESULT_dict['SumVFD']=np.sum(VFD_info_dict['scalar'])
            
            #Calculate relative angles
            absolute_ang, relative_ang = Calculate_relative_angles(df_vowel, additional_infos=False)
            [RESULT_dict['absAng_a'], RESULT_dict['absAng_u'], RESULT_dict['absAng_i']]=absolute_ang
            [RESULT_dict['ang_ai'], RESULT_dict['ang_iu'], RESULT_dict['ang_ua']]=relative_ang
            [RESULT_dict['dcov_12'], RESULT_dict['dcorr_12'], RESULT_dict['dvar_1'], RESULT_dict['dvar_2']],\
                [RESULT_dict['dcor_a'],RESULT_dict['dcor_u'],RESULT_dict['dcor_i']], RESULT_dict['pear_12']=Calculate_distanceCorr(df_vowel)
            RESULT_dict['dist_tot']=Calculate_pointDistsTotal(df_vowel)
            RESULT_dict['pointDistsTotal']=Calculate_pointDistsTotal(df_vowel)
            RESULT_dict['repulsive_force']=Calculate_pair_distrib_dist(df_vowel, vowelCol_name='vowel')
            
            
            ''' Get partial between/within class variances, (A:, i:) (A:, u:) (i:, u:)'''
            # comb2 = combinations(F12_raw_dict.keys(), 2)
            # comb3 = combinations(F12_raw_dict.keys(), 3)
            # cluster_vars=list(comb2) + list(comb3)
            # cluster_vars=list(comb3)
            # for cluster_lsts in cluster_vars:
            #     F12_tmp={cluster:F12_raw_dict[cluster] for cluster in cluster_lsts}
            #     df_vowel=Get_DfVowels(F12_tmp,self.Inspect_features)
            #     cluster_str=','.join(sorted(F12_tmp.keys()))
            #     RESULT_dict=Store_FeatVals(RESULT_dict,df_vowel,self.Inspect_features, cluster_str=cluster_str)  
            
            
            ''' End of feature calculation '''
            # =============================================================================
            df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict)
            df_RESULT_list.index=[people]
            df_formant_statistic=df_formant_statistic.append(df_RESULT_list)
            
            ''' Especially return the scatter matrices '''
            if RETURN_scatter_matrix!=False:
                within_class_scatter_matrix, between_class_scatter_matrix, linear_discriminant,Total_scatter_matrix,\
                                    within_class_scatter_matrix_norm, between_class_scatter_matrix_norm, linear_discriminant_norm, Total_scatter_matrix_norm = LDA_scatter_matrices(df_vowel)
                
                SCATTER_matrixBookeep_dict[people]['Norm(WC)']=within_class_scatter_matrix_norm
                SCATTER_matrixBookeep_dict[people]['Norm(BC)']=between_class_scatter_matrix_norm
                SCATTER_matrixBookeep_dict[people]['Norm(B_CRatio)']=linear_discriminant_norm
                SCATTER_matrixBookeep_dict[people]['Norm(TotalVar)']=Total_scatter_matrix_norm
            
        if RETURN_scatter_matrix!=False:
            return df_formant_statistic, SCATTER_matrixBookeep_dict
        else:
            return df_formant_statistic
    
    
    
    def BasicFilter_byNum(self,df_formant_statistic,N=1):
        filter_bool=np.logical_and(df_formant_statistic['u_num']>N,df_formant_statistic['a_num']>N)
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['i_num']>N)
        df_formant_statistic_limited=df_formant_statistic[filter_bool]
        return df_formant_statistic_limited
    
    def Silhouette_filtering(self,df_vowel,FILTER_overlap_thrld=0):
        X=df_vowel[self.Inspect_features]
        labels=df_vowel['vowel']
        silhouette_score=sklearn.metrics.silhouette_samples(X, labels,  metric='euclidean')
        df_vowel_filtered=df_vowel[silhouette_score>=FILTER_overlap_thrld]
        return df_vowel_filtered
    
    def KDE_Filtering(self,df_vowel,THRESHOLD=10,scale_factor=100):
        X=df_vowel[self.Inspect_features].values
        labels=df_vowel['vowel']
        
        df_vowel_calibrated=pd.DataFrame([])
        for phone in set(labels):
            
            df=df_vowel[df_vowel['vowel']==phone][self.Inspect_features]
            data_array=df_vowel[df_vowel['vowel']==phone][self.Inspect_features].values
    
            x=data_array[:,0]
            y=data_array[:,1]
            xmin = x.min()
            xmax = x.max()        
            ymin = y.min()
            ymax = y.max()
            
            image_num=1j
            X, Y = np.mgrid[xmin:xmax:image_num*scale_factor, ymin:ymax:image_num*scale_factor]
            
            positions = np.vstack([X.ravel(), Y.ravel()])
            
            values = np.vstack([x, y])
            
            kernel = stats.gaussian_kde(values)
                    
            Z = np.reshape(kernel(positions).T, X.shape)
            normalized_z = preprocessing.normalize(Z)
            
            df['x_to_scale'] = (100*(x - np.min(x))/np.ptp(x)).astype(int) 
            df['y_to_scale'] = (100*(y - np.min(y))/np.ptp(y)).astype(int) 
            
            normalized_z=(100*(Z - np.min(Z.ravel()))/np.ptp(Z.ravel())).astype(int)
            to_delete = zip(*np.where((normalized_z<THRESHOLD) == True))
            
            # The indexes that are smaller than threshold
            deletepoints_bool=df.apply(lambda x: (x['x_to_scale'], x['y_to_scale']), axis=1).isin(to_delete)
            df_calibrated=df.loc[(deletepoints_bool==False).values]
            df_deleted_after_calibrated=df.loc[(deletepoints_bool==True).values]
            
            df_vowel_calibrated_tmp=df_calibrated.drop(columns=['x_to_scale','y_to_scale'])
            df_vowel_calibrated_tmp['vowel']=phone
            df_vowel_output=df_vowel_calibrated_tmp.copy()
            df_vowel_calibrated=df_vowel_calibrated.append(df_vowel_output)
            
            
            # Data prepare for plotting 
            # df_calibrated_tocombine=df_calibrated.copy()
            # df_calibrated_tocombine['cal']='calibrated'
            # df_deleted_after_calibrated['cal']='deleted'
            # df_calibratedcombined=df_calibrated_tocombine.append(df_deleted_after_calibrated)
            
            # #Plotting code
            # fig = plt.figure(figsize=(8,8))
            # ax = fig.gca()
            # ax.set_xlim(xmin, xmax)
            # ax.set_ylim(ymin, ymax)
            # # cfset = ax.contourf(X, Y, Z, cmap='coolwarm')
            # # ax.imshow(Z, cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
            # # cset = ax.contour(X, Y, Z, colors='k')
            # cfset = ax.contourf(X, Y, normalized_z, cmap='coolwarm')
            # ax.imshow(normalized_z, cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
            # cset = ax.contour(X, Y, normalized_z, colors='k')
            # ax.clabel(cset, inline=1, fontsize=10)
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # plt.title('2D Gaussian Kernel density estimation')
            
            # sns.scatterplot(data=df_vowel[df_vowel['vowel']==phone], x="F1", y="F2")
            # sns.scatterplot(data=df_calibratedcombined, x="F1", y="F2",hue='cal')
        return df_vowel_calibrated