
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
import scipy.stats as st
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
    def __init__(self, Stat_med_str_VSA='mean'):
        
        self.Stat_med_str_VSA=Stat_med_str_VSA
        self.Inspect_features=['F1','F2']
        
    def calculate_features(self,Vowels_AUI,Label,PhoneOfInterest):
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
        
            
            if len(u)==0 or len(a)==0 or len(i)==0:
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
            RESULT_dict['F2i_u']= u[1]/i[1]
            RESULT_dict['F1a_u']= u[0]/a[0]
            # assert FCR <=2
            
            RESULT_dict['VSA1']=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
            
            RESULT_dict['LnVSA']=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
            
            EDiu=np.sqrt((u[1]-i[1])**2+(u[0]-i[0])**2)
            EDia=np.sqrt((a[1]-i[1])**2+(a[0]-i[0])**2)
            EDau=np.sqrt((u[1]-a[1])**2+(u[0]-a[0])**2)
            S=(EDiu+EDia+EDau)/2
            RESULT_dict['VSA2']=np.sqrt(S*(S-EDiu)*(S-EDia)*(S-EDau))
            
            RESULT_dict['LnVSA']=np.sqrt(np.log(S)*(np.log(S)-np.log(EDiu))*(np.log(S)-np.log(EDia))*(np.log(S)-np.log(EDau)))
            
            ''' a u i distance '''
            RESULT_dict['dau1'] = np.abs(a[0] - u[0])
            RESULT_dict['dai1'] = np.abs(a[0] - i[0])
            RESULT_dict['diu1'] = np.abs(i[0] - u[0])
            RESULT_dict['daudai1'] = RESULT_dict['dau1'] + RESULT_dict['dai1']
            RESULT_dict['daudiu1'] = RESULT_dict['dau1'] + RESULT_dict['diu1']
            RESULT_dict['daidiu1'] = RESULT_dict['dai1'] + RESULT_dict['diu1']
            RESULT_dict['daidiudau1'] = RESULT_dict['dai1'] + RESULT_dict['diu1']+ RESULT_dict['dau1']
            
            RESULT_dict['dau2'] = np.abs(a[1] - u[1])
            RESULT_dict['dai2'] = np.abs(a[1] - i[1])
            RESULT_dict['diu2'] = np.abs(i[1] - u[1])
            RESULT_dict['daudai2'] = RESULT_dict['dau2'] + RESULT_dict['dai2']
            RESULT_dict['daudiu2'] = RESULT_dict['dau2'] + RESULT_dict['diu2']
            RESULT_dict['daidiu2'] = RESULT_dict['dai2'] + RESULT_dict['diu2']
            RESULT_dict['daidiudau2'] = RESULT_dict['dai2'] + RESULT_dict['diu2']+ RESULT_dict['dau2']
            
            # =============================================================================
            ''' F-value, Valid Formant measure '''
            
            # =============================================================================
            # Get data
            F12_raw_dict=Vowels_AUI[people]
            u=F12_raw_dict['u:']
            a=F12_raw_dict['A:']
            i=F12_raw_dict['i:']
            df_vowel = pd.DataFrame(np.vstack([u,a,i]),columns=self.Inspect_features)
            df_vowel['vowel'] = np.hstack([np.repeat('u:',len(u)),np.repeat('A:',len(a)),np.repeat('i:',len(i))])
            df_vowel['target']=pd.Categorical(df_vowel['vowel'])
            df_vowel['target']=df_vowel['target'].cat.codes
            # F-test
            print("utt number of group u = {0}, utt number of group i = {1}, utt number of group A = {2}".format(\
                len(u),len(a),len(i)))
            F_vals=f_classif(df_vowel[self.Inspect_features].values,df_vowel['target'].values)[0]
            RESULT_dict['F_vals_f1']=F_vals[0]
            RESULT_dict['F_vals_f2']=F_vals[1]
            RESULT_dict['F_val_mix']=RESULT_dict['F_vals_f1'] + RESULT_dict['F_vals_f2']
            
            msb=f_classif(df_vowel[self.Inspect_features].values,df_vowel['target'].values)[2]
            msw=f_classif(df_vowel[self.Inspect_features].values,df_vowel['target'].values)[3]
            ssbn=f_classif(df_vowel[self.Inspect_features].values,df_vowel['target'].values)[4]
            
            
            
            RESULT_dict['MSB_f1']=msb[0]
            RESULT_dict['MSB_f2']=msb[1]
            MSB_f1 , MSB_f2 = RESULT_dict['MSB_f1'], RESULT_dict['MSB_f2']
            RESULT_dict['MSB_mix']=MSB_f1 + MSB_f2
            RESULT_dict['MSW_f1']=msw[0]
            RESULT_dict['MSW_f2']=msw[1]
            MSW_f1 , MSW_f2 = RESULT_dict['MSW_f1'], RESULT_dict['MSW_f2']
            RESULT_dict['MSW_mix']=MSW_f1 + MSW_f2
            RESULT_dict['SSBN_f1']=ssbn[0]
            RESULT_dict['SSBN_f2']=ssbn[1]
            
            # =============================================================================
            # criterion
            # F1u < F1a
            # F2u < F2a
            # F2u < F2i
            # F1i < F1a
            # F2a < F2i
            # =============================================================================
            u_mean=F12_val_dict['u:']
            a_mean=F12_val_dict['A:']
            i_mean=F12_val_dict['i:']
            
            F1u, F2u=u_mean[0], u_mean[1]
            F1a, F2a=a_mean[0], a_mean[1]
            F1i, F2i=i_mean[0], i_mean[1]
            
            filt1 = [1 if F1u < F1a else 0]
            filt2 = [1 if F2u < F2a else 0]
            filt3 = [1 if F2u < F2i else 0]
            filt4 = [1 if F1i < F1a else 0]
            filt5 = [1 if F2a < F2i else 0]
            RESULT_dict['criterion_score']=np.sum([filt1,filt2,filt3,filt4,filt5])
        
            df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict)
            df_RESULT_list.index=[people]
            df_formant_statistic=df_formant_statistic.append(df_RESULT_list)
        return df_formant_statistic