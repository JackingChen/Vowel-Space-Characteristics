
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

import math
import pysptk
try:
    from .phonation_functions import jitter_env, logEnergy, shimmer_env, APQ, PPQ
except:
    from phonation_functions import jitter_env, logEnergy, shimmer_env, APQ, PPQ
import scipy.stats as st
import uuid
import pandas as pd
path_app = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path_app+'/../')
from utils_jack import dynamic2statict, save_dict_kaldimat
import praat.praat_functions as praat_functions
# from script_mananger import script_manager
import torch
from tqdm import tqdm
from itertools import combinations
class Phonation:
    def __init__(self, Inspect_features):
        self.Inspect_features=Inspect_features
        self.N=1
        self.vowel_min_num=1
        self.ISSegment_feature=False
    def _updateISSegmentFeature(self,bool):
        self.ISSegment_feature=bool
    def calculate_features(self,Vowels_AUI,Label,PhoneOfInterest,label_choose_lst=['ADOS_C']):
        # =============================================================================
        # Code calculate vowel features
        # =============================================================================
        df_formant_statistic=pd.DataFrame()
        for people in Vowels_AUI.keys(): #update 2021/05/27 fixed 
            RESULT_dict={}
            RESULT_dict['u_num'], RESULT_dict['a_num'], RESULT_dict['i_num']=\
                len(Vowels_AUI[people]['u:']),len(Vowels_AUI[people]['A:']),len(Vowels_AUI[people]['i:'])
            
            for label_choose in label_choose_lst:
                RESULT_dict[label_choose]=Label.label_raw[label_choose][Label.label_raw['name']==people].values    
            RESULT_dict['sex']=Label.label_raw['sex'][Label.label_raw['name']==people].values[0]
            RESULT_dict['age']=Label.label_raw['age_year'][Label.label_raw['name']==people].values[0]
            RESULT_dict['Module']=Label.label_raw['Module'][Label.label_raw['name']==people].values[0]
            # Get data
            F12_raw_dict=Vowels_AUI[people]

            def Get_DfVowels(F12_raw_dict,Inspect_features=['F1','F2']):
                df_vowel = pd.DataFrame()
                for keys in F12_raw_dict.keys(): #Shoulb be bounded by PhoneOfInterest
                    if len(df_vowel) == 0:
                        df_vowel=F12_raw_dict[keys]
                        df_vowel['vowel']=keys
                    else:
                        df_=F12_raw_dict[keys]
                        df_['vowel']=keys
                        # df_vowel=df_vowel.append(df_)
                        df_vowel = pd.concat([df_vowel, df_], ignore_index=True)
                df_vowel['target']=pd.Categorical(df_vowel['vowel'])
                df_vowel['target']=df_vowel['target'].cat.codes
                return df_vowel
            # df_vowel=Get_DfVowels(F12_raw_dict,self.Inspect_features)
            # cluster_str=','.join(sorted(F12_raw_dict.keys()))
            
            def MeanVar_features(df_vowel, Sessional_pools=['mean','var']):
                # =============================================================================
                '''
                
                    Phonation feature calculation module
                
                '''                
                # =============================================================================
                df_mean=df_vowel[self.Inspect_features].mean(axis=0)
                df_var=df_vowel[self.Inspect_features].var(axis=0)
                df_max=df_vowel[self.Inspect_features].max(axis=0)
                
                df_Sessional_feat=pd.concat([df_mean,df_var,df_max],axis=0).T
                columns=['{0}_{1}'.format(c,Sp)  for Sp in Sessional_pools for c in df_vowel[self.Inspect_features].columns]
                df_Sessional_feat.index=columns
                return df_Sessional_feat.to_dict()
            
            def Store_FeatVals(RESULT_dict,df_vowel,Inspect_features=['stdevF0','localJitter', 'localabsoluteJitter','localShimmer'],\
                               cluster_str='u:,i:,A:'):
                df_Sessional_feat = MeanVar_features(df_vowel, Sessional_pools=['mean','var', 'max'])
                
                for keys, values in df_Sessional_feat.items():
                    RESULT_dict[keys+'({0})'.format(cluster_str)]=values
                return RESULT_dict
            
            
            comb = combinations(PhoneOfInterest, 1)
            comb3 = combinations(PhoneOfInterest, len(PhoneOfInterest))
            if self.ISSegment_feature == True:
                cluster_vars=list(comb3)
            else:
                cluster_vars=list(comb) + list(comb3)
                cluster_vars=list(set(cluster_vars)) # to avoid 
            for cluster_lsts in cluster_vars:
                F12_tmp={cluster:F12_raw_dict[cluster] for cluster in cluster_lsts}
                df_vowel=Get_DfVowels(F12_tmp,self.Inspect_features)
                cluster_str=','.join(sorted(F12_tmp.keys()))

                RESULT_dict=Store_FeatVals(RESULT_dict,df_vowel,self.Inspect_features, cluster_str=cluster_str) 
                
            ''' End of feature calculation '''
            # =============================================================================
            df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict)
            df_RESULT_list.index=[people]
            # df_formant_statistic=df_formant_statistic.append(df_RESULT_list)
            df_formant_statistic = pd.concat([df_formant_statistic, df_RESULT_list]) # index 很重要不能ignore
        return df_formant_statistic
class Phonation_disvoice:
    """
    Compute phonation features from sustained vowels and continuous speech.

    For continuous speech, the features are computed over voiced segments

    Seven descriptors are computed:

    1. First derivative of the fundamental Frequency

    2. Second derivative of the fundamental Frequency

    3. Jitter

    4. Shimmer

    5. Amplitude perturbation quotient

    6. Pitch perturbation quotient

    7. Logaritmic Energy

    Static or dynamic matrices can be computed:

    Static matrix is formed with 29 features formed with (seven descriptors) x (4 functionals: mean, std, skewness, kurtosis) + degree of Unvoiced

    Dynamic matrix is formed with the seven descriptors computed for frames of 40 ms.

    Notes:

    1. In dynamic features the first 11 frames of each recording are not considered to be able to stack the APQ and PPQ descriptors with the remaining ones.
    2. The fundamental frequency is computed the RAPT algorithm. To use the PRAAT method,  change the "self.pitch method" variable in the class constructor.

    Script is called as follows

    >>> python phonation.py <file_or_folder_audio> <file_features> <static (true or false)> <plots (true or false)> <format (csv, txt, npy, kaldi, torch)>

    Examples command line:

    >>> python phonation.py "../audios/001_a1_PCGITA.wav" "phonationfeaturesAst.txt" "true" "true" "txt"
    >>> python phonation.py "../audios/098_u1_PCGITA.wav" "phonationfeaturesUst.csv" "true" "true" "csv"
    >>> python phonation.py "../audios/098_u1_PCGITA.wav" "phonationfeaturesUdyn.pt" "false" "true" "torch"

    >>> python phonation.py "../audios/" "phonationfeaturesst.txt" "true" "false" "txt"
    >>> python phonation.py "../audios/" "phonationfeaturesst.csv" "true" "false" "csv"
    >>> python phonation.py "../audios/" "phonationfeaturesdyn.pt" "false" "false" "torch"

    Examples directly in Python

    >>> from disvoice.phonation import Phonation
    >>> phonation=Phonation()
    >>> file_audio="../audios/001_a1_PCGITA.wav"
    >>> features=phonation.extract_features_file(file_audio, static, plots=True, fmt="numpy")
    >>> features2=phonation.extract_features_file(file_audio, static, plots=True, fmt="dataframe")
    >>> features3=phonation.extract_features_file(file_audio, dynamic, plots=True, fmt="torch")
    
    >>> path_audios="../audios/"
    >>> features1=phonation.extract_features_path(path_audios, static, plots=False, fmt="numpy")
    >>> features2=phonation.extract_features_path(path_audios, static, plots=False, fmt="torch")
    >>> features3=phonation.extract_features_path(path_audios, static, plots=False, fmt="dataframe")

    """
    def __init__(self, maxf0=350, minf0=60, pitch_method="rapt"):
        self.pitch_method=pitch_method
        self.size_frame=0.04
        self.size_step=0.02
        self.minf0=minf0
        self.maxf0=maxf0
        self.voice_bias=-0.2
        self.energy_thr_percent=0.025
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        self.head=["DF0", "DDF0", "Jitter", "Shimmer", "apq", "ppq", "logE"]
        # self.head=["Jitter", "Shimmer", "apq", "ppq", "logE"]
    def _update_pitch_method(self, pitch_method):
        self.pitch_method=pitch_method


    def plot_phon(self, data_audio,fs,F0,logE):
        """Plots of the phonation features

        :param data_audio: speech signal.
        :param fs: sampling frequency
        :param F0: contour of the fundamental frequency
        :param logE: contour of the log-energy
        :returns: plots of the phonation features.
        """
        plt.figure(figsize=(6,6))
        plt.subplot(211)
        ax1=plt.gca()
        t=np.arange(len(data_audio))/float(fs)
        ax1.plot(t, data_audio, 'k', label="speech signal", alpha=0.8)
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_xlim([0, t[-1]])
        plt.grid(True)
        ax2 = ax1.twinx()
        fsp=len(F0)/t[-1]
        t2=np.arange(len(F0))/fsp
        ax2.plot(t2, F0, 'r', linewidth=2,label=r"F_0")
        ax2.set_ylabel(r'$F_0$ (Hz)', color='r', fontsize=12)
        ax2.tick_params('y', colors='r')

        plt.grid(True)

        plt.subplot(212)
        Esp=len(logE)/t[-1]
        t2=np.arange(len(logE))/float(Esp)
        plt.plot(t2, logE, color='k', linewidth=2.0)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Energy (dB)', fontsize=14)
        plt.xlim([0, t[-1]])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def extract_features_file(self, audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        """Extract the phonation features from an audio file
        
        :param audio: .wav audio file.
        :param static: whether to compute and return statistic functionals over the feature matrix, or return the feature matrix computed over frames
        :param plots: timeshift to extract the features
        :param fmt: format to return the features (npy, dataframe, torch, kaldi)
        :param kaldi_file: file to store kaldi features, only valid when fmt=="kaldi"
        :returns: features computed from the audio file.

        >>> phonation=Phonation()
        >>> file_audio="../audios/001_a1_PCGITA.wav"
        >>> features1=phonation.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
        >>> features2=phonation.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
        >>> features3=phonation.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
        >>> phonation.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test")
        """
        fs, data_audio=read(audio)
        data_audio=data_audio-np.mean(data_audio)
        data_audio=data_audio/float(np.max(np.abs(data_audio)))
        size_frameS=self.size_frame*float(fs)
        size_stepS=self.size_step*float(fs)
        overlap=size_stepS/size_frameS
        if self.pitch_method == 'praat':
            name_audio=audio.split('/')
            temp_uuid='phon'+name_audio[-1][0:-4]
            if not os.path.exists(self.PATH+'/../tempfiles/'):
                os.makedirs(self.PATH+'/../tempfiles/')
            temp_filename_vuv=self.PATH+'/../tempfiles/tempVUV'+temp_uuid+'.txt'
            temp_filename_f0=self.PATH+'/../tempfiles/tempF0'+temp_uuid+'.txt'
            praat_functions.praat_vuv(audio, temp_filename_f0, temp_filename_vuv, time_stepF0=self.size_step, minf0=self.minf0, maxf0=self.maxf0)
            F0,_=praat_functions.decodeF0(temp_filename_f0,len(data_audio)/float(fs),self.size_step)
            os.remove(temp_filename_vuv)
            os.remove(temp_filename_f0)
        elif self.pitch_method == 'rapt':
            data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
            F0=pysptk.sptk.rapt(data_audiof, fs, int(size_stepS), min=self.minf0, max=self.maxf0, voice_bias=self.voice_bias, otype='f0')
        F0nz=F0[F0!=0]
        Jitter=jitter_env(F0nz, len(F0nz))
        nF=int((len(data_audio)/size_frameS/overlap))-1
        Amp=[]
        logE=[]
        apq=[]
        ppq=[]
        DF0=np.diff(F0nz, 1)
        DDF0=np.diff(DF0,1)
        F0z=F0[F0==0]
        totaldurU=len(F0z)
        thresholdE=10*logEnergy([self.energy_thr_percent])
        degreeU=100*float(totaldurU)/len(F0)
        lnz=0
        for l in range(nF):
            data_frame=data_audio[int(l*size_stepS):int(l*size_stepS+size_frameS)]
            energy=10*logEnergy(data_frame)
            if F0[l]!=0:
                Amp.append(np.max(np.abs(data_frame)))
                logE.append(energy)
                if lnz>=12: # TODO:
                    amp_arr=np.asarray([Amp[j] for j in range(lnz-12, lnz)])
                    #print(amp_arr)
                    apq.append(APQ(amp_arr))
                if lnz>=6: # TODO:
                    f0arr=np.asarray([F0nz[j] for j in range(lnz-6, lnz)])
                    ppq.append(PPQ(1/f0arr))
                lnz=lnz+1

        Shimmer=shimmer_env(Amp, len(Amp))
        apq=np.asarray(apq)
        ppq=np.asarray(ppq)
        logE=np.asarray(logE)


        if len(apq)==0:
            print("warning, there is not enough long voiced segments to compute the APQ, in this case APQ=shimmer")
            apq=Shimmer

        if plots:
            self.plot_phon(data_audio,fs,F0,logE)

        if len(Shimmer)==len(apq):
            feat_mat=np.vstack((DF0[5:], DDF0[4:], Jitter[6:], Shimmer[6:], apq[6:], ppq, logE[6:])).T
        else:
            feat_mat=np.vstack((DF0[11:], DDF0[10:], Jitter[12:], Shimmer[12:], apq, ppq[6:], logE[12:])).T

        feat_v=dynamic2statict([DF0, DDF0, Jitter, Shimmer, apq, ppq, logE])
        # feat_v=dynamic2statict([Jitter, Shimmer, apq, ppq, logE])


        if fmt=="npy" or fmt=="txt":
            if static:
                return feat_v
            else:
                return feat_mat
        elif fmt=="dataframe" or fmt=="csv":
            if static:
                head_st=[]
                df={}
                for k in ["avg", "std", "skewness", "kurtosis"]:
                    for h in self.head:
                        head_st.append(k+" "+h)
                for e, k in enumerate(head_st):
                    df[k]=[feat_v[e]]
                            
                return pd.DataFrame(df)
            else:
                df={}
                for e, k in enumerate(self.head):
                    df[k]=feat_mat[:,e]
                return pd.DataFrame(df)
        elif fmt=="torch":
            if static:
                feat_t=torch.from_numpy(feat_v)
                return feat_t
            else:
                return torch.from_numpy(feat_mat)

        elif fmt=="kaldi":
            if static:
                raise ValueError("Kaldi is only supported for dynamic features")
            else:
                name_all=audio.split('/')
                dictX={name_all[-1]:feat_mat}
                save_dict_kaldimat(dictX, kaldi_file)
        else:
            raise ValueError(fmt+" is not supported")

    def extract_features_path(self, path_audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        """Extract the phonation features for audios inside a path
        
        :param path_audio: directory with (.wav) audio files inside, sampled at 16 kHz
        :param static: whether to compute and return statistic functionals over the feature matrix, or return the feature matrix computed over frames
        :param plots: timeshift to extract the features
        :param fmt: format to return the features (npy, dataframe, torch, kaldi)
        :param kaldi_file: file to store kaldifeatures, only valid when fmt=="kaldi"
        :returns: features computed from the audio file.

        >>> phonation=Phonation()
        >>> path_audio="../audios/"
        >>> features1=phonation.extract_features_path(path_audio, static=True, plots=False, fmt="npy")
        >>> features2=phonation.extract_features_path(path_audio, static=True, plots=False, fmt="csv")
        >>> features3=phonation.extract_features_path(path_audio, static=False, plots=True, fmt="torch")
        >>> phonation.extract_features_path(path_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test.ark")
        """
        hf=os.listdir(path_audio)
        hf.sort()

        pbar=tqdm(range(len(hf)))
        ids=[]

        Features=[]
        for j in pbar:
            pbar.set_description("Processing %s" % hf[j])
            audio_file=path_audio+hf[j]
            feat=self.extract_features_file(audio_file, static=static, plots=plots, fmt="npy")
            Features.append(feat)
            if static:
                ids.append(hf[j])
            else:
                ids.append(np.repeat(hf[j], feat.shape[0]))
        
        Features=np.vstack(Features)
        ids=np.hstack(ids)
        if fmt=="npy" or fmt=="txt":
            return Features
        elif fmt=="dataframe" or fmt=="csv":
            if static:
                head_st=[]
                df={}
                for k in ["avg", "std", "skewness", "kurtosis"]:
                    for h in self.head:
                        head_st.append(k+" "+h)
                for e, k in enumerate(head_st):
                    df[k]=Features[:,e]
            else:
                df={}
                for e, k in enumerate(self.head):
                    df[k]=Features[:,e]
            df["id"]=ids
            return pd.DataFrame(df)
        elif fmt=="torch":
            return torch.from_numpy(Features)
        elif fmt=="kaldi":
            if static:
                raise ValueError("Kaldi is only supported for dynamic features")
            else:
                dictX=get_dict(Features, ids)
                save_dict_kaldimat(dictX, kaldi_file)
        else:
            raise ValueError(fmt+" is not supported")

class Phonation_jack:
    """
    Compute phonation features from sustained vowels and continuous speech.

    For continuous speech, the features are computed over voiced segments

    Seven descriptors are computed:

    1. First derivative of the fundamental Frequency

    2. Second derivative of the fundamental Frequency

    3. Jitter

    4. Shimmer

    5. Amplitude perturbation quotient

    6. Pitch perturbation quotient

    7. Logaritmic Energy

    Static or dynamic matrices can be computed:maxf0

    Static matrix is formed with 29 features formed with (seven descriptors) x (4 functionals: mean, std, skewness, kurtosis) + degree of Unvoiced

    Dynamic matrix is formed with the seven descriptors computed for frames of 40 ms.

    Notes:

    1. In dynamic features the first 11 frames of each recording are not considered to be able to stack the APQ and PPQ descriptors with the remaining ones.
    2. The fundamental frequency is computed the RAPT algorithm. To use the PRAAT method,  change the "self.pitch method" variable in the class constructor.

    Script is called as follows

    >>> python phonation.py <file_or_folder_audio> <file_features> <static (true or false)> <plots (true or false)> <format (csv, txt, npy, kaldi, torch)>

    Examples command line:

    >>> python phonation.py "../audios/001_a1_PCGITA.wav" "phonationfeaturesAst.txt" "true" "true" "txt"
    >>> python phonation.py "../audios/098_u1_PCGITA.wav" "phonationfeaturesUst.csv" "true" "true" "csv"
    >>> python phonation.py "../audios/098_u1_PCGITA.wav" "phonationfeaturesUdyn.pt" "false" "true" "torch"

    >>> python phonation.py "../audios/" "phonationfeaturesst.txt" "true" "false" "txt"
    >>> python phonation.py "../audios/" "phonationfeaturesst.csv" "true" "false" "csv"
    >>> python phonation.py "../audios/" "phonationfeaturesdyn.pt" "false" "false" "torch"

    Examples directly in Python

    >>> from disvoice.phonation import Phonation
    >>> phonation=Phonation()
    >>> file_audio="../audios/001_a1_PCGITA.wav"
    >>> features=phonation.extract_features_file(file_audio, static, plots=True, fmt="numpy")
    >>> features2=phonation.extract_features_file(file_audio, static, plots=True, fmt="dataframe")
    >>> features3=phonation.extract_features_file(file_audio, dynamic, plots=True, fmt="torch")
    
    >>> path_audios="../audios/"
    >>> features1=phonation.extract_features_path(path_audios, static, plots=False, fmt="numpy")
    >>> features2=phonation.extract_features_path(path_audios, static, plots=False, fmt="torch")
    >>> features3=phonation.extract_features_path(path_audios, static, plots=False, fmt="dataframe")

    """
    def __init__(self, maxf0=350, minf0=60):
        self.pitch_method="rapt"
        self.size_frame=0.04
        self.size_step=0.02
        self.minf0=minf0
        self.maxf0=maxf0
        self.voice_bias=-0.2
        self.energy_thr_percent=0.025
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        self.head=["DF0", "DDF0", "Jitter", "Shimmer", "apq", "ppq", "logE"]

    def extract_features_file(self, audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        """Extract the phonation features from an audio file
        
        :param audio: .wav audio file.
        :param static: whether to compute and return statistic functionals over the feature matrix, or return the feature matrix computed over frames
        :param plots: timeshift to extract the features
        :param fmt: format to return the features (npy, dataframe, torch, kaldi)
        :param kaldi_file: file to store kaldi features, only valid when fmt=="kaldi"
        :returns: features computed from the audio file.

        >>> phonation=Phonation()
        >>> file_audio="../audios/001_a1_PCGITA.wav"
        >>> features1=phonation.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
        >>> features2=phonation.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
        >>> features3=phonation.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
        >>> phonation.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test")
        """
        fs, data_audio=read(audio)
        data_audio=data_audio-np.mean(data_audio)
        data_audio=data_audio/float(np.max(np.abs(data_audio)))
        size_frameS=self.size_frame*float(fs)
        size_stepS=self.size_step*float(fs)
        overlap=size_stepS/size_frameS
        if self.pitch_method == 'praat':
            name_audio=audio.split('/')
            temp_uuid='phon'+name_audio[-1][0:-4]
            if not os.path.exists(self.PATH+'/../tempfiles/'):
                os.makedirs(self.PATH+'/../tempfiles/')
            temp_filename_vuv=self.PATH+'/../tempfiles/tempVUV'+temp_uuid+'.txt'
            temp_filename_f0=self.PATH+'/../tempfiles/tempF0'+temp_uuid+'.txt'
            praat_functions.praat_vuv(audio, temp_filename_f0, temp_filename_vuv, time_stepF0=self.size_step, minf0=self.minf0, maxf0=self.maxf0)
            F0,_=praat_functions.decodeF0(temp_filename_f0,len(data_audio)/float(fs),self.size_step)
            os.remove(temp_filename_vuv)
            os.remove(temp_filename_f0)
        elif self.pitch_method == 'rapt':
            data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
            F0=pysptk.sptk.rapt(data_audiof, fs, int(size_stepS), min=self.minf0, max=self.maxf0, voice_bias=self.voice_bias, otype='f0')
        F0nz=F0[F0!=0]
        Jitter=jitter_env(F0nz, len(F0nz))
        nF=int((len(data_audio)/size_frameS/overlap))-1
        Amp=[]
        logE=[]
        apq=[]
        ppq=[]
        DF0=np.diff(F0nz, 1)
        DDF0=np.diff(DF0,1)
        F0z=F0[F0==0]
        totaldurU=len(F0z)
        thresholdE=10*logEnergy([self.energy_thr_percent])
        degreeU=100*float(totaldurU)/len(F0)
        lnz=0
        for l in range(nF):
            data_frame=data_audio[int(l*size_stepS):int(l*size_stepS+size_frameS)]
            energy=10*logEnergy(data_frame)
            if F0[l]!=0:
                Amp.append(np.max(np.abs(data_frame)))
                logE.append(energy)
                if lnz>=12: # TODO:
                    amp_arr=np.asarray([Amp[j] for j in range(lnz-12, lnz)])
                    #print(amp_arr)
                    apq.append(APQ(amp_arr))
                if lnz>=6: # TODO:
                    f0arr=np.asarray([F0nz[j] for j in range(lnz-6, lnz)])
                    ppq.append(PPQ(1/f0arr))
                lnz=lnz+1

        Shimmer=shimmer_env(Amp, len(Amp))
        apq=np.asarray(apq)
        ppq=np.asarray(ppq)
        logE=np.asarray(logE)


        if len(apq)==0:
            print("warning, there is not enough long voiced segments to compute the APQ, in this case APQ=shimmer")
            apq=Shimmer


        if len(Shimmer)==len(apq):
            feat_mat=np.vstack((DF0[5:], DDF0[4:], Jitter[6:], Shimmer[6:], apq[6:], ppq, logE[6:])).T
        else:
            feat_mat=np.vstack((DF0[11:], DDF0[10:], Jitter[12:], Shimmer[12:], apq, ppq[6:], logE[12:])).T

        feat_v=dynamic2statict([DF0, DDF0, Jitter, Shimmer, apq, ppq, logE])
        
        head_st=[]
        df={}
        for k in ["avg", "std", "skewness", "kurtosis"]:
            for h in self.head:
                head_st.append(k+" "+h)
        for e, k in enumerate(head_st):
            df[k]=[feat_v[e]]
                    
        return pd.DataFrame(df)

# if __name__=="__main__":

#     if len(sys.argv)!=6:
#         print("python phonation.py <file_or_folder_audio> <file_features> <static (true, false)> <plots (true,  false)> <format (csv, txt, npy, kaldi, torch)>")
#         sys.exit()

#     phonation=Phonation()
#     script_manager(sys.argv, phonation)