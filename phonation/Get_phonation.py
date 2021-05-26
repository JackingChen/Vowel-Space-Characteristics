
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa
modified by Jackchen 20210524

This script is to extract phonation feature 

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

import torch
from tqdm import tqdm
from addict import Dict
import glob 
import argparse
import pickle
from scipy import stats
from pydub import AudioSegment
class Phonation:
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
    def __init__(self):
        self.pitch_method="rapt"
        self.size_frame=0.04
        self.size_step=0.02
        self.minf0=60
        self.maxf0=350
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


def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--base_path_phf', default='/homes/ssd1/jackchen/gop_prediction',
                        help='path of the base directory')
    parser.add_argument('--filepath', default='/mnt/sdd/jackchen/egs/formosa/s6/Segmented_ADOS_emotion',
                        help='path of the base directory')
    parser.add_argument('--trnpath', default='/mnt/sdd/jackchen/egs/formosa/s6/Audacity_Word',
                        help='path of the base directory')
    parser.add_argument('--checkpointpath', default='/homes/ssd1/jackchen/DisVoice/phonation/features',
                        help='path of the base directory')
    parser.add_argument('--outpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--avgmethod', default='middle',
                        help='path of the base directory')
    parser.add_argument('--check', default=True,
                        help='path of the base directory')
    parser.add_argument('--checkreliability', default=False,
                            help='path of the base directory')
    args = parser.parse_args()
    return args

args = get_args()
base_path=args.base_path
filepath=args.filepath
trnpath=args.trnpath
outpath=args.outpath
AVERAGEMETHOD=args.avgmethod
path_app = base_path
sys.path.append(path_app)
checkpointpath_manual=args.checkpointpath
PhonationPath=base_path + "/phonation" 


import praat.praat_functions as praat_functions
from script_mananger import script_manager
from utils_jack import dynamic2statict_artic, dynamic2statict, save_dict_kaldimat, get_dict
from script_mananger import script_manager
# =============================================================================
'''

    Manual area

'''
Formants_utt_symb=Dict()
Formants_people_symb=Dict()
# =============================================================================
# audiopath='/homes/ssd1/jackchen/DisVoice/data/Segmented_ADOS_TD_normalized_untranscripted'
audiopath='/homes/ssd1/jackchen/DisVoice/data/Segmented_ADOS_normalized'
dataset_name=os.path.basename(audiopath)
dataset_name=dataset_name[dataset_name.find('ADOS'):dataset_name.find('normalized')-1]


files=glob.glob(audiopath+"/*.wav")

silence_duration=0.02 #0.1s

silence_duration_ms=silence_duration*1000
silence = AudioSegment.silent(duration=silence_duration_ms)
if os.path.exists('Gen_formant_multiprocess.log'):
    os.remove('Gen_formant_multiprocess.log')

chkptpath=PhonationPath+"/Phonation_dict_bag_{}.pkl".format(dataset_name)
if not os.path.exists(chkptpath):
    Phonation_dict_bag=Dict()
    for file in tqdm(files):
        # print(file)
        name='_'.join(os.path.basename(file).replace(".wav","").split("_")[:-1])
        # filename=os.path.basename(file).split(".")[0]
        # spkr_name='_'.join(filename.split("_")[:-3])
        # utt='_'.join(filename.split("_")[:])
        audiofile=file
        # trn=trnpath+"/{name}.txt".format(name=filename)
        # df_segInfo=pd.read_csv(trn, header=None,delimiter='\t')
        
        # audio = AudioSegment.from_wav(audiofile)
        phonation_extractor=Phonation()
        
        df_feat_utt=phonation_extractor.extract_features_file(audiofile)
        
        if name not in Phonation_dict_bag.keys():
            Phonation_dict_bag[name]=pd.DataFrame()
    
        Phonation_dict_bag[name]=Phonation_dict_bag[name].append(df_feat_utt)
    
    
    
    tmpPath=PhonationPath + "/features"
    if not os.path.exists(tmpPath):
        os.makedirs(tmpPath)
    pickle.dump(Phonation_dict_bag,open(chkptpath,"wb"))
else:
    Phonation_dict_bag=pickle.load(open(chkptpath,"rb"))

chkptpath_kid=PhonationPath+"/df_phonation_kid_{}.pkl".format(dataset_name)
chkptpath_doc=PhonationPath+"/df_phonation_doc_{}.pkl".format(dataset_name)
if not os.path.exists(chkptpath_kid) or \
   not os.path.exists(chkptpath_doc):
    Phonation_role_dict=Dict()
    for keys, values in Phonation_dict_bag.items():
        if '_K' in keys:
            Phonation_role_dict['K'][keys.replace("_K","")]=values.mean(axis=0)
        elif '_D' in keys:
            Phonation_role_dict['D'][keys.replace("_D","")]=values.mean(axis=0)
    
    df_phonation_kid=pd.DataFrame.from_dict(Phonation_role_dict['K']).T
    df_phonation_doc=pd.DataFrame.from_dict(Phonation_role_dict['D']).T
    pickle.dump(df_phonation_kid,open(chkptpath_kid,"wb"))
    pickle.dump(df_phonation_doc,open(chkptpath_doc,"wb"))
else:
    df_phonation_kid=pickle.load(open(chkptpath_kid,"rb"))
    df_phonation_doc=pickle.load(open(chkptpath_doc,"rb"))

chkptpath_kid_TD=args.checkpointpath + "/df_phonation_kid_ADOS_TD.pkl"
chkptpath_kid_ASD=args.checkpointpath + "/df_phonation_kid_ADOS.pkl"
chkptpath_doc_ASD=args.checkpointpath + "/df_phonation_doc_ADOS.pkl"
chkptpath_doc_TD=args.checkpointpath + "/df_phonation_doc_ADOS_TD.pkl"

# df_phonation_kid_TD=pickle.load(open(chkptpath_kid_TD,"rb"))
# df_phonation_kid_ASD=pickle.load(open(chkptpath_kid_ASD,"rb"))
# df_phonation_doc_ASD=pickle.load(open(chkptpath_doc_ASD,"rb"))
# df_phonation_doc_TD=pickle.load(open(chkptpath_doc_TD,"rb"))



def TTest_Cmp_checkpoint(chkptpath_1,chkptpath_2):
    df_phonation_1=pickle.load(open(chkptpath_1,"rb"))
    df_phonation_2=pickle.load(open(chkptpath_2,"rb"))
    name1=os.path.basename(chkptpath_1).replace(".pkl","")
    name2=os.path.basename(chkptpath_2).replace(".pkl","")
    
    dataset_str='{name1}vs{name2}'.format(name1=name1,name2=name2)
    df_ttest_result=pd.DataFrame()
    for col in df_phonation_1.columns:
        df_ttest_result.loc[dataset_str+"-p-val",col]=stats.ttest_ind(df_phonation_1[col].dropna(),df_phonation_2[col].dropna())[1].astype(float)
    df_ttest_result=df_ttest_result.T 
    
    result_path="RESULTS/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    df_ttest_result.to_excel(result_path+dataset_str+".xlsx")

TTest_Cmp_checkpoint(chkptpath_doc_ASD,chkptpath_doc_TD)
TTest_Cmp_checkpoint(chkptpath_kid_ASD,chkptpath_kid_TD)

# dataset_str='{ds}_DvsK'.format(ds=dataset_name)
# df_ttest_result=pd.DataFrame()
# for col in df_phonation_kid.columns:
#     df_ttest_result.loc[dataset_str+"-p-val",col]=stats.ttest_ind(df_phonation_kid[col].dropna(),df_phonation_doc[col].dropna())[1].astype(float)
# df_ttest_result=df_ttest_result.T 

# result_path="RESULTS/"
# if not os.path.exists(result_path):
#     os.makedirs(result_path)
# df_ttest_result.to_excel(result_path+dataset_str+".xlsx")
