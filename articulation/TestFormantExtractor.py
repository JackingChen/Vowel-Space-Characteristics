#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 18:17:35 2021

@author: jackchen
"""

import glob
import numpy as np
import pandas as pd
import parselmouth 
import statistics
from parselmouth.praat import call

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
from addict import Dict
import glob
import argparse
import math
from pydub import AudioSegment

from tqdm import tqdm
from multiprocessing import Pool, current_process
import pickle

import re
import tgre

def praat_formants(audio_filename, results_filename,sizeframe,step, n_formants=5, max_formant=5500):
    
    """
    runs FormantsPraat script to obtain the formants for a wav file.
    It writes the results into a text file.
    These results can then be read using the function decodeFormants.

    :param audio_filaname: Full path to the wav file, string
    :param results_filename: Full path to the resulting file with the formants
    :param sizeframe: window size 
    :param step: time step to compute formants
    :param n_formants: number of formants to look for
    :param max_formant: maximum frequencyof formants to look for
    :returns: nothing
    """
    path_praat_script=os.path.dirname(os.path.abspath(__file__)) + "/../praat/"
    command='praat --run '+path_praat_script+'/FormantsPraat.praat '
    command+=audio_filename + ' '+results_filename+' '
    command+=str(n_formants)+' '+ str(max_formant) + ' '
    command+=str(float(sizeframe)/2)+' '
    command+=str(float(step))
    os.system(command) #formant extraction praat

def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice/articulation',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--base_path_phf', default='/homes/ssd1/jackchen/gop_prediction/data',
                        help='path of the base directory')
    parser.add_argument('--filepath', default='/homes/ssd1/jackchen/DisVoice/data/Segmented_ADOS_emotion_normalized',
                        help='/homes/ssd1/jackchen/DisVoice/data/{Segmented_ADOS_normalized|Session_ADOS_normalized|Segmented_ADOS_TD_normalized}')
    parser.add_argument('--trnpath', default='/mnt/sdd/jackchen/egs/formosa/s6/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/kid/ADOS_tdnn_fold_transfer',
                        help='/mnt/sdd/jackchen/egs/formosa/s6/{Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/{kid|kid_TD}/ADOS_tdnn_fold_transfer | Alignment_human/kid/Audacity_phone|')
    parser.add_argument('--outpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Inspect',
                        help='path of the base directory')
    parser.add_argument('--avgmethod', default='mean',
                        help='path of the base directory')
    parser.add_argument('--check', default=False,
                        help='path of the base directory')
    parser.add_argument('--checkreliability', default=False,
                            help='path of the base directory')
    parser.add_argument('--PoolFormantWindow', default=3, type=int,
                            help='path of the base directory')
    parser.add_argument('--Inspect_features', default=['F1','F2'],
                            help='')
    parser.add_argument('--Exportwav', default=True,
                            help='path of the base directory')
    parser.add_argument('--Exportpraat', default=True,
                            help='path of the base directory')
    args = parser.parse_args()
    return args

args = get_args()
base_path=args.base_path
filepath=args.filepath
trnpath=args.trnpath
outpath=args.outpath
AVERAGEMETHOD=args.avgmethod
path_app = base_path+'/../'
sys.path.append(path_app)
PoolFormantWindow=args.PoolFormantWindow
import praat.praat_functions as praat_functions
from script_mananger import script_manager
from utils_jack import *
from utils_jack  import functional_method

def decodeFormants(fileTxt):
	"""
	Read the praat textgrid file for formants and return the array
	
	:param fileTxt: File with the formants, which can be generated using the
		 			function praat_formants
	:returns F1: Numpy array containing the values for the first formant
	:returns F2: Numpy array containing the values for the second formant
	"""
	fid=open(fileTxt)
	datam=fid.read()
	end_line1=multi_find(datam, '\n')
	F1=[]
	F2=[]
	ji=10
	while (ji<len(end_line1)-1):
		line1=datam[end_line1[ji]+1:end_line1[ji+1]]
		cond=(line1=='3' or line1=='4' or line1=='5')
# 		cond2=(line1 in [str(i) for i in range(10)])
		if (cond):
			F1.append(float(datam[end_line1[ji+1]+1:end_line1[ji+2]]))
			F2.append(float(datam[end_line1[ji+3]+1:end_line1[ji+4]]))
		ji=ji+1
	F1=np.asarray(F1)
	F2=np.asarray(F2)
	return F1, F2

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
        # print("Length of the audio file = {audio}, length of F1 = {F1}, length of F2 = {F2}, length of F0 = {F0}".format(\
        #                                                             audio=int(len(data_audio)/fs/0.02), F1=len(F1), F2=len(F2), F0= len(F0)))
        # assert int(len(data_audio)/fs/0.02)- len(F1) ==2
        
        os.remove(temp_filename)
        
        return [F1, F2]

# This function measures formants using Formant Position formula
def measureFormants(sound, wave_file, f0min,f0max, time_step=0.0025,MaxnumForm=5,Maxformant=5000,framesize=0.025):
    sound = parselmouth.Sound(sound) # read the sound
    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    
    formants = call(sound, "To Formant (burg)", time_step, MaxnumForm, Maxformant, framesize, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    
    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)
    
    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
    
    return f1_list, f2_list

    
    
from HYPERPARAM import phonewoprosody, Label
PhoneMapp_dict={'u:':phonewoprosody.Phoneme_sets['u_'],\
                'i:':phonewoprosody.Phoneme_sets['i_'],\
                'A:':phonewoprosody.Phoneme_sets['A_']}
PhoneOI=['i:','u:','A:']
# =============================================================================
role_str=trnpath.split("/")[-2]
role= '_D_' if role_str == 'doc' else '_K_'

files=glob.glob(trnpath+"/*{}*.txt".format(role))

silence_duration=0.02 #0.1s
silence_duration_ms=silence_duration*1000
silence = AudioSegment.silent(duration=silence_duration_ms)

praat_outpath=outpath+'/praat/'

# =============================================================================
'''

    phone sequence extractor/creator

'''
inspect_people=['2015_12_06_01_097', '2015_12_13_01_153', '2016_01_26_02_108_1',
       '2016_06_25_01_070_1', '2016_06_27_02_017_1', '2016_07_05_01_135_1',
       '2016_07_06_01_078_1', '2016_07_30_01_148', '2016_08_12_01_179_1',
       '2016_08_18_01_163_1', '2016_08_26_01_168_1', '2016_08_27_01_044_1',
       '2016_09_21_01_191_1', '2016_09_24_01_174_1', '2016_10_21_01_202_1']  # manual select certain person
# inspect_people=list(set([os.path.basename(file)[:re.search("_[K|D]_", os.path.basename(file)).start()] for file in files]))

inspect_phone=PhoneOI
files= [file for file in files for insp in inspect_people  if insp in file]
Wav_collect=Dict()
Trn_collect=Dict()
for s in PhoneOI:
    for insp in inspect_people:
        Trn_collect[insp][s].basetime=0.0


# =============================================================================
'''

    Manual area

'''        
max_formant=5500
utterance_collect_bag=Dict()
# =============================================================================
Formants_people_symb=Dict()
Formants_utt_symb=Dict()
for file in tqdm(sorted(files)[:]):
    filename=os.path.basename(file).split(".")[0]
    spkr_name=filename[:re.search("_[K|D]_", filename).start()]
    utt='_'.join(filename.split("_")[:])
    
    
    trn=trnpath+"/{name}.txt".format(name=filename)
    df_segInfo=pd.read_csv(trn, header=None,delimiter='\t')
    if 'Session' in filepath:
        audiofile=filepath+"/{name}.wav".format(name='_'.join(filename.split("_")[:-1]))
    elif 'Segment' in filepath:
        audiofile=filepath+"/{name}.wav".format(name=filename)
    else:
        raise OSError(os.strerror, 'not allowed filepath')
    audio = AudioSegment.from_wav(audiofile)
    
    gender_query_str=filename[:re.search("_[K|D]_", filename).start()]
    role=filename[re.search("[K|D]", filename).start()]
    if role =='D':
        gender='female'
    elif role =='K':
        series_gend=Info_name_sex[Info_name_sex['name']==gender_query_str]['sex']
        gender=series_gend.values[0]
    
    minf0=F0_parameter_dict[gender]['f0_min']
    maxf0=F0_parameter_dict[gender]['f0_max']
    
    F1F2_extractor=Extract_F1F2(maxf0=maxf0, minf0=minf0)

    for i, (st,ed,symb) in enumerate(df_segInfo.values):
        if symb not in [x for s in PhoneOI for x in  PhoneMapp_dict[s] ]:
            continue
        
        ''' Allow an extention of a half window length  for audio segment calculation'''
        st_ext= max(st - F1F2_extractor.sizeframe/2,0)
        ed_ext= min(ed + F1F2_extractor.sizeframe/2,max(df_segInfo[1]))
        # segment_lengths.append((ed-st)) # np.quatile(segment_lengths,0.05)=0.08
        # st_ms=st * 1000 #Works in milliseconds
        # ed_ms=ed * 1000 #Works in milliseconds
        st_ms=st_ext * 1000 #Works in milliseconds
        ed_ms=ed_ext * 1000 #Works in milliseconds

        audio_segment = silence + audio[st_ms:ed_ms] + silence
        temp_outfile=F1F2_extractor.PATH+'/../tempfiles/tempwav{}.wav'.format(utt+symb)
        
        audio_segment.export(temp_outfile, format="wav")
        # =============================================================================
        ''' Method Disvoice '''         
        # =============================================================================
        
        # [F1,F2]=F1F2_extractor.extract_features_file(temp_outfile)
        
        name_audio=temp_outfile.split('/')
        temp_uuid='artic'+name_audio[-1][0:-4]

        temp_filename=F1F2_extractor.PATH+'/../tempfiles/tempFormants'+temp_uuid+'.txt'
        praat_formants(temp_outfile, temp_filename,F1F2_extractor.sizeframe,F1F2_extractor.step, n_formants=5, max_formant=max_formant)
        ''' # [F1_, F2_]=decodeFormants(temp_filename)''' 
        fid=open(temp_filename)
        datam=fid.read()
        end_line1=multi_find(datam, '\n')
        F1=[]
        F2=[]
        ji=10
        while (ji<len(end_line1)-1):
            line1=datam[end_line1[ji]+1:end_line1[ji+1]]
            cond=(line1=='3' or line1=='4' or line1=='5')
    # 		cond2=(line1 in [str(i) for i in range(10)])
            if (cond):
                F1.append(float(datam[end_line1[ji+1]+1:end_line1[ji+2]]))
                F2.append(float(datam[end_line1[ji+3]+1:end_line1[ji+4]]))
            ji=ji+1
        F1=np.asarray(F1)
        F2=np.asarray(F2)
        
        os.remove(temp_filename)
        
        # =============================================================================
        ''' Method Praat '''         
        # =============================================================================
        
        wave_file=    temp_outfile
        sound = parselmouth.Sound(wave_file)
        f0min,f0max=minf0, maxf0
        # F1_list,F2_list=measureFormants(sound,wave_file,f0min,f0max,time_step=F1F2_extractor.step,MaxnumForm=5,Maxformant=max_formant,framesize=F1F2_extractor.sizeframe)
        
        
        time_step=F1F2_extractor.step
        MaxnumForm=5
        Maxformant=max_formant
        framesize=F1F2_extractor.sizeframe
        sound = parselmouth.Sound(sound) # read the sound
        pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        
        formants = call(sound, "To Formant (burg)", time_step, MaxnumForm, Maxformant, framesize, 50)
        numPoints = call(pointProcess, "Get number of points")
    
        f1_list = []
        f2_list = []
        f3_list = []
        f4_list = []
        
        # Measure formants only at glottal pulses
        for point in range(0, numPoints):
            point += 1
            t = call(pointProcess, "Get time from index", point)
            f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
            f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
            f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
            f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)
            f4_list.append(f4)
        
        f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
        f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
        f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
        f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
        
        F1_list=f1_list
        F2_list=f2_list
        
        os.remove(temp_outfile)
        
        
        
        
        if len(F1_list) > 0:
            assert len(F1_list) == len(F2_list)
            utt_phone_name='{0}{1}'.format(utt,symb)
            if utt_phone_name not in utterance_collect_bag.keys():
                utterance_collect_bag[utt_phone_name]=[]
            utterance_collect_bag[utt_phone_name].append(F1_list)
        
        
        if len(F1) == 0 or len(F2)==0:
            F1_static,F2_static = -1,-1
        else:
            F1_static=functional_method(F1,method=AVERAGEMETHOD,window=PoolFormantWindow)
            F2_static=functional_method(F2,method=AVERAGEMETHOD,window=PoolFormantWindow)
        