#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:14:50 2020

@author: jackchen

THis script collects the wav data and their utterance name and merge it to one 
file, into a praat file for user to have a deeper analysis on phone level feature analyses
                           
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
    from articulation_functions import extractTrans, V_UV, measureFormants
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
from articulation import Extract_F1F2
import re
import tgre

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
    parser.add_argument('--formantmethod', default='praat',
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


from HYPERPARAM import phonewoprosody, Label
PhoneMapp_dict=phonewoprosody.PhoneMapp_dict
PhoneOI=phonewoprosody.PhoneOI
functional_method_window=PoolFormantWindow
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
inspect_people=['2016_06_27_02_017_1', '2016_07_30_01_148', '2016_08_26_01_168_1',
       '2016_09_24_01_174_1']  # manual select certain person
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
# utterance_collect_bag=[]
# =============================================================================
Formants_people_symb=Dict()
Formants_utt_symb=Dict()
error_msg_bag=[]
for file in tqdm(files):
    if '2ndPass' in file:
        namecode='2nd_pass'
    else:
        namecode='1st_pass'
        
    filename=os.path.basename(file).split(".")[0]
    spkr_name='_'.join(filename.split("_")[:-3])
    utt='_'.join(filename.split("_")[:])
    if 'Session' in filepath:
        audiofile=filepath+"/{name}.wav".format(name='_'.join(filename.split("_")[:-1]))
    elif 'Segment' in filepath:
        audiofile=filepath+"/{name}.wav".format(name=filename)
    else:
        raise OSError(os.strerror, 'not allowed filepath')
    trn=trnpath+"/{name}.txt".format(name=filename)
    df_segInfo=pd.read_csv(trn, header=None,delimiter='\t')
    
    audio = AudioSegment.from_wav(audiofile)
    
    gender_query_str='_'.join(filename.split("_")[:Namecode_dict[namecode]['role']])
    role=filename.split("_")[Namecode_dict[namecode]['role']]
    if role =='D':
        gender='female'
    elif role =='K':
        series_gend=Info_name_sex[Info_name_sex['name']==gender_query_str]['sex']
        gender=series_gend.values[0]
    
    minf0=F0_parameter_dict[gender]['f0_min']
    maxf0=F0_parameter_dict[gender]['f0_max']
    
    F1F2_extractor=Extract_F1F2(maxf0=maxf0, minf0=minf0)
    
    for st,ed,symb in df_segInfo.values:
        ''' Allow an extention of a half window length  for audio segment calculation'''
        st_ext= max(st - F1F2_extractor.sizeframe/2,0)
        ed_ext= min(ed + F1F2_extractor.sizeframe/2,max(df_segInfo[1]))
        # segment_lengths.append((ed-st)) # np.quatile(segment_lengths,0.05)=0.08
        st_ms=st * 1000 #Works in milliseconds
        ed_ms=ed * 1000 #Works in milliseconds
        # st_ms=st_ext * 1000 #Works in milliseconds
        # ed_ms=ed_ext * 1000 #Works in milliseconds

        audio_segment = silence + audio[st_ms:ed_ms] + silence
        temp_outfile=F1F2_extractor.PATH+'/../tempfiles/tempwav{}.wav'.format(utt+symb)
        
        audio_segment.export(temp_outfile, format="wav")
        if args.formantmethod == 'Disvoice':
            [F1,F2]=F1F2_extractor.extract_features_file(temp_outfile)
        elif args.formantmethod == 'praat':
            try:
                MaxnumForm=5
                if 'u:' in symb:
                    maxFormant=3000
                else:
                    maxFormant=5000
                [F1,F2]=measureFormants(temp_outfile,minf0,maxf0,time_step=F1F2_extractor.step,MaxnumForm=MaxnumForm,Maxformant=maxFormant,framesize=F1F2_extractor.sizeframe)
            except :
                print("Error processing ",utt+"__"+symb)
                error_msg_bag.append(utt+"__"+symb)
        
        
        if len(F1) == 0 or len(F2)==0:
            F1_static, F2_static= -1, -1
        else:
            F1_static=functional_method(F1,method=AVERAGEMETHOD,window=functional_method_window)
            F2_static=functional_method(F2,method=AVERAGEMETHOD,window=functional_method_window)
        
        
        assert  math.isnan(F1_static) == False and math.isnan(F2_static) == False
        os.remove(temp_outfile)
        
        tmp_dict=Dict()
        tmp_dict[symb].F1=F1_static
        tmp_dict[symb].F2=F2_static
        df_tmp=pd.DataFrame.from_dict(tmp_dict)
        if utt not in  Formants_utt_symb.keys():
            Formants_utt_symb[utt]=df_tmp
        else:
            Formants_utt_symb[utt]=pd.concat([Formants_utt_symb[utt],df_tmp],axis=1)
        
        if len(F1) != 0 and len(F2)!=0:
            if spkr_name not in Formants_people_symb.keys():
                if symb not in Formants_people_symb[spkr_name].keys():
                    Formants_people_symb[spkr_name][symb]=[[F1_static, F2_static]]
                elif symb in Formants_people_symb[spkr_name].keys():
                    Formants_people_symb[spkr_name][symb].append([F1_static, F2_static])
            else:
                if symb not in Formants_people_symb[spkr_name].keys():
                    Formants_people_symb[spkr_name][symb]=[[F1_static, F2_static]]
                elif symb in Formants_people_symb[spkr_name].keys(): 
                    Formants_people_symb[spkr_name][symb].append([F1_static, F2_static])
    Formants_utt_symb[utt] = Formants_utt_symb[utt].T
    df=pd.DataFrame(df_segInfo[[0,1]].values,index=df_segInfo[2])
    Formants_utt_symb[utt]['start']=df[0]
    Formants_utt_symb[utt]['end']=df[1]
                
    dur=ed_ext-st_ext
    for s in PhoneOI:
        if symb in [x for x in  PhoneMapp_dict[s]]:
            if s not in Wav_collect[spkr_name].keys():
                Wav_collect[spkr_name][s]=audio[st_ms:ed_ms]
                start=Trn_collect[spkr_name][s].basetime
                end=Trn_collect[spkr_name][s].basetime+dur
                Trn_collect[spkr_name][s].trn='{0}\t{1}\t{2}:{3}\n'.format(start,end,dur,utt,symb)
                Trn_collect[spkr_name][s].praat=[tgre.Interval(start, end, str(utt+symb))]
                Trn_collect[spkr_name][s].basetime+=dur
            else:
                Wav_collect[spkr_name][s]=Wav_collect[spkr_name][s] + silence + audio[st_ms:ed_ms] + silence
                start=Trn_collect[spkr_name][s].basetime + silence_duration
                end=Trn_collect[spkr_name][s].basetime + silence_duration + dur
                
                Trn_collect[spkr_name][s].trn+='{0}\t{1}\t{2}:{3}\n'.format(start,end,utt,symb)
                Trn_collect[spkr_name][s].praat.append(tgre.Interval(start, end, str(utt+symb)))
                Trn_collect[spkr_name][s].basetime+=silence_duration + dur + silence_duration                    
    Formants_utt_symb[utt] = Formants_utt_symb[utt].T
# =============================================================================
'''
# Collect the segment wavfile of phone of interest and stick them together separated by a silence
'''
# =============================================================================
if not os.path.exists(praat_outpath):
    os.makedirs(praat_outpath)


for people in Wav_collect.keys():
    praat_person_path=praat_outpath+people+'/'
    if not os.path.exists(praat_person_path):
        os.makedirs(praat_person_path)
    for symb in Wav_collect[people].keys():
        outfilename=people+symb
        
        if args.Exportwav:
            Wav_collect[people][symb].export(praat_person_path+"/"+outfilename+".wav", format="wav")
        
        if args.Exportpraat:
            max_length=Trn_collect[people][symb].praat[-1].xmax+0.1
            tier_phone = tgre.IntervalTier('phone', 0, max_length, items=Trn_collect[people][symb].praat)
            new_tg = tgre.TextGrid(0, max_length, tiers=[tier_phone])
            new_tg.to_praat(path=praat_person_path+"/"+outfilename+".praat")
        
# =============================================================================
'''
# Collect utterances in person
# the transcripts are converted to praat format using function: Audacity2Praat
'''
import shutil
outputpath_filefilter=outpath+'/fileOfInterest'

# out path structure $outpath/people/symb
def Audacity2Praat(origin_trn_file):
    df_segInfo=pd.read_csv(origin_trn_file, header=None,delimiter='\t')
    
    overlapp_identifier=np.where(df_segInfo[1].shift() - df_segInfo[0] >0.0)[0]
    if len(overlapp_identifier) > 0:
        for j in range(len(overlapp_identifier)):
            df_segInfo.loc[overlapp_identifier[j]-1,1] = df_segInfo.iloc[overlapp_identifier[j]][0]
    
    interval_lst=[]
    for i, (st,ed,symb) in enumerate(df_segInfo.values):
        interval_lst.append(tgre.Interval(st, ed, str(symb)))
    
    
    max_length=ed+0.1
    tier_phone = tgre.IntervalTier('phone', 0, max_length, items=interval_lst)
    new_tg = tgre.TextGrid(0, max_length, tiers=[tier_phone])
    return new_tg
# =============================================================================
for utt in Formants_utt_symb.keys():
    if len(Formants_utt_symb) == 0:
        continue
    people=utt[:re.search("_[K|D]_", utt).start()]
    df_tmp=Formants_utt_symb[utt]
    df_tmp['text']=df_tmp.index
    for symb in PhoneOI:
        outpath_ffl="{root}/{people}/{symb}/".format(root=outputpath_filefilter,people=people,symb=symb)
        if not os.path.exists(outpath_ffl):
            os.makedirs(outpath_ffl)
        origin_wav_file=args.filepath+'/'+utt+".wav"
        origin_trn_file=args.trnpath+'/'+utt+".txt"
        new_tg=Audacity2Praat(origin_trn_file)
        
        
        
        shutil.copy(origin_wav_file,outpath_ffl+utt+".wav")
        # shutil.copy(origin_trn_file,outpath_ffl+utt+".txt")
        new_tg.to_praat(path=outpath_ffl+utt+".praat")
        

        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        