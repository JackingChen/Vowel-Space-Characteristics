
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa
modified by Jackchen 20210524

This script is to extract word level phonation feature 
    input: {trnpath,filepath} NOTE! utterance name of .wav and .txt should match
    
    or user can use precomputed checkpoints

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
from phonation import Phonation_jack as Phonation
    
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

def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--filepath', default='/homes/ssd1/jackchen/DisVoice/data/Segmented_ADOS_emotion_normalized',
                        help='path of the base directory')
    parser.add_argument('--trnpath', default='/mnt/sdd/jackchen/egs/formosa/s6/Alignment_ADOShappyDAAIKidallDeceiptformosaCSRC_chain/kid/ADOS_tdnn_fold_transfer_word',
                        help='path of the base directory')
    parser.add_argument('--checkpointpath', default='/homes/ssd1/jackchen/DisVoice/phonation/features',
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
Phonation_utt_symb=Dict()
Phonation_people_symb=Dict()
# =============================================================================
dataset_name=os.path.basename(filepath)
dataset_name=dataset_name[dataset_name.find('ADOS'):dataset_name.find('normalized')-1]
role=trnpath.split("/")[-2]
dataset_name=dataset_name+"_"+role

files=glob.glob(trnpath+"/*.txt")
# if os.path.exists('Gen_formant_multiprocess.log'):
#     os.remove('Gen_formant_multiprocess.log')

'''
    Phonation_utt_symb[utt] -> df_feat_word: index= word, columns= 28 dim phonation features
    process: input: {trnfiles, audio_files} 
        1. get the Alignment file and get the word occurance timings
            a. the word occurance timings are extended by one window size
        2. cut out the word occurances -> output: temp_outfile
        3. calculate the phonation features (7*4 dim)
        
        

'''
chkptpath=PhonationPath+"/Phonation_dict_bag_{}_WordLevel.pkl".format(dataset_name)
segment_lengths=[]
if not os.path.exists(chkptpath):
    Phonation_dict_bag=Dict()
    for file in tqdm(files[:]):
        filename=os.path.basename(file).split(".")[0]
        spkr_name='_'.join(filename.split("_")[:-3])
        utt='_'.join(filename.split("_")[:])
        audiofile=filepath+"/{name}.wav".format(name=filename)
        trn=trnpath+"/{name}.txt".format(name=filename)
        df_segInfo=pd.read_csv(trn, header=None,delimiter='\t')
        
        audio = AudioSegment.from_wav(audiofile)
        phonation_extractor=Phonation()
        
        
        for st,ed,symb in df_segInfo.values:
            ''' Allow an extention of a window length  for audio segment calculation'''
            st_ext= np.max(st - phonation_extractor.size_frame,0)
            ed_ext= min(ed + phonation_extractor.size_frame,max(df_segInfo[1]))
            # segment_lengths.append((ed-st)) # np.quantile(segment_lengths,0.05)=0.08
            st_ms=st * 1000 #Works in milliseconds
            ed_ms=ed_ext * 1000 #Works in milliseconds
    
            audio_segment = audio[st_ms:ed_ms] 
            if audio_segment.duration_seconds <0.05:
                continue
            temp_outfile=phonation_extractor.PATH+'/../tempfiles/tempwav{}.wav'.format(utt+symb)
            
            audio_segment.export(temp_outfile, format="wav")
            df_feat_word=phonation_extractor.extract_features_file(temp_outfile)

            

            os.remove(temp_outfile)
            if np.isnan(df_feat_word.values).any() == False :            
                df_feat_word.index=[symb]
                if utt not in  Phonation_utt_symb.keys():
                    Phonation_utt_symb[utt]=df_feat_word
                else:
                    Phonation_utt_symb[utt]=pd.concat([Phonation_utt_symb[utt],df_feat_word],axis=0)
    tmpPath=PhonationPath + "/features"
    if not os.path.exists(tmpPath):
        os.makedirs(tmpPath)
    pickle.dump(Phonation_utt_symb,open(chkptpath,"wb"))
else:
    Phonation_utt_symb=pickle.load(open(chkptpath,"rb"))

'''

    Reconstruct Phonation_utt_symb[utt] into Phonation_people_symb[people]

'''
chkptpath=PhonationPath+"/features/Phonation_people_symb_{}_WordLevel.pkl".format(dataset_name)
if not os.path.exists(chkptpath):
    for keys, values in Phonation_utt_symb.items():
        spkr_name='_'.join(keys.split("_")[:-3])
        role=keys.split("_")[-3]
        emotion=keys.split("_")[-1]
        if spkr_name not in  Phonation_people_symb.keys():
            Phonation_people_symb[spkr_name]=values
        else:
            Phonation_people_symb[spkr_name]=pd.concat([Phonation_people_symb[spkr_name],values],axis=0)
    pickle.dump(Phonation_people_symb,open(chkptpath,"wb"))
else:
    Phonation_people_symb=pickle.load(open(chkptpath,"rb"))

'''
    
    Get session level word phonation features
    df_session_phonation_role: index= people, columns= 28 dim phonation features
    process: mean over all phonation words
'''
chkptpath_role=PhonationPath+"/features/df_phonation_{}_WordLevel.pkl".format(dataset_name)
df_session_phonation_role=pd.DataFrame([],columns=Phonation_people_symb[list(Phonation_people_symb.keys())[0]].columns)
for keys, values in Phonation_people_symb.items():
    df_session_phonation_role.loc[keys]=values.mean(axis=0)
pickle.dump(df_session_phonation_role,open(chkptpath_role,"wb"))


# =============================================================================
'''

    t-test area

'''
# =============================================================================

chkptpath_doc_ASD=PhonationPath+"/features/df_phonation_ADOS_emotion_doc_WordLevel.pkl"
chkptpath_kid_ASD=PhonationPath+"/features/df_phonation_ADOS_emotion_kid_WordLevel.pkl"
TTest_Cmp_checkpoint(chkptpath_doc_ASD,chkptpath_kid_ASD)


