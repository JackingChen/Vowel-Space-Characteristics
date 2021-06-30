
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa
modified by Jackchen 20210524

This script is to extract word level phonation feature 
    input: {trnpath,filepath} NOTE! utterance name of .wav and .txt should match
    
    or user can use precomputed checkpoints


Wordlevel: np.quantile(segment_lengths,0.05)=0.08 
Single_Wordlevel: np.quantile(segment_lengths,0.3)=0.09
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
from multiprocessing import Pool, current_process
import   re

def TTest_Cmp_checkpoint(chkptpath_1,chkptpath_2):
    df_phonation_1=pickle.load(open(chkptpath_1,"rb"))
    df_phonation_2=pickle.load(open(chkptpath_2,"rb"))
    name1=os.path.basename(chkptpath_1).replace(".pkl","")
    name2=os.path.basename(chkptpath_2).replace(".pkl","")
    
    dataset_str='{name1}vs{name2}'.format(name1=name1,name2=name2)
    df_ttest_result=pd.DataFrame()
    for col in df_phonation_1.columns:
        df_ttest_result.loc[dataset_str+"-p-val",col]=stats.ttest_ind(df_phonation_1[col].dropna(),df_phonation_2[col].dropna())[1].astype(float)
        df_ttest_result.loc['{0}-{1}'.format(name1,name2),col] = df_phonation_1[col].dropna().mean() - df_phonation_2[col].dropna().mean()
    df_ttest_result=df_ttest_result.T 
    
    result_path="RESULTS/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    df_ttest_result.to_excel(result_path+dataset_str+".xlsx")
    
    return df_ttest_result

def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--filepath', default='/homes/ssd1/jackchen/DisVoice/data/Segmented_ADOS_TD_normalized',
                        help='path of the base directory')
    parser.add_argument('--trnpath', default='/mnt/sdd/jackchen/egs/formosa/s6/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/kid_TD/ADOS_tdnn_fold_transfer',
                        help='path of the base directory')
    parser.add_argument('--checkpointpath', default='/homes/ssd1/jackchen/DisVoice/phonation/features',
                        help='path of the base directory')
    parser.add_argument('--ADOSlabel', default='/homes/ssd1/jackchen/gop_prediction/label_ADOS_77.xlsx',
                        help='path of the base directory')
    parser.add_argument('--check', default=False,
                        help='path of the base directory')
    parser.add_argument('--checkreliability', default=False,
                            help='path of the base directory')
    parser.add_argument('--method', default='praat',
                            help='Disvoice|praat')
    args = parser.parse_args()
    return args

args = get_args()
base_path=args.base_path
filepath=args.filepath
trnpath=args.trnpath
method=args.method
path_app = base_path
sys.path.append(path_app)
checkpointpath_manual=args.checkpointpath
PhonationPath=base_path + "/phonation" 


import praat.praat_functions as praat_functions
from script_mananger import script_manager
from utils_jack import *
from script_mananger import script_manager


# =============================================================================
'''

    Manual area

'''
F0_parameter_dict=Dict()
F0_parameter_dict['male']={'f0_min':75,'f0_max':400}
F0_parameter_dict['female']={'f0_min':75,'f0_max':800}

Phonation_utt_symb=Dict()
Phonation_people_symb=Dict()
# =============================================================================
if '2ndpass' in filepath:
    dataset_name=os.path.basename(filepath)
    namecode='2nd_pass'
else:
    dataset_name=os.path.basename(filepath)
    dataset_name=dataset_name[dataset_name.find('ADOS'):dataset_name.find('normalized')-1]
    namecode='1st_pass'


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

import parselmouth 
from parselmouth.praat import call
import statistics
import inspect



def measurePitch(voiceID, f0min, f0max, unit):
    columns=['duration', 'meanF0', 'stdevF0', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'aqpq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    sound = parselmouth.Sound(voiceID) # read the sound
    duration = call(sound, "Get total duration") # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    df=pd.DataFrame(np.array([duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer]),index=columns)
    return df.T



def Get_phonationdictbag_map(files, Info_name_sex, Namecode_dict):
    print(" process PID", os.getpid(), " running")
    Phonation_utt_symb=Dict()
    for file in files:
        filename=os.path.basename(file).split(".")[0]
        spkr_name=filename[:re.search("_[K|D]_", filename).start()]
        utt='_'.join(filename.split("_")[:])
        audiofile=filepath+"/{name}.wav".format(name=filename)
        trn=trnpath+"/{name}.txt".format(name=filename)
        df_segInfo=pd.read_csv(trn, header=None,delimiter='\t')
        
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
        
        phonation_extractor=Phonation(maxf0=maxf0, minf0=minf0)
        
        
        for st,ed,symb in df_segInfo.values:
            ''' Allow an extention of a window length  for audio segment calculation'''
            st_ext= np.max(st - phonation_extractor.size_frame,0)
            ed_ext= min(ed + phonation_extractor.size_frame,max(df_segInfo[1]))
            # segment_lengths.append((ed-st)) # np.quantile(segment_lengths,0.05)=0.08
            if method == "Disvoice":
                st_ms=st * 1000 #Works in milliseconds
                ed_ms=ed_ext * 1000 #Works in milliseconds
            elif method == "praat":
                st_ms=st_ext * 1000 #Works in milliseconds
                ed_ms=ed_ext * 1000 #Works in milliseconds
                
            audio_segment = audio[st_ms:ed_ms] 
            if audio_segment.duration_seconds <0.05:
                continue
            temp_outfile=phonation_extractor.PATH+'/../tempfiles/tempwav{}.wav'.format(utt+symb)
            
            audio_segment.export(temp_outfile, format="wav")
            
            if method == "Disvoice":
                df_feat_word=phonation_extractor.extract_features_file(temp_outfile)
            elif method == "praat":
                df_feat_word=measurePitch(temp_outfile, minf0, maxf0, "Hertz")

            
            os.remove(temp_outfile)
            if np.isnan(df_feat_word.values).any() == False :            
                df_feat_word.index=[symb]
                if utt not in  Phonation_utt_symb.keys():
                    Phonation_utt_symb[utt]=df_feat_word
                else:
                    Phonation_utt_symb[utt]=pd.concat([Phonation_utt_symb[utt],df_feat_word],axis=0)
    print("PID {} Getting ".format(os.getpid()), "Done")
    return  Phonation_utt_symb

''' multiprocessin area  '''

interval=20
pool = Pool(int(os.cpu_count()))


keys=[]
for i in range(0,len(files),interval):
    # print(list(combs_tup.keys())[i:i+interval])
    keys.append(files[i:i+interval])
flat_keys=[item for sublist in keys for item in sublist]
assert len(flat_keys) == len(files)

final_result = pool.starmap(Get_phonationdictbag_map, [([file_block,Info_name_sex, Namecode_dict]) \
                                                  for file_block in tqdm(keys)])

    
Phonation_utt_symb
for fin_resut_dict in final_result:    
    for keys, values in fin_resut_dict.items():
        Phonation_utt_symb[keys]=values

if args.check:
    chkptpath_cmp=PhonationPath+"/Phonation_dict_bag_{}_WordLevel_cmp.pkl".format(dataset_name)
    Phonation_utt_symb_cmp=pickle.load(open(chkptpath_cmp,"rb"))
    for keys, values in Phonation_utt_symb.items():
        assert (sum((Phonation_utt_symb_cmp[keys] - values).values) == np.zeros(Phonation_utt_symb_cmp[keys].values.shape)).all()

chkptpath=PhonationPath+"/Phonation_dict_bag_{}_SegmentLevel.pkl".format(dataset_name)
pickle.dump(Phonation_utt_symb,open(chkptpath,"wb"))
print("Finished creating Phonation_utt_symb in", chkptpath)
''' multiprocessin area   done'''



chkptpath=PhonationPath+"/Phonation_dict_bag_{}_SegmentLevel.pkl".format(dataset_name)


''' manual area '''
# segment_lengths=[]
# for file in tqdm(files[:]):
#     if '2ndPass' in file:
#         namecode='2nd_pass'
#     else:
#         namecode='1st_pass'
    
#     filename=os.path.basename(file).split(".")[0]
#     spkr_name='_'.join(filename.split("_")[:Namecode_dict[namecode]['role']])
#     utt='_'.join(filename.split("_")[:])
#     audiofile=filepath+"/{name}.wav".format(name=filename)
#     trn=trnpath+"/{name}.txt".format(name=filename)
#     df_segInfo=pd.read_csv(trn, header=None,delimiter='\t')
    
#     audio = AudioSegment.from_wav(audiofile)        

#     gender_query_str='_'.join(filename.split("_")[:Namecode_dict[namecode]['role']])
#     role=filename.split("_")[Namecode_dict[namecode]['role']]
#     if role =='D':
#         gender='female'
#     elif role =='K':
#         series_gend=Info_name_sex[Info_name_sex['name']==gender_query_str]['sex']
#         gender=series_gend.values[0]
    
#     minf0=F0_parameter_dict[gender]['f0_min']
#     maxf0=F0_parameter_dict[gender]['f0_max']
    
    
#     phonation_extractor=Phonation(maxf0=maxf0, minf0=minf0)

    
#     for st,ed,symb in df_segInfo.values:
#         ''' Allow an extention of a window length  for audio segment calculation'''
#         st_ext= np.max(st - phonation_extractor.size_frame,0)
#         ed_ext= min(ed + phonation_extractor.size_frame,max(df_segInfo[1]))
#         # segment_lengths.append((ed-st)) # np.quantile(segment_lengths,0.05)=0.08
#         if method == "Disvoice":
#             st_ms=st * 1000 #Works in milliseconds
#             ed_ms=ed_ext * 1000 #Works in milliseconds
#         elif method == "praat":
#             st_ms=st_ext * 1000 #Works in milliseconds
#             ed_ms=ed_ext * 1000 #Works in milliseconds

#         audio_segment = audio[st_ms:ed_ms] 
#         if audio_segment.duration_seconds <0.05:
#             continue
#         temp_outfile=phonation_extractor.PATH+'/../tempfiles/tempwav{}.wav'.format(utt+symb)
        
#         audio_segment.export(temp_outfile, format="wav")
        
#         if method == "Disvoice":
#             df_feat_word=phonation_extractor.extract_features_file(temp_outfile)
#         elif method == "praat":
#             df_feat_word=measurePitch(temp_outfile, minf0, maxf0, "Hertz")

        
#         os.remove(temp_outfile)
#         if np.isnan(df_feat_word.values).any() == False :            
#             df_feat_word.index=[symb]
#             if utt not in  Phonation_utt_symb.keys():
#                 Phonation_utt_symb[utt]=df_feat_word
#             else:
#                 Phonation_utt_symb[utt]=pd.concat([Phonation_utt_symb[utt],df_feat_word],axis=0)
# tmpPath=PhonationPath + "/features"
# if not os.path.exists(tmpPath):
#     os.makedirs(tmpPath)
# pickle.dump(Phonation_utt_symb,open(chkptpath,"wb"))
''' Manual area end '''
    
Phonation_utt_symb=pickle.load(open(chkptpath,"rb"))

'''

    Reconstruct Phonation_utt_symb[utt] into Phonation_people_symb[people]

'''
chkptpath=PhonationPath+"/features/Phonation_people_symb_{}_WordLevel.pkl".format(dataset_name)

for keys, values in tqdm(Phonation_utt_symb.items()):
    spkr_name='_'.join(keys.split("_")[:Namecode_dict[namecode]['role']])
    role=keys.split("_")[Namecode_dict[namecode]['role']]
    emotion=keys.split("_")[Namecode_dict[namecode]['emotion']]
    if spkr_name not in  Phonation_people_symb.keys():
        Phonation_people_symb[spkr_name]=values
    else:
        Phonation_people_symb[spkr_name]=pd.concat([Phonation_people_symb[spkr_name],values],axis=0)
pickle.dump(Phonation_people_symb,open(chkptpath,"wb"))


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

chkptpath_doc_ASD=PhonationPath+"/features/df_phonation_Segments_info_2ndpass_doc_WordLevel.pkl"
chkptpath_kid_ASD=PhonationPath+"/features/df_phonation_Segments_info_2ndpass_kid_WordLevel.pkl"
df_ttest_result_a=TTest_Cmp_checkpoint(chkptpath_doc_ASD,chkptpath_kid_ASD)


