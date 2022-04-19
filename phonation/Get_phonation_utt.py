
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa
modified by Jackchen 20210524

This script is to extract utterance level phonation feature 
    input: filepath
    
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
from phonation import Phonation_jack as Phonation

import parselmouth 
from parselmouth.praat import call
import statistics
import inspect
from multiprocessing import Pool, current_process

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
    parser.add_argument('--filepath', default='data/Segmented_ADOS_normalized',
                        help='data/{Segmented_ADOS_TD_normalized_untranscripted|Segmented_ADOS_normalized}')
    parser.add_argument('--checkpointpath', default='/homes/ssd1/jackchen/DisVoice/phonation/features',
                        help='path of the base directory')
    parser.add_argument('--check', default=False,
                        help='path of the base directory')
    parser.add_argument('--checkreliability', default=False,
                            help='path of the base directory')
    parser.add_argument('--method', default='Disvoice',
                            help='Disvoice|praat')
    args = parser.parse_args()
    return args

args = get_args()
base_path=args.base_path
filepath=args.filepath
path_app = base_path
method=args.method
sys.path.append(path_app)
checkpointpath_manual=args.checkpointpath
PhonationPath=base_path + "/phonation/features" 


import praat.praat_functions as praat_functions
from utils_jack import *
# =============================================================================
'''

    Manual area

'''
F0_parameter_dict=Dict()
F0_parameter_dict['male']={'f0_min':75,'f0_max':400}
F0_parameter_dict['female']={'f0_min':75,'f0_max':550}

Formants_utt_symb=Dict()
Formants_people_symb=Dict()
# =============================================================================
audiopath = os.path.join(base_path,filepath)
dataset_name=os.path.basename(audiopath)
dataset_name=dataset_name[dataset_name.find('ADOS'):dataset_name.find('normalized')-1]

files=glob.glob(audiopath+"/*.wav")

# if os.path.exists('Gen_formant_multiprocess.log'):
#     os.remove('Gen_formant_multiprocess.log')

''' multiprocessin area   done'''

def Get_phonationdictbag_map(files,Info_name_sex):
    print(" process PID", os.getpid(), " running")
    Phonation_dict_bag=Dict()
    for file in tqdm(files):
        name='_'.join(os.path.basename(file).replace(".wav","").split("_")[:-1])
        audiofile=file
        gender_query_str='_'.join(name.split("_")[:-1])
        role=name.split("_")[-1]
        if role =='D':
            gender='female'
        elif role =='K':
            gender=Info_name_sex[Info_name_sex['name']==gender_query_str]['sex'].values[0]
        
        minf0=F0_parameter_dict[gender]['f0_min']
        maxf0=F0_parameter_dict[gender]['f0_max']
        
        if method == "Disvoice":
            phonation_extractor=Phonation(maxf0=maxf0, minf0=minf0)
            # toy_files=glob.glob("data/Segmented_ADOS_normalized/2015_12_06_01_097_K_*.wav")
            # for f in toy_files:
            #     print(phonation_extractor.extract_features_file(audiofile))
            df_feat_utt=phonation_extractor.extract_features_file(audiofile)
        elif method == "praat":
            df_feat_utt=measurePitch(audiofile, minf0, maxf0, "Hertz")
        
        
        if name not in Phonation_dict_bag.keys():
            Phonation_dict_bag[name]=pd.DataFrame()
        Phonation_dict_bag[name]=Phonation_dict_bag[name].append(df_feat_utt)
    # tmpPath=PhonationPath + "/features"
    # if not os.path.exists(tmpPath):
    #     os.makedirs(tmpPath)
    print("PID {} Getting ".format(os.getpid()), "Done")
    return Phonation_dict_bag

interval=20
pool = Pool(int(os.cpu_count()))


keys=[]
for i in range(0,len(files),interval):
    # print(list(combs_tup.keys())[i:i+interval])
    keys.append(files[i:i+interval])
flat_keys=[item for sublist in keys for item in sublist]
assert len(flat_keys) == len(files)

final_result = pool.starmap(Get_phonationdictbag_map, [([file_block,Info_name_sex]) \
                                                  for file_block in tqdm(keys)])

    
Phonation_utt_symb=Dict()
for fin_resut_dict in final_result:    
    for keys, values in fin_resut_dict.items():
        if keys not in Phonation_utt_symb.keys():
            Phonation_utt_symb[keys]=pd.DataFrame()
        Phonation_utt_symb[keys]=Phonation_utt_symb[keys].append(values)
            
        

if args.check:
    chkptpath_cmp=PhonationPath+"/Phonation_dict_bag_{}_cmp.pkl".format(dataset_name)
    Phonation_utt_symb_cmp=pickle.load(open(chkptpath_cmp,"rb"))
    for keys, values in Phonation_utt_symb.items():
        assert ((Phonation_utt_symb_cmp[keys].dropna() - values.dropna()).values == np.zeros(Phonation_utt_symb_cmp[keys].dropna().values.shape)).all()

chkptpath=PhonationPath+"/Phonation_dict_bag_{}.pkl".format(dataset_name)
pickle.dump(Phonation_utt_symb,open(chkptpath,"wb"))

''' multiprocessin area  '''


'''
    Phonation_dict_bag[utt] -> df_feat_utt: columns= 28 dim phonation features
    process: input: audio_files
'''

chkptpath=PhonationPath+"/Phonation_dict_bag_{}.pkl".format(dataset_name)

if not os.path.exists(chkptpath):
    Phonation_dict_bag=Dict()
    for file in tqdm(files):
        name='_'.join(os.path.basename(file).replace(".wav","").split("_")[:-1])
        audiofile=file
        gender_query_str='_'.join(name.split("_")[:-1])
        role=name.split("_")[-1]
        if role =='D':
            gender='female'
        elif role =='K':
            series_gend=Info_name_sex[Info_name_sex['name']==gender_query_str]['sex']
            gender=series_gend.values[0]
        
        minf0=F0_parameter_dict[gender]['f0_min']
        maxf0=F0_parameter_dict[gender]['f0_max']
        
        if method == "Disvoice":
            phonation_extractor=Phonation(maxf0=maxf0, minf0=minf0)
            df_feat_utt=phonation_extractor.extract_features_file(audiofile)
        elif method == "praat":
            df_feat_utt=measurePitch(audiofile, minf0, maxf0, "Hertz")
        
        
        if name not in Phonation_dict_bag.keys():
            Phonation_dict_bag[name]=pd.DataFrame()
        Phonation_dict_bag[name]=Phonation_dict_bag[name].append(df_feat_utt)
    tmpPath=PhonationPath + "/features"
    if not os.path.exists(tmpPath):
        os.makedirs(tmpPath)
    pickle.dump(Phonation_dict_bag,open(chkptpath,"wb"))
else:
    Phonation_dict_bag=pickle.load(open(chkptpath,"rb"))


'''
    df_phonation_{role}[utt], role={doc|kid} 
    process: input: audio_files
    
    這邊只是把所有utterence level的feature做一個mean的動作
    
'''

chkptpath_kid=PhonationPath+"/df_phonation_kid_{}.pkl".format(dataset_name)
chkptpath_doc=PhonationPath+"/df_phonation_doc_{}.pkl".format(dataset_name)

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


chkptpath_kid_TD=args.checkpointpath + "/df_phonation_kid_ADOS_TD.pkl"
chkptpath_kid_ASD=args.checkpointpath + "/df_phonation_kid_ADOS.pkl"
chkptpath_doc_ASD=args.checkpointpath + "/df_phonation_doc_ADOS.pkl"
chkptpath_doc_TD=args.checkpointpath + "/df_phonation_doc_ADOS_TD.pkl"


# =============================================================================
'''

    t-test area

'''
# =============================================================================

df_ttest_result_a=TTest_Cmp_checkpoint(chkptpath_doc_ASD,chkptpath_doc_TD)
df_ttest_result_b=TTest_Cmp_checkpoint(chkptpath_kid_ASD,chkptpath_kid_TD)

df_ttest_result_c=TTest_Cmp_checkpoint(chkptpath_doc_ASD,chkptpath_kid_ASD)
df_ttest_result_d=TTest_Cmp_checkpoint(chkptpath_doc_TD,chkptpath_kid_TD)






aaa=ccc
