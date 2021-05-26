
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa
modified by Jackchen 20210524

This script is to extract word level phonation feature 
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
    parser.add_argument('--filepath', default='data/Segmented_ADOS_normalized',
                        help='data/{Segmented_ADOS_TD_normalized_untranscripted|Segmented_ADOS_normalized}')
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
audiopath = os.path.join(base_path,filepath)
dataset_name=os.path.basename(audiopath)
dataset_name=dataset_name[dataset_name.find('ADOS'):dataset_name.find('normalized')-1]

files=glob.glob(audiopath+"/*.wav")
# if os.path.exists('Gen_formant_multiprocess.log'):
#     os.remove('Gen_formant_multiprocess.log')

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


'''
    df_phonation_{role}[utt], role={doc|kid} 
    process: input: audio_files
'''

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


# =============================================================================
'''

    t-test area

'''
# =============================================================================

TTest_Cmp_checkpoint(chkptpath_doc_ASD,chkptpath_doc_TD)
TTest_Cmp_checkpoint(chkptpath_kid_ASD,chkptpath_kid_TD)






aaa=ccc
# dataset_str='{ds}_DvsK'.format(ds=dataset_name)
# df_ttest_result=pd.DataFrame()
# for col in df_phonation_kid.columns:
#     df_ttest_result.loc[dataset_str+"-p-val",col]=stats.ttest_ind(df_phonation_kid[col].dropna(),df_phonation_doc[col].dropna())[1].astype(float)
# df_ttest_result=df_ttest_result.T 

# result_path="RESULTS/"
# if not os.path.exists(result_path):
#     os.makedirs(result_path)
# df_ttest_result.to_excel(result_path+dataset_str+".xlsx")
