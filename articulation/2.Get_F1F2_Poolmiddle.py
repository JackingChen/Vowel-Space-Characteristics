#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:14:50 2020

@author: jackchen


"""
'''

    Input: 
        wav files: <path-to-wav-files>/Segmented_ADOS_ASD_emotion_normalized
        trn files: <path-to-trn-files>/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain
    Output:
        Formants_utt_symb
        Formants_people_symb
        Phonation_utt_symb
'''

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
import Multiprocess
import re
import statistics
import shutil
import seaborn as sns
from HYPERPARAM import phonewoprosody, Label
def GetBetweenPhoneDistance(df_top_dict,\
                            subtract_columns=['mean', 'min', '25%', '50%', '75%', 'max'],\
                            people_index=['2016_10_12_01_219_1','2017_07_08_01_317']):
    # =============================================================================
    '''
    
        Calculate the distributional distance between a u i
    
    '''
    BetweenPhoneDistance=Dict()
    # =============================================================================
    for symb in df_top_dict.keys():
        for feat in df_top_dict[symb].keys():
            print(df_top_dict[symb][feat])
    df_subtract_asubu_F1=df_top_dict['A:']['F1'][subtract_columns].subtract(df_top_dict['u:']['F1'][subtract_columns])
    df_subtract_asubu_F1['origin_A:_F1_std']=df_top_dict['A:']['F1']['std']
    df_subtract_asubu_F1['origin_u:_F1_std']=df_top_dict['u:']['F1']['std']
    dfsubtract_asubu_F1_certianpeople=df_subtract_asubu_F1.loc[people_index]
    df_subtract_asubi_F1=df_top_dict['A:']['F1'][subtract_columns].subtract(df_top_dict['i:']['F1'][subtract_columns])
    df_subtract_asubi_F1['origin_A:_F1_std']=df_top_dict['A:']['F1']['std']
    df_subtract_asubi_F1['origin_i:_F1_std']=df_top_dict['i:']['F1']['std']
    df_subtract_asubi_F1_certianpeople=df_subtract_asubi_F1.loc[people_index]
    df_subtract_isubu_F2=df_top_dict['i:']['F2'][subtract_columns].subtract(df_top_dict['u:']['F2'][subtract_columns])
    df_subtract_isubu_F2['origin_i:_F2_std']=df_top_dict['i:']['F2']['std']
    df_subtract_isubu_F2['origin_u:_F2_std']=df_top_dict['u:']['F2']['std']
    df_subtract_isubu_F2_certianpeople=df_subtract_isubu_F2.loc[people_index]
    BetweenPhoneDistance['F1(a-u)']=dfsubtract_asubu_F1_certianpeople
    BetweenPhoneDistance['F1(a-u)']=df_subtract_asubi_F1_certianpeople
    BetweenPhoneDistance['F2(i-u)']=df_subtract_isubu_F2_certianpeople
    return BetweenPhoneDistance

def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice/articulation',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--base_path_phf', default='/homes/ssd1/jackchen/gop_prediction/data',
                        help='path of the base directory')
    parser.add_argument('--filepath', default='/homes/ssd1/jackchen/DisVoice/data/Segmented_ADOS_normalized',
                        help='''/homes/ssd1/jackchen/DisVoice/data/{Segmented_ADOS_ASD_emotion_normalized|Segmented_ADOS_emotion_normalized|Segmented_ADOS_TD_normalized|Segmented_ADOS_TD_emotion_normalized}
                              注意！trnpath 是依賴filepath的，兩者的{filename}.txt, {filename}.wav filename要一模一樣''')
    parser.add_argument('--trnpath', default='/mnt/sdd/jackchen/egs/formosa/s6/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/KID_FromASD_DOCKID/ADOS_tdnn_fold_transfer',
                        help='''/mnt/sdd/jackchen/egs/formosa/s6/{Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/{kid88|kid_TD|ASD_DOCKID|ASD_DOCKID_emotion|TD_DOCKID_emotion}/ADOS_tdnn_fold_transfer | Alignment_human/kid/Audacity_phone|
                              注意！trnpath 是依賴filepath的，兩者的{filename}.txt, {filename}.wav filename要一模一樣''')
    parser.add_argument('--outpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--plot_outpath', default='Plot/',
                        help='path of the base directory')
    parser.add_argument('--formantmethod', default='praat',
                        help='path of the base directory')
    parser.add_argument('--avgmethod', default='middle',
                        help='path of the base directory')
    parser.add_argument('--check', default=False,
                        help='path of the base directory')
    parser.add_argument('--checkreliability', default=False,
                            help='path of the base directory')
    parser.add_argument('--PoolFormantWindow', default=3, type=int,
                            help='path of the base directory')
    parser.add_argument('--Inspect_features', default=['F1','F2'],
                            help='')
    
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
plot_outpath=args.plot_outpath
import praat.praat_functions as praat_functions
from utils_jack  import functional_method, Info_name_sex, F0_parameter_dict


# =============================================================================
'''
Check with phf features

'''
# =============================================================================
if args.check:
    Utt_phf_dict=pickle.load(open(args.base_path_phf+"/Utt_phf_dict.pkl","rb"))
# =============================================================================
'''

This is an data collector with format


Formants_utt_symb[utt][phone] = [F1, F2] record F1, F2's of each utterances'
Formants_people_symb[spkr_name][phone] = [F1, F2] record F1, F2's of each people'
'''

def Rolestr2role(role_str):
    Rolestr2role_map=Dict()
    Rolestr2role_map['DOC_FromASD_DOCKID']='_D_'
    Rolestr2role_map['KID_FromASD_DOCKID']='_K_'
    Rolestr2role_map['DOCKID']=''
    Rolestr2role_map['kid88']='_K_'
    Rolestr2role_map['kid']='_K_'
    
    return Rolestr2role_map[role_str]
# =============================================================================
role_str=trnpath.split("/")[-2]
# This function replaced by Rolestr2role function
# if role_str == 'DOC_FromASD_DOCKID':
#     role= '_D_'
# elif 'DOCKID' in role_str and 'From' not in role_str:
#     role= '' 
# else :
#     role= '_K_'

role=Rolestr2role(role_str)
files=glob.glob(trnpath+"/*{}*.txt".format(role))

silence_duration=0.02 #0.1s
silence_duration_ms=silence_duration*1000
silence = AudioSegment.silent(duration=silence_duration_ms)
if os.path.exists('Gen_formant_multiprocess.log'):
    os.remove('Gen_formant_multiprocess.log')




''' Multithread processing start '''
pool = Pool(int(os.cpu_count()))
# pool = Pool(1)
keys=[]
interval=2
for i in range(0,len(files),interval):
    # print(list(combs_tup.keys())[i:i+interval])
    keys.append(files[i:i+interval])
flat_keys=[item for sublist in keys for item in sublist]
assert len(flat_keys) == len(files)

multi=Multiprocess.Multi(filepath, MaxnumForm=5, AVERAGEMETHOD=AVERAGEMETHOD)
multi._updatePhonedict(phonewoprosody.Phoneme_sets)
multi._updateLeftSymbMapp(phonewoprosody.LeftSymbMapp)
# multi.MEASURE_PHONATION=True
multi._measurephonation()
# final_results=pool.starmap(process_audio, [([file_block,silence,trnpath,PoolFormantWindow]) for file_block in tqdm(keys)])
final_results=pool.starmap(multi.process_audio, [([file_block,silence,trnpath,PoolFormantWindow]) for file_block in tqdm(keys)])

Formants_people_symb=Dict()
for _, load_file_tmp, _ in final_results:        
    for spkr_name, phone_dict in load_file_tmp.items():
        for phone, values in phone_dict.items():
            symb=phone
            if spkr_name not in Formants_people_symb.keys():
                if symb not in Formants_people_symb[spkr_name].keys():
                    Formants_people_symb[spkr_name][symb]=values
                elif symb in Formants_people_symb[spkr_name].keys():
                    Formants_people_symb[spkr_name][symb].extend(values)
            else:
                if symb not in Formants_people_symb[spkr_name].keys():
                    Formants_people_symb[spkr_name][symb]=values
                elif symb in Formants_people_symb[spkr_name].keys(): 
                    Formants_people_symb[spkr_name][symb].extend(values)
            

count=0
Formants_utt_symb=Dict()
Phonation_utt_symb=Dict()
for load_file_tmp ,_,  load_file_tmp_p in final_results:
    for utt in load_file_tmp.keys():
        Formants_utt_symb[utt]=load_file_tmp[utt]
        Phonation_utt_symb[utt]=load_file_tmp_p[utt]
if not os.path.exists(outpath):
    os.makedirs(outpath)


pickle.dump(Formants_utt_symb,open(outpath+"/Formants_utt_symb_by{avgmed}_window{wind}.pkl".format(avgmed=AVERAGEMETHOD,wind=PoolFormantWindow),"wb"))

print("Finished creating Formants_utt_symb in", outpath+"/Formants_utt_symb_by{avgmed}_window{wind}.pkl".format(avgmed=AVERAGEMETHOD,wind=PoolFormantWindow))

pickle.dump(Phonation_utt_symb,open(outpath+"/Phonation_utt_symb.pkl".format(),"wb"))

print("Finished creating Formants_people_symb in", outpath+"/Phonation_utt_symb.pkl".format())

pickle.dump(Formants_people_symb,open(outpath+"/Formants_people_symb_by{avgmed}_window{wind}.pkl".format(avgmed=AVERAGEMETHOD,wind=PoolFormantWindow),"wb"))

print("Finished creating Formants_people_symb in", outpath+"/Formants_people_symb_by{avgmed}_window{wind}.pkl".format(avgmed=AVERAGEMETHOD,wind=PoolFormantWindow))





uttpath=outpath+"/Formants_utt_symb_by{avgmed}_window{wind}.pkl".format(avgmed=AVERAGEMETHOD,wind=PoolFormantWindow)
utt_outpath=outpath+"/Formants_utt_symb_by{avgmed}_window{wind}_{role}.pkl".format(avgmed=AVERAGEMETHOD,\
                                                                                   wind=PoolFormantWindow,\
                                                                                   role=role_str)
uttPhonation=outpath+"/Phonation_utt_symb.pkl".format()
uttPhonation_outpath=outpath+"/Phonation_utt_symb_{role}.pkl".format(role=role_str)
    
peoplepath=outpath+"/Formants_people_symb_by{avgmed}_window{wind}.pkl".format(avgmed=AVERAGEMETHOD,wind=PoolFormantWindow)
people_outpath=outpath+"/Formants_people_symb_by{avgmed}_window{wind}_{role}.pkl".format(avgmed=AVERAGEMETHOD,\
                                                                                   wind=PoolFormantWindow,\
                                                                                   role=role_str)


    
shutil.copy(uttpath, utt_outpath)
shutil.copy(uttPhonation, uttPhonation_outpath)
shutil.copy(peoplepath, people_outpath)

print("Copied ", uttpath, " to ", utt_outpath)
