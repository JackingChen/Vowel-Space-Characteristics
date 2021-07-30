#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 21:18:09 2021

@author: jackchen
"""

import os, glob
import pandas as pd
import argparse
import re
import pickle
def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--ASR_path', default='/mnt/sdd/jackchen/egs/formosa/s6',
                        help='path of the base directory', dest='ASR_path')
    parser.add_argument('--columns', default=['utt','st','ed','txt','spk'],
                        help='path of the base directory')
    parser.add_argument('--out_path', default='Features/Other',
                        help='path of the base directory')
    args = parser.parse_args()
    return args


args = get_args()
ASR_path=args.ASR_path



TD_info_path=ASR_path + '/Segments_info_ADOS_TD.txt' 
ASD_info_path=ASR_path + '/Segments_info_ADOS_ASD.txt' 
df_ASD_info=pd.read_csv(ASD_info_path, sep='\t',header=None)
df_ASD_info.columns=args.columns
df_ASD_info_kid=df_ASD_info[df_ASD_info['spk'].str.contains("_K")]


df_TD_info=pd.read_csv(TD_info_path, sep='\t',header=None)
df_TD_info.columns=args.columns
df_TD_info_kid=df_TD_info[df_TD_info['spk'].str.contains("_K")]

def GetSpeechSpeed(df_TD_info_kid, namestr_end_position="_emotion"):
    people_set=set(df_TD_info_kid['spk'])
    df_speed=pd.DataFrame()
    for people in people_set:
        df_TD_info_kid_utt=df_TD_info_kid[df_TD_info_kid['spk']==people]
        name=people[:re.search(namestr_end_position,people).start()]
        
        dur = df_TD_info_kid_utt['ed'] - df_TD_info_kid_utt['st']
        Strlength = df_TD_info_kid_utt['txt'].str.len()
        speed=dur/Strlength
        
        df_speed.loc[name,'dur']=dur.mean()
        df_speed.loc[name,'strlen']=Strlength.mean()
        df_speed.loc[name,'speed']=speed.mean()
        df_speed.loc[name,'totalword']=Strlength.sum()
    return df_speed

df_speed_TDkid= GetSpeechSpeed(df_TD_info_kid, namestr_end_position="_[D|K]")
df_speed_ASDkid= GetSpeechSpeed(df_ASD_info_kid, namestr_end_position="_[D|K]")

if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)
pickle.dump(df_speed_TDkid, open(args.out_path + "/df_dur_strlen_speed_{role}.pkl".format(role='TD'),"wb"))
pickle.dump(df_speed_ASDkid, open(args.out_path + "/df_dur_strlen_speed_{role}.pkl".format(role='ASD'),"wb"))