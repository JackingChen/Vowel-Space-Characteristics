#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:45:10 2022

@author: jack
"""

import glob
import os
import soundfile as sf
import re
from addict import Dict
from tqdm import tqdm
import numpy as np

ASD_dir_path='data/Segmented_ADOS_normalized'
TD_dir_path='data/Segmented_ADOS_TD_normalized'
suffix='*.wav'

ASD_DK_files=glob.glob(os.path.join(ASD_dir_path, suffix))
TD_DK_files=glob.glob(os.path.join(TD_dir_path, suffix))


len(ASD_DK_files)
len(TD_DK_files)
len(ASD_DK_files)+len(TD_DK_files)

Wavtime_person_ASD=Dict()
Total_time_ASD=0
for filefullPathname in tqdm(ASD_DK_files):
    filename=os.path.basename(filefullPathname)
    
    
    spkr_name=filename[:re.search("_[K|D]_", filename).start()]
    role=filename[re.search("[K|D]", filename).start()]
    
    s, fs = sf.read(filefullPathname)
    SoundFile_time=s.shape[0]/fs
    
    Total_time_ASD += SoundFile_time
    
    if spkr_name not in Wavtime_person_ASD.keys():
        Wavtime_person_ASD[spkr_name].time=0
        Wavtime_person_ASD[spkr_name].number=0
    Wavtime_person_ASD[spkr_name].time+=SoundFile_time
    Wavtime_person_ASD[spkr_name].number+=1
print("Total_time_ASD: ", Total_time_ASD)

# Wavtime_person_ASD=Dict()
Total_time_TD=0
for filefullPathname in tqdm(TD_DK_files):
    filename=os.path.basename(filefullPathname)
    
    
    spkr_name=filename[:re.search("_[K|D]_", filename).start()]
    role=filename[re.search("[K|D]", filename).start()]
    
    s, fs = sf.read(filefullPathname)
    SoundFile_time=s.shape[0]/fs
    
    Total_time_TD += SoundFile_time
    
    if spkr_name not in Wavtime_person_ASD.keys():
        Wavtime_person_ASD[spkr_name].time=0
        Wavtime_person_ASD[spkr_name].number=0
    Wavtime_person_ASD[spkr_name].time+=SoundFile_time
    Wavtime_person_ASD[spkr_name].number+=1
print("Total_time_TD: ", Total_time_TD)


average_utterances=np.mean([v1.number for spkr, v1 in Wavtime_person_ASD.items()])
print("Each session contains ",average_utterances," utterances")