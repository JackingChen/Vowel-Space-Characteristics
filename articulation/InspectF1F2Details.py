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
    parser.add_argument('--filepath', default='/homes/ssd1/jackchen/DisVoice/data/Segmented_ADOS_TD_normalized',
                        help='/homes/ssd1/jackchen/DisVoice/data/{Segmented_ADOS_normalized|Session_ADOS_normalized|Segmented_ADOS_TD_normalized}')
    parser.add_argument('--trnpath', default='/mnt/sdd/jackchen/egs/formosa/s6/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/kid_TD/ADOS_tdnn_fold_transfer',
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
PhoneOI=PhoneMapp_dict.keys()
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
# inspect_people=['2016_06_27_02_017_1', '2016_07_30_01_148', '2016_08_26_01_168_1',
#        '2016_09_24_01_174_1']  # manual select certain person
inspect_people=list(set([os.path.basename(file)[:re.search("_[K|D]_", os.path.basename(file)).start()] for file in files]))

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
self=Dict()
self.MaxnumForm=5
self.filepath=filepath
self.formantmethod='praat'
self.check=False
self.AVERAGEMETHOD=AVERAGEMETHOD
# =============================================================================
Formants_people_symb=Dict()
Formants_utt_symb=Dict()
error_msg_bag=[]

for file in tqdm(files):
    filename=os.path.basename(file).split(".")[0]
    spkr_name=filename[:re.search("_[K|D]_", filename).start()]
    utt='_'.join(filename.split("_")[:])
    
    trn=trnpath+"/{name}.txt".format(name=filename)
    df_segInfo=pd.read_csv(trn, header=None,delimiter='\t')
    if 'Session' in self.filepath:
        audiofile=self.filepath+"/{name}.wav".format(name=filename[:re.search("_[K|D]_", filename).end()-1])
    elif 'Segment' in self.filepath:
        audiofile=self.filepath+"/{name}.wav".format(name=filename)
    else:
        raise OSError(os.strerror, 'not allowed filepath')
    audio = AudioSegment.from_wav(audiofile)
    
    gender_query_str=filename[:re.search("_[K|D]_", filename).start()]
    role=filename[re.search("[K|D]", filename).start()]
    
    if role =='D':
        gender='female'
        age_year=pd.Series([28])
    elif role =='K':
        series_gend=Info_name_sex[Info_name_sex['name']==gender_query_str]['sex']
        gender=series_gend.values[0]
        age_year=Info_name_sex[Info_name_sex['name']==gender_query_str]['age_year']
    
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
        if self.formantmethod == 'Disvoice':
            [F1,F2]=F1F2_extractor.extract_features_file(temp_outfile)
        elif self.formantmethod == 'praat':
            try:
                def GetmaxFormant(symb, age, sex):
                    if age <= 12 or sex=='female':
                        maxFormant=6000
                    else:
                        maxFormant=5000
                    if 'u:' in symb:
                        maxFormant=3000
                    return maxFormant
    
                    
                maxFormant=GetmaxFormant(symb=symb, age=age_year.values[0], sex=gender)

                [F1,F2]=measureFormants(temp_outfile,minf0,maxf0,time_step=F1F2_extractor.step,MaxnumForm=self.MaxnumForm,Maxformant=maxFormant,framesize=F1F2_extractor.sizeframe)
            except :
                F1, F2 = [], []
                print("Error processing ",utt+"__"+symb)
                error_msg_bag.append(utt+"__"+symb)
        
        
        if len(F1) == 0 or len(F2)==0:
            F1_static, F2_static= -1, -1
        else:
            F1_static=functional_method(F1,method=self.AVERAGEMETHOD,window=functional_method_window)
            F2_static=functional_method(F2,method=self.AVERAGEMETHOD,window=functional_method_window)
        
        
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
        
        
                    
        dur=ed-st
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
    df=pd.DataFrame(df_segInfo[[0,1]].values,index=df_segInfo[2])
    Formants_utt_symb[utt]['start']=df[0]
    Formants_utt_symb[utt]['end']=df[1]        

        

# =============================================================================
'''
    First Filter by 1.5*IQR method and
    Inspect the formant values
    
    Please check the AUI_info

'''
from utils_jack  import  Formant_utt2people_reshape, Gather_info_certainphones, \
                         FilterUttDictsByCriterion, GetValuelimit_IQR, \
                         Get_aligned_sequences, WER 
from datetime import datetime as dt
import pathlib
import Multiprocess
def Process_IQRFiltering_Multi(Formants_utt_symb, limit_people_rule, outpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles'):
    pool = Pool(int(os.cpu_count()))
    keys=[]
    interval=20
    for i in range(0,len(Formants_utt_symb.keys()),interval):
        # print(list(combs_tup.keys())[i:i+interval])
        keys.append(list(Formants_utt_symb.keys())[i:i+interval])
    flat_keys=[item for sublist in keys for item in sublist]
    assert len(flat_keys) == len(Formants_utt_symb.keys())
    muti=Multiprocess.Multi()
    final_results=pool.starmap(muti.FilterUttDictsByCriterion_map, [([Formants_utt_symb,Formants_utt_symb,file_block,limit_people_rule]) for file_block in tqdm(keys)])
    # final_results=pool.starmap(FilterUttDictsByCriterion_map, [([Formants_utt_symb,Formants_utt_symb,file_block,limit_people_rule]) for file_block in tqdm(keys)])
    
    Formants_utt_symb_limited=Dict()
    for load_file_tmp,_ in final_results:        
        for utt, df_utt in load_file_tmp.items():
            Formants_utt_symb_limited[utt]=df_utt
    
    pickle.dump(Formants_utt_symb_limited,open(outpath+"/[Analyzing]Formants_utt_symb_limited.pkl","wb"))
    print('Formants_utt_symb saved to ',outpath+"/[Analyzing]Formants_utt_symb_limited.pkl")

# =============================================================================
PhoneOfInterest=list(PhoneMapp_dict.keys())
''' Vowel AUI rule is using phonewoprosody '''


Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
AUI_info=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)
limit_people_rule=GetValuelimit_IQR(AUI_info,PhoneMapp_dict,args.Inspect_features)



''' multi processing start '''
date_now='{0}-{1}-{2} {3}'.format(dt.now().year,dt.now().month,dt.now().day,dt.now().hour)
outpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles'
filepath=outpath+"/[Analyzing]Formants_utt_symb_limited.pkl"
if os.path.exists(filepath):
    fname = pathlib.Path(filepath)
    mtime = dt.fromtimestamp(fname.stat().st_mtime)
    filemtime='{0}-{1}-{2} {3}'.format(mtime.year,mtime.month,mtime.day,mtime.hour)
    
    # If file last modify time is not now (precisions to the hours) than we create new one
    if filemtime != date_now:
        Process_IQRFiltering_Multi(Formants_utt_symb,limit_people_rule) # the results will be output as pkl file at outpath+"/[Analyzing]Formants_utt_symb_limited.pkl"
else:
    Process_IQRFiltering_Multi(Formants_utt_symb,limit_people_rule)
Formants_utt_symb_limited=pickle.load(open(filepath,"rb"))
''' multi processing end '''
if len(limit_people_rule) >0:
    Formants_utt_symb=Formants_utt_symb_limited



Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
AUI_info=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)
        
        
        
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        