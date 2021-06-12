#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:14:50 2020

@author: jackchen

Te purpose of this scrupt is to use praat function to calcualte F1, F2 values of each phonemes
The procedures are in the following:
    1. read the phone timestamps (audacity format, you should generate it from ASR alignments)
    2. slice the time boundaries out of the audio files and run praat functions on the short audio sample

Because this procedure will take multiple hours!!!!!! we implemented multiprocessing
approach but still leave a "Manual area" at the bottom

Note !!!!!!!!!!!!  with unknown reason, The output of gop.*.txt will have less SHORT 'SIL' than forced alignment results
ex:
    2015_12_05_01_063_1_K_95_angry: utt in Utt_phf_dict 1 Not Match utt in Formants_utt_symb 3
    2015_12_13_01_153_K_2_angry: utt in Utt_phf_dict 51 Not Match utt in Formants_utt_symb 52
    2017_07_05_01_310_1_K_38_afraid: utt in Utt_phf_dict 38 Not Match utt in Formants_utt_symb 39
    2016_12_04_01_188_1_K_40_angry: utt in Utt_phf_dict 70 Not Match utt in Formants_utt_symb 71
    2016_12_24_01_226_K_17_angry: utt in Utt_phf_dict 33 Not Match utt in Formants_utt_symb 35
    ...
Here I put an log to report the unmatched files:
    len(Utt_phf_dict[utt][Utt_phf_dict[utt].index != 'SIL']) != len(Formants_utt_symb[utt][Formants_utt_symb[utt].index != "SIL"]):
    at line 343

update 2021/05/27 :  extend audio segments with half a window
    st_ext= max(st - F1F2_extractor.sizeframe/2,0)
    ed_ext= min(ed + F1F2_extractor.sizeframe/2,max(df_segInfo[1]))
    
       2021/06/10 :  changed the muli-process method to starmap:
                           code: final_results=pool.starmap
                     added an argument for formant funcational method:
                           functional_method(data, method='middle', window=3)
                           
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
from articulation import Extract_F1F2
import re
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
                        help='/homes/ssd1/jackchen/DisVoice/data/{Segmented_ADOS_normalized|Session_ADOS_normalized}')
    parser.add_argument('--trnpath', default='/mnt/sdd/jackchen/egs/formosa/s6/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/kid_TD/ADOS_tdnn_fold_transfer/',
                        help='/mnt/sdd/jackchen/egs/formosa/s6/Alignment_ADOShappyDAAIKidallDeceiptformosaCSRC_chain/kid/ADOS_tdnn_fold_transfer | /mnt/sdd/jackchen/egs/formosa/s6/Alignment_human/kid/Audacity_phone')
    parser.add_argument('--outpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--avgmethod', default='middle',
                        help='path of the base directory')
    parser.add_argument('--check', default=False,
                        help='path of the base directory')
    parser.add_argument('--checkreliability', default=False,
                            help='path of the base directory')
    parser.add_argument('--PoolFormantWindow', default=1, type=int,
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
Formants_people_symb=Dict()
Formants_utt_symb=Dict()
# =============================================================================
role_str=trnpath.split("/")[-2]
role= '_D_' if role_str == 'doc' else '_K_'

files=glob.glob(trnpath+"/*{}*.txt".format(role))

silence_duration=0.02 #0.1s
silence_duration_ms=silence_duration*1000
silence = AudioSegment.silent(duration=silence_duration_ms)
if os.path.exists('Gen_formant_multiprocess.log'):
    os.remove('Gen_formant_multiprocess.log')

def process_audio(files,silence,trnpath,functional_method_window):
    Formants_people_symb=Dict()
    Formants_utt_symb=Dict()
    print("Process {} executing".format(files))
    for file in files:
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
    
        for st,ed,symb in df_segInfo.values:
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
            [F1,F2]=F1F2_extractor.extract_features_file(temp_outfile)
            if len(F1) == 0 or len(F2)==0:
                F1_static,F2_static = -1,-1
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
        if args.check:
            if len(Utt_phf_dict[utt][Utt_phf_dict[utt].index != 'SIL']) != len(Formants_utt_symb[utt][Formants_utt_symb[utt].index != "SIL"]):
                with open('Gen_formant_multiprocess.log', 'a') as f:
                    string=utt + ": utt in Utt_phf_dict " + str(len(Utt_phf_dict[utt])) + " Not Match utt in Formants_utt_symb "+  str(len(Formants_utt_symb[utt])) + "\n"
                    
                    f.write(string)
            assert len(Formants_utt_symb[utt]) !=0
    
    return Formants_utt_symb, Formants_people_symb

''' Multithread processing start '''
# multi_ppl_path= args.outpath + "/multijobs/"
# rmfiles = glob.glob(multi_ppl_path+"*")
# for file in rmfiles:    
#     os.remove(file)
pool = Pool(int(os.cpu_count()))
keys=[]
interval=2
for i in range(0,len(files),interval):
    # print(list(combs_tup.keys())[i:i+interval])
    keys.append(files[i:i+interval])
flat_keys=[item for sublist in keys for item in sublist]
assert len(flat_keys) == len(files)
final_results=pool.starmap(process_audio, [([file_block,silence,trnpath,PoolFormantWindow]) for file_block in tqdm(keys)])


Formants_people_symb=Dict()
for _, load_file_tmp in final_results:        
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
for load_file_tmp ,_ in final_results:
    for utt, df_phone in load_file_tmp.items():
        Formants_utt_symb[utt]=df_phone
if not os.path.exists(outpath):
    os.makedirs(outpath)


pickle.dump(Formants_utt_symb,open(outpath+"/Formants_utt_symb_by{avgmed}_window{wind}.pkl".format(avgmed=AVERAGEMETHOD,wind=PoolFormantWindow),"wb"))
pickle.dump(Formants_people_symb,open(outpath+"/Formants_people_symb_by{avgmed}_window{wind}.pkl".format(avgmed=AVERAGEMETHOD,wind=PoolFormantWindow),"wb"))
    
''' Multithread processing end '''

# =============================================================================
'''

Check area
    check if Formants_utt_symb and Formants_people_symb match
    
    randomly check 4 phones for check operation
'''

phones_check=['A:4','ax5','ax4','A:5']
# =============================================================================
df_template=pd.DataFrame([],columns=[a for sublist in [['F1','F2']] for a in sublist])
check_dict=Dict()
for keys, values in tqdm(Formants_utt_symb.items()):
    emotion=keys.split("_")[-1]
    n=keys.split("_")[-2]
    speaker='_'.join(keys.split("_")[:-3])
    for phone in phones_check:
        if phone not in check_dict[speaker].keys():
            check_dict[speaker][phone]=df_template
        check_dict[speaker][phone]=check_dict[speaker][phone].append(values[values.index==phone])

spk='2015_12_05_01_063_1'
for phone in phones_check:
    assert len(check_dict[spk][phone]) == len(Formants_people_symb[spk][phone])

    
# =============================================================================
'''

    Manual area
    You can use it to debug

'''
# Formants_utt_symb=Dict()
# Formants_people_symb=Dict()
# # =============================================================================
# for file in tqdm(files[:]):
#     filename=os.path.basename(file).split(".")[0]
#     spkr_name=filename[:re.search("_[K|D]_", filename).start()]
#     utt='_'.join(filename.split("_")[:])
    
#     trn=trnpath+"/{name}.txt".format(name=filename)
#     df_segInfo=pd.read_csv(trn, header=None,delimiter='\t')
#     if 'Session' in filepath:
#         audiofile=filepath+"/{name}.wav".format(name='_'.join(filename.split("_")[:-1]))
#     elif 'Segment' in filepath:
#         audiofile=filepath+"/{name}.wav".format(name=filename)
#     else:
#         raise OSError(os.strerror, 'not allowed filepath')
#     audio = AudioSegment.from_wav(audiofile)
    
#     gender_query_str=filename[:re.search("_[K|D]_", filename).start()]
#     role=filename[re.search("[K|D]", filename).start()]
#     if role =='D':
#         gender='female'
#     elif role =='K':
#         series_gend=Info_name_sex[Info_name_sex['name']==gender_query_str]['sex']
#         gender=series_gend.values[0]
    
#     minf0=F0_parameter_dict[gender]['f0_min']
#     maxf0=F0_parameter_dict[gender]['f0_max']
    
#     F1F2_extractor=Extract_F1F2(maxf0=maxf0, minf0=minf0)
#     for st,ed,symb in tqdm(df_segInfo.values):
#         st_ext= max(st - F1F2_extractor.sizeframe/2,0)
#         ed_ext= min(ed + F1F2_extractor.sizeframe/2,max(df_segInfo[1]))
#         # segment_lengths.append((ed-st)) # np.quatile(segment_lengths,0.05)=0.08
#         # st_ms=st * 1000 #Works in milliseconds
#         # ed_ms=ed * 1000 #Works in milliseconds
#         st_ms=st_ext * 1000 #Works in milliseconds
#         ed_ms=ed_ext * 1000 #Works in milliseconds
        
        
        
#         audio_segment = silence + audio[st_ms:ed_ms] + silence
#         temp_outfile=F1F2_extractor.PATH+'/../tempfiles/tempwav{}.wav'.format(utt+symb)
        
        
#         audio_segment.export(temp_outfile, format="wav")
        
        
#         [F1,F2]=F1F2_extractor.extract_features_file(temp_outfile)
#         if len(F1) == 0 or len(F2)==0:
#             F1_static,F2_static = -1,-1
#         else:
#             F1_static=functional_method(F1,method=AVERAGEMETHOD)
#             F2_static=functional_method(F2,method=AVERAGEMETHOD)
        
        
#         assert  math.isnan(F1_static) == False and math.isnan(F2_static) == False
#         os.remove(temp_outfile)
        
#         tmp_dict=Dict()
#         tmp_dict[symb].F1=F1_static
#         tmp_dict[symb].F2=F2_static
#         df_tmp=pd.DataFrame.from_dict(tmp_dict)
#         if utt not in  Formants_utt_symb.keys():
#             Formants_utt_symb[utt]=df_tmp
#         else:
#             Formants_utt_symb[utt]=pd.concat([Formants_utt_symb[utt],df_tmp],axis=1)
        
#         if len(F1) != 0 and len(F2)!=0:
#             if spkr_name not in Formants_people_symb.keys():
#                 if symb not in Formants_people_symb[spkr_name].keys():
#                     Formants_people_symb[spkr_name][symb]=[[F1_static, F2_static]]
#                 elif symb in Formants_people_symb[spkr_name].keys():
#                     Formants_people_symb[spkr_name][symb].append([F1_static, F2_static])
#             else:
#                 if symb not in Formants_people_symb[spkr_name].keys():
#                     Formants_people_symb[spkr_name][symb]=[[F1_static, F2_static]]
#                 elif symb in Formants_people_symb[spkr_name].keys(): 
#                     Formants_people_symb[spkr_name][symb].append([F1_static, F2_static])
#     Formants_utt_symb[utt] = Formants_utt_symb[utt].T
#     if args.check:
#         if len(Utt_phf_dict[utt][Utt_phf_dict[utt].index != 'SIL']) != len(Formants_utt_symb[utt][Formants_utt_symb[utt].index != "SIL"]):
#             with open('Gen_formant_multiprocess.log', 'a') as f:
#                 string=utt + ": utt in Utt_phf_dict " + str(len(Utt_phf_dict[utt])) + " Not Match utt in Formants_utt_symb "+  str(len(Formants_utt_symb[utt])) + "\n"
                
#                 f.write(string)
#         assert len(Formants_utt_symb[utt]) !=0

# pickle.dump(Formants_utt_symb,open(outpath+"/Formants_utt_symb_cmp.pkl","wb"))
# pickle.dump(Formants_people_symb,open(outpath+"/Formants_people_symb_cmp.pkl","wb"))


# # =============================================================================
# '''

# Check reliability 

# Assert multiprocess == single
# '''
# # =============================================================================
if args.checkreliability:
    # Formants_utt_symb_cmp=pickle.load(open(outpath+"/Formants_utt_symb_cmp.pkl","rb"))
    Formants_people_symb_cmp=pickle.load(open(outpath+"/Formants_people_symb_bymiddle_window1.pkl","rb"))
    
    # Formants_utt_symb=pickle.load(open(outpath+"/Formants_utt_symb.pkl","rb"))
    Formants_people_symb=pickle.load(open(outpath+"/Formants_people_symb_bymiddle_window1_ASDkid.pkl","rb"))

    for keys in Formants_utt_symb_cmp.keys():
        print(keys)
        assert np.sum(Formants_utt_symb_cmp[keys].values - Formants_utt_symb[keys].values) == 0
        
        
    for spkr_name in Formants_people_symb_cmp.keys():
        for symb in Formants_people_symb_cmp[spkr_name].keys():
            assert np.sum(np.vstack(Formants_people_symb_cmp[spkr_name][symb]) - np.vstack(Formants_people_symb[spkr_name][symb])) ==0


# =============================================================================
'''

    Compare Formant values between human labels and aligner

''' 
# =============================================================================
# ''' 1. load data '''
# Formants_utt_symb_cmp=pickle.load(open(outpath+"/Formants_utt_symb_humanlabel_ASDkid.pkl","rb"))
# Formants_utt_symb=pickle.load(open(outpath+"/Formants_utt_symb_bymiddle_ASDkid.pkl","rb"))

# ''' 2. gather data '''
# ''' Formant_people_symb_total['cmp'][people] = df: index = phone, column = [F1, F2]'''
# import re
# Formant_people_symb_total=Dict()
# Formant_people_symb_total['ori']=Dict()
# Formant_people_symb_total['cmp']=Dict()
# for keys, values in Formants_utt_symb_cmp.items():
#     people=keys[:keys.find(re.findall("_[K|D]",keys)[0])]
#     if people not in Formant_people_symb_total['cmp'].keys():
#         Formant_people_symb_total['cmp'][people]=pd.DataFrame()
#     if people not in Formant_people_symb_total['ori'].keys():
#         Formant_people_symb_total['ori'][people]=pd.DataFrame()
#     Formant_people_symb_total['cmp'][people]=Formant_people_symb_total['cmp'][people].append(values)
#     Formant_people_symb_total['ori'][people]=Formant_people_symb_total['ori'][people].append(Formants_utt_symb[keys])

''' manage data '''
