#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 11:33:51 2021

@author: jackchen
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
    from articulation_functions import extractTrans, V_UV, measureFormants, measurePitch
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

from utils_jack  import functional_method, Info_name_sex, F0_parameter_dict

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
                        help='/homes/ssd1/jackchen/DisVoice/data/{Segmented_ADOS_ASD_emotion_normalized|Segmented_ADOS_emotion_normalized|Segmented_ADOS_TD_normalized|Segmented_ADOS_TD_emotion_normalized}')
    parser.add_argument('--trnpath', default='/mnt/sdd/jackchen/egs/formosa/s6/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/ASD_DOCKID/ADOS_tdnn_fold_transfer',
                        help='/mnt/sdd/jackchen/egs/formosa/s6/{Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/{kid88|kid_TD|ASD_DOCKID|ASD_DOCKID_emotion|TD_DOCKID_emotion}/ADOS_tdnn_fold_transfer | Alignment_human/kid/Audacity_phone|')
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
from script_mananger import script_manager
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
# =============================================================================
role_str=trnpath.split("/")[-2]
if role_str == 'doc':
    role= '_D_'
elif 'DOCKID' in role_str:
    role= '' 
else :
    role= '_K_'

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

self=multi

file_block=['/mnt/sdd/jackchen/egs/formosa/s6/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/ASD_DOCKID/ADOS_tdnn_fold_transfer/2016_09_24_01_174_1_D_58.txt',
 '/mnt/sdd/jackchen/egs/formosa/s6/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/ASD_DOCKID/ADOS_tdnn_fold_transfer/2016_07_06_01_078_1_K_68.txt']

files,silence,trnpath = file_block,silence,trnpath
functional_method_window=3
record_WavTrn=False
# =============================================================================
'''
    Multiprocess/Multi/process_audio
    Code start here

'''
# =============================================================================

if record_WavTrn:
    inspect_people=list(set([os.path.basename(file)[:re.search("_[K|D]_", os.path.basename(file)).start()] for file in files]))
    if not self.phoneMappdict:
        raise Exception('You should update self.phoneMappdict')
    PhoneOI=self.phoneMappdict.keys()
    Wav_collect=Dict()
    Trn_collect=Dict()
    for s in PhoneOI:
        for insp in inspect_people:
            Trn_collect[insp][s].basetime=0.0
Formants_people_symb=Dict()
Formants_utt_symb=Dict()
Phonation_utt_symb=Dict()
error_msg_bag=[]
print("Process {} executing".format(files))
for file in files:
    filename=os.path.basename(file).split(".")[0]
    spkr_name=filename[:re.search("_[K|D]_", filename).start()]
    utt='_'.join(filename.split("_")[:])
    
    trn=trnpath+"/{name}.txt".format(name=filename)
    df_segInfo=pd.read_csv(trn, header=None,delimiter='\t')
    df_segInfo.columns=['st','ed','txt']
    # In our transcripts we have Biphones like ueI, uA, we divide those phones to two phones
    regexp = re.compile(r'u(A|aI|ax|eI|O)_') #['uA_', 'uaI_', 'uax_', 'ueI_', 'uO_']
    u_Biphoneset = [symbbset for symbbset in self.Phoneme_sets.keys() if regexp.search(symbbset)]
    def ScanBiphone(df_segInfo,Phoneme_sets,Biphoneset):
        df_bool=df_segInfo.copy()
        for idx in df_segInfo.index:
            Biphone_lst= [symb for symbset in u_Biphoneset for symb in Phoneme_sets[symbset]]
            if df_segInfo.loc[idx,'txt'] in Biphone_lst:
                df_bool.loc[idx,'txt']=True
            else:
                df_bool.loc[idx,'txt']=False
            
        return df_bool
    def GetmaxFormant(symb, age, sex):
        if age <= 12 or sex=='female':
            maxFormant=5500
        else:
            maxFormant=5000
        if 'u:' in symb or 'w' in symb:
            maxFormant=3000
        return maxFormant
    df_u_BiphonBool=ScanBiphone(df_segInfo,self.Phoneme_sets,u_Biphoneset)
    
    for ind in df_segInfo[df_u_BiphonBool['txt']].index:
        st,ed,symb = df_segInfo.loc[ind]
        old_idx=df_segInfo.loc[ind].name
        dur_distrubute=(ed - st)/2
        
        left_symb=symb[:re.search('u',symb).end()]
        right_symb=symb[re.search('u',symb).end():]
        df_segInfo=df_segInfo.append(pd.DataFrame({'st': st, 'ed': st + dur_distrubute, \
                                                   'txt': '{0}'.format(self.LeftSymbMapp[left_symb])}, index=[old_idx+len(df_segInfo)]))
        df_segInfo=df_segInfo.append(pd.DataFrame({'st': st + dur_distrubute, 'ed': ed, \
                                                   'txt': '{0}-{1}'.format(self.LeftSymbMapp[left_symb],right_symb)}, index=[old_idx+len(df_segInfo)]))
        df_segInfo=df_segInfo.drop(index=old_idx)
    df_segInfo=df_segInfo.sort_values(by='st')
    df_segInfo=df_segInfo.reset_index(drop=True)
    
    
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
        age_year=pd.Series([28]) # default docs are all 28 years old
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
        ed_ext= min(ed + F1F2_extractor.sizeframe/2,max(df_segInfo['ed']))
        # segment_lengths.append((ed-st)) # np.quatile(segment_lengths,0.05)=0.08
        st_ms=st * 1000 #Works in milliseconds
        ed_ms=ed * 1000 #Works in milliseconds
        # st_ms=st_ext * 1000 #Works in milliseconds
        # ed_ms=ed_ext * 1000 #Works in milliseconds

        audio_segment = silence + audio[st_ms:ed_ms] + silence
        temp_outfile=F1F2_extractor.PATH+'/../tempfiles/tempwav{}.wav'.format(utt+symb)
        
        audio_segment.export(temp_outfile, format="wav")
        # =============================================================================
        # Formants Calculation start here
        if self.formantmethod == 'Disvoice':
            [F1,F2]=F1F2_extractor.extract_features_file(temp_outfile)
        elif self.formantmethod == 'praat':
            try:
                
                maxFormant=GetmaxFormant(symb=symb, age=age_year.values[0], sex=gender)

                [F1,F2]=measureFormants(temp_outfile,minf0,maxf0,time_step=F1F2_extractor.step,MaxnumForm=self.MaxnumForm,Maxformant=maxFormant,framesize=F1F2_extractor.sizeframe)
            except :
                F1, F2 = [], []
                print("Error processing ",utt+"__"+symb)
                error_msg_bag.append(utt+"__"+symb)
            
        
        F1=self._self_constraint(F1,feat='F1')
        F2=self._self_constraint(F2,feat='F2')

        if len(F1) < 2 or len(F2)<2: # don't accept the data with length = 0 for 1
            F1_static, F2_static= -1, -1
        else:
            import warnings
            F1_static=functional_method(F1,method=self.AVERAGEMETHOD,window=functional_method_window)
            F2_static=functional_method(F2,method=self.AVERAGEMETHOD,window=functional_method_window)
            
            warnings.filterwarnings("ignore")
            #Average of a masked array will be nan
            if math.isnan(F1_static) == True:
                #Second trial
                F1_tmp=functional_method(F1,method='mean')
                if math.isnan(F1_tmp) == True:
                    F1_static=-1
                else:
                    F1_static=F1_tmp
                    
            if math.isnan(F2_static) == True:
                F2_tmp=functional_method(F2,method='mean')
                if math.isnan(F2_tmp) == True:
                    F2_static=-1
                else:
                    F2_static=F2_tmp
            # warnings.filterwarnings('default')
        assert math.isnan(F1_static) == False or math.isnan(F2_static) == False
        
        # F0 Calculation start here
        df_feat_utt=measurePitch(temp_outfile, minf0, maxf0, "Hertz")
        df_feat_utt.index=[symb]
        if utt not in Phonation_utt_symb.keys():
            Phonation_utt_symb[utt]=pd.DataFrame()
        Phonation_utt_symb[utt]=Phonation_utt_symb[utt].append(df_feat_utt)

        
        # =============================================================================
        os.remove(temp_outfile)
        
        
        def Add_FormantsUttSymb(F1_static,F2_static,F1,F2,symb,Formants_utt_symb, Formants_people_symb):
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
            return Formants_utt_symb, Formants_people_symb
        Formants_utt_symb, Formants_people_symb= Add_FormantsUttSymb(F1_static,F2_static,F1,F2,symb,Formants_utt_symb, Formants_people_symb)

        if record_WavTrn:
            dur=audio[st_ms:ed_ms].duration_seconds
            if dur >=0.00001:
                silence_real_duration=silence.duration_seconds
                for s in PhoneOI:
                    if symb in [x for x in  self.phoneMappdict[s]]:
                        if s not in Wav_collect[spkr_name].keys():
                            Wav_collect[spkr_name][s]=audio[st_ms:ed_ms]
                            start=Trn_collect[spkr_name][s].basetime
                            end=Trn_collect[spkr_name][s].basetime+dur
                            Trn_collect[spkr_name][s].trn='{0}\t{1}\t{2}:{3}\n'.format(start,end,dur,utt,symb)
                            Trn_collect[spkr_name][s].praat=[tgre.Interval(start, end, str(utt+symb))]
                            Trn_collect[spkr_name][s].basetime+=dur
                        else:
                            Wav_collect[spkr_name][s]=Wav_collect[spkr_name][s] + silence + audio[st_ms:ed_ms] + silence
                            start=Trn_collect[spkr_name][s].basetime + silence_real_duration
                            end=Trn_collect[spkr_name][s].basetime + silence_real_duration + dur
                            
                            Trn_collect[spkr_name][s].trn+='{0}\t{1}\t{2}:{3}\n'.format(start,end,utt,symb)
                            Trn_collect[spkr_name][s].praat.append(tgre.Interval(start, end, str(utt+symb)))
                            Trn_collect[spkr_name][s].basetime+=silence_real_duration + dur + silence_real_duration  
    Formants_utt_symb[utt] = Formants_utt_symb[utt].T
    df=pd.DataFrame(df_segInfo[['st','ed']].values,index=df_segInfo['txt'])
    Formants_utt_symb[utt]['start']=df[0]
    Formants_utt_symb[utt]['end']=df[1]
    Phonation_utt_symb[utt]['start']=df[0]
    Phonation_utt_symb[utt]['end']=df[1]


# =============================================================================
'''

    Code end here

'''
# =============================================================================