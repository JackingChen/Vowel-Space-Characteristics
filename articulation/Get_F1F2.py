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


def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice/articulation',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--base_path_phf', default='/homes/ssd1/jackchen/gop_prediction',
                        help='path of the base directory')
    parser.add_argument('--filepath', default='/mnt/sdd/jackchen/egs/formosa/s6/Segmented_ADOS_emotion',
                        help='path of the base directory')
    parser.add_argument('--trnpath', default='/mnt/sdd/jackchen/egs/formosa/s6/Audacity',
                        help='path of the base directory')
    parser.add_argument('--outpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--avgmethod', default='middle',
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
path_app = base_path+'/../'
sys.path.append(path_app)
import praat.praat_functions as praat_functions
from script_mananger import script_manager
from utils_jack import dynamic2statict_artic, save_dict_kaldimat, get_dict


# =============================================================================
'''
Check with phf features

'''
# =============================================================================
if args.check:
    Utt_phf_dict=pickle.load(open(args.base_path_phf+"/Utt_phf_dict.pkl","rb"))


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def time2frame(time,unit=0.02):
    return int(time/unit)

def functional_method(data, method='middle'):
    
    if method=="mean":
        ret=np.mean(data)
    elif method=="middle":
        ret=np.mean(data[int(len(data)/2)-3: int(len(data)/2)+3])
    
    return ret

class Extract_F1F2:

    """Extract the articulation features from an audio file
    
    >>> articulation=Articulation()
    >>> file_audio="../audios/001_ddk1_PCGITA.wav"
    >>> features1=articulation.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
    >>> features2=articulation.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
    >>> features3=articulation.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
    >>> articulation.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test")
    
    >>> path_audio="../audios/"
    >>> features1=articulation.extract_features_path(path_audio, static=True, plots=False, fmt="npy")
    >>> features2=articulation.extract_features_path(path_audio, static=True, plots=False, fmt="csv")
    >>> features3=articulation.extract_features_path(path_audio, static=False, plots=True, fmt="torch")
    >>> articulation.extract_features_path(path_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test.ark")
    """
    
    def __init__(self):
        self.pitch_method="praat"
        self.sizeframe=0.04
        self.step=0.02
        self.nB=22
        self.nMFCC=12
        self.minf0=60
        self.maxf0=350
        self.voice_bias=-0.2
        self.len_thr_miliseconds=270.0
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        
    def extract_features_file(self,audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        fs, data_audio=read(audio)
        #normalize raw wavform
        data_audio=data_audio-np.mean(data_audio)
        data_audio=data_audio/float(np.max(np.abs(data_audio)))
        size_frameS=self.sizeframe*float(fs)
        size_stepS=self.step*float(fs)
        overlap=size_stepS/size_frameS
    
        if self.pitch_method == 'praat':
            name_audio=audio.split('/')
            temp_uuid='articulation'+name_audio[-1][0:-4]
            if not os.path.exists(self.PATH+'/../tempfiles/'):
                os.makedirs(self.PATH+'/../tempfiles/')
            temp_filename_vuv=self.PATH+'/../tempfiles/tempVUV'+temp_uuid+'.txt'
            temp_filename_f0=self.PATH+'/../tempfiles/tempF0'+temp_uuid+'.txt'
            praat_functions.praat_vuv(audio, temp_filename_f0, temp_filename_vuv, time_stepF0=self.step, minf0=self.minf0, maxf0=self.maxf0)
            F0,_=praat_functions.decodeF0(temp_filename_f0,len(data_audio)/float(fs),self.step)
            segmentsFull,segmentsOn,segmentsOff=praat_functions.read_textgrid_trans(temp_filename_vuv,data_audio,fs,self.sizeframe)
            os.remove(temp_filename_vuv)
            os.remove(temp_filename_f0)
        elif self.pitch_method == 'rapt':
            data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
            F0=pysptk.sptk.rapt(data_audiof, fs, int(size_stepS), min=self.minf0, max=self.maxf0, voice_bias=self.voice_bias, otype='f0')

            segmentsOn=V_UV(F0, data_audio, fs, 'onset')
            segmentsOff=V_UV(F0, data_audio, fs, 'offset')
            
            
        name_audio=audio.split('/')
        temp_uuid='artic'+name_audio[-1][0:-4]
        if not os.path.exists(self.PATH+'/../tempfiles/'):
            os.makedirs(self.PATH+'/../tempfiles/')
        temp_filename=self.PATH+'/../tempfiles/tempFormants'+temp_uuid+'.txt'
        praat_functions.praat_formants(audio, temp_filename,self.sizeframe,self.step)
        [F1, F2]=praat_functions.decodeFormants(temp_filename)
        # print("Length of the audio file = {audio}, length of F1 = {F1}, length of F2 = {F2}, length of F0 = {F0}".format(\
        #                                                             audio=int(len(data_audio)/fs/0.02), F1=len(F1), F2=len(F2), F0= len(F0)))
        # assert int(len(data_audio)/fs/0.02)- len(F1) ==2
        
        os.remove(temp_filename)
        
        return [F1, F2]

# =============================================================================
'''

This is an data collector with format


Formants_utt_symb[utt][phone] = [F1, F2] record F1, F2's of each utterances'
Formants_people_symb[spkr_name][phone] = [F1, F2] record F1, F2's of each people'
'''

Formants_people_symb=Dict()
Formants_utt_symb=Dict()
# =============================================================================
files=glob.glob(trnpath+"/*_K_*.txt")

silence_duration=0.02 #0.1s

silence_duration_ms=silence_duration*1000
silence = AudioSegment.silent(duration=silence_duration_ms)
if os.path.exists('Gen_formant_multiprocess.log'):
    os.remove('Gen_formant_multiprocess.log')

def process_audio(file):
    print("Process {} executing".format(file))
    filename=os.path.basename(file).split(".")[0]
    spkr_name='_'.join(filename.split("_")[:-3])
    utt='_'.join(filename.split("_")[:])
    audiofile=filepath+"/{name}.wav".format(name=filename)
    trn=trnpath+"/{name}.txt".format(name=filename)
    df_segInfo=pd.read_csv(trn, header=None,delimiter='\t')
    
    audio = AudioSegment.from_wav(audiofile)
    F1F2_extractor=Extract_F1F2()
    
    
    
    
    for st,ed,symb in df_segInfo.values:
        st_ms=st * 1000 #Works in milliseconds
        ed_ms=ed * 1000 #Works in milliseconds
        
        audio_segment = silence + audio[st_ms:ed_ms] + silence
        temp_outfile=F1F2_extractor.PATH+'/../tempfiles/tempwav{}.wav'.format(utt+symb)
        
        audio_segment.export(temp_outfile, format="wav")
        [F1,F2]=F1F2_extractor.extract_features_file(temp_outfile)
        if len(F1) == 0 or len(F2)==0:
            F1_static,F2_static = -1,-1
        else:
            F1_static=functional_method(F1,method=AVERAGEMETHOD)
            F2_static=functional_method(F2,method=AVERAGEMETHOD)
        
        
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
    
    outpath= args.outpath + "/multijobs/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    
    current_process_id=current_process()
    pickle.dump(Formants_utt_symb,open(outpath+"/Formants_utt_symb_{}.pkl".format(current_process_id),"wb"))
    pickle.dump(Formants_people_symb,open(outpath+"/Formants_people_symb_{}.pkl".format(current_process_id),"wb"))
    print("Process {} done".format(file))

multi_ppl_path= args.outpath + "/multijobs/"
rmfiles = glob.glob(multi_ppl_path+"*")
for file in rmfiles:    
    os.remove(file)
pool = Pool(os.cpu_count())
pool.map(process_audio, files)
pool.close()
pool.join()

mlti_ppl_pkls=glob.glob("Pickles/multijobs/Formants_people_symb*.pkl*")

Formants_people_symb=Dict()
for pkl_file in mlti_ppl_pkls:
    load_file_tmp=pickle.load(open(pkl_file,"rb"))
    
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


mlti_utt_pkls=glob.glob("Pickles/multijobs/Formants_utt_symb*.pkl*")
count=0
Formants_utt_symb=Dict()
for pkl_file in mlti_utt_pkls:
    load_file_tmp=pickle.load(open(pkl_file,"rb"))
    
    for utt, df_phone in load_file_tmp.items():
        Formants_utt_symb[utt]=df_phone



            
            
if not os.path.exists(outpath):
    os.makedirs(outpath)

pickle.dump(Formants_utt_symb,open(outpath+"/Formants_utt_symb_by{}.pkl".format(AVERAGEMETHOD),"wb"))
pickle.dump(Formants_people_symb,open(outpath+"/Formant_people_symb_by{}.pkl".format(AVERAGEMETHOD),"wb"))
    
# =============================================================================
'''

Check area
    check if Formants_utt_symb and Formants_people_symb match
    
    randomly check 4 phones for check operation
'''

phones_check=['A:4','ax5','ax4','A:5']
# =============================================================================
# symb2int_table=args.base_path_phf + '/gop_exp_ADOShappyDAAIKidallDeceiptformosaCSRC/nnet3/ADOS_tdnn_fold_transfer_final_xent/' + "phones.txt"
# symbtab=pd.read_csv(symb2int_table, header=None, delimiter=" ")
# symbs=symbtab[1:133][0].values
# valid_symbs=np.delete(symbs,[symbs.tolist().index('b'),symbs.tolist().index('SIL')])
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

'''
# Formants_utt_symb=Dict()
# Formants_people_symb=Dict()
# # =============================================================================
# for file in tqdm(files[13:20]):
#     filename=os.path.basename(file).split(".")[0]
#     spkr_name='_'.join(filename.split("_")[:-3])
#     utt='_'.join(filename.split("_")[:])
#     audiofile=filepath+"/{name}.wav".format(name=filename)
#     trn=trnpath+"/{name}.txt".format(name=filename)
#     df_segInfo=pd.read_csv(trn, header=None,delimiter='\t')
    
#     audio = AudioSegment.from_wav(audiofile)
#     F1F2_extractor=Extract_F1F2()
    
    
    
    
#     for st,ed,symb in df_segInfo.values:
#         st_ms=st * 1000 #Works in milliseconds
#         ed_ms=ed * 1000 #Works in milliseconds
        
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
            # if spkr_name not in Formants_people_symb.keys():
            #     if symb not in Formants_people_symb[spkr_name].keys():
            #         Formants_people_symb[spkr_name][symb]=[[F1_static, F2_static]]
            #     elif symb in Formants_people_symb[spkr_name].keys():
            #         Formants_people_symb[spkr_name][symb].append([F1_static, F2_static])
            # else:
            #     if symb not in Formants_people_symb[spkr_name].keys():
            #         Formants_people_symb[spkr_name][symb]=[[F1_static, F2_static]]
            #     elif symb in Formants_people_symb[spkr_name].keys(): 
            #         Formants_people_symb[spkr_name][symb].append([F1_static, F2_static])
#     Formants_utt_symb[utt] = Formants_utt_symb[utt].T
#     if args.check:
#         if len(Utt_phf_dict[utt][Utt_phf_dict[utt].index != 'SIL']) != len(Formants_utt_symb[utt][Formants_utt_symb[utt].index != "SIL"]):
#             with open('Gen_formant_multiprocess.log', 'a') as f:
#                 string=utt + ": utt in Utt_phf_dict " + str(len(Utt_phf_dict[utt])) + " Not Match utt in Formants_utt_symb "+  str(len(Formants_utt_symb[utt])) + "\n"
                
#                 f.write(string)
#         assert len(Formants_utt_symb[utt]) !=0
    
# pickle.dump(Formants_utt_symb,open(outpath+"/Formants_utt_symb_cmp.pkl","wb"))
# pickle.dump(Formants_people_symb,open(outpath+"/Formants_people_symb_cmp.pkl","wb"))


# =============================================================================
'''

Check reliability 

Assert multiprocess == single
'''
# =============================================================================
if args.checkreliability:
    Formants_utt_symb_cmp=pickle.load(open(outpath+"/Formants_utt_symb_cmp.pkl","rb"))
    Formants_people_symb_cmp=pickle.load(open(outpath+"/Formants_people_symb_cmp.pkl","rb"))
    
    Formants_utt_symb=pickle.load(open(outpath+"/Formants_utt_symb.pkl","rb"))
    Formants_people_symb=pickle.load(open(outpath+"/Formants_people_symb.pkl","rb"))

    for keys in Formants_utt_symb_cmp.keys():
        print(keys)
        assert np.sum(Formants_utt_symb_cmp[keys].values - Formants_utt_symb[keys].values) == 0
        
        
    for spkr_name in Formants_people_symb_cmp.keys():
        for symb in Formants_people_symb_cmp[spkr_name].keys():
            assert np.sum(np.vstack(Formants_people_symb_cmp[spkr_name][symb]) - np.vstack(Formants_people_symb[spkr_name][symb])) ==0
