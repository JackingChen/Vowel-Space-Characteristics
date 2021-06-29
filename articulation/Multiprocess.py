#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jun 29 11:57:01 2021

@author: jackchen
"""
from addict import Dict
import re 
import pandas as pd
import numpy as np
import os, sys
from articulation import Extract_F1F2
from pydub import AudioSegment
try:
    from .articulation_functions import measureFormants
except: 
    from articulation_functions import measureFormants
import math
from tqdm import tqdm
from utils_wer.wer import  wer as WER
from utils_jack  import  Get_aligned_sequences


path_app = '/homes/ssd1/jackchen/DisVoice'
sys.path.append(path_app)
from utils_jack  import functional_method, Info_name_sex, F0_parameter_dict

class Multi:
    def __init__(self, filepath, MaxnumForm=5,AVERAGEMETHOD='middle'):
        self.MaxnumForm=MaxnumForm
        self.filepath=filepath
        self.formantmethod='praat'
        self.check=False
        self.AVERAGEMETHOD=AVERAGEMETHOD
    def process_audio(self,files,silence,trnpath,functional_method_window=3):
        Formants_people_symb=Dict()
        Formants_utt_symb=Dict()
        error_msg_bag=[]
        print("Process {} executing".format(files))
        for file in files:
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
                        MaxnumForm=5
                        if 'u:' in symb:
                            maxFormant=3000
                        else:
                            maxFormant=5000
                        [F1,F2]=measureFormants(temp_outfile,minf0,maxf0,time_step=F1F2_extractor.step,MaxnumForm=self.MaxnumForm,Maxformant=maxFormant,framesize=F1F2_extractor.sizeframe)
                    except :
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
            Formants_utt_symb[utt] = Formants_utt_symb[utt].T
            df=pd.DataFrame(df_segInfo[[0,1]].values,index=df_segInfo[2])
            Formants_utt_symb[utt]['start']=df[0]
            Formants_utt_symb[utt]['end']=df[1]

        
        if len(error_msg_bag) !=0:
            import warnings
            warnings.warn("Warning..files in ..{0}...is not sucessfully computed".format(error_msg_bag))
        
        return Formants_utt_symb, Formants_people_symb
    def FilterUttDictsByCriterion_map(Formants_utt_symb,Formants_utt_symb_cmp,keys,limit_people_rule):
        # Masks will be generated by setting criterion on Formants_utt_symb
        # and Formants_utt_symb_cmp will be masked by the same mask as Formants_utt_symb
        # we need to make sure two things:
        #   1. the length of Formants_utt_symb_cmp and Formants_utt_symb are the same
        #   2. the phone sequences are aligned correctly
        Formants_utt_symb_limited=Dict()
        Formants_utt_symb_cmp_limited=Dict()
        for utt in tqdm(keys):
            people=utt[:utt.find(re.findall("_[K|D]",utt)[0])]
            df_ori=Formants_utt_symb[utt].sort_values(by="start")
            df_cmp=Formants_utt_symb_cmp[utt].sort_values(by="start")
            df_ori['text']=df_ori.index
            df_cmp['text']=df_cmp.index
            
            r=df_cmp.index.astype(str)
            h=df_ori.index.astype(str)
            
            error_info, WER_value=WER(r,h)
            utt_human_ali, utt_hype_ali=Get_aligned_sequences(ref=df_cmp, hype=df_ori ,error_info=error_info) # This step cannot gaurentee hype and human be exact the same string
                                                                                                              # because substitude error also counts when selecting the optimal 
                                                                                                              # matched string         
            utt_human_ali.index=utt_human_ali['text']
            utt_human_ali=utt_human_ali.drop(columns=["text"])
            utt_hype_ali.index=utt_hype_ali['text']
            utt_hype_ali=utt_hype_ali.drop(columns=["text"])
            
            assert len(utt_human_ali) == len(utt_hype_ali)
            limit_rule=limit_people_rule[people]
            SymbRuleChecked_bookkeep={}
            for symb_P in limit_rule.keys():
                values_limit=limit_rule[symb_P]
        
                filter_bool=utt_hype_ali.index.str.contains(symb_P)  #  1. select the phone with criterion
                filter_bool_inv=np.invert(filter_bool)           #  2. create a mask for unchoosed phones
                                                                 #  we want to make sure that only 
                                                                 #  the phones not match the criterion will be False
                for feat in values_limit.keys():
                    feat_max_value=values_limit[feat]['max']
                    filter_bool=np.logical_and(filter_bool , (utt_hype_ali[feat]<=feat_max_value))
                    feat_min_value=values_limit[feat]['min']
                    filter_bool=np.logical_and(filter_bool , (utt_hype_ali[feat]>=feat_min_value))
                    
                filter_bool=np.logical_or(filter_bool_inv,filter_bool)
                
                # check & debug
                if not filter_bool.all():
                    print(utt,filter_bool[filter_bool==False])
                
                SymbRuleChecked_bookkeep[symb_P]=filter_bool.to_frame()
            
            df_True=pd.DataFrame(np.array([True]*len(utt_hype_ali)))
            for keys, values in SymbRuleChecked_bookkeep.items():
                df_True=np.logical_and(values,df_True)
            
            Formants_utt_symb_limited[utt]=utt_hype_ali[df_True[0].values]
            Formants_utt_symb_cmp_limited[utt]=utt_human_ali[df_True[0].values]
        return Formants_utt_symb_limited,Formants_utt_symb_cmp_limited

class Multi_WithPosteriorgram(Multi):
    def process_audio(self,files,silence,trnpath,functional_method_window=3, AVERAGEMETHOD='middle'):
        Formants_people_symb=Dict()
        Formants_utt_symb=Dict()
        error_msg_bag=[]
        print("Process {} executing".format(files))
        for file in files:
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
                        MaxnumForm=5
                        if 'u:' in symb:
                            maxFormant=3000
                        else:
                            maxFormant=5000
                        [F1,F2]=measureFormants(temp_outfile,minf0,maxf0,time_step=F1F2_extractor.step,MaxnumForm=self.MaxnumForm,Maxformant=maxFormant,framesize=F1F2_extractor.sizeframe)
                    except :
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
            Formants_utt_symb[utt] = Formants_utt_symb[utt].T
            df=pd.DataFrame(df_segInfo[[0,1]].values,index=df_segInfo[2])
            Formants_utt_symb[utt]['start']=df[0]
            Formants_utt_symb[utt]['end']=df[1]
            if self.check:
                if len(Utt_phf_dict[utt][Utt_phf_dict[utt].index != 'SIL']) != len(Formants_utt_symb[utt][Formants_utt_symb[utt].index != "SIL"]):
                    with open('Gen_formant_multiprocess.log', 'a') as f:
                        string=utt + ": utt in Utt_phf_dict " + str(len(Utt_phf_dict[utt])) + " Not Match utt in Formants_utt_symb "+  str(len(Formants_utt_symb[utt])) + "\n"
                        
                        f.write(string)
                assert len(Formants_utt_symb[utt]) !=0
        
        if len(error_msg_bag) !=0:
            import warnings
            warnings.warn("Warning..files in ..{0}...is not sucessfully computed".format(error_msg_bag))
            
            
            



