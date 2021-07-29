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

# path_app = '/homes/ssd1/jackchen/DisVoice/articulation'
# sys.path.append(path_app)
try:
    from .articulation import Extract_F1F2
except:
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
import numpy.ma as ma
import tgre
# path_app = '/homes/ssd1/jackchen/DisVoice'
# sys.path.append(path_app)
from utils_jack  import functional_method, Info_name_sex, F0_parameter_dict


class Multi:
    def __init__(self, filepath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles', MaxnumForm=5,AVERAGEMETHOD='middle'):
        self.MaxnumForm=MaxnumForm
        self.filepath=filepath
        self.formantmethod='praat'
        self.check=False
        self.AVERAGEMETHOD=AVERAGEMETHOD
        self.Constraint={"F1":[0,1000],"F2":[0,2500]}
    def _updatePhonemapp(self,phoneMappdict):
        self.phoneMappdict = phoneMappdict
    def _updatePhonedict(self,Phoneme_sets):
        self.Phoneme_sets = Phoneme_sets
    def _updateLeftSymbMapp(self,LeftSymbMapp):
        self.LeftSymbMapp = LeftSymbMapp
    def _self_constraint(self,data_one_dim,feat='F1'):
        lower_bound, upper_bound=self.Constraint[feat]
        
        bool_mask=np.logical_or(np.array(data_one_dim)>upper_bound,np.array(data_one_dim)<lower_bound)
        
        data_clipped=ma.array(data_one_dim,mask=bool_mask)
        return data_clipped
    def process_audio(self,files,silence,trnpath,functional_method_window=3,record_WavTrn=False):
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
                ed_ext= min(ed + F1F2_extractor.sizeframe/2,max(df_segInfo['ed']))
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
                    def GetmaxFormant(symb, age, sex):
                        if age <= 12 or sex=='female':
                            maxFormant=5500
                        else:
                            maxFormant=5000
                        if 'u:' in symb or 'w' in symb:
                            maxFormant=3000
                        return maxFormant
        
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

        
        if len(error_msg_bag) !=0:
            import warnings
            warnings.warn("Warning..files in ..{0}...is not sucessfully computed".format(error_msg_bag))
        if record_WavTrn:
            return Formants_utt_symb, Formants_people_symb, [Wav_collect, Trn_collect]
        else:
            return Formants_utt_symb, Formants_people_symb
    def FilterUttDictsByCriterion_map(self,Formants_utt_symb,Formants_utt_symb_cmp,keys,limit_people_rule):
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
            
            
            



