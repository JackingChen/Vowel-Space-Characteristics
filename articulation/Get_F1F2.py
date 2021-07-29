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
      2021/06/10 : Added some Analyses codes that does the following things:
          1. Plot the F1/F2 data distribution
          2. Filter out the outlier and make condition mask for unqualified people
                           
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
                        help='/homes/ssd1/jackchen/DisVoice/data/{Segmented_ADOS_normalized|Segmented_ADOS_emotion_normalized|Segmented_ADOS_TD_normalized}')
    parser.add_argument('--trnpath', default='/mnt/sdd/jackchen/egs/formosa/s6/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/kid88/ADOS_tdnn_fold_transfer',
                        help='/mnt/sdd/jackchen/egs/formosa/s6/{Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/{kid88|kid_TD}/ADOS_tdnn_fold_transfer | Alignment_human/kid/Audacity_phone|')
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
role= '_D_' if role_str == 'doc' else '_K_'

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

# final_results=pool.starmap(process_audio, [([file_block,silence,trnpath,PoolFormantWindow]) for file_block in tqdm(keys)])
final_results=pool.starmap(multi.process_audio, [([file_block,silence,trnpath,PoolFormantWindow]) for file_block in tqdm(keys)])

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


filedir=os.path.basename(filepath)
role_rawstr=filedir[re.search("Segmented_",filedir).end():re.search("_normalized",filedir).start()]
if 'TD' in role_rawstr:
    role='ASDTD'
else:
    role='ASDkid'

pickle.dump(Formants_utt_symb,open(outpath+"/Formants_utt_symb_by{avgmed}_window{wind}.pkl".format(avgmed=AVERAGEMETHOD,wind=PoolFormantWindow),"wb"))

print("Finished creating Formants_utt_symb in", outpath+"/Formants_utt_symb_by{avgmed}_window{wind}.pkl".format(avgmed=AVERAGEMETHOD,wind=PoolFormantWindow))

pickle.dump(Formants_people_symb,open(outpath+"/Formants_people_symb_by{avgmed}_window{wind}.pkl".format(avgmed=AVERAGEMETHOD,wind=PoolFormantWindow),"wb"))

print("Finished creating Formants_people_symb in", outpath+"/Formants_people_symb_by{avgmed}_window{wind}.pkl".format(avgmed=AVERAGEMETHOD,wind=PoolFormantWindow))




uttpath=outpath+"/Formants_utt_symb_by{avgmed}_window{wind}.pkl".format(avgmed=AVERAGEMETHOD,wind=PoolFormantWindow)
utt_outpath=outpath+"/Formants_utt_symb_by{avgmed}_window{wind}_{role}.pkl".format(avgmed=AVERAGEMETHOD,\
                                                                                   wind=PoolFormantWindow,\
                                                                                   role=role)
peoplepath=outpath+"/Formants_people_symb_by{avgmed}_window{wind}.pkl".format(avgmed=AVERAGEMETHOD,wind=PoolFormantWindow)
people_outpath=outpath+"/Formants_people_symb_by{avgmed}_window{wind}_{role}.pkl".format(avgmed=AVERAGEMETHOD,\
                                                                                   wind=PoolFormantWindow,\
                                                                                   role=role)


    
shutil.copy(uttpath, utt_outpath)
shutil.copy(peoplepath, people_outpath)


 
''' Multithread processing end '''


# =============================================================================
'''

    Manual area
    You can use it to debug


    If you want to manually debug, you copy the code from the function
'''

# # =============================================================================
# pickle.dump(Formants_utt_symb,open(outpath+"/Formants_utt_symb_cmp.pkl","wb"))
# pickle.dump(Formants_people_symb,open(outpath+"/Formants_people_symb_cmp.pkl","wb"))


# =============================================================================
'''

    Check data distribution of Vowel Of Interest

''' 
PhoneMapp_dict=phonewoprosody.PhoneMapp_dict
PhoneOI=PhoneMapp_dict.keys()
# =============================================================================
Formants_utt_symb=pickle.load(open(outpath+"/Formants_utt_symb_by{avgmed}_window{wind}.pkl".format(avgmed=AVERAGEMETHOD,wind=PoolFormantWindow),"rb"))
Formants_people_symb=pickle.load(open(outpath+"/Formants_people_symb_by{avgmed}_window{wind}.pkl".format(avgmed=AVERAGEMETHOD,wind=PoolFormantWindow),"rb"))


# Use inter quartile range to decide the Formant limits    
# First: gather all data and get statistic values
def Get_PersonalPhonedata(Formants_people_symb,PhoneOI):
    Person_IQR_dict=Dict()
    Person_IQR_all_dict=Dict()
    for p, v in Formants_people_symb.items():
        for symb in PhoneOI:
            phones_comb=Formants_people_symb[p]
            for phone, values in phones_comb.items():
                if phone in [x for x in  PhoneMapp_dict[symb]]:
                    df_phone_values=pd.DataFrame(phones_comb[phone],columns=args.Inspect_features)
                    df_phone_values.index=[phone]*len(values)
                    
                    gender_query_str=p
                    series_gend=Info_name_sex[Info_name_sex['name']==gender_query_str]['sex']
                    gender=series_gend.values[0]
                    
                    df_phone_values['sex']=gender
                    # Gather all data of all single person
                    if symb not in Person_IQR_dict[p].keys():
                        Person_IQR_dict[p][symb]=pd.DataFrame()
                    Person_IQR_dict[p][symb]=Person_IQR_dict[p][symb].append(df_phone_values)
                    
                    
                    # Gather all data of all people
                    if symb not in Person_IQR_all_dict.keys():
                        Person_IQR_all_dict[symb]=pd.DataFrame()
                    Person_IQR_all_dict[symb]=Person_IQR_all_dict[symb].append(df_phone_values)
    return Person_IQR_dict, Person_IQR_all_dict
AUI_dict, _=Get_PersonalPhonedata(Formants_people_symb,PhoneOI=PhoneOI)

# =============================================================================
# Plot boxplots
# PeopleOfInterest=['2016_06_27_02_017_1', '2016_07_30_01_148', '2016_08_26_01_168_1',
#        '2016_09_24_01_174_1'] # Manual choose
PeopleOfInterest=Formants_people_symb.keys()
# =============================================================================
Name2num=Dict()
for i,k in enumerate(sorted(AUI_dict.keys())):
    Name2num[k]=i

a4_dims = (13.7, 8.27)
figureOutpath="Inspect/"
sex='male'
for feat in args.Inspect_features:
    for symb in PhoneOI:
        plt.figure()
        df_data_top=pd.DataFrame([])
        for person in AUI_dict.keys():
            if person not in PeopleOfInterest:
                continue
            if symb in AUI_dict[person].keys():
                df_data=AUI_dict[person][symb]
                df_data['people']=Name2num[person]
                if len(sex)>0:
                    df_data=df_data[df_data['sex']==sex]
                df_data_top=df_data_top.append(df_data)
                # bx = sns.boxplot(x="people", y=feat, data=df_data)
        # cx = sns.boxplot(x="people", y=feat, data=df_data_top.iloc[:100])
        fig, ax = plt.subplots(figsize=a4_dims)
        ax = sns.boxplot(ax=ax, x="people", y=feat, data=df_data_top)
        title='{0}_{1}'.format('distribution single people boxplot ' + symb,'feature: ' +feat)
        plt.title( title )
        plt.savefig(figureOutpath+'{0}_{1}.png'.format(symb,feat))
        
# =============================================================================
# Joint plot those people of data
# =============================================================================
plot_outpath=plot_outpath+trnpath[re.search("new_system",trnpath).end()+1:re.search("ADOS_tdnn",trnpath).start()-1]
for people in PeopleOfInterest:
    df_samples_AUI=pd.DataFrame()
    for symb in AUI_dict[people].keys():
        df_tmp=AUI_dict[people][symb]
        df_tmp['phone']=symb
        df_samples_AUI=df_samples_AUI.append(df_tmp)
    sns.jointplot(data=df_samples_AUI, x='F1',y='F2',hue='phone')
    # for line in range(0,AUI_dict[people][symb].shape[0]):
    #     plt.text(AUI_dict[people][symb].F1.iloc[line]+0.2, AUI_dict[people][symb].F1.iloc[line], AUI_dict[people][symb].utt.iloc[line], horizontalalignment='left', size='medium', color='black', weight='semibold')

    
    
    if not os.path.exists(plot_outpath):
        os.makedirs(plot_outpath)
    # info_str="""f1MSB:Vowel: {0}""".format(symb)
    title='{0}_{1}'.format(people, symb)
    plt.title( title )
    plt.savefig (plot_outpath+"/{0}.png".format(title))
    # plt.text(x=0, y=0,s=info_str)

# =============================================================================
'''

    Filter data (by 1.5*IQR) and Generate Qualified people condition dataframe

'''
# find not reasonable data by functionals 
# =============================================================================
df_top_dict=Dict()
N=1
FilterOutlier=True
for feat in args.Inspect_features:
    for symb in PhoneOI:
        df_people_statistics=pd.DataFrame([],columns=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        for person in AUI_dict.keys():
            if symb not in AUI_dict[person].keys():
                continue
            df_data=AUI_dict[person][symb][[feat]]
            if FilterOutlier:
                q25, q75 = df_data.quantile(q=0.25).values[0], df_data.quantile(q=0.75).values[0]
                iqr=q75 - q25
                cut_off = iqr * 1.5
                lower, upper = q25 - cut_off, q75 + cut_off
                
                df_data = df_data[df_data <= upper]
                df_data = df_data[df_data >= lower]
                
            
            df_people_statistics.loc[person,df_data.describe().index]=df_data[feat].describe()
            df_people_statistics['symb']=symb
            
        df_top_dict[symb][feat]=df_people_statistics

cond_N=(df_top_dict['A:'].F1['count'] > N) & (df_top_dict['u:'].F1['count'] > N) & (df_top_dict['i:'].F1['count'] > N)
# find those people that F1 
cond_feat1=(df_top_dict['A:'].F1['50%'] - df_top_dict['u:'].F1['50%']).dropna() <0
cond_feat2=(df_top_dict['A:'].F1['50%'] - df_top_dict['i:'].F1['50%']).dropna() <0
cond_feat3=(df_top_dict['i:'].F2['50%'] - df_top_dict['u:'].F2['50%']).dropna() <0
cond_feat_unreasonable=cond_feat1 | cond_feat2 | cond_feat3
cond=cond_N & cond_feat_unreasonable
cond.name=cond_feat1.name
cond_feat1.index[cond_feat1==True]
cond_feat2.index[cond_feat2==True]
cond_feat3.index[cond_feat3==True]
cond.index[cond==True]


condition_path="Inspect/condition/"
if not os.path.exists(condition_path):
    os.makedirs(condition_path)
cond.to_frame().to_excel(condition_path+"unreasonable_all.xlsx")
cond_feat1.to_frame().to_excel(condition_path+"F1A_lessThan_u.xlsx")
cond_feat2.to_frame().to_excel(condition_path+"F1A_lessThan_i.xlsx")
cond_feat3.to_frame().to_excel(condition_path+"F2i_lessThan_u.xlsx")

# BetweenPhoneDistance= GetBetweenPhoneDistance(df_top_dict)




    
    
    
    
    













# =============================================================================
'''

    Old function codes that are not in use currently

'''
# =============================================================================
def process_audio(files,silence,trnpath,functional_method_window):

    

# Person_IQR_dict=Dict()
# Person_IQR_all_dict=Dict()
# for p, v in Formants_people_symb.items():
#     for symb in PhoneOI:
#         phones_comb=Formants_people_symb[p]
#         for phone, values in phones_comb.items():
#             if phone in [x for x in  PhoneMapp_dict[symb]]:
#                 df_phone_values=pd.DataFrame(phones_comb[phone],columns=args.Inspect_features)
#                 df_phone_values.index=[phone]*len(values)
                
#                 gender_query_str=p
#                 series_gend=Info_name_sex[Info_name_sex['name']==gender_query_str]['sex']
#                 gender=series_gend.values[0]
                
#                 df_phone_values['sex']=gender
#                 # Gather all data of all single person
#                 if symb not in Person_IQR_dict[p].keys():
#                     Person_IQR_dict[p][symb]=pd.DataFrame()
#                 Person_IQR_dict[p][symb]=Person_IQR_dict[p][symb].append(df_phone_values)
                
                
#                 # Gather all data of all people
#                 if symb not in Person_IQR_all_dict.keys():
#                     Person_IQR_all_dict[symb]=pd.DataFrame()
#                 Person_IQR_all_dict[symb]=Person_IQR_all_dict[symb].append(df_phone_values)
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
        if 'Session' in filepath:
            audiofile=filepath+"/{name}.wav".format(name=filename[:re.search("_[K|D]_", filename).end()-1])
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
            st_ms=st * 1000 #Works in milliseconds
            ed_ms=ed * 1000 #Works in milliseconds
            # st_ms=st_ext * 1000 #Works in milliseconds
            # ed_ms=ed_ext * 1000 #Works in milliseconds
    
            audio_segment = silence + audio[st_ms:ed_ms] + silence
            temp_outfile=F1F2_extractor.PATH+'/../tempfiles/tempwav{}.wav'.format(utt+symb)
            
            audio_segment.export(temp_outfile, format="wav")
            if args.formantmethod == 'Disvoice':
                [F1,F2]=F1F2_extractor.extract_features_file(temp_outfile)
            elif args.formantmethod == 'praat':
                try:
                    MaxnumForm=5
                    # if 'u:' in symb or 'A:' in symb:
                    if 'u:' in symb:
                        maxFormant=3000
                    else:
                        maxFormant=5000
                    [F1,F2]=measureFormants(temp_outfile,minf0,maxf0,time_step=F1F2_extractor.step,MaxnumForm=MaxnumForm,Maxformant=maxFormant,framesize=F1F2_extractor.sizeframe)
                except :
                    print("Error processing ",utt+"__"+symb)
                    error_msg_bag.append(utt+"__"+symb)
            if len(F1) == 0 or len(F2)==0:
                F1_static, F2_static= -1, -1
            else:
                F1_static=functional_method(F1,method=AVERAGEMETHOD,window=functional_method_window)
                F2_static=functional_method(F2,method=AVERAGEMETHOD,window=functional_method_window)
            
            
            assert  math.isnan(F1_static) == False and math.isnan(F2_static) == False
            # if symb !='u:1':
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
        if args.check:
            if len(Utt_phf_dict[utt][Utt_phf_dict[utt].index != 'SIL']) != len(Formants_utt_symb[utt][Formants_utt_symb[utt].index != "SIL"]):
                with open('Gen_formant_multiprocess.log', 'a') as f:
                    string=utt + ": utt in Utt_phf_dict " + str(len(Utt_phf_dict[utt])) + " Not Match utt in Formants_utt_symb "+  str(len(Formants_utt_symb[utt])) + "\n"
                    
                    f.write(string)
            assert len(Formants_utt_symb[utt]) !=0
    
    return Formants_utt_symb, Formants_people_symb