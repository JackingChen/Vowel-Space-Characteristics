#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:14:50 2020

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
    from articulation_functions import extractTrans, V_UV, measureFormants
    
try:
    from .articulation import Extract_F1F2
except:
    from articulation import Extract_F1F2
    
import uuid
import pandas as pd
import torch
from tqdm import tqdm
from addict import Dict
import glob
import argparse
import math
from pydub import AudioSegment

from HYPERPARAM import phonewoprosody, Label
from multiprocessing import Pool, current_process
import pickle
import subprocess
import re
import seaborn as sns
from matplotlib import pyplot as plt
import parselmouth 
from parselmouth.praat import call

path_app = '/mnt/sdd/jackchen/egs/formosa/s6/local'
sys.path.append(path_app)
from utils_wer.wer import  wer as WER
def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice/articulation',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--base_path_phf', default='/homes/ssd1/jackchen/gop_prediction/data',
                        help='path of the base directory')
    parser.add_argument('--filepath', default='/homes/ssd1/jackchen/DisVoice/data/Segmented_ADOS_emotion_normalized',
                        help='/homes/ssd1/jackchen/DisVoice/data/{Segmented_ADOS_normalized|Session_ADOS_normalized}')
    parser.add_argument('--trnpath', default='/mnt/sdd/jackchen/egs/formosa/s6/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/kid_cmpWithHuman/ADOS_tdnn_fold_transfer',
                        help='/mnt/sdd/jackchen/egs/formosa/s6/Alignment_ADOShappyDAAIKidallDeceiptformosaCSRC_chain/old_system/kid/ADOS_tdnn_fold_transfer | /mnt/sdd/jackchen/egs/formosa/s6/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/kid_cmpWithHuman/ADOS_tdnn_fold_transfer')
    parser.add_argument('--outpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--formantmethod', default='praat',
                        help='path of the base directory')
    parser.add_argument('--avgmethod', default='middle',
                        help='path of the base directory')
    parser.add_argument('--check', default=False,
                        help='path of the base directory')
    parser.add_argument('--checkreliability', default=False,
                            help='path of the base directory')
    parser.add_argument('--Stat_med_str_VSA', default='mean',
                            help='path of the base directory')
    parser.add_argument('--PoolFormantWindow', default=3, type=int,
                            help='path of the base directory')
    parser.add_argument('--Calculate_alignment_score', default=True, type=bool,
                            help='')
    parser.add_argument('--phonesInspect', default='A:|u:|i:|j',
                            help='')
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
# path_app = base_path+'/../'
# sys.path.append(path_app)
Stat_med_str=args.Stat_med_str_VSA
PoolFormantWindow=args.PoolFormantWindow
Inspect_features=args.Inspect_features
import praat.praat_functions as praat_functions
# from script_mananger import script_manager
from utils_jack import *
from utils_jack  import functional_method, Formant_utt2people_reshape, \
                        Gather_info_certainphones,GetValuelimit_IQR,\
                        Get_aligned_sequences, f_classif
from metric import EvaluateAlignments
import Multiprocess

if args.Calculate_alignment_score:
    phonesInspect = args.phonesInspect


# =============================================================================
'''
Data generation from raw


This is an data collector with format


Formants_utt_symb[utt][phone] = [F1, F2] record F1, F2's of each utterances'
Formants_people_symb[spkr_name][phone] = [F1, F2] record F1, F2's of each people'
'''

# =============================================================================
def Process_F1F2_Multi(trnpath,trntype='human',AVERAGEMETHOD='middle',functional_method_window=3):
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
    keys=[]
    interval=2
    for i in range(0,len(files),interval):
        # print(list(combs_tup.keys())[i:i+interval])
        keys.append(files[i:i+interval])
    flat_keys=[item for sublist in keys for item in sublist]
    assert len(flat_keys) == len(files)
    # final_results=pool.starmap(process_audio, [([file_block,silence,trnpath,functional_method_window]) for file_block in tqdm(keys)])
    multi=Multiprocess.Multi(filepath, MaxnumForm=5, AVERAGEMETHOD=AVERAGEMETHOD)
    multi._updatePhonedict(phonewoprosody.Phoneme_sets)
    multi._updateLeftSymbMapp(phonewoprosody.LeftSymbMapp)
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
    pickle.dump(Formants_utt_symb,open(outpath+"/[Testing]Formants_utt_{role}_symb_by{avg}.pkl".format(role=trntype,avg=AVERAGEMETHOD),"wb"))
    print('Formants_utt_symb saved to ',outpath+"/[Testing]Formants_utt_{role}_symb_by{avg}.pkl".format(role=trntype,avg=AVERAGEMETHOD))
    pickle.dump(Formants_people_symb,open(outpath+"/[Testing]Formants_people_{role}_symb_by{avg}.pkl".format(role=trntype,avg=AVERAGEMETHOD),"wb"))
    print('Formants_people_symb saved to ',outpath+"/[Testing]Formants_people_{role}_symb_by{avg}.pkl".format(role=trntype,avg=AVERAGEMETHOD))
    
    
#################  Code executed here   #############################
trnpath_human='/mnt/sdd/jackchen/egs/formosa/s6/Alignment_human/kid/Audacity_phone'
Process_F1F2_Multi(trnpath_human,trntype='human',AVERAGEMETHOD=AVERAGEMETHOD,functional_method_window=PoolFormantWindow)
trnpath_Auto=args.trnpath
Process_F1F2_Multi(trnpath_Auto,trntype='Auto',AVERAGEMETHOD=AVERAGEMETHOD,functional_method_window=PoolFormantWindow)

''' Multithread processing end '''

# =============================================================================













# =============================================================================
'''

    Report agreement with human at tolarance [ 10 20 30 ] ms

''' 
# =============================================================================
''' 1. load data '''
Formants_utt_symb_cmp=pickle.load(open(outpath+"/[Testing]Formants_utt_{role}_symb_by{avg}.pkl".format(role='human',avg=AVERAGEMETHOD),"rb"))
Formants_utt_symb=pickle.load(open(outpath+"/[Testing]Formants_utt_{role}_symb_by{avg}.pkl".format(role='Auto',avg=AVERAGEMETHOD),"rb"))
# Formants_people_symb_cmp=pickle.load(open(outpath+"/Formants_people_symb_bymiddle_window3_ASDTD.pkl","rb"))
# Formants_people_symb=pickle.load(open(outpath+"/Formants_people_symb_bymiddle_window3_ASDkid.pkl","rb"))
# =============================================================================
'''

    Filter data using the data distribution itself (1.5 IQR range)
'''


# =============================================================================

PhoneMapp_dict=phonewoprosody.PhoneMapp_dict
PhoneOI=PhoneMapp_dict.keys()
PhoneOfInterest=list(PhoneMapp_dict.keys())
# =============================================================================
# Use inter quartile range to decide the Formant limits    
# First: gather all data and get statistic values
Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb_cmp,Align_OrinCmp=False)
AUI_info=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)
# =============================================================================
# Filter data by IQR
# =============================================================================
limit_people_rule=GetValuelimit_IQR(AUI_info,PhoneMapp_dict,args.Inspect_features)
Formants_utt_symb_limited,Formants_utt_symb_cmp_limited=FilterUttDictsByCriterion(Formants_utt_symb,Formants_utt_symb_cmp,limit_people_rule)

if len(limit_people_rule) >0:
    Formants_utt_symb=Formants_utt_symb_limited
    Formants_utt_symb_cmp=Formants_utt_symb_cmp_limited

Formants_utt_symb_cmp_limited,Formants_utt_symb_limited=FilterUttDictsByCriterion(Formants_utt_symb_cmp,Formants_utt_symb,limit_people_rule)
if len(limit_people_rule) >0:
    Formants_utt_symb=Formants_utt_symb_limited
    Formants_utt_symb_cmp=Formants_utt_symb_cmp_limited        

# =============================================================================
'''

    Calculate the human-aligner agreement (focusing on certain phone)
    
    Note. we measure this after filtering out the outliers
'''
# =============================================================================
Dict_phone_Inspect=Dict()
for phonesInspect in ['A:','u:','i:|j','A:|u:|i:|j','A:|u:|i:|j|w']:
# for phonesInspect in ['uA:','uO:','uaI','ueI']:
    df_evaluation_metric=EvaluateAlignments(Formants_utt_symb_cmp,Formants_utt_symb,phonesInspect)
    df_evaluation_metric.loc['average']=df_evaluation_metric.mean()
    print('phones Inspect',phonesInspect)
    print(df_evaluation_metric)
    Dict_phone_Inspect[phonesInspect]=df_evaluation_metric












# =============================================================================
'''

    Compare Formant values between human labels and aligner

''' 
# =============================================================================

''' 1. gather data '''
''' Formant_people_symb_total['cmp'][people] = df: index = phone, column = [F1, F2]'''
Formant_people_symb_total=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb_cmp,Align_OrinCmp=True)
''' 2. manage data '''
AUI_dict=Gather_info_certainphones(Formant_people_symb_total,PhoneMapp_dict,PhoneOfInterest)
# =============================================================================
'''

    Application 1
    
    Inspect the deviation when using forced ALignment from human label
    
    we create the following two information:
        1. joint plot
        2. absolute difference

'''
# =============================================================================
''' Plot distributions '''
plot_outpath=base_path+'/Plot_cmp_ori'
if not os.path.exists(plot_outpath):
    os.makedirs(plot_outpath)

df_ttest_p=pd.DataFrame([],columns=args.Inspect_features)
for people in AUI_dict.keys():
    for symb in AUI_dict[people].keys():
        g= sns.JointGrid(data=AUI_dict[people][symb], x='F1',y='F2',hue='cmps')
        g.plot(sns.scatterplot, sns.histplot)

        
        # for line in range(0,AUI_dict[people][symb].shape[0]):
        # plt.text(AUI_dict[people][symb].F1.iloc[line]+0.2, AUI_dict[people][symb].F1.iloc[line], AUI_dict[people][symb].utt.iloc[line], horizontalalignment='left', size='medium', color='black', weight='semibold')
        # info_str="""f1MSB:Vowel: {0}""".format(symb)
        title='{0}_{1}'.format(people, symb)
        plt.title( title )
        plt.savefig (plot_outpath+"/{0}.png".format(title))
        # plt.text(x=0, y=0,s=info_str)
        
        ''' ttest to test if two distributions have the same mean'''
        df_data_cmp=AUI_dict[people][symb][AUI_dict[people][symb]['cmps']=='cmp']
        df_data_ori=AUI_dict[people][symb][AUI_dict[people][symb]['cmps']=='ori']
        
        for feat in args.Inspect_features:
            data_cmp,data_ori=df_data_cmp[feat], df_data_ori[feat]
            df_ttest_p.loc['{0}_{1}'.format(people,symb),feat]=stats.ttest_ind(data_cmp,data_ori)[1]

''' Calculate feature distances'''
Dict_people_formant_diff=Dict()
for symb in PhoneOfInterest:
    Dict_people_formant_diff[symb]=pd.DataFrame([],columns=Inspect_features)
    for people in AUI_dict.keys():    
        ori_data=AUI_dict[people][symb][AUI_dict[people][symb]['cmps']=='ori']
        cmp_data=AUI_dict[people][symb][AUI_dict[people][symb]['cmps']=='cmp']
        ori_data.index, cmp_data.index=ori_data['utt'].map(str) + '_' + ori_data.index.map(str), cmp_data['utt'].map(str) + "_" + cmp_data.index.map(str)
        df_subtract=ori_data[Inspect_features].subtract(cmp_data[Inspect_features])
        
        df_FeatDist_average=df_subtract.abs().mean(axis=0)

        Dict_people_formant_diff[symb].loc[people]=df_FeatDist_average.values
print("average deviation of features:",Inspect_features," is \n",df_FeatDist_average)
aaa=ccc
























# =============================================================================
'''
    [Not Gonna use this anymore] !!!
    
    
    Application 2
    
    Inspect the deviation when using forced ALignment from human label
    
    we create the following information:
        1. the deviation of final features (LOC)
        2. the deviation of rank of final features


'''
# =============================================================================
# AUI_dict to Vowels_AUI
Vowels_AUI_top=Dict()
for people in AUI_dict.keys():
    for symb in AUI_dict[people].keys():
        Vowels_AUI_top['cmp'][people][symb]=AUI_dict[people][symb][AUI_dict[people][symb]['cmps']=="cmp"].values[:,[0,1]]
        Vowels_AUI_top['ori'][people][symb]=AUI_dict[people][symb][AUI_dict[people][symb]['cmps']=="ori"].values[:,[0,1]]
df_formant_statistic77_path='Pickles/Session_formants_people_vowel_feat/Formant_AUI_tVSAFCRFvals_ASDkid.pkl'
df_formant_statistic_77=pickle.load(open(df_formant_statistic77_path,'rb'))
FeatOfInterests=['F_vals_f1','F_vals_f2','F_val_mix','MSB_f1','MSB_f2','MSB_mix']
df_quantile_diff_total=pd.DataFrame([],index=FeatOfInterests,columns=df_formant_statistic_cmp.index)
for feat in FeatOfInterests:
    rank_diff_mean, df_rank_diff=Compare_feature_by_quantile_diff(df_formant_statistic_77,\
                                     df_formant_statistic_cmp,\
                                     df_formant_statistic_ori,feat=feat)
    print(feat,rank_diff_mean)
    df_quantile_diff_total.loc[feat,'abs_diff_mean'] = rank_diff_mean.values
    df_quantile_diff_total.loc[feat,df_rank_diff.index.values.astype(str)] = df_rank_diff.values.reshape(-1)
outpath="RESULTS/"
if not os.path.exists(outpath):
    os.makedirs(outpath)
exp_path=trnpath[trnpath.find("Alignment"):trnpath.find("_chain")]
argument_str='_'.join([exp_path,args.avgmethod,'windowsize{}'.format(args.PoolFormantWindow)])
df_quantile_diff_total.to_excel(outpath+"diff_human_alignment{arguments}.xlsx".format(arguments='_'+argument_str))

















def calculate_features(Vowels_AUI,Label):
    # =============================================================================
    # Code calculate vowel features
    Statistic_method={'mean':np.mean,'median':np.median,'mode':stats.mode}
    label_choose='ADOS_C'
    # =============================================================================
    df_formant_statistic=pd.DataFrame()
    for people in Vowels_AUI.keys(): #update 2021/05/27 fixed 
        RESULT_dict={}
        F12_raw_dict=Vowels_AUI[people]
        F12_val_dict={k:[] for k in ['u','a','i']}
        for k,v in F12_raw_dict.items():
            if Stat_med_str == 'mode':
                F12_val_dict[k]=Statistic_method[Stat_med_str](v,axis=0)[0].ravel()
            else:
                F12_val_dict[k]=Statistic_method[Stat_med_str](v,axis=0)
        RESULT_dict['u_num'], RESULT_dict['a_num'], RESULT_dict['i_num']=\
            len(Vowels_AUI[people]['u']),len(Vowels_AUI[people]['a']),len(Vowels_AUI[people]['i'])
        
        RESULT_dict['ADOS']=Label.label_raw[label_choose][Label.label_raw['name']==people].values    
        RESULT_dict['sex']=Label.label_raw['sex'][Label.label_raw['name']==people].values[0]
        RESULT_dict['age']=Label.label_raw['age_year'][Label.label_raw['name']==people].values[0]
        RESULT_dict['Module']=Label.label_raw['Module'][Label.label_raw['name']==people].values[0]
        
        u=F12_val_dict['u']
        a=F12_val_dict['a']
        i=F12_val_dict['i']
    
        
        if len(u)==0 or len(a)==0 or len(i)==0:
            u_num= RESULT_dict['u_num'] if type(RESULT_dict['u_num'])==int else 0
            i_num= RESULT_dict['i_num'] if type(RESULT_dict['i_num'])==int else 0
            a_num= RESULT_dict['a_num'] if type(RESULT_dict['a_num'])==int else 0
            df_RESULT_list=pd.DataFrame(np.zeros([1,len(df_formant_statistic.columns)]),columns=df_formant_statistic.columns)
            df_RESULT_list.index=[people]
            df_RESULT_list['FCR']=10
            df_RESULT_list['ADOS']=RESULT_dict['ADOS'][0]
            df_RESULT_list[['u_num','a_num','i_num']]=[u_num, i_num, a_num]
            df_formant_statistic=df_formant_statistic.append(df_RESULT_list)
            continue
        
        numerator=u[1] + a[1] + i[0] + u[0]
        demominator=i[1] + a[0]
        RESULT_dict['FCR']=np.float(numerator/demominator)
        RESULT_dict['F2i_u']= u[1]/i[1]
        RESULT_dict['F1a_u']= u[0]/a[0]
        # assert FCR <=2
        
        RESULT_dict['VSA1']=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
        
        RESULT_dict['LnVSA']=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
        
        EDiu=np.sqrt((u[1]-i[1])**2+(u[0]-i[0])**2)
        EDia=np.sqrt((a[1]-i[1])**2+(a[0]-i[0])**2)
        EDau=np.sqrt((u[1]-a[1])**2+(u[0]-a[0])**2)
        S=(EDiu+EDia+EDau)/2
        RESULT_dict['VSA2']=np.sqrt(S*(S-EDiu)*(S-EDia)*(S-EDau))
        
        RESULT_dict['LnVSA']=np.sqrt(np.log(S)*(np.log(S)-np.log(EDiu))*(np.log(S)-np.log(EDia))*(np.log(S)-np.log(EDau)))
        
        ''' a u i distance '''
        RESULT_dict['dau1'] = np.abs(a[0] - u[0])
        RESULT_dict['dai1'] = np.abs(a[0] - i[0])
        RESULT_dict['diu1'] = np.abs(i[0] - u[0])
        RESULT_dict['daudai1'] = RESULT_dict['dau1'] + RESULT_dict['dai1']
        RESULT_dict['daudiu1'] = RESULT_dict['dau1'] + RESULT_dict['diu1']
        RESULT_dict['daidiu1'] = RESULT_dict['dai1'] + RESULT_dict['diu1']
        RESULT_dict['daidiudau1'] = RESULT_dict['dai1'] + RESULT_dict['diu1']+ RESULT_dict['dau1']
        
        RESULT_dict['dau2'] = np.abs(a[1] - u[1])
        RESULT_dict['dai2'] = np.abs(a[1] - i[1])
        RESULT_dict['diu2'] = np.abs(i[1] - u[1])
        RESULT_dict['daudai2'] = RESULT_dict['dau2'] + RESULT_dict['dai2']
        RESULT_dict['daudiu2'] = RESULT_dict['dau2'] + RESULT_dict['diu2']
        RESULT_dict['daidiu2'] = RESULT_dict['dai2'] + RESULT_dict['diu2']
        RESULT_dict['daidiudau2'] = RESULT_dict['dai2'] + RESULT_dict['diu2']+ RESULT_dict['dau2']
        
        # =============================================================================
        ''' F-value, Valid Formant measure '''
        
        # =============================================================================
        # Get data
        F12_raw_dict=Vowels_AUI[people]
        u=F12_raw_dict['u']
        a=F12_raw_dict['a']
        i=F12_raw_dict['i']
        df_vowel = pd.DataFrame(np.vstack([u,a,i]),columns=Inspect_features)
        df_vowel['vowel'] = np.hstack([np.repeat('u',len(u)),np.repeat('a',len(a)),np.repeat('i',len(i))])
        df_vowel['target']=pd.Categorical(df_vowel['vowel'])
        df_vowel['target']=df_vowel['target'].cat.codes
        # F-test
        print("utt number of group u = {0}, utt number of group i = {1}, utt number of group A = {2}".format(\
            len(u),len(a),len(i)))
        F_vals=f_classif(df_vowel[Inspect_features].values,df_vowel['target'].values)[0]
        RESULT_dict['F_vals_f1']=F_vals[0]
        RESULT_dict['F_vals_f2']=F_vals[1]
        RESULT_dict['F_val_mix']=RESULT_dict['F_vals_f1'] + RESULT_dict['F_vals_f2']
        
        msb=f_classif(df_vowel[Inspect_features].values,df_vowel['target'].values)[2]
        msw=f_classif(df_vowel[Inspect_features].values,df_vowel['target'].values)[3]
        ssbn=f_classif(df_vowel[Inspect_features].values,df_vowel['target'].values)[4]
        
        
        
        RESULT_dict['MSB_f1']=msb[0]
        RESULT_dict['MSB_f2']=msb[1]
        MSB_f1 , MSB_f2 = RESULT_dict['MSB_f1'], RESULT_dict['MSB_f2']
        RESULT_dict['MSB_mix']=MSB_f1 + MSB_f2
        RESULT_dict['MSW_f1']=msw[0]
        RESULT_dict['MSW_f2']=msw[1]
        MSW_f1 , MSW_f2 = RESULT_dict['MSW_f1'], RESULT_dict['MSW_f2']
        RESULT_dict['MSW_mix']=MSW_f1 + MSW_f2
        RESULT_dict['SSBN_f1']=ssbn[0]
        RESULT_dict['SSBN_f2']=ssbn[1]
        
        # =============================================================================
        # criterion
        # F1u < F1a
        # F2u < F2a
        # F2u < F2i
        # F1i < F1a
        # F2a < F2i
        # =============================================================================
        u_mean=F12_val_dict['u']
        a_mean=F12_val_dict['a']
        i_mean=F12_val_dict['i']
        
        F1u, F2u=u_mean[0], u_mean[1]
        F1a, F2a=a_mean[0], a_mean[1]
        F1i, F2i=i_mean[0], i_mean[1]
        
        filt1 = [1 if F1u < F1a else 0]
        filt2 = [1 if F2u < F2a else 0]
        filt3 = [1 if F2u < F2i else 0]
        filt4 = [1 if F1i < F1a else 0]
        filt5 = [1 if F2a < F2i else 0]
        RESULT_dict['criterion_score']=np.sum([filt1,filt2,filt3,filt4,filt5])
    
        df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict)
        df_RESULT_list.index=[people]
        df_formant_statistic=df_formant_statistic.append(df_RESULT_list)
    return df_formant_statistic


df_formant_statistic_cmp=calculate_features(Vowels_AUI_top['cmp'],Label)
df_formant_statistic_ori=calculate_features(Vowels_AUI_top['ori'],Label)
aaa=ccc



def Compare_feature_by_quantile_diff(df_formant_statistic,df_formant_statistic_cmp,df_formant_statistic_ori,feat='F_vals_f1',N=5,Plot=True):
    filter_bool=np.logical_and(df_formant_statistic['u_num']>N,df_formant_statistic['a_num']>N)
    filter_bool=np.logical_and(filter_bool,df_formant_statistic['i_num']>N)
    filter_bool=np.logical_and(filter_bool,df_formant_statistic['ADOS'].isna()!=True)
    df_formant_statistic = df_formant_statistic[filter_bool]
    df_feature=df_formant_statistic[[feat]]
    
    
    df_rank_diff=pd.DataFrame()
    for people in df_formant_statistic_cmp.index:
        x_cmp=df_formant_statistic_cmp.loc[people,feat]
        rank_cmp=stats.percentileofscore(df_feature,x_cmp)
        
        x_ori=df_formant_statistic_ori.loc[people,feat]
        rank_ori=stats.percentileofscore(df_feature,x_ori)
        
        rank_diff=rank_cmp - rank_ori
        title='{0}_{1}'.format(people, feat)
        df_rank_diff.loc[people,'rank_diff']=rank_diff
        if Plot:
            plot_outpath='Plot_diff_distrib'
            if not os.path.exists(plot_outpath):
                os.makedirs(plot_outpath)
            plot = sns.distplot(df_feature[[feat]], hist=True, kde=True, 
                      bins=int(180/5), color = 'darkblue', 
                      hist_kws={'edgecolor':'black'},
                      kde_kws={'linewidth': 4})
            plt.plot([x_cmp, x_cmp], [0, plot.get_ylim()[1]])
            plt.plot([x_ori, x_ori], [0, plot.get_ylim()[1]])
            plt.title( title )
            fig = plot.get_figure()
            fig.savefig(plot_outpath+'/'+"{0}_{1}.png".format(people,feat))
            fig.clear()
    return df_rank_diff.abs().mean(), df_rank_diff
def process_audio(files,silence,trnpath,functional_method_window=3):
    Formants_people_symb=Dict()
    Formants_utt_symb=Dict()
    error_msg_bag=[]
    print("Process {} executing".format(files))
    for file in files:
        if '2ndPass' in file:
            namecode='2nd_pass'
        else:
            namecode='1st_pass'
            
        filename=os.path.basename(file).split(".")[0]
        spkr_name='_'.join(filename.split("_")[:-3])
        utt='_'.join(filename.split("_")[:])
        if 'Session' in filepath:
            audiofile=filepath+"/{name}.wav".format(name='_'.join(filename.split("_")[:-1]))
        elif 'Segment' in filepath:
            audiofile=filepath+"/{name}.wav".format(name=filename)
        else:
            raise OSError(os.strerror, 'not allowed filepath')
        trn=trnpath+"/{name}.txt".format(name=filename)
        df_segInfo=pd.read_csv(trn, header=None,delimiter='\t')
        
        audio = AudioSegment.from_wav(audiofile)
        
        gender_query_str='_'.join(filename.split("_")[:Namecode_dict[namecode]['role']])
        role=filename.split("_")[Namecode_dict[namecode]['role']]
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
    
    if len(error_msg_bag) !=0:
        import warnings
        warnings.warn("Warning..files in ..{0}...is not sucessfully computed".format(error_msg_bag))
    
    return Formants_utt_symb, Formants_people_symb
# =============================================================================
'''

    Manual area
    You can use it to debug

'''


# pickle.dump(Formants_utt_symb,open(outpath+"/Formants_utt_symb_cmp.pkl","wb"))
# pickle.dump(Formants_people_symb,open(outpath+"/Formants_people_symb_cmp.pkl","wb"))

# =============================================================================

# =============================================================================
'''

    Check the distribution of everyone

'''
# Person_IQR_dict=Dict()
# Person_IQR_all_dict=Dict()
# PhoneOI=['i:','u:','A:']
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

# import scipy
# for symb in PhoneOI:
#     # g= sns.jointplot(data=Person_IQR_all_dict[symb], x='F1',y='F2',hue='sex')
#     # title='{0}_{1}'.format('distrivution all people', symb)
#     # plt.title( title )
    
#     male_data=Person_IQR_all_dict[symb][Person_IQR_all_dict[symb]['sex']=='male']
#     female_data=Person_IQR_all_dict[symb][Person_IQR_all_dict[symb]['sex']=='female']
    
#     for feat in args.Inspect_features:
#         print('testing for feature ', symb,feat)
#         # stat, p = scipy.stats.shapiro(male_data[args.Inspect_features])
#         stat, p = scipy.stats.normaltest(male_data[feat])
#         alpha = 0.05
#         if p > alpha:
#         	print('Sample looks Gaussian (fail to reject H0)', 'male data')
#         else:
#         	print('Sample does not look Gaussian (reject H0)', 'male data')
        
#         # stat, p = scipy.stats.shapiro(female_data[feat])  
#         stat, p = scipy.stats.normaltest(female_data[feat])
#         alpha = 0.05
#         if p > alpha:
#         	print('Sample looks Gaussian (fail to reject H0)', 'female data')
#         else:
#         	print('Sample does not look Gaussian (reject H0)', 'female data')
            
        ### most of the data are not normal
        #             testing for feature  i: F1
        # Sample does not look Gaussian (reject H0) male data
        # Sample does not look Gaussian (reject H0) female data
        # testing for feature  i: F2
        # Sample does not look Gaussian (reject H0) male data
        # Sample looks Gaussian (fail to reject H0) female data
        # testing for feature  u: F1
        # Sample does not look Gaussian (reject H0) male data
        # Sample does not look Gaussian (reject H0) female data
        # testing for feature  u: F2
        # Sample does not look Gaussian (reject H0) male data
        # Sample does not look Gaussian (reject H0) female data
        # testing for feature  A: F1
        # Sample does not look Gaussian (reject H0) male data
        # Sample does not look Gaussian (reject H0) female data
        # testing for feature  A: F2
        # Sample does not look Gaussian (reject H0) male data
        # Sample does not look Gaussian (reject H0) female data

# sex=''
# feat='F1'
# for feat in args.Inspect_features:
#     for symb in PhoneOI:
#         plt.figure()
#         df_data_top=pd.DataFrame([])
#         for person in Person_IQR_dict.keys():
#             if symb in Person_IQR_dict[person].keys():
#                 df_data=Person_IQR_dict[person][symb]
#                 df_data['people']=person
#                 if len(sex)>0:
#                     df_data=df_data[df_data['sex']==sex]
#                 df_data_top=df_data_top.append(df_data)
#                 # bx = sns.boxplot(x="people", y=feat, data=df_data)
#         # cx = sns.boxplot(x="people", y=feat, data=df_data_top.iloc[:100])
#         ax = sns.boxplot(x="people", y=feat, data=df_data_top)
#         title='{0}_{1}'.format('distribution single people boxplot ' + symb,'feature: ' +feat)
#         plt.title( title )

# =============================================================================

# =============================================================================
'''

    Inspect area

''' 
# =============================================================================
# N=5
# feat='MSB_f2'
# df_formant_statistic=df_formant_statistic_77
# ######
# filter_bool=np.logical_and(df_formant_statistic['u_num']>N,df_formant_statistic['a_num']>N)
# filter_bool=np.logical_and(filter_bool,df_formant_statistic['i_num']>N)
# filter_bool=np.logical_and(filter_bool,df_formant_statistic['ADOS'].isna()!=True)
# df_formant_statistic = df_formant_statistic[filter_bool]
# df_feature=df_formant_statistic[[feat]]




# df_rank_diff=pd.DataFrame()
# for people in df_formant_statistic_cmp.index:
#     x_cmp=df_formant_statistic_cmp.loc[people,feat]
#     rank_cmp=stats.percentileofscore(df_feature,x_cmp)
    
#     x_ori=df_formant_statistic_ori.loc[people,feat]
#     rank_ori=stats.percentileofscore(df_feature,x_ori)
    
#     rank_diff=rank_cmp - rank_ori
    
#     df_rank_diff.loc[people,'rank_diff']=rank_diff
    
    
#     plot_outpath='Plot_diff_distrib'
#     if not os.path.exists(plot_outpath):
#         os.makedirs(plot_outpath)
#     plot = sns.distplot(df_feature[feat], hist=True, kde=True, 
#               bins=int(180/5), color = 'darkblue', 
#               hist_kws={'edgecolor':'black'},
#               kde_kws={'linewidth': 4})
#     plt.plot([x_cmp, x_cmp], [0, plot.get_ylim()[1]])
#     plt.plot([x_ori, x_ori], [0, plot.get_ylim()[1]])
#     fig = plot.get_figure()
#     fig.savefig(plot_outpath+'/'+"{0}_{1}.png".format(people,feat))
#     fig.clear()
    