#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 12:50:44 2020

@author: jackchen

This script do the following work:
    1. Group the phone sequence by lexicon
        * update 2021/07/02: We do not group by lexicon by now
    2. Gather formant information from Formants_utt_symb by the order [utt][grp] (<- to reshape the data structure)
    
    3. single phone to context dependant phone
    4. Generate (triphones) context dependant phones (multi thread approach)
    
    5. Get Bi-phones from triphones

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
import pathlib
import re


import articulation.Multiprocess as Multiprocess

import praat.praat_functions as praat_functions
from script_mananger import script_manager
from articulation.HYPERPARAM import phonewoprosody, Label
from CtxDepPhone_merger import Filter_CtxPhone2Biphone, CtxDepPhone_merger


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

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1

def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--base_path_phf', default='/homes/ssd1/jackchen/gop_prediction',
                        help='path of the base directory')
    parser.add_argument('--outpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--reFilter', default=False,
                            help='')
    parser.add_argument('--check', default=True,
                        help='path of the base directory')
    parser.add_argument('--manual', default=False,
                        help='path of the base directory')
    parser.add_argument('--use_ckpt', default=True,
                        help='')
    parser.add_argument('--Inspect_features', default=['F1','F2'],
                            help='')
    args = parser.parse_args()
    return args

args = get_args()
base_path=args.base_path
outpath=args.outpath
feat_column=args.Inspect_features
manual=args.manual

path_app = base_path
# sys.path.append(path_app)
# sys.path.append(path_app+'/articulation')


# ============================================================================
'''

    single phone to context dependant phone
    Utt_ctxdepP_dict[utt] -> df: index = contetxt-dependant phone([s]-j+aU4), columns = [F1, F2] of any features
'''
def FindCentralPhone(symb):
    regexp = re.compile(r'.*[-+].*')
    if regexp.search(symb):
        if '-' in symb:
            critical_P=symb[symb.find('-')+1:]
        elif '+' in symb:
            critical_P=symb[:symb.find('+')]
        else:
            raise ValueError
    else:
        critical_P=symb
    return critical_P

def GetCtxPhoneFromUtt_map(keys, Formants_utt_symb):
    print(" process PID", os.getpid(), " running")
    Utt_ctxdepP_dict=Dict()
    Set_lst=[]
    for utt in keys:
        df_Utt_phfContextDep=pd.DataFrame([],columns=feat_column)
        
        values = Formants_utt_symb[utt]
        phoneSeq=list(values.index.astype(str))    
        if len(phoneSeq) == 1:
            df_ctxdepP=values.iloc[[0],:]
            df_ctxdepP.index=['{0}-{1}+{2}'.format('[s]',phoneSeq[0],'[\s]')]
            df_Utt_phfContextDep=df_Utt_phfContextDep.append(df_ctxdepP)
        else:    
            for i in range(len(phoneSeq)): # df_Utt_phfContextDep append each word
                
                df_ctxdepP=values.iloc[[i],:]
                critical_P=FindCentralPhone(phoneSeq[i])
                if i==0:
                    right_critical_P=FindCentralPhone(phoneSeq[i+1])
                    df_ctxdepP.index=['{0}-{1}+{2}'.format('[s]',critical_P,right_critical_P)]
                    # df_ctxdepP=pd.DataFrame(,index=['{0}-{1}+{2}'.format('[s]',phoneSeq[i],phoneSeq[i+1])],columns=values.columns)
                elif i==len(phoneSeq)-1:
                    left_critical_P=FindCentralPhone(phoneSeq[i-1])
                    df_ctxdepP.index=['{0}-{1}+{2}'.format(left_critical_P,critical_P,'[\s]')]
                else:
                    left_critical_P=FindCentralPhone(phoneSeq[i-1])
                    right_critical_P=FindCentralPhone(phoneSeq[i+1])
                    df_ctxdepP.index=['{0}-{1}+{2}'.format(left_critical_P,critical_P,right_critical_P)]
                df_Utt_phfContextDep=df_Utt_phfContextDep.append(df_ctxdepP)
        assert len(df_Utt_phfContextDep) == len(phoneSeq) #check if the appended df: 'df_Utt_phfContextDep' ,atches phone sequence
    
        Set_lst.append(list(df_Utt_phfContextDep.index))
        Utt_ctxdepP_dict[utt]=df_Utt_phfContextDep  
    
    print("PID {} Getting ".format(os.getpid()), "Done")
    return Utt_ctxdepP_dict, Set_lst

# =============================================================================
Formants_utt_symb=pickle.load(open(outpath+"/Formants_utt_symb_bymiddle_window3_ASDkid.pkl","rb"))
# =============================================================================
'''
    Stage 0: preprocessing data
    Filter out data using by 1.5*IQR

'''
PhoneMapp_dict=phonewoprosody.PhoneMapp_dict
PhoneOfInterest=list(PhoneMapp_dict.keys())


''' Vowel AUI rule is using phonewoprosody '''
from utils_jack  import  Formant_utt2people_reshape, Gather_info_certainphones,  GetValuelimit_IQR
from datetime import datetime as dt

Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
AUI_info=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)
limit_people_rule=GetValuelimit_IQR(AUI_info,PhoneMapp_dict,args.Inspect_features)

''' multi processing start '''
date_now='{0}-{1}-{2} {3}'.format(dt.now().year,dt.now().month,dt.now().day,dt.now().hour)
ckpt_outpath='Features/Formants_utt_symb'
if not os.path.exists(ckpt_outpath) and args.reFilter==False:
    os.makedirs(ckpt_outpath)
filepath=ckpt_outpath+"/[Analyzing]Formants_utt_symb_limited.pkl"
# # If file last modify time is not now (precisions to the hours) than we create new one
if os.path.exists(filepath):
    fname = pathlib.Path(filepath)
    mtime = dt.fromtimestamp(fname.stat().st_mtime)
    filemtime='{0}-{1}-{2} {3}'.format(mtime.year,mtime.month,mtime.day,mtime.hour)
    if filemtime != date_now:
        Process_IQRFiltering_Multi(Formants_utt_symb,limit_people_rule,outpath=ckpt_outpath) # the results will be output as pkl file at outpath+"/[Analyzing]Formants_utt_symb_limited.pkl"
else:
    Process_IQRFiltering_Multi(Formants_utt_symb,limit_people_rule,outpath=ckpt_outpath) # the results will be output as pkl file at outpath+"/[Analyzing]Formants_utt_symb_limited.pkl"


Formants_utt_symb_limited=pickle.load(open(filepath,"rb"))
''' multi processing end '''
if len(limit_people_rule) >0:
    Formants_utt_symb=Formants_utt_symb_limited

pickle.dump(Formants_utt_symb_limited,open(ckpt_outpath+"/[Analyzing]Formants_utt_symb_limited.pkl","wb"))



# =============================================================================
'''
    Stage 1: Sliding window over all phones and save them to dict (and also all possible ctx phones)
    
'''
# =============================================================================
''' multiprocessing start  '''
keys=[]
interval=20
for i in range(0,len(Formants_utt_symb),interval):
    # print(list(Utt_ctxdepP_dict.keys())[i:i+interval])
    keys.append(list(Formants_utt_symb.keys())[i:i+interval])
flat_keys=[item for sublist in keys for item in sublist]
assert len(flat_keys) == len(Formants_utt_symb)

pool = Pool(os.cpu_count())
# pool = Pool(2)
final_result = pool.starmap(GetCtxPhoneFromUtt_map, [(key,Formants_utt_symb) for key in keys])

Utt_ctxdepP_dict=Dict()
df_template=pd.DataFrame([],columns=feat_column)
Set_lst=[]
for d, s_l in tqdm(final_result):
    for utt in d.keys():
        if utt not in Utt_ctxdepP_dict.keys():
            Utt_ctxdepP_dict[utt]=df_template
        Utt_ctxdepP_dict[utt]=Utt_ctxdepP_dict[utt].append(d[utt])
    Set_lst.extend([sl for sl in s_l])

if args.check:
    Triphoneset = set([item for sublist in Set_lst for item in sublist])
    for key in Utt_ctxdepP_dict.keys(): # check if all triphones in Triphoneset
        values=Utt_ctxdepP_dict[key]
        for phone in values.index:
            assert phone in Triphoneset

''' multiprocessing end '''
pickle.dump(Utt_ctxdepP_dict,open(outpath+"/Utt_ctxdepP_dict.pkl","wb"))
pickle.dump(Set_lst,open(outpath+"/CxtDepPhone_Setlst.pkl","wb"))

Formant_people_information=Formant_utt2people_reshape(Utt_ctxdepP_dict,Utt_ctxdepP_dict,Align_OrinCmp=False)
AUI=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)

if args.check:  #Check if certainphone like 'w' in the CtxPhones    
    for utt in Utt_ctxdepP_dict.keys():
        for CtxPhone in Utt_ctxdepP_dict[utt].index:
            left_P=CtxPhone[:CtxPhone.find('-')]
            right_P=CtxPhone[CtxPhone.find('+')+1:]
            critical_P=CtxPhone[CtxPhone.find('-')+1:CtxPhone.find('+')]

# =============================================================================
'''
    Stage 2: Turning Utt_ctxdepP_dict[utt] -> df.loc[CtxPhone] = feat   into   PeopleCtxDepPhoneFunctional_dict[spk][phone] -> df.loc[CtxPhone]= feat
    Accumulate the appearance of each context dependant phones (wanna control the phone environment)
    
'''

def GetEachCtxDepPhoneFromUtt_map_oldone(keys,Triphoneset,Utt_ctxdepP_dict):
    print(" process PID", os.getpid(), " running")
    PeopleCtxDepPhoneFunctional_dict=Dict()
    for key in keys:
        values=Utt_ctxdepP_dict[key]
        df_template=pd.DataFrame([],columns=feat_column)
        spk=key[:re.search("_(K|D)_",key).start()]#+"_K"
        # for phone in Triphoneset:
        #     if phone not in PeopleCtxDepPhoneFunctional_dict[spk].keys():
        #         PeopleCtxDepPhoneFunctional_dict[spk][phone]=df_template
        #     PeopleCtxDepPhoneFunctional_dict[spk][phone]=PeopleCtxDepPhoneFunctional_dict[spk][phone].append(values[values.index==phone])
        for phone in Triphoneset:
            if len(values[values.index==phone])>0:
                if phone not in PeopleCtxDepPhoneFunctional_dict[spk].keys():
                    PeopleCtxDepPhoneFunctional_dict[spk][phone]=df_template
                PeopleCtxDepPhoneFunctional_dict[spk][phone]=PeopleCtxDepPhoneFunctional_dict[spk][phone].append(values[values.index==phone])

    print("PID {} Getting ".format(os.getpid()), list(Utt_ctxdepP_dict.keys()).index(key), "Done")
    # print(" process PID", os.getpid(), " done")
    return PeopleCtxDepPhoneFunctional_dict
def GetEachCtxDepPhoneFromUtt_map(keys,Utt_ctxdepP_dict):
    print(" process PID", os.getpid(), " running")
    Triphoneset=set([ctxP for k in keys for ctxP in list(Utt_ctxdepP_dict[k].index)])
    PeopleCtxDepPhoneFunctional_dict=Dict()
    for phone in Triphoneset:
        for key in keys:
            values=Utt_ctxdepP_dict[key]
            if len(values[values.index==phone])>0:
                df_template=pd.DataFrame([],columns=feat_column)
                spk=key[:re.search("_(K|D)_",key).start()]#+"_K"
                if phone not in PeopleCtxDepPhoneFunctional_dict[spk].keys():
                    PeopleCtxDepPhoneFunctional_dict[spk][phone]=df_template
                PeopleCtxDepPhoneFunctional_dict[spk][phone]=PeopleCtxDepPhoneFunctional_dict[spk][phone].append(values[values.index==phone])
    # print("PID {} Getting ".format(os.getpid()), list(Utt_ctxdepP_dict.keys()).index(key), "Done")
    # print(" process PID", os.getpid(), " done")
    return PeopleCtxDepPhoneFunctional_dict

# THis part is really slow even if we use multi-processing technique, so use checkpoint if available
    
if os.path.exists(outpath+"/PeopleCtxDepPhoneFunctional_dict.pkl") and args.use_ckpt:
    PeopleCtxDepPhoneFunctional_dict=pickle.load(open(outpath+"/PeopleCtxDepPhoneFunctional_dict.pkl","rb"))
    Utt_ctxdepP_dict=pickle.load(open(outpath+"/Utt_ctxdepP_dict.pkl","rb"))
    Set_lst=pickle.load(open(outpath+"/CxtDepPhone_Setlst.pkl","rb"))
    Triphoneset = set([item for sublist in Set_lst for item in sublist])
else:    
    Utt_ctxdepP_dict=pickle.load(open(outpath+"/Utt_ctxdepP_dict.pkl","rb"))
    Set_lst=pickle.load(open(outpath+"/CxtDepPhone_Setlst.pkl","rb"))
    Triphoneset = set([item for sublist in Set_lst for item in sublist])
    df_template=pd.DataFrame([],columns=feat_column)

    keys=[]
    interval=5
    for i in range(0,len(Utt_ctxdepP_dict),interval):
        # print(list(Utt_ctxdepP_dict.keys())[i:i+interval])
        keys.append(list(Utt_ctxdepP_dict.keys())[i:i+interval])
    flat_keys=[item for sublist in keys for item in sublist]
    assert len(flat_keys) == len(Utt_ctxdepP_dict)
    
    
    pool = Pool(os.cpu_count())
    # pool = Pool(2)
    
    final_result = pool.starmap(GetEachCtxDepPhoneFromUtt_map, [(key,Utt_ctxdepP_dict) for key in tqdm(keys)])
    print('GetEachCtxDepPhoneFromUtt_map done')
    PeopleCtxDepPhoneFunctional_dict=Dict()
    for d in tqdm(final_result):
        for spk in d.keys():
            for phone,values in d[spk].items():
                if phone not in PeopleCtxDepPhoneFunctional_dict[spk].keys():
                    PeopleCtxDepPhoneFunctional_dict[spk][phone]=df_template
                PeopleCtxDepPhoneFunctional_dict[spk][phone]=PeopleCtxDepPhoneFunctional_dict[spk][phone].append(values)
    
    
    # ''' The total number of context dendant phones in "PeopleCtxDepPhoneFunctional_dict" should match "Utt_ctxdepP_dict" '''
    if args.check:
        PeopleCtxDepPhoneFunctional_dict_num_dict=Dict()
        PeopleCtxDepPhoneFunctional_dict_num=0
        for spk, v in PeopleCtxDepPhoneFunctional_dict.items():
            if spk not in PeopleCtxDepPhoneFunctional_dict_num_dict.keys():
                PeopleCtxDepPhoneFunctional_dict_num_dict[spk]=0
            for phone, values in PeopleCtxDepPhoneFunctional_dict[spk].items():
                PeopleCtxDepPhoneFunctional_dict_num+=len(values)
                PeopleCtxDepPhoneFunctional_dict_num_dict[spk]+=len(values)
                
                    
        Utt_ctxdepP_dict_num_dict=Dict()
        Utt_ctxdepP_dict_num=0
        for utt, v in Utt_ctxdepP_dict.items():
            spk=utt[:re.search("_(K|D)_",utt).start()]#+"_K"
            Utt_ctxdepP_dict_num+=len(v)
            if spk not in Utt_ctxdepP_dict_num_dict.keys():
                Utt_ctxdepP_dict_num_dict[spk]=0
            Utt_ctxdepP_dict_num_dict[spk]+=len(v)
        
        assert Utt_ctxdepP_dict_num == PeopleCtxDepPhoneFunctional_dict_num
    pickle.dump(PeopleCtxDepPhoneFunctional_dict,open(outpath+"/PeopleCtxDepPhoneFunctional_dict.pkl","wb"))





# =============================================================================
'''
    Stage 3: Generate context dependant phones by rules. Here is some rules for example
    
    1. bi-phone
    2. limit {a or u or i} only, 
    
    Previous steps get the tri-phone from word level utterance, 
    We now extract bi-gram phones from the above "PeopleCtxDepPhoneFunctional_dict"

'''
# df_template=pd.DataFrame([],columns=feat_column)
# # =============================================================================

    
filt_CtxPhone=Filter_CtxPhone2Biphone(PeopleCtxDepPhoneFunctional_dict,Triphoneset)
import time
start = time.time()
PeopleLeftDepPhoneFunctional_dict=filt_CtxPhone.Process_multi(filt_CtxPhone.Get_LBiphone_map)
PeopleRightDepPhoneFunctional_dict=filt_CtxPhone.Process_multi(filt_CtxPhone.Get_RBiphone_map)


# Just a little unit test
filt_CtxPhone.check_totnum(Utt_ctxdepP_dict,PeopleLeftDepPhoneFunctional_dict)
filt_CtxPhone.check_totnum(Utt_ctxdepP_dict,PeopleRightDepPhoneFunctional_dict)
# PeopleLeftDepPhoneFunctional_dict, PeopleRightDepPhoneFunctional_dict = filt_CtxPhone.Get_Biphone()
end = time.time()
print(end - start)


pickle.dump(PeopleLeftDepPhoneFunctional_dict,open(outpath+"/PeopleLeftDepPhoneFunctional_dict.pkl","wb"))
pickle.dump(PeopleRightDepPhoneFunctional_dict,open(outpath+"/PeopleRightDepPhoneFunctional_dict.pkl","wb"))


            


# =============================================================================
'''
    Stage 4: 
    
    Merge context depenant phones by self creating rules
    
    Note1: there're sets of ordered procedure you might/(or might not) want to follow'
    Take a look in the top of CtxDepPhone_merger script for details
    
    Note2: you may want to check the final outcome (which phones are selected, and how much numbers 
    of people/samples are left )
'''

# =============================================================================

Phoneset_sets={'Manner_simp1':phonewoprosody.Manner_sets_simple1, \
                'Manner_simp2':phonewoprosody.Manner_sets_simple2, \
                'Place_simp1':phonewoprosody.Place_sets_simple1,  \
                'Place_simp2':phonewoprosody.Place_sets_simple2}

PeopleCtxDepPhoneFunctional_dict=pickle.load(open(outpath+"/PeopleCtxDepPhoneFunctional_dict.pkl","rb"))
PeopleLeftDepPhoneFunctional_dict=pickle.load(open(outpath+"/PeopleLeftDepPhoneFunctional_dict.pkl","rb"))
PeopleRightDepPhoneFunctional_dict=pickle.load(open(outpath+"/PeopleRightDepPhoneFunctional_dict.pkl","rb"))


# if args.check:  #Check if certainphone like 'w' in the CtxPhones    
#     for people in PeopleCtxDepPhoneFunctional_dict.keys():
#         for CtxPhone in PeopleCtxDepPhoneFunctional_dict[people].keys():
#             left_P=CtxPhone[:CtxPhone.find('-')]
#             right_P=CtxPhone[CtxPhone.find('+')+1:]
#             critical_P=CtxPhone[CtxPhone.find('-')+1:CtxPhone.find('+')]
#             if critical_P == 'w':
#                 aaa=ccc
#     for people in PeopleLeftDepPhoneFunctional_dict.keys():
#         for CtxPhone in PeopleLeftDepPhoneFunctional_dict[people].keys():
#             left_P=CtxPhone[:CtxPhone.find('-')]
#             critical_P=CtxPhone[CtxPhone.find('-')+1:]
#             if critical_P == 'w':
#                 aaa=ccc
#     for people in PeopleRightDepPhoneFunctional_dict.keys():
#         for CtxPhone in PeopleRightDepPhoneFunctional_dict[people].keys():
#             critical_P=CtxPhone[:CtxPhone.find('+')]
#             right_P=CtxPhone[CtxPhone.find('+')+1:]
#             if critical_P == 'w':
#                 aaa=ccc

PhonesOfInterest=phonewoprosody.PhoneMapp_dict.keys()
ctxdepphone_mger=CtxDepPhone_merger(phonewoprosody)   
AUI_ContextDepPhones=Dict()
AUI_ContextDepPhones['CtxDepVowel_AUI']=ctxdepphone_mger.get_Dep_AUI(PeopleCtxDepPhoneFunctional_dict,\
                                                                     PhoneMapp_dict,PhonesOfInterest,mode='Ctx')
AUI_ContextDepPhones['LeftDepVowel_AUI']=ctxdepphone_mger.get_Dep_AUI(PeopleLeftDepPhoneFunctional_dict,\
                                                                      PhoneMapp_dict,PhonesOfInterest,mode='Left')
AUI_ContextDepPhones['RightDepVowel_AUI']=ctxdepphone_mger.get_Dep_AUI(PeopleRightDepPhoneFunctional_dict,\
                                                                       PhoneMapp_dict,PhonesOfInterest,mode='Right')


# if args.check: 
#     for CtxPhone in AUI_ContextDepPhones['CtxDepVowel_AUI'].keys():
#         for people in AUI_ContextDepPhones['CtxDepVowel_AUI'][CtxPhone].keys():
#             left_P=CtxPhone[:CtxPhone.find('-')]
#             right_P=CtxPhone[CtxPhone.find('+')+1:]
#             critical_P=CtxPhone[CtxPhone.find('-')+1:CtxPhone.find('+')]
#             if critical_P == 'w':
#                 aaa=ccc
#     for CtxPhone in AUI_ContextDepPhones['LeftDepVowel_AUI'].keys():
#         for people in AUI_ContextDepPhones['LeftDepVowel_AUI'][CtxPhone].keys():
#             left_P=CtxPhone[:CtxPhone.find('-')]
#             critical_P=CtxPhone[CtxPhone.find('-')+1:]
#             if critical_P == 'w':
#                 aaa=cc
#     for CtxPhone in AUI_ContextDepPhones['RightDepVowel_AUI'].keys():
#         for people in AUI_ContextDepPhones['RightDepVowel_AUI'][CtxPhone].keys():
#             critical_P=CtxPhone[:CtxPhone.find('+')]
#             right_P=CtxPhone[CtxPhone.find('+')+1:]
#             if critical_P == 'w':
#                 aaa=ccc

if args.check:
    Check_dict=Dict()

# Get phonetic environment defined by Manner and Place
for keys,Phoneset_Here in tqdm(Phoneset_sets.items()):
    AUI_ContextDepPhonesMerge_MannerPlace_uwij=Dict()
    for feat in AUI_ContextDepPhones.keys():
        
        Feature_dict=AUI_ContextDepPhones[feat]
        DepPhonesMerge_MannerPlace=ctxdepphone_mger.get_Dep_MannernPlace_AUI(Feature_dict,Phoneset_Here,feat)
        DepPhonesMerge_MannerPlace_wopros=ctxdepphone_mger.get_Dep_wopros_AUI(DepPhonesMerge_MannerPlace,feat=feat)
        DepPhonesMerge_MannerPlacewoprosuwij_n_AllMerged=Dict()
        DepPhonesMerge_MannerPlace_wopros_uwij=ctxdepphone_mger.get_Dep_uwij_AUI(DepPhonesMerge_MannerPlace_wopros,feat=feat)
        DepPhonesMerge_MannerPlace_wopros_AllMerged=ctxdepphone_mger.get_AllMerged_AUI(DepPhonesMerge_MannerPlace_wopros_uwij,feat=feat)
        if args.check:
            Check_dict[feat]=DepPhonesMerge_MannerPlace_wopros_AllMerged
        DepPhonesMerge_MannerPlacewoprosuwij_n_AllMerged.update(DepPhonesMerge_MannerPlace_wopros_AllMerged)
        DepPhonesMerge_MannerPlacewoprosuwij_n_AllMerged.update(DepPhonesMerge_MannerPlace_wopros_uwij)
        AUI_ContextDepPhonesMerge_MannerPlace_uwij[feat]=DepPhonesMerge_MannerPlacewoprosuwij_n_AllMerged
    pickle.dump(AUI_ContextDepPhonesMerge_MannerPlace_uwij,open(outpath+"/AUI_ContextDepPhonesMerge_{0}_uwij.pkl".format(keys),"wb"))

# if args.check:  #Check if certainphone like 'w' in the CtxPhones    
#     for feat in AUI_ContextDepPhonesMerge_MannerPlace_uwij.keys():
#         CtxDepVowel_AUI_dict=AUI_ContextDepPhonesMerge_MannerPlace_uwij[feat]
#         for CtxP in CtxDepVowel_AUI_dict.keys():
#             for people in CtxDepVowel_AUI_dict[CtxP].keys():
#                 for CtxPhone in CtxDepVowel_AUI_dict[CtxP][people].index:
#                     if feat == 'CtxDepVowel_AUI':
#                         left_P=CtxPhone[:CtxPhone.find('-')]
#                         right_P=CtxPhone[CtxPhone.find('+')+1:]
#                         critical_P=CtxPhone[CtxPhone.find('-')+1:CtxPhone.find('+')]
                        
#                         if critical_P == 'w':
#                             aaa=ccc



AUI_ContextDepPhonesMerge_uwij=Dict()
for feat in AUI_ContextDepPhones.keys():
    Feature_dict=AUI_ContextDepPhones[feat]
    DepPhonesMerge_wopros=ctxdepphone_mger.get_Dep_wopros_AUI(Feature_dict,feat=feat)
    DepPhonesMerge_woprosuwij_n_AllMerged=Dict()
    DepPhonesMerge_wopros_uwij=ctxdepphone_mger.get_Dep_uwij_AUI(DepPhonesMerge_wopros,feat=feat)
    DepPhonesMerge_wopros_AllMerged=ctxdepphone_mger.get_AllMerged_AUI(DepPhonesMerge_wopros_uwij,feat=feat)
    if args.check:
        Check_dict[feat]=DepPhonesMerge_wopros_AllMerged
    DepPhonesMerge_woprosuwij_n_AllMerged.update(DepPhonesMerge_wopros_AllMerged)
    DepPhonesMerge_woprosuwij_n_AllMerged.update(DepPhonesMerge_wopros_uwij)
    AUI_ContextDepPhonesMerge_uwij[feat]=DepPhonesMerge_woprosuwij_n_AllMerged
pickle.dump(AUI_ContextDepPhonesMerge_uwij,open(outpath+"/AUI_ContextDepPhonesMerge_uwij.pkl","wb"))

if args.check:  #Check if certainphone like 'w' in the CtxPhones    
    for feat in AUI_ContextDepPhonesMerge_uwij.keys():
        CtxDepVowel_AUI_dict=AUI_ContextDepPhonesMerge_uwij[feat]
        for CtxP in CtxDepVowel_AUI_dict.keys():
            for people in CtxDepVowel_AUI_dict[CtxP].keys():
                for CtxPhone in CtxDepVowel_AUI_dict[CtxP][people].index:
                    if feat == 'CtxDepVowel_AUI':
                        left_P=CtxPhone[:CtxPhone.find('-')]
                        right_P=CtxPhone[CtxPhone.find('+')+1:]
                        critical_P=CtxPhone[CtxPhone.find('-')+1:CtxPhone.find('+')]
                        
                        if critical_P == 'w':
                            aaa=ccc


# =============================================================================
'''  TODO: write script that inspects the phones like uA: '''
# =============================================================================
Feature_dict=AUI_ContextDepPhones['CtxDepVowel_AUI']
























''' This script should end here '''

aaa=ccc
# =============================================================================
# check area
# =============================================================================

def Reload_dict2_phonespk(Feature_dict,feat='CtxDepVowel_AUI'):
    Feature_merge_dict=Dict()
    for phone in tqdm(Feature_dict.keys()):
        for spk in Feature_dict[phone].keys():
            values= Feature_dict[phone][spk]
            feat_type=feat[:feat.find("Dep")]
            if feat_type == 'Ctx':
                left_P=phone[:phone.find('-')]
                right_P=phone[phone.find('+')+1:]
                critical_P=phone[phone.find('-')+1:phone.find('+')]
                
                Depphone_merged=critical_P
                
            elif feat_type == 'Left':
                left_P=phone[:phone.find('-')]
                critical_P=phone[phone.find('-')+1:]
            
                Depphone_merged=critical_P
                
            elif feat_type == 'Right':
                critical_P=phone[:phone.find('+')]
                right_P=phone[phone.find('+')+1:]
                
                Depphone_merged=critical_P
            
            if Depphone_merged not in Feature_merge_dict.keys():
                if spk not in Feature_merge_dict[Depphone_merged].keys():
                    Feature_merge_dict[Depphone_merged][spk]=values
                elif spk in Feature_merge_dict[Depphone_merged].keys():
                    Feature_merge_dict[Depphone_merged][spk]=Feature_merge_dict[Depphone_merged][spk].append(values)
            else:
                if spk not in Feature_merge_dict[Depphone_merged].keys():
                    Feature_merge_dict[Depphone_merged][spk]=values
                elif spk in Feature_merge_dict[Depphone_merged].keys(): 
                    Feature_merge_dict[Depphone_merged][spk]=Feature_merge_dict[Depphone_merged][spk].append(values)
    return Feature_merge_dict


if args.check:
    CtxDepVwl=Dict()
    for feat in Check_dict.keys():
        Feature_dict=Check_dict[feat]
        CtxDepVwl[feat]=Reload_dict2_phonespk(Feature_dict,Phoneset_Here,feat=feat)
    
    feat1='CtxDepVowel_AUI'
    feat2='LeftDepVowel_AUI'
    feat3='RightDepVowel_AUI'
    
    for keys in CtxDepVwl[feat1].keys():
        for people in CtxDepVwl[feat1][keys].keys():
            assert (CtxDepVwl[feat1][keys][people].sort_values('F1').values == CtxDepVwl[feat2][keys][people].sort_values('F1').values).all().all()
            assert (CtxDepVwl[feat1][keys][people].sort_values('F1').values == CtxDepVwl[feat3][keys][people].sort_values('F1').values).all().all()


# =============================================================================
''' DEBUGING , remove soon '''
# =============================================================================

def lookup_Psets(s,Phoneme_sets):
    for p,v in Phoneme_sets.items():
        if s in v:
            return p
    return s  


Feature_dict=PeopleCtxDepPhoneFunctional_dict
def Reload_dict(Feature_dict):
    Feature_merge_dict=Dict()
    for spk in tqdm(Feature_dict.keys()):
        for phone in Feature_dict[spk].keys():
            values= Feature_dict[spk][phone]
            left_P=phone[:phone.find('-')]
            right_P=phone[phone.find('+')+1:]
            critical_P=phone[phone.find('-')+1:phone.find('+')]
            
            if critical_P in [x for symb in PhoneOfInterest for x in  PhoneMapp_dict[symb] ]:
                
                ph_wopros_center=lookup_Psets(critical_P,phonewoprosody.Phoneme_sets)
                ph_wopros_left='All'
                ph_wopros_right='All'
                
                if ph_wopros_center == 'j':
                    ph_wopros_center='i_'
                
                ph_wopros_center=ph_wopros_center.replace("_",":")
                wopros_Depphone="{0}-{1}+{2}".format(ph_wopros_left,ph_wopros_center,ph_wopros_right)
                
                
                Depphone_merged=wopros_Depphone
                if spk not in Feature_merge_dict.keys():
                    if Depphone_merged not in Feature_merge_dict[spk].keys():
                        Feature_merge_dict[spk][Depphone_merged]=values
                    elif Depphone_merged in Feature_merge_dict[spk].keys():
                        Feature_merge_dict[spk][Depphone_merged]=Feature_merge_dict[spk][Depphone_merged].append(values)
                else:
                    if Depphone_merged not in Feature_merge_dict[spk].keys():
                        Feature_merge_dict[spk][Depphone_merged]=values
                    elif Depphone_merged in Feature_merge_dict[spk].keys(): 
                        Feature_merge_dict[spk][Depphone_merged]=Feature_merge_dict[spk][Depphone_merged].append(values)
    return Feature_merge_dict



#assert CtxDepVwl1 = CtxDepVwl2 =CtxDepVwl3
def Reload_dict2(Feature_dict,feat='CtxDepVowel_AUI'):
    Feature_merge_dict=Dict()
    for spk in tqdm(Feature_dict.keys()):
        for phone in Feature_dict[spk].keys():
            values= Feature_dict[spk][phone]
            # left_P=phone[:phone.find('-')]
            # right_P=phone[phone.find('+')+1:]
            critical_P=phone[phone.find('-')+1:phone.find('+')]
            
            Depphone_merged=critical_P
            if spk not in Feature_merge_dict.keys():
                if Depphone_merged not in Feature_merge_dict[spk].keys():
                    Feature_merge_dict[spk][Depphone_merged]=values
                elif Depphone_merged in Feature_merge_dict[spk].keys():
                    Feature_merge_dict[spk][Depphone_merged]=Feature_merge_dict[spk][Depphone_merged].append(values)
            else:
                if Depphone_merged not in Feature_merge_dict[spk].keys():
                    Feature_merge_dict[spk][Depphone_merged]=values
                elif Depphone_merged in Feature_merge_dict[spk].keys(): 
                    Feature_merge_dict[spk][Depphone_merged]=Feature_merge_dict[spk][Depphone_merged].append(values)
    return Feature_merge_dict

Formants_utt_symb_cmp=pickle.load(open('articulation/Pickles/[Analyzing]Formants_utt_symb_limited.pkl',"rb"))
Formants_utt_symb=pickle.load(open('Features/Formants_utt_symb/[Analyzing]Formants_utt_symb_limited.pkl',"rb"))
for utt in Formants_utt_symb.keys():
    assert (Formants_utt_symb[utt] == Formants_utt_symb_cmp[utt]).all().all()
    
''' Debug remove soon Compare  Feature_dict'''
from articulation.articulation import Articulation
from metric import Evaluation_method     

Feature_dict_to_cmp= Reload_dict(Feature_dict)
Feature_dict_to_cmp_single= Reload_dict2(Feature_dict_to_cmp)


Vowels_AUI=pickle.load(open("articulation/Pickles/Session_formants_people_vowel_feat/Vowels_AUI_{}.pkl".format('ASDkid'),"rb"))
for people in Feature_dict_to_cmp_single.keys():
    for phone in Feature_dict_to_cmp_single[people].keys():
        df_Feature_dict=Feature_dict_to_cmp_single[people][phone].sort_values(['F1','F2'])
        df_Vowels_AUI=Vowels_AUI[people][phone].sort_values(['F1','F2'])
        assert (df_Feature_dict[args.Inspect_features].values == df_Vowels_AUI[args.Inspect_features].values).all().all()
        print((df_Feature_dict[args.Inspect_features].values == df_Vowels_AUI[args.Inspect_features].values))
        

''' Debug remove soon  Compare df_formant_statistic '''
articulation=Articulation()
for people in Feature_dict_to_cmp_single.keys():
    for p in PhoneOfInterest:
        if p not in Feature_dict_to_cmp_single[people].keys():
            Feature_dict_to_cmp_single[people][p]=pd.DataFrame([])

df_formant_statistic=articulation.calculate_features(Feature_dict_to_cmp_single,Label,PhoneOfInterest=PhoneOfInterest)

Eval_med=Evaluation_method()
df_formant_statistic=Eval_med._Postprocess_dfformantstatistic(df_formant_statistic)
df_formant_statistic_to_cmp=pickle.load(open("articulation/Pickles/Session_formants_people_vowel_feat/Formant_AUI_tVSAFCRFvals_ASDkid.pkl","rb"))

df_formant_statistic_to_cmp_sorted=df_formant_statistic_to_cmp.sort_index()
df_formant_statistic=df_formant_statistic.sort_index()
Aaa=(df_formant_statistic_to_cmp_sorted - df_formant_statistic)
print(len(np.where(Aaa>0.000001)))







# LeftBiPhoneSet=[]
# RightBiPhoneSet=[]
# for e in Triphoneset:
#     LeftBiPhoneSet.append(e.split("+")[0])
#     RightBiPhoneSet.append(e.split("-")[1])

# PeopleLeftDepPhoneFunctional_dict=Dict()
# PeopleRightDepPhoneFunctional_dict=Dict()
# for spk, v in tqdm(PeopleCtxDepPhoneFunctional_dict.items()):
#     for phone, values in PeopleCtxDepPhoneFunctional_dict[spk].items():
#         LPhone, RPhone=phone.split("+")[0], phone.split("-")[1]
#         if LPhone not in PeopleLeftDepPhoneFunctional_dict[spk].keys():
#             PeopleLeftDepPhoneFunctional_dict[spk][LPhone]=df_template
#         PeopleLeftDepPhoneFunctional_dict[spk][LPhone]=PeopleLeftDepPhoneFunctional_dict[spk][LPhone].append(\
#                                                 pd.DataFrame(values.values,columns=values.columns,index=[LPhone]*len(values)))
        
#         if RPhone not in PeopleRightDepPhoneFunctional_dict[spk].keys():
#             PeopleRightDepPhoneFunctional_dict[spk][RPhone]=df_template
#         PeopleRightDepPhoneFunctional_dict[spk][RPhone]=PeopleRightDepPhoneFunctional_dict[spk][RPhone].append(\
#                                                 pd.DataFrame(values.values,columns=values.columns,index=[RPhone]*len(values)))
# if args.check:
#     PeopleLeftDepPhoneFunctional_dict_num=0
#     for spk, v in PeopleLeftDepPhoneFunctional_dict.items():
#         for phone, values in PeopleLeftDepPhoneFunctional_dict[spk].items():
#             PeopleLeftDepPhoneFunctional_dict_num+=len(values)
            
#     PeopleRightDepPhoneFunctional_dict_num=0
#     for spk, v in PeopleLeftDepPhoneFunctional_dict.items():
#         for phone, values in PeopleLeftDepPhoneFunctional_dict[spk].items():
#             PeopleRightDepPhoneFunctional_dict_num+=len(values)
    
#     Utt_ctxdepP_dict_num=0
#     for utt, v in Utt_ctxdepP_dict.items():
#         Utt_ctxdepP_dict_num+=len(v)
    
#     assert Utt_ctxdepP_dict_num == PeopleLeftDepPhoneFunctional_dict_num
#     assert Utt_ctxdepP_dict_num == PeopleRightDepPhoneFunctional_dict_num



# =============================================================================
# Manual for GetCtxPhoneFromUtt_map
# =============================================================================
# if manual:
#     Utt_ctxdepP_dict=Dict()
#     Set_lst=[]
#     for utt in tqdm(list(Formants_utt_symb.keys())):
#         df_Utt_phfContextDep=pd.DataFrame([],columns=feat_column)
        
#         values = Formants_utt_symb[utt]
#         phoneSeq=list(values.index.astype(str))    
#         if len(phoneSeq) == 1:
#             df_ctxdepP=values.iloc[[0],:]
#             df_ctxdepP.index=['{0}-{1}+{2}'.format('[s]',phoneSeq[0],'[\s]')]
#             df_Utt_phfContextDep=df_Utt_phfContextDep.append(df_ctxdepP)
#         else:    
#             for i in range(len(phoneSeq)): # df_Utt_phfContextDep append each word
#                 df_ctxdepP=values.iloc[[i],:]
#                 if i==0:
#                     df_ctxdepP.index=['{0}-{1}+{2}'.format('[s]',phoneSeq[i],phoneSeq[i+1])]
#                     # df_ctxdepP=pd.DataFrame(,index=['{0}-{1}+{2}'.format('[s]',phoneSeq[i],phoneSeq[i+1])],columns=values.columns)
#                 elif i==len(phoneSeq)-1:
#                     df_ctxdepP.index=['{0}-{1}+{2}'.format(phoneSeq[i-1],phoneSeq[i],'[\s]')]
#                 else:
#                     df_ctxdepP.index=['{0}-{1}+{2}'.format(phoneSeq[i-1],phoneSeq[i],phoneSeq[i+1])]
#                 df_Utt_phfContextDep=df_Utt_phfContextDep.append(df_ctxdepP)
#         assert len(df_Utt_phfContextDep) == len(phoneSeq) #check if the appended df: 'df_Utt_phfContextDep' ,atches phone sequence
    
#         Set_lst.append(list(df_Utt_phfContextDep.index))
#         Utt_ctxdepP_dict[utt]=df_Utt_phfContextDep    
    
