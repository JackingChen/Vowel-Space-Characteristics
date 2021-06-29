#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 12:50:44 2020

@author: jackchen

This script do the following work:
    1. Group the phone sequence by lexicon
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
    parser.add_argument('--filepath', default='/mnt/sdd/jackchen/egs/formosa/s6/Segmented_ADOS_emotion',
                        help='path of the base directory')
    parser.add_argument('--trnpath_w', default='/mnt/sdd/jackchen/egs/formosa/s6/Audacity_Word',
                        help='path of the base directory')
    parser.add_argument('--trnpath_p', default='/mnt/sdd/jackchen/egs/formosa/s6/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/kid/ADOS_tdnn_fold_transfer',
                        help='path of the base directory')
    parser.add_argument('--asr_path', default='/mnt/sdd/jackchen/egs/formosa/s6',
                        help='path of the base directory')
    parser.add_argument('--outpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--check', default=True,
                        help='path of the base directory')
    parser.add_argument('--feat_column', default=['F1','F2'],
                        help='path of the base directory')
    parser.add_argument('--manual', default=False,
                        help='path of the base directory')
    parser.add_argument('--use_ckpt', default=True,
                        help='')
    args = parser.parse_args()
    return args

args = get_args()
base_path=args.base_path
filepath=args.filepath
trnpath_w=args.trnpath_w
trnpath_p=args.trnpath_p
asr_path=args.asr_path
outpath=args.outpath
feat_column=args.feat_column
manual=args.manual

path_app = base_path
sys.path.append(path_app)
import praat.praat_functions as praat_functions
from script_mananger import script_manager

files_p=glob.glob(trnpath_p+"/*_K_*.txt")

# =============================================================================
'''

    single phone to context dependant phone
    Utt_ctxdepP_dict[utt] -> df: index = contetxt-dependant phone([s]-j+aU4), columns = [F1, F2] of any features
'''
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
                if i==0:
                    df_ctxdepP.index=['{0}-{1}+{2}'.format('[s]',phoneSeq[i],phoneSeq[i+1])]
                    # df_ctxdepP=pd.DataFrame(,index=['{0}-{1}+{2}'.format('[s]',phoneSeq[i],phoneSeq[i+1])],columns=values.columns)
                elif i==len(phoneSeq)-1:
                    df_ctxdepP.index=['{0}-{1}+{2}'.format(phoneSeq[i-1],phoneSeq[i],'[\s]')]
                else:
                    df_ctxdepP.index=['{0}-{1}+{2}'.format(phoneSeq[i-1],phoneSeq[i],phoneSeq[i+1])]
                df_Utt_phfContextDep=df_Utt_phfContextDep.append(df_ctxdepP)
        assert len(df_Utt_phfContextDep) == len(phoneSeq) #check if the appended df: 'df_Utt_phfContextDep' ,atches phone sequence
    
        Set_lst.append(list(df_Utt_phfContextDep.index))
        Utt_ctxdepP_dict[utt]=df_Utt_phfContextDep  
    
    print("PID {} Getting ".format(os.getpid()), "Done")
    return Utt_ctxdepP_dict, Set_lst
# =============================================================================

Formants_utt_symb=pickle.load(open(outpath+"/Formants_utt_symb_bymiddle.pkl","rb"))
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

''' multiprocessing end '''


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
    

pickle.dump(Utt_ctxdepP_dict,open(outpath+"/Utt_ctxdepP_dict_cmp.pkl","wb"))
pickle.dump(Set_lst,open(outpath+"/CxtDepPhone_Setlst.pkl","wb"))


# =============================================================================
'''

    Accumulate the appearance of each context dependant phones (wanna control the phone environment)
    
'''

def GetEachCtxDepPhoneFromUtt_map(keys,Triphoneset,Utt_ctxdepP_dict):
    print(" process PID", os.getpid(), " running")
    PeopleCtxDepPhoneFunctional_dict=Dict()
    for key in keys:
        values=Utt_ctxdepP_dict[key]
        df_template=pd.DataFrame([],columns=feat_column)
        spk='_'.join(key.split("_")[:-2])
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
# =============================================================================

# THis part is really slow even if we use multi-processing technique, so use checkpoint if available
if os.path.exists(outpath+"/PeopleCtxDepPhoneFunctional_dict.pkl") and args.use_ckpt:
    PeopleCtxDepPhoneFunctional_dict=pickle.load(open(outpath+"/PeopleCtxDepPhoneFunctional_dict.pkl","rb"))
else:    
    Utt_ctxdepP_dict=pickle.load(open(outpath+"/Utt_ctxdepP_dict.pkl","rb"))
    Set_lst=pickle.load(open(outpath+"/CxtDepPhone_Setlst.pkl","rb"))
    Triphoneset = set([item for sublist in Set_lst for item in sublist])
    df_template=pd.DataFrame([],columns=feat_column)
    
    # =============================================================================
    # Manual for GetEachCtxDepPhoneFromUtt_map
    # =============================================================================
    # if manual:
    #     PeopleCtxDepPhoneFunctional_dict=Dict()
    #     ''' Gathering phones from utt level,  Utt_ctxdepP_dict -> PeopleCtxDepPhoneFunctional_dict '''
    #     for keys, values in tqdm(Utt_ctxdepP_dict.items()):
    #         emotion=keys.split("_")[-1]
    #         n=keys.split("_")[-2]
    #         spk='_'.join(keys.split("_")[:-2])
            
    #         for phone in Triphoneset:
    #             if len(values[values.index==phone])>0:
    #                 if phone not in PeopleCtxDepPhoneFunctional_dict[spk].keys():
    #                     PeopleCtxDepPhoneFunctional_dict[spk][phone]=df_template
    #                 PeopleCtxDepPhoneFunctional_dict[spk][phone]=PeopleCtxDepPhoneFunctional_dict[spk][phone].append(values[values.index==phone])
    keys=[]
    interval=20
    for i in range(0,len(Utt_ctxdepP_dict),interval):
        # print(list(Utt_ctxdepP_dict.keys())[i:i+interval])
        keys.append(list(Utt_ctxdepP_dict.keys())[i:i+interval])
    flat_keys=[item for sublist in keys for item in sublist]
    assert len(flat_keys) == len(Utt_ctxdepP_dict)
    
    
    pool = Pool(os.cpu_count())
    # pool = Pool(2)
    final_result = pool.starmap(GetEachCtxDepPhoneFromUtt_map, [(key,Triphoneset,Utt_ctxdepP_dict) for key in keys])
    
    PeopleCtxDepPhoneFunctional_dict=Dict()
    for d in tqdm(final_result):
        for spk in d.keys():
            for phone,values in d[spk].items():
                if phone not in PeopleCtxDepPhoneFunctional_dict[spk].keys():
                    PeopleCtxDepPhoneFunctional_dict[spk][phone]=df_template
                PeopleCtxDepPhoneFunctional_dict[spk][phone]=PeopleCtxDepPhoneFunctional_dict[spk][phone].append(values)
    
    
    # ''' The total number of context dendant phones in "PeopleCtxDepPhoneFunctional_dict" should match "Utt_ctxdepP_dict" '''
    # if args.check:
    #     PeopleCtxDepPhoneFunctional_dict_num=0
    #     for spk, v in PeopleCtxDepPhoneFunctional_dict.items():
    #         for phone, values in PeopleCtxDepPhoneFunctional_dict[spk].items():
    #             PeopleCtxDepPhoneFunctional_dict_num+=len(values)
    #     Utt_ctxdepP_dict_num=0
    #     for utt, v in Utt_ctxdepP_dict.items():
    #         Utt_ctxdepP_dict_num+=len(v)
        
    #     assert Utt_ctxdepP_dict_num == PeopleCtxDepPhoneFunctional_dict_num
    pickle.dump(PeopleCtxDepPhoneFunctional_dict,open(outpath+"/PeopleCtxDepPhoneFunctional_dict.pkl","wb"))

# =============================================================================
'''

    Previous steps get the tri-phone from word level utterance, 
    We now extract bi-gram phones from the above "PeopleCtxDepPhoneFunctional_dict"

'''
# =============================================================================


LeftBiPhoneSet=[]
RightBiPhoneSet=[]
for e in Triphoneset:
    LeftBiPhoneSet.append(e.split("+")[0])
    RightBiPhoneSet.append(e.split("-")[1])

PeopleLeftDepPhoneFunctional_dict=Dict()
PeopleRightDepPhoneFunctional_dict=Dict()
for spk, v in tqdm(PeopleCtxDepPhoneFunctional_dict.items()):
    for phone, values in PeopleCtxDepPhoneFunctional_dict[spk].items():
        LPhone, RPhone=phone.split("+")[0], phone.split("-")[1]
        if LPhone not in PeopleLeftDepPhoneFunctional_dict[spk].keys():
            PeopleLeftDepPhoneFunctional_dict[spk][LPhone]=df_template
        PeopleLeftDepPhoneFunctional_dict[spk][LPhone]=PeopleLeftDepPhoneFunctional_dict[spk][LPhone].append(\
                                                pd.DataFrame(values.values,columns=values.columns,index=[LPhone]*len(values)))
        
        if RPhone not in PeopleRightDepPhoneFunctional_dict[spk].keys():
            PeopleRightDepPhoneFunctional_dict[spk][RPhone]=df_template
        PeopleRightDepPhoneFunctional_dict[spk][RPhone]=PeopleRightDepPhoneFunctional_dict[spk][RPhone].append(\
                                                pd.DataFrame(values.values,columns=values.columns,index=[RPhone]*len(values)))
if args.check:
    PeopleLeftDepPhoneFunctional_dict_num=0
    for spk, v in PeopleLeftDepPhoneFunctional_dict.items():
        for phone, values in PeopleLeftDepPhoneFunctional_dict[spk].items():
            PeopleLeftDepPhoneFunctional_dict_num+=len(values)
            
    PeopleRightDepPhoneFunctional_dict_num=0
    for spk, v in PeopleLeftDepPhoneFunctional_dict.items():
        for phone, values in PeopleLeftDepPhoneFunctional_dict[spk].items():
            PeopleRightDepPhoneFunctional_dict_num+=len(values)
    
    Utt_ctxdepP_dict_num=0
    for utt, v in Utt_ctxdepP_dict.items():
        Utt_ctxdepP_dict_num+=len(v)
    
    assert Utt_ctxdepP_dict_num == PeopleLeftDepPhoneFunctional_dict_num
    assert Utt_ctxdepP_dict_num == PeopleRightDepPhoneFunctional_dict_num


pickle.dump(PeopleLeftDepPhoneFunctional_dict,open(outpath+"/PeopleLeftDepPhoneFunctional_dict.pkl","wb"))
pickle.dump(PeopleRightDepPhoneFunctional_dict,open(outpath+"/PeopleRightDepPhoneFunctional_dict.pkl","wb"))