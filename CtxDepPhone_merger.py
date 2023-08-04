#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 22:51:34 2020

@author: jackchen


This script function like this:
    first, change the dict structure from PeopleCtxDepPhoneFunctional_dict[spk][phone] to AUI_ContextDepPhones[phone][spk], and take only /a/ /u/ /i/
    Second, we treat different tones (e.g. A:1 A:2 A:3 ...) to single tone (e.g. A) for both critical phone and surrounding phones: AUI_ContextDepPhonesWoPros[phone][spk]
    Third, we treat the similar phones /w/ /u/ and /j/ /i/ as same phone
    Forth, we Collapse the surrounding phones into catagorical phones like "fricative"

The code here will be executed really fast so no need for multiprocessing

    
**Notice that In this script, reorganizing finctions such as "get_Dep_MannernPlace_AUI", "get_Dep_wopros_AUI"
, "get_Dep_uwij_AUI" should be written in the form of function, and then be used in Execution area



    
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
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.feature_selection import f_classif
import re

path_app = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path_app+"/utils")

from articulation.HYPERPARAM import phonewoprosody, Label

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
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice/articulation',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--trnpath_w', default='/mnt/sdd/jackchen/egs/formosa/s6/Audacity_Word',
                        help='path of the base directory')
    parser.add_argument('--outpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--check', default=True,
                        help='path of the base directory')
    args = parser.parse_args()
    return args

args = get_args()
base_path=args.base_path
trnpath_w=args.trnpath_w
outpath=args.outpath



# path_app = base_path+'/../'
# sys.path.append(path_app)
class CtxDepPhone_merger:
    def __init__(self,phonewoprosody,AUI_ContextDepPhone=None):
        self.AUI_ContextDepPhone=AUI_ContextDepPhone
        self.phonewoprosody=phonewoprosody
    def install(self,AUI_ContextDepPhone):
        self.AUI_ContextDepPhone=AUI_ContextDepPhone
    def _lookup_Psets(self,s,Phoneme_sets):
        for p,v in Phoneme_sets.items():
            if s in v:
                return p
        return s  
    def get_Dep_AUI(self,PeopleCtxDepPhoneFunctional_dict,PhoneMapp_dict,PhonesOfInterest,mode="Ctx"):
        ''' Filter out all critical phonemes instead of A U I'''
        ''' mode can be [Ctx, Left, Right]                     '''
        ''' and change PeopleCtxDepPhoneFunctional_dict[spk][phone] to AUI_ContextDepPhones[phone][spk] '''
        CtxDepVowel_AUI=Dict()
        for spk,v in tqdm(PeopleCtxDepPhoneFunctional_dict.items()):
            for phone,values in PeopleCtxDepPhoneFunctional_dict[spk].items():
                assert len(values)>0
                assert phone == values.index.values[0]
                
                if mode == 'Ctx':
                    critical_P=phone[phone.find('-')+1:phone.find('+')]
                elif mode == 'Left':
                    critical_P=phone[phone.find('-')+1:]
                elif mode == 'Right':
                    critical_P=phone[:phone.find('+')]
                
    
                if critical_P in [e for phoneme in phonewoprosody.PhoneMapp_dict.keys() for e in PhoneMapp_dict[phoneme]]:
                    
                    if phone not in CtxDepVowel_AUI.keys():
                        if spk not in CtxDepVowel_AUI[phone].keys():
                            CtxDepVowel_AUI[phone][spk]=values
                        elif spk in CtxDepVowel_AUI[phone].keys():
                            CtxDepVowel_AUI[phone][spk].append(values)
                    else:
                        if spk not in CtxDepVowel_AUI[phone].keys():
                            CtxDepVowel_AUI[phone][spk]=values
                        elif spk in CtxDepVowel_AUI[phone].keys(): 
                            CtxDepVowel_AUI[phone][spk].append(values)
        return CtxDepVowel_AUI
    def get_Dep_wopros_AUI(self,Feature_dict,feat='CtxDepVowel_AUI',check=True):
        # =============================================================================
        '''
            Turn DepVowel_AUI into []DepVowelwopros_AUI
            
            Here we take off the prosody label for both surrounding phones and critical phones
        '''
        Feature_wopros_dict=Dict()
        for phone in tqdm(Feature_dict.keys()):
            for spk in Feature_dict[phone].keys():
                values= Feature_dict[phone][spk]
                feat_type=feat[:feat.find("Dep")]
                
                
                if feat_type == 'Ctx':
                    left_P=phone[:phone.find('-')]
                    right_P=phone[phone.find('+')+1:]
                    critical_P=phone[phone.find('-')+1:phone.find('+')]
                    
                    ph_wopros_center=self._lookup_Psets(critical_P,self.phonewoprosody.Phoneme_sets).replace("_","")
                    ph_wopros_left=self._lookup_Psets(left_P,self.phonewoprosody.Phoneme_sets).replace("_","") if left_P not in ['[s]','[\s]'] else left_P
                    ph_wopros_right=self._lookup_Psets(right_P,self.phonewoprosody.Phoneme_sets).replace("_","") if right_P not in ['[s]','[\s]'] else right_P
                    
                    wopros_Depphone="{0}-{1}+{2}".format(ph_wopros_left,ph_wopros_center,ph_wopros_right)
                    
                elif feat_type == 'Left':
                    left_P=phone[:phone.find('-')]
                    critical_P=phone[phone.find('-')+1:]
                    
                    ph_wopros_center=self._lookup_Psets(critical_P,self.phonewoprosody.Phoneme_sets).replace("_","")
                    ph_wopros_left=self._lookup_Psets(left_P,self.phonewoprosody.Phoneme_sets).replace("_","") if left_P not in ['[s]','[\s]'] else left_P
                    
                    wopros_Depphone="{0}-{1}".format(ph_wopros_left,ph_wopros_center)
                elif feat_type == 'Right':
                    critical_P=phone[:phone.find('+')]
                    right_P=phone[phone.find('+')+1:]
                    
                    ph_wopros_center=self._lookup_Psets(critical_P,self.phonewoprosody.Phoneme_sets).replace("_","")
                    ph_wopros_right=self._lookup_Psets(right_P,self.phonewoprosody.Phoneme_sets).replace("_","") if right_P not in ['[s]','[\s]'] else right_P
                    
                    wopros_Depphone="{0}+{1}".format(ph_wopros_center,ph_wopros_right)
                    

                if wopros_Depphone not in Feature_wopros_dict.keys():
                    if spk not in Feature_wopros_dict[wopros_Depphone].keys():
                        Feature_wopros_dict[wopros_Depphone][spk]=values
                    elif spk in Feature_wopros_dict[wopros_Depphone].keys():
                        Feature_wopros_dict[wopros_Depphone][spk]=Feature_wopros_dict[wopros_Depphone][spk].append(values)
                else:
                    if spk not in Feature_wopros_dict[wopros_Depphone].keys():
                        Feature_wopros_dict[wopros_Depphone][spk]=values
                    elif spk in Feature_wopros_dict[wopros_Depphone].keys(): 
                        Feature_wopros_dict[wopros_Depphone][spk]=Feature_wopros_dict[wopros_Depphone][spk].append(values)
        
        ''' The total number of context dendant phones in "Feature_wopros_dict" should match "Feature_dict" '''
        if check:
            Feature_wopros_dict_num=0
            for phone, v in Feature_wopros_dict.items():
                for spk, values in Feature_wopros_dict[phone].items():
                    Feature_wopros_dict_num+=len(values)
            Feature_dict_num=0
            for phone, v in Feature_dict.items():
                for spk, values in Feature_dict[phone].items():
                    Feature_dict_num+=len(values)
            
            assert Feature_dict_num == Feature_wopros_dict_num
        return Feature_wopros_dict
    def get_Dep_uwij_AUI(self,Feature_dict,feat='CtxDepVowel_AUI'):
        # =============================================================================
        '''
        
            Merge /w/ /u/ and /j/ /i/
        
        '''
        # =============================================================================
        Feature_merge_dict=Dict()
        for phone in tqdm(Feature_dict.keys()):
            for spk in Feature_dict[phone].keys():
                values= Feature_dict[phone][spk]
                feat_type=feat[:feat.find("Dep")]
                
                if feat_type == 'Ctx':
                    left_P=phone[:phone.find('-')]
                    right_P=phone[phone.find('+')+1:]
                    critical_P=phone[phone.find('-')+1:phone.find('+')]
                    if critical_P == 'j':
                        critical_P='i'
                    elif critical_P == 'w':
                        critical_P='u'
                        
                    Depphone_merged="{0}-{1}+{2}".format(left_P,critical_P,right_P)
                    
                elif feat_type == 'Left':
                    left_P=phone[:phone.find('-')]
                    critical_P=phone[phone.find('-')+1:]
                    if critical_P == 'j':
                        critical_P='i'
                    elif critical_P == 'w':
                        critical_P='u'
                    
                    Depphone_merged="{0}-{1}".format(left_P,critical_P)
                    
                elif feat_type == 'Right':
                    critical_P=phone[:phone.find('+')]
                    right_P=phone[phone.find('+')+1:]
                    if critical_P == 'j':
                        critical_P='i'
                    elif critical_P == 'w':
                        critical_P='u'
                    
                    Depphone_merged="{0}+{1}".format(critical_P,right_P)
                
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
        ''' The total number of context dendant phones in "Feature_wopros_dict" should match "Feature_dict" '''
        if args.check:
            Feature_merge_dict_num=0
            for phone, v in Feature_merge_dict.items():
                for spk, values in Feature_merge_dict[phone].items():
                    Feature_merge_dict_num+=len(values)
            Feature_dict_num=0
            for phone, v in Feature_dict.items():
                for spk, values in Feature_dict[phone].items():
                    Feature_dict_num+=len(values)
            
            assert Feature_dict_num == Feature_merge_dict_num
        return Feature_merge_dict
    def get_Dep_ij_AUI(self,Feature_dict,feat='CtxDepVowel_AUI'):
        # =============================================================================
        '''
        
            Merge /w/ /u/ and /j/ /i/
        
        '''
        # =============================================================================
        Feature_merge_dict=Dict()
        for phone in tqdm(Feature_dict.keys()):
            for spk in Feature_dict[phone].keys():
                values= Feature_dict[phone][spk]
                feat_type=feat[:feat.find("Dep")]
                
                if feat_type == 'Ctx':
                    left_P=phone[:phone.find('-')]
                    right_P=phone[phone.find('+')+1:]
                    critical_P=phone[phone.find('-')+1:phone.find('+')]
                    if critical_P == 'j':
                        critical_P='i'
                        
                    Depphone_merged="{0}-{1}+{2}".format(left_P,critical_P,right_P)
                    
                elif feat_type == 'Left':
                    left_P=phone[:phone.find('-')]
                    critical_P=phone[phone.find('-')+1:]
                    if critical_P == 'j':
                        critical_P='i'
                    
                    Depphone_merged="{0}-{1}".format(left_P,critical_P)
                    
                elif feat_type == 'Right':
                    critical_P=phone[:phone.find('+')]
                    right_P=phone[phone.find('+')+1:]
                    if critical_P == 'j':
                        critical_P='i'
                    
                    Depphone_merged="{0}+{1}".format(critical_P,right_P)
                
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
        ''' The total number of context dendant phones in "Feature_wopros_dict" should match "Feature_dict" '''
        if args.check:
            Feature_merge_dict_num=0
            for phone, v in Feature_merge_dict.items():
                for spk, values in Feature_merge_dict[phone].items():
                    Feature_merge_dict_num+=len(values)
            Feature_dict_num=0
            for phone, v in Feature_dict.items():
                for spk, values in Feature_dict[phone].items():
                    Feature_dict_num+=len(values)
            
            assert Feature_dict_num == Feature_merge_dict_num
        return Feature_merge_dict
    def get_Dep_MannernPlace_AUI(self, Feature_dict,Phoneset_Here,feat='CtxDepVowel_AUI'):
        Feature_merge_dict=Dict()
        for phone in tqdm(Feature_dict.keys()):
            for spk in Feature_dict[phone].keys():
                values= Feature_dict[phone][spk]
                feat_type=feat[:feat.find("Dep")]
                
                if feat_type == 'Ctx':
                    left_P=phone[:phone.find('-')]
                    right_P=phone[phone.find('+')+1:]
                    critical_P=phone[phone.find('-')+1:phone.find('+')]
                    
                    # if left_P in ['A','O','i','u']:
                    #     left_P=left_P+':'
                    # if right_P in ['A','O','i','u']:
                    #     right_P=right_P+':'
                    
                    
                    left_P_catagory=self._lookup_Psets(left_P,Phoneset_Here)
                    right_P_catagory=self._lookup_Psets(right_P,Phoneset_Here)
                    assert left_P_catagory != -1 and right_P_catagory != -1
                    
                        
                    Depphone_merged="{0}-{1}+{2}".format(left_P_catagory,critical_P,right_P_catagory)
                    
                elif feat_type == 'Left':
                    left_P=phone[:phone.find('-')]
                    critical_P=phone[phone.find('-')+1:]
                    
                    # if left_P in ['A','O','i','u']:
                    #     left_P=left_P+':'
                    
                    left_P_catagory=self._lookup_Psets(left_P,Phoneset_Here)
                    assert left_P_catagory != -1 
        
                    Depphone_merged="{0}-{1}".format(left_P_catagory,critical_P)
                    
                elif feat_type == 'Right':
                    critical_P=phone[:phone.find('+')]
                    right_P=phone[phone.find('+')+1:]
                    # if right_P in ['A','O','i','u']:
                    #     right_P=right_P+':'
                    
                    right_P_catagory=self._lookup_Psets(right_P,Phoneset_Here)
                    assert right_P_catagory != -1
                    
                    Depphone_merged="{0}+{1}".format(critical_P,right_P_catagory)
                    
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
        ''' The total number of context dendant phones in "Feature_wopros_dict" should match "Feature_dict" '''
        if args.check:
            Feature_merge_dict_num=0
            for phone, v in Feature_merge_dict.items():
                for spk, values in Feature_merge_dict[phone].items():
                    Feature_merge_dict_num+=len(values)
            Feature_dict_num=0
            for phone, v in Feature_dict.items():
                for spk, values in Feature_dict[phone].items():
                    Feature_dict_num+=len(values)
            assert Feature_dict_num == Feature_merge_dict_num
        return Feature_merge_dict
    def get_AllMerged_AUI(self, Feature_dict,feat='CtxDepVowel_AUI'):
        Feature_merge_dict=Dict()
        for phone in tqdm(Feature_dict.keys()):
            for spk in Feature_dict[phone].keys():
                values= Feature_dict[phone][spk]
                feat_type=feat[:feat.find("Dep")]
                
                if feat_type == 'Ctx':
                    critical_P=phone[phone.find('-')+1:phone.find('+')]


                    Depphone_merged="{0}-{1}+{2}".format('All',critical_P,'All')
                    
                elif feat_type == 'Left':
                    critical_P=phone[phone.find('-')+1:]

                    Depphone_merged="{0}-{1}".format('All',critical_P)
                    
                elif feat_type == 'Right':
                    critical_P=phone[:phone.find('+')]

                    Depphone_merged="{0}+{1}".format(critical_P,'All')
                    
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
        ''' The total number of context dendant phones in "Feature_wopros_dict" should match "Feature_dict" '''
        if args.check:
            Feature_merge_dict_num=0
            for phone, v in Feature_merge_dict.items():
                for spk, values in Feature_merge_dict[phone].items():
                    Feature_merge_dict_num+=len(values)
            Feature_dict_num=0
            for phone, v in Feature_dict.items():
                for spk, values in Feature_dict[phone].items():
                    Feature_dict_num+=len(values)
            assert Feature_dict_num == Feature_merge_dict_num
        return Feature_merge_dict

# # =============================================================================
# '''

#     Execution area

# '''
# # =============================================================================

# Phoneset_sets={'Manner_simp1':phonewoprosody.Manner_sets_simple1, \
#                 'Manner_simp2':phonewoprosody.Manner_sets_simple2, \
#                 'Place_simp1':phonewoprosody.Place_sets_simple1,  \
#                 'Place_simp2':phonewoprosody.Place_sets_simple2}

# PeopleCtxDepPhoneFunctional_dict=pickle.load(open(outpath+"/PeopleCtxDepPhoneFunctional_dict.pkl","rb"))
# PeopleLeftDepPhoneFunctional_dict=pickle.load(open(outpath+"/PeopleLeftDepPhoneFunctional_dict.pkl","rb"))
# PeopleRightDepPhoneFunctional_dict=pickle.load(open(outpath+"/PeopleRightDepPhoneFunctional_dict.pkl","rb"))

    
# ctxdepphone_mger=CtxDepPhone_merger(phonewoprosody)
 

   
# AUI_ContextDepPhones=Dict()
# AUI_ContextDepPhones['CtxDepVowel_AUI']=ctxdepphone_mger.get_Dep_AUI(PeopleCtxDepPhoneFunctional_dict,mode='Ctx')
# AUI_ContextDepPhones['LeftDepVowel_AUI']=ctxdepphone_mger.get_Dep_AUI(PeopleLeftDepPhoneFunctional_dict,mode='Left')
# AUI_ContextDepPhones['RightDepVowel_AUI']=ctxdepphone_mger.get_Dep_AUI(PeopleRightDepPhoneFunctional_dict,mode='Right')


# for keys,Phoneset_Here in tqdm(Phoneset_sets.items()):
#     AUI_ContextDepPhonesMerge_MannerPlace_uwij=Dict()
#     for feat in AUI_ContextDepPhones.keys():
#         Feature_dict=AUI_ContextDepPhones[feat]
#         DepPhonesMerge_MannerPlace=ctxdepphone_mger.get_Dep_MannernPlace_AUI(Feature_dict,Phoneset_Here,feat)
#         DepPhonesMerge_MannerPlace_wopros=ctxdepphone_mger.get_Dep_wopros_AUI(DepPhonesMerge_MannerPlace,feat=feat)
#         DepPhonesMerge_MannerPlace_wopros_uwij=ctxdepphone_mger.get_Dep_uwij_AUI(DepPhonesMerge_MannerPlace_wopros,feat=feat)
#         AUI_ContextDepPhonesMerge_MannerPlace_uwij[feat]=DepPhonesMerge_MannerPlace_wopros_uwij
#     pickle.dump(AUI_ContextDepPhonesMerge_MannerPlace_uwij,open(outpath+"/AUI_ContextDepPhonesMerge_{0}_uwij.pkl".format(keys),"wb"))



class Filter_CtxPhone2Biphone:
    def __init__(self, PeopleCtxDepPhoneFunctional_dict,Triphoneset,feat_column=['F1','F2']):
        self.PeopleCtxDepPhoneFunctional_dict=PeopleCtxDepPhoneFunctional_dict
        self.Triphoneset=Triphoneset
        self.df_template=pd.DataFrame([],columns=feat_column)
    def Get_Biphone_Strset(self):
        LeftBiPhoneSet=[]
        RightBiPhoneSet=[]
        for e in self.Triphoneset:
            LeftBiPhoneSet.append(e.split("+")[0])
            RightBiPhoneSet.append(e.split("-")[1])
        return LeftBiPhoneSet, RightBiPhoneSet
    def Get_Biphone(self, check=True):
        PeopleLeftDepPhoneFunctional_dict=Dict()
        PeopleRightDepPhoneFunctional_dict=Dict()
        for spk, v in tqdm(self.PeopleCtxDepPhoneFunctional_dict.items()):
            for phone, values in self.PeopleCtxDepPhoneFunctional_dict[spk].items():
                LPhone, RPhone=phone.split("+")[0], phone.split("-")[1]
                if LPhone not in PeopleLeftDepPhoneFunctional_dict[spk].keys():
                    PeopleLeftDepPhoneFunctional_dict[spk][LPhone]=self.df_template
                PeopleLeftDepPhoneFunctional_dict[spk][LPhone]=PeopleLeftDepPhoneFunctional_dict[spk][LPhone].append(\
                                                        pd.DataFrame(values.values,columns=values.columns,index=[LPhone]*len(values)))
                
                if RPhone not in PeopleRightDepPhoneFunctional_dict[spk].keys():
                    PeopleRightDepPhoneFunctional_dict[spk][RPhone]=self.df_template
                PeopleRightDepPhoneFunctional_dict[spk][RPhone]=PeopleRightDepPhoneFunctional_dict[spk][RPhone].append(\
                                                        pd.DataFrame(values.values,columns=values.columns,index=[RPhone]*len(values)))

        return PeopleLeftDepPhoneFunctional_dict, PeopleRightDepPhoneFunctional_dict
    def Get_LBiphone_map(self, keys,  check=True):
        PeopleLeftDepPhoneFunctional_dict=Dict()
        # PeopleRightDepPhoneFunctional_dict=Dict()
        print(" process PID", os.getpid(), " running")
        for spk in tqdm(keys):
            for phone, values in self.PeopleCtxDepPhoneFunctional_dict[spk].items():
                LPhone=phone.split("+")[0]
                if LPhone not in PeopleLeftDepPhoneFunctional_dict[spk].keys():
                    PeopleLeftDepPhoneFunctional_dict[spk][LPhone]=self.df_template
                PeopleLeftDepPhoneFunctional_dict[spk][LPhone]=PeopleLeftDepPhoneFunctional_dict[spk][LPhone].append(\
                                                        pd.DataFrame(values.values,columns=values.columns,index=[LPhone]*len(values)))
        print(" process PID", os.getpid(), " done")                    
        return PeopleLeftDepPhoneFunctional_dict
    def Get_RBiphone_map(self, keys,  check=True):
        PeopleRightDepPhoneFunctional_dict=Dict()
        print(" process PID", os.getpid(), " running")
        for spk in tqdm(keys):
            for phone, values in self.PeopleCtxDepPhoneFunctional_dict[spk].items():
                RPhone=phone.split("-")[1]

                if RPhone not in PeopleRightDepPhoneFunctional_dict[spk].keys():
                    PeopleRightDepPhoneFunctional_dict[spk][RPhone]=self.df_template
                PeopleRightDepPhoneFunctional_dict[spk][RPhone]=PeopleRightDepPhoneFunctional_dict[spk][RPhone].append(\
                                                        pd.DataFrame(values.values,columns=values.columns,index=[RPhone]*len(values)))
        print(" process PID", os.getpid(), " done")                    
        return PeopleRightDepPhoneFunctional_dict
    def Process_multi(self,func):
        keys=[]
        interval=20
        for i in range(0,len(self.PeopleCtxDepPhoneFunctional_dict),interval):
            keys.append(list(self.PeopleCtxDepPhoneFunctional_dict.keys())[i:i+interval])
        flat_keys=[item for sublist in keys for item in sublist]
        assert len(flat_keys) == len(self.PeopleCtxDepPhoneFunctional_dict)
        
        
        pool = Pool(os.cpu_count())
        final_result = pool.starmap(func, [([key]) for key in tqdm(keys)])
        
        PeopleCtxDepPhoneFunctional_dict=Dict()
        for d in tqdm(final_result):
            for spk in d.keys():
                for phone,values in d[spk].items():
                    if phone not in PeopleCtxDepPhoneFunctional_dict[spk].keys():
                        PeopleCtxDepPhoneFunctional_dict[spk][phone]=self.df_template
                    PeopleCtxDepPhoneFunctional_dict[spk][phone]=PeopleCtxDepPhoneFunctional_dict[spk][phone].append(values)
                
        return PeopleCtxDepPhoneFunctional_dict
        
        
    def check_totnum(self,Utt_ctxdepP_dict,dict_to_test,check=True):
        # ''' The total number of context dendant phones in "PeopleCtxDepPhoneFunctional_dict" should match "Utt_ctxdepP_dict" '''
        if check:
            PeopleCtxDepPhoneFunctional_dict_num_dict=Dict()
            PeopleCtxDepPhoneFunctional_dict_num=0
            for spk, v in dict_to_test.items():
                if spk not in PeopleCtxDepPhoneFunctional_dict_num_dict.keys():
                    PeopleCtxDepPhoneFunctional_dict_num_dict[spk]=0
                for phone, values in dict_to_test[spk].items():
                    PeopleCtxDepPhoneFunctional_dict_num+=len(values)
                    PeopleCtxDepPhoneFunctional_dict_num_dict[spk]+=len(values)
                    
                        
            Utt_ctxdepP_dict_num_dict=Dict()
            Utt_ctxdepP_dict_num=0
            for utt, v in Utt_ctxdepP_dict.items():
                spk=utt[:re.search("_(K|D)_",utt).start()]+"_K"
                Utt_ctxdepP_dict_num+=len(v)
                if spk not in Utt_ctxdepP_dict_num_dict.keys():
                    Utt_ctxdepP_dict_num_dict[spk]=0
                Utt_ctxdepP_dict_num_dict[spk]+=len(v)
            assert Utt_ctxdepP_dict_num == PeopleCtxDepPhoneFunctional_dict_num
    def Check(self, data_dict,data_dict_cmp):
        for key1 in data_dict.keys():
            for key2 in data_dict[key1].keys():
                assert data_dict[key1][key2].values == data_dict_cmp[key1][key2]