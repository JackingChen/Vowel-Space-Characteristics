#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 12:41:30 2021

@author: jackchen

This module is for unit-tests

"""
import pickle
import pandas as pd
import numpy as np

file_basepath='/homes/ssd1/jackchen/DisVoice/articulation'
# Formants_utt_symb_path=file_basepath + '/Pickles/Formants_utt_symb_bymiddle_window3.pkl'
# Formants_people_symb_path=file_basepath + '/Pickles/Formants_people_symb_bymiddle_window3.pkl'

# Formants_utt_symb_path=file_basepath + '/Pickles/[Testing]Formants_utt_Auto_symb_bymiddle.pkl'
Formants_utt_symb_path=file_basepath + '/Pickles/[Analyzing]Formants_utt_symb_limited.pkl'
Formants_people_symb_path=file_basepath + '/Pickles/[Testing]Formants_people_Auto_symb_bymiddle.pkl'
df_formant_statistic_path=file_basepath + '/Pickles/Session_formants_people_vowel_feat/Formant_AUI_tVSAFCRFvals_ASDkid.pkl'

class Testing:
    def __init__(self):
        self.Formants_utt_symb_path=Formants_utt_symb_path
        self.Formants_people_symb_path=Formants_people_symb_path
        self.df_formant_statistic_path=df_formant_statistic_path
        self.Formants_utt_symb_cmp_path=self.Formants_utt_symb_path.replace(".pkl","_cmp.pkl")
        self.Formants_people_symb_cmp_path=self.Formants_people_symb_path.replace(".pkl","_cmp.pkl")
        self.df_formant_statistic_cmp_path=self.df_formant_statistic_path.replace(".pkl","_cmp.pkl")
        
        
    def TestFormants_utt(self):
        Formants_utt_symb=pickle.load(open(self.Formants_utt_symb_path,"rb"))
        Formants_utt_symb_cmp=pickle.load(open(self.Formants_utt_symb_cmp_path,"rb"))
        for utt, df_values in Formants_utt_symb.items():
            df_values_cmp=Formants_utt_symb_cmp[utt]
            assert ((df_values == df_values_cmp).all()).all()
        print('passed TestFormants_utt') 
            
    def TestFormants_people(self):
        Formants_people_symb=pickle.load(open(self.Formants_people_symb_path,"rb"))
        Formants_people_symb_cmp=pickle.load(open(self.Formants_people_symb_cmp_path,"rb"))
        for people, df_values in Formants_people_symb.items():
            for symb, feat_array in df_values.items():
                feat_array_cmp=Formants_people_symb_cmp[people][symb]
                assert np.sum((np.array(feat_array) - np.array(feat_array_cmp))) == 0
            
        print('passed TestFormants_people')    
    def Test_Formantstatistic(self):
        df_formant_statistic=pickle.load(open(self.df_formant_statistic_path,"rb"))
        df_formant_statistic_cmp=pickle.load(open(self.df_formant_statistic_cmp_path,"rb"))
        assert (df_formant_statistic - df_formant_statistic_cmp).sum().sum() ==0
        print('passed Test_Formantstatistic')   
test=Testing()
    
self=test

self.TestFormants_utt()
self.TestFormants_people()
self.Test_Formantstatistic()