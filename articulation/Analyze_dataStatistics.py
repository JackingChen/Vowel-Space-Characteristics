#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 12:20:34 2021

@author: jackchen
"""

import pickle
from addict import Dict
import pandas as pd
Vowels_AUIpath='Pickles/Session_formants_people_vowel_feat/'
role="ASDkid"
Vowels_AUI=pickle.load(open(Vowels_AUIpath+"Vowels_AUI_{}.pkl".format(role),"rb"))


Formant_value_people=Dict()
PhoneNum_people=Dict()
for people in Vowels_AUI.keys():
    for cirtical_P in Vowels_AUI[people].keys():
        data_df=Vowels_AUI[people][cirtical_P]
        
        Formant_value_people[cirtical_P][people]=data_df.mean(axis=0)
        PhoneNum_people[cirtical_P][people]=len(data_df)

Formant_value_Meanpeople_phone=Dict()
Formant_value_Meanpeople_phone_raw=Dict()
for cirtical_P in Formant_value_people.keys():
    Formant_value_Meanpeople_phone[cirtical_P].max=pd.DataFrame.from_dict(Formant_value_people[cirtical_P]).T.max(axis=0)
    Formant_value_Meanpeople_phone[cirtical_P].min=pd.DataFrame.from_dict(Formant_value_people[cirtical_P]).T.min(axis=0)
    Formant_value_Meanpeople_phone_raw[cirtical_P]=pd.DataFrame.from_dict(Formant_value_people[cirtical_P]).T
    
df_Formant_value_Meanpeople_phone_dict=Dict()
for cirtical_P in Formant_value_Meanpeople_phone.keys():
    df_Formant_value_Meanpeople_phone_dict[cirtical_P]=pd.DataFrame.from_dict(Formant_value_Meanpeople_phone[cirtical_P])

df_PhoneNum_people_dict=Dict()
for cirtical_P in PhoneNum_people.keys():
    df_PhoneNum_people_dict[cirtical_P]=pd.DataFrame.from_dict(PhoneNum_people[cirtical_P],orient='index').describe()
