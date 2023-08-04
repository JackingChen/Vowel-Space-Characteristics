#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 22:21:32 2020

@author: jackchen

Note, we only analyze the vowels using F1 F2. We didn't include other phonemes other than that'

The purpose of this script is to generate Sesssional Formant features (four descriptors of F1 and F2 (at most 8 dimensions))
THe functional is taken among each phonemes of one person

"""

import os
import pickle
import argparse
from addict import Dict
import numpy as np
from HYPERPARAM import phonewoprosody, Label
import matplotlib.pyplot as plt
import pandas as pd
from utils_jack import dynamic2statict_artic_formant, save_dict_kaldimat, get_dict

def NameMatchAssertion(Formants_people_symb,name):
    ''' check the name in  Formants_people_symb matches the names in label'''
    for name in Formants_people_symb.keys():
        assert name in name

def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice/articulation',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--filepath', default='/mnt/sdd/jackchen/egs/formosa/s6/Segmented_ADOS_emotion',
                        help='path of the base directory')
    parser.add_argument('--trnpath', default='/mnt/sdd/jackchen/egs/formosa/s6/Audacity',
                        help='path of the base directory')
    parser.add_argument('--pklpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')

    args = parser.parse_args()
    return args



args = get_args()
base_path=args.base_path
filepath=args.filepath
trnpath=args.trnpath
pklpath=args.pklpath


Vowels_single=['i_','E','axr','A_','u_','ax','O_']
Vowels_prosody_single=[phonewoprosody.Phoneme_sets[v]  for v in Vowels_single]
Vowels_prosody_single=[item for sublist in Vowels_prosody_single for item in sublist]


Formants_utt_symb=pickle.load(open(pklpath+"/Formants_utt_symb.pkl","rb"))
Formants_people_symb=pickle.load(open(pklpath+"/Formants_people_symb.pkl","rb"))

people='2015_12_05_01_063_1'
person_vowel_symb=Dict()

X_lst=[]
y_lst=[]

func=["mean",'std','skew','skew_sign','skew_abs','kurtosis']
Feat=['F1','F2']

Static_cols=[]

for fun in func:
    for f in Feat:
        Static_cols.append(f+"+"+fun)

# =============================================================================
''' grouping severity '''
label_set=['ADOS_C','ADOS_S','ADOS_SC']
# =============================================================================


# =============================================================================
'''

Df_formants_people_vowel[people][phone] = df(index={single_phone},columns={Feat+func})


'''
def lookup_Psets(s,Phoneme_sets):
    for p,v in Phoneme_sets.items():
        if s in v:
            return p
    return -1
# =============================================================================



# =============================================================================
''' here we use average method for the calculation of phonewoprosodys'''
# =============================================================================
Df_formants_people_vowel=Dict()
Combine_formants_people_vowelwoprosody=Dict()
Combine_formants_people_vowelwoprosody01=Dict()
for people in sorted(Formants_people_symb.keys()):
    Df_formants_people_vowel[people]=pd.DataFrame([],index=Vowels_prosody_single,columns=Static_cols)
    one_count, zero_count = 0,0
    for phone, values in sorted(Formants_people_symb[people].items()):
        
        if phone in Vowels_prosody_single:
            '''     Collecting F1 F2 statics for single phone       '''
            if len(Formants_people_symb[people][phone])==1:
                feature=np.repeat(Formants_people_symb[people][phone],2,axis=0)
                one_count+=1
            elif len(Formants_people_symb[people][phone])==0:
                feature=np.repeat(np.zeros((1,2)),2,axis=0)
                zero_count+=1
            else:
                feature=Formants_people_symb[people][phone]
            static_func=dynamic2statict_artic_formant(np.vstack(feature).T)
            Df_formants_people_vowel[people].loc[phone]=static_func
            
            '''     Collecting F1 F2 statics for phonewoprosody       '''
            
            ph_wopros=lookup_Psets(phone,phonewoprosody.Phoneme_sets)
            assert ph_wopros != -1
            if ph_wopros not in Combine_formants_people_vowelwoprosody[people].keys():
                assert len(Combine_formants_people_vowelwoprosody[people][ph_wopros]) == 0
                Combine_formants_people_vowelwoprosody[people][ph_wopros]=np.vstack(feature)
                
            else:
                Combine_formants_people_vowelwoprosody[people][ph_wopros]=np.vstack((Combine_formants_people_vowelwoprosody[people][ph_wopros],feature))
           
            ph_wopros=lookup_Psets(phone,phonewoprosody.Phoneme01_sets)
            if ph_wopros != -1:
                if ph_wopros not in Combine_formants_people_vowelwoprosody01[people].keys():
                    assert len(Combine_formants_people_vowelwoprosody01[people][ph_wopros]) == 0
                    Combine_formants_people_vowelwoprosody01[people][ph_wopros]=np.vstack(feature)
                    
                else:
                    Combine_formants_people_vowelwoprosody01[people][ph_wopros]=np.vstack((Combine_formants_people_vowelwoprosody01[people][ph_wopros],feature))
            
            
            
    num_feature_vowel=np.vstack([values for phone, values in Formants_people_symb[people].items() if phone in Vowels_prosody_single]).shape[0]
    num_feature_vowelwoprosody=np.vstack([values for phone, values in Combine_formants_people_vowelwoprosody[people].items() if phone in Vowels_single]).shape[0]
    assert num_feature_vowel +one_count + zero_count == num_feature_vowelwoprosody
pickle.dump(Df_formants_people_vowel,open(pklpath+"/Df_formants_people_vowel.pkl","wb"))




Df_formants_people_vowelwoprosody=Dict()
for people in sorted(Combine_formants_people_vowelwoprosody.keys()):
    Df_formants_people_vowelwoprosody[people]=pd.DataFrame([],index=Vowels_single,columns=Static_cols)
    for ph_wopros, values in sorted(Combine_formants_people_vowelwoprosody[people].items()):
        static_func=dynamic2statict_artic_formant(values.T)
        Df_formants_people_vowelwoprosody[people].loc[ph_wopros]=static_func
pickle.dump(Df_formants_people_vowelwoprosody,open(pklpath+"/Df_formants_people_vowelwoprosody.pkl","wb"))

Df_formants_people_vowelwoprosody01=Dict()
for people in sorted(Combine_formants_people_vowelwoprosody01.keys()):
    Df_formants_people_vowelwoprosody01[people]=pd.DataFrame([],index=Vowels_single,columns=Static_cols)
    for ph_wopros, values in sorted(Combine_formants_people_vowelwoprosody01[people].items()):
        static_func=dynamic2statict_artic_formant(values.T)
        Df_formants_people_vowelwoprosody01[people].loc[ph_wopros]=static_func
pickle.dump(Df_formants_people_vowelwoprosody01,open(pklpath+"/Df_formants_people_vowelwoprosody01.pkl","wb"))
# =============================================================================
'''

Prepare for EN regression code

feature dimensions are 
['F1+mean',
 'F1+std',
 'F1+skew',
 'F1+kurtosis',
 'F2+mean',
 'F2+std',
 'F2+skew',
 'F2+kurtosis']


'''

# =============================================================================

PhoneDefect_num=Dict()
for phone in Vowels_prosody_single:
    Session_formants_people_vowel=Dict()
    defect_num=0
    for people in sorted(Df_formants_people_vowel.keys()):
        Session_formants_people_vowel[phone][people]=Df_formants_people_vowel[people].loc[phone].fillna(-1).values
        if (Session_formants_people_vowel[phone][people] == np.array([-1]*len(Session_formants_people_vowel[phone][people]))).all():
            defect_num+=1
    
    outpath=pklpath+"/Session_formants_people_vowel/{}.pkl"
    if not os.path.exists(pklpath+"/Session_formants_people_vowel/"):
        os.makedirs(pklpath+"/Session_formants_people_vowel/")
    if defect_num < 57:
        pickle.dump(Session_formants_people_vowel[phone],open(outpath.format(phone),"wb"))
    PhoneDefect_num[phone]=defect_num
    
    ''' stop condition for inspectation '''
    if phone == 'ax5':
        ADOS_label=Label.label_raw['ADOS_C']
        NameMatchAssertion(Session_formants_people_vowel[phone],Label.label_raw['name'].values)
        df_formants_people_vowel=pd.DataFrame.from_dict(Session_formants_people_vowel[phone]).T
        df_formants_people_vowel.columns=Static_cols
        df_formants_people_vowel['ADOS']=ADOS_label.values
        outpath=base_path+"/excels/"
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        df_formants_people_vowel.to_excel(outpath+"/{}.xlsx".format(phone.replace(":","Hyphen")))

    
    
PhonewoprosDefect_num=Dict()
for ph_wopros in Vowels_single:
    Session_formants_people_vowelwoprosody=Dict()
    defect_num=0
    for people in sorted(Df_formants_people_vowelwoprosody.keys()):
        Session_formants_people_vowelwoprosody[ph_wopros][people]=Df_formants_people_vowelwoprosody[people].loc[ph_wopros].fillna(-1).values
        if (Session_formants_people_vowelwoprosody[ph_wopros][people] == np.array([-1]*len(Session_formants_people_vowelwoprosody[ph_wopros][people]))).all():
            defect_num+=1
    print("ph_wopros", ph_wopros, "defect_num", defect_num)
    outpath=pklpath+"/Session_formants_people_vowelwoprosody/{}.pkl"
    if not os.path.exists(pklpath+"/Session_formants_people_vowelwoprosody/"):
        os.makedirs(pklpath+"/Session_formants_people_vowelwoprosody/")
    if defect_num < 57:
        pickle.dump(Session_formants_people_vowelwoprosody[ph_wopros],open(outpath.format(ph_wopros),"wb"))
    PhonewoprosDefect_num[ph_wopros]=defect_num
    
    ''' stop condition for inspectation '''
    if ph_wopros == 'O_':
        NameMatchAssertion(Session_formants_people_vowelwoprosody[ph_wopros],Label.label_raw['name'].values)
        ADOS_label=Label.label_raw['ADOS_C']
        NameMatchAssertion(Session_formants_people_vowelwoprosody[ph_wopros],Label.label_raw['name'].values)
        df_formants_people_vowel=pd.DataFrame.from_dict(Session_formants_people_vowelwoprosody[ph_wopros]).T
        df_formants_people_vowel.columns=Static_cols
        df_formants_people_vowel['ADOS']=ADOS_label.values
        outpath=base_path+"/excels/"
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        df_formants_people_vowel.to_excel(outpath+"/{}.xlsx".format(ph_wopros.replace(":","Hyphen")))


Phonewopros01Defect_num=Dict()
for ph_wopros in Vowels_single:
    Session_formants_people_vowelwoprosody01=Dict()
    defect_num=0
    for people in sorted(Df_formants_people_vowelwoprosody01.keys()):
        Session_formants_people_vowelwoprosody01[ph_wopros][people]=Df_formants_people_vowelwoprosody01[people].loc[ph_wopros].fillna(-1).values
        if (Session_formants_people_vowelwoprosody01[ph_wopros][people] == np.array([-1]*len(Session_formants_people_vowelwoprosody01[ph_wopros][people]))).all():
            defect_num+=1
    print("ph_wopros", ph_wopros, "defect_num", defect_num)
    outpath=pklpath+"/Session_formants_people_vowelwoprosody01/{}.pkl"
    if not os.path.exists(pklpath+"/Session_formants_people_vowelwoprosody01/"):
        os.makedirs(pklpath+"/Session_formants_people_vowelwoprosody01/")
    if defect_num < 57:
        pickle.dump(Session_formants_people_vowelwoprosody01[ph_wopros],open(outpath.format(ph_wopros),"wb"))
    Phonewopros01Defect_num[ph_wopros]=defect_num