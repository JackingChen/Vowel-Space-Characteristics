
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa
modified by Jackchen 20210524

This script is to extract utterance level phonation feature 
    input: filepath
    
    or user can use precomputed checkpoints

"""

from scipy.io.wavfile import read
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import math
import pysptk
try:
    from .phonation_functions import jitter_env, logEnergy, shimmer_env, APQ, PPQ
except:
    from phonation.phonation_functions import jitter_env, logEnergy, shimmer_env, APQ, PPQ
import scipy.stats as st
import uuid
import pandas as pd
from scipy.stats import spearmanr,pearsonr 

import torch
from tqdm import tqdm
from addict import Dict
import glob 
import argparse
import pickle
from scipy import stats
from pydub import AudioSegment
# from phonation.phonation import Phonation_jack as Phonation
from phonation.phonation import Phonation_disvoice as Phonation
from prosody.prosody import Prosody_jack as Prosody

import parselmouth 
from parselmouth.praat import call
import statistics
import inspect
from multiprocessing import Pool, current_process
from metric import Evaluation_method 
from articulation.HYPERPARAM import phonewoprosody, Label
import matplotlib.pyplot as plt



def Add_label(df_formant_statistic,Label,label_choose='ADOS_cate_C'):
    for people in df_formant_statistic.index:
        bool_ind=Label.label_raw['name']==people
        df_formant_statistic.loc[people,label_choose]=Label.label_raw.loc[bool_ind,label_choose].values
    return df_formant_statistic

def criterion_filter(df_formant_statistic,N=10,\
                     constrain_sex=-1, constrain_module=-1,constrain_agemax=-1,constrain_ADOScate=-1,constrain_agemin=-1,\
                     evictNamelst=[]):
    # filter by number of phones
    
    
    
    filter_bool=np.logical_and(df_formant_statistic['u_num']>N,df_formant_statistic['a_num']>N)
    filter_bool=np.logical_and(filter_bool,df_formant_statistic['i_num']>N)
    
    # filer by other biological information
    if constrain_sex != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['sex']==constrain_sex)
    if constrain_module != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['Module']==constrain_module)
    if constrain_agemax != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['age']<=constrain_agemax)
    if constrain_agemin != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['age']>=constrain_agemin)
    if constrain_ADOScate != -1:
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['ADOS_cate_C']==constrain_ADOScate)
    
    # filter the names given the name list
    if len(evictNamelst)>0:
        for name in evictNamelst:
            filter_bool.loc[name]=False
    
    # print("filter bool")
    # print(filter_bool)
    # print("df_formant_statistic")
    # print(~df_formant_statistic.isna().T.any())
    # get rid of nan values
    filter_bool=np.logical_and(filter_bool,~df_formant_statistic.isna().T.any())
    return df_formant_statistic[filter_bool]


def measurePitch(voiceID, f0min, f0max, unit):
    columns=['duration', 'meanF0', 'stdevF0', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'aqpq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    sound = parselmouth.Sound(voiceID) # read the sound
    duration = call(sound, "Get total duration") # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    df=pd.DataFrame(np.array([duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer]),index=columns)
    return df.T


def TTest_Cmp_checkpoint(chkptpath_1,chkptpath_2):
    df_phonation_1=pickle.load(open(chkptpath_1,"rb"))
    df_phonation_2=pickle.load(open(chkptpath_2,"rb"))
    name1=os.path.basename(chkptpath_1).replace(".pkl","")
    name2=os.path.basename(chkptpath_2).replace(".pkl","")
    
    dataset_str='{name1}vs{name2}'.format(name1=name1,name2=name2)
    df_ttest_result=pd.DataFrame()
    for col in df_phonation_1.columns:
        df_ttest_result.loc[dataset_str+"-p-val",col]=stats.ttest_ind(df_phonation_1[col].dropna(),df_phonation_2[col].dropna())[1].astype(float)
        df_ttest_result.loc['{0}-{1}'.format(name1,name2),col] = df_phonation_1[col].dropna().mean() - df_phonation_2[col].dropna().mean()
    df_ttest_result=df_ttest_result.T 
    
    result_path="RESULTS/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    df_ttest_result.to_excel(result_path+dataset_str+".xlsx")
    return df_ttest_result
    
def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--base_path', default='/homes/ssd1/jackchen/DisVoice',
                        help='path of the base directory', dest='base_path')
    parser.add_argument('--filepath', default='data/Segmented_ADOS_normalized',
                        help='data/{Segmented_ADOS_TD_normalized_untranscripted|Segmented_ADOS_normalized}')
    parser.add_argument('--inpklpath', default='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',
                        help='path of the base directory')
    parser.add_argument('--checkpointpath', default='/homes/ssd1/jackchen/DisVoice/phonation/features',
                        help='path of the base directory')
    parser.add_argument('--check', default=False,
                        help='path of the base directory')
    parser.add_argument('--Plot', default=True,
                        help='')
    parser.add_argument('--checkreliability', default=False,
                            help='path of the base directory')
    parser.add_argument('--method', default='Disvoice_prosody_energy',
                            help='Disvoice_phonation|Disvoice_prosody_energy|praat')
    args = parser.parse_args()
    return args

args = get_args()

base_path=args.base_path
filepath=args.filepath
pklpath=args.inpklpath
path_app = base_path
method=args.method
sys.path.append(path_app)
checkpointpath_manual=args.checkpointpath
PhonationPath=base_path + "/phonation/features" 


import praat.praat_functions as praat_functions
from utils_jack import *
# =============================================================================
'''

    Manual area

'''
F0_parameter_dict=Dict()
F0_parameter_dict['male']={'f0_min':75,'f0_max':400}
F0_parameter_dict['female']={'f0_min':75,'f0_max':550}

Formants_utt_symb=Dict()
Formants_people_symb=Dict()
# =============================================================================
audiopath = os.path.join(base_path,filepath)
dataset_name=os.path.basename(audiopath)
dataset_name=dataset_name[dataset_name.find('ADOS'):dataset_name.find('normalized')-1]

files=glob.glob(audiopath+"/*.wav")


# if os.path.exists('Gen_formant_multiprocess.log'):
#     os.remove('Gen_formant_multiprocess.log')

''' multiprocessin area   done'''

def Get_phonationdictbag_map(files,Info_name_sex):
    print(" process PID", os.getpid(), " running")
    Phonation_dict_bag=Dict()
    Skipped_audio_file=[]
    for file in tqdm(files):
        name='_'.join(os.path.basename(file).replace(".wav","").split("_")[:-1])

        audiofile=file
        gender_query_str='_'.join(name.split("_")[:-1])
        role=name.split("_")[-1]
        if role =='D':
            gender='female'
        elif role =='K':
            gender=Info_name_sex[Info_name_sex['name']==gender_query_str]['sex'].values[0]
        
        minf0=F0_parameter_dict[gender]['f0_min']
        maxf0=F0_parameter_dict[gender]['f0_max']
        
        if method == "Disvoice_phonation":
            phonation_extractor=Phonation(maxf0=maxf0, minf0=minf0)
            # phonation_extractor._update_pitch_method('praat')
            # static = True : static feature, False = dynamic feature
            try:
                df_feat_utt=phonation_extractor.extract_features_file(audiofile,fmt="dataframe", static=True)
            except IndexError: 
                print("Skipped audio file: ", audiofile)
                Skipped_audio_file.append(audiofile)
                continue
        elif method == "Disvoice_prosody_energy":
            prosody_extractor=Prosody(maxf0=maxf0, minf0=minf0)
            # static = True : static feature, False = dynamic feature
            try:
                df_feat_utt=prosody_extractor.extract_features_file(audiofile,fmt="dataframe" , static=True)
            except IndexError: 
                print("Skipped audio file: ", audiofile)
                Skipped_audio_file.append(audiofile)
                continue
        elif method == "praat":
            df_feat_utt=measurePitch(audiofile, minf0, maxf0, "Hertz")
        
        
        if name not in Phonation_dict_bag.keys():
            Phonation_dict_bag[name]=pd.DataFrame()
        Phonation_dict_bag[name]=Phonation_dict_bag[name].append(df_feat_utt)
    # tmpPath=PhonationPath + "/features"
    # if not os.path.exists(tmpPath):
    #     os.makedirs(tmpPath)
    print("PID {} Getting ".format(os.getpid()), "Done")
    print("Total skipped audio files :", Skipped_audio_file)
    return Phonation_dict_bag

interval=20
pool = Pool(int(os.cpu_count()))


# DEBUG: for specific person: 2015_12_06_01_097_K
# file_097=[]
# for file in files:
#     if '2015_12_06_01_097_K' in file:
#         file_097.append(file)
# gender=Info_name_sex[Info_name_sex['name']=='2015_12_06_01_097']['sex'].values[0]
# minf0=F0_parameter_dict[gender]['f0_min']
# maxf0=F0_parameter_dict[gender]['f0_max']
# DF_dict=Dict()
# for f_097 in file_097:
#     phonation_extractor=Phonation(maxf0=maxf0, minf0=minf0)
#     phonation_extractor._update_pitch_method('praat')
#     try:
#         df_feat_utt=phonation_extractor.extract_features_file(f_097,fmt="dataframe" , static=True)
#     except IndexError:
#         continue
    
#     DF_dict[os.path.basename(f_097)]=df_feat_utt



keys=[]
for i in range(0,len(files),interval):
    # print(list(combs_tup.keys())[i:i+interval])
    keys.append(files[i:i+interval])
flat_keys=[item for sublist in keys for item in sublist]
assert len(flat_keys) == len(files)

final_result = pool.starmap(Get_phonationdictbag_map, [([file_block,Info_name_sex]) \
                                                  for file_block in tqdm(keys)])


Phonation_utt_symb=Dict()
for fin_resut_dict in final_result:    
    for keys, values in fin_resut_dict.items():
        if keys not in Phonation_utt_symb.keys():
            Phonation_utt_symb[keys]=pd.DataFrame()
        # if '2015_12_06_01_097_K' in keys:
        #     print(values)
        Phonation_utt_symb[keys]=Phonation_utt_symb[keys].append(values)
            

if args.check:
    chkptpath_cmp=PhonationPath+"/Phonation_dict_bag_{}_cmp.pkl".format(dataset_name)
    Phonation_utt_symb_cmp=pickle.load(open(chkptpath_cmp,"rb"))
    for keys, values in Phonation_utt_symb.items():
        assert ((Phonation_utt_symb_cmp[keys].dropna() - values.dropna()).values == np.zeros(Phonation_utt_symb_cmp[keys].dropna().values.shape)).all()

chkptpath=PhonationPath+"/Phonation_dict_bag_{}.pkl".format(dataset_name)
pickle.dump(Phonation_utt_symb,open(chkptpath,"wb"))

''' multiprocessin area  '''


'''
    Phonation_dict_bag[utt] -> df_feat_utt: columns= 28 dim phonation features
    process: input: audio_files
'''

chkptpath=PhonationPath+"/Phonation_dict_bag_{}.pkl".format(dataset_name)

if not os.path.exists(chkptpath):
    Phonation_dict_bag=Dict()
    for file in tqdm(files):
        name='_'.join(os.path.basename(file).replace(".wav","").split("_")[:-1])
        audiofile=file
        gender_query_str='_'.join(name.split("_")[:-1])
        role=name.split("_")[-1]
        if role =='D':
            gender='female'
        elif role =='K':
            series_gend=Info_name_sex[Info_name_sex['name']==gender_query_str]['sex']
            gender=series_gend.values[0]
        
        minf0=F0_parameter_dict[gender]['f0_min']
        maxf0=F0_parameter_dict[gender]['f0_max']
        
        if method == "Disvoice":
            phonation_extractor=Phonation(maxf0=maxf0, minf0=minf0)
            df_feat_utt=phonation_extractor.extract_features_file(audiofile)
        elif method == "praat":
            df_feat_utt=measurePitch(audiofile, minf0, maxf0, "Hertz")
        
        
        if name not in Phonation_dict_bag.keys():
            Phonation_dict_bag[name]=pd.DataFrame()
        Phonation_dict_bag[name]=Phonation_dict_bag[name].append(df_feat_utt)
    tmpPath=PhonationPath + "/features"
    if not os.path.exists(tmpPath):
        os.makedirs(tmpPath)
    pickle.dump(Phonation_dict_bag,open(chkptpath,"wb"))
else:
    Phonation_dict_bag=pickle.load(open(chkptpath,"rb"))


'''
    df_phonation_{role}[utt], role={doc|kid} 
    process: input: audio_files
    
    這邊只是把所有utterence level的feature做一個mean的動作
    
'''

chkptpath_kid=PhonationPath+"/df_phonation_kid_{}.pkl".format(dataset_name)
chkptpath_doc=PhonationPath+"/df_phonation_doc_{}.pkl".format(dataset_name)

Phonation_role_dict=Dict()
for keys, values in Phonation_dict_bag.items():
    if '_K' in keys:
        Phonation_role_dict['K'][keys.replace("_K","")]=values.mean(axis=0)
    elif '_D' in keys:
        Phonation_role_dict['D'][keys.replace("_D","")]=values.mean(axis=0)

df_phonation_kid=pd.DataFrame.from_dict(Phonation_role_dict['K']).T
df_phonation_doc=pd.DataFrame.from_dict(Phonation_role_dict['D']).T


# Remove NaN person
def FilterNan_criteria(df_input):
    df_=df_input[~df_input.isna().any(axis=1)]
    return df_

df_phonation_kid=FilterNan_criteria(df_phonation_kid)
df_phonation_doc=FilterNan_criteria(df_phonation_doc)

pickle.dump(df_phonation_kid,open(chkptpath_kid,"wb"))
pickle.dump(df_phonation_doc,open(chkptpath_doc,"wb"))


chkptpath_kid_TD=args.checkpointpath + "/df_phonation_kid_ADOS_TD.pkl"
chkptpath_kid_ASD=args.checkpointpath + "/df_phonation_kid_ADOS.pkl"
chkptpath_doc_ASD=args.checkpointpath + "/df_phonation_doc_ADOS.pkl"
chkptpath_doc_TD=args.checkpointpath + "/df_phonation_doc_ADOS_TD.pkl"


# =============================================================================
'''

    t-test area

'''
# =============================================================================

# df_ttest_result_a=TTest_Cmp_checkpoint(chkptpath_doc_ASD,chkptpath_doc_TD)
# df_ttest_result_b=TTest_Cmp_checkpoint(chkptpath_kid_ASD,chkptpath_kid_TD)

# df_ttest_result_c=TTest_Cmp_checkpoint(chkptpath_doc_ASD,chkptpath_kid_ASD)
# df_ttest_result_d=TTest_Cmp_checkpoint(chkptpath_doc_TD,chkptpath_kid_TD)




# =============================================================================
''' 

    Workspace start from this section

'''
# =============================================================================

''' Add formant information to df '''
def TBMEB1Preparation_LoadForFromOtherData(dfFormantStatisticpath,\
                                           prefix='Formant_AUI_tVSAFCRFvals',\
                                           suffix='KID_FromASD_DOCKID'):
    '''
        
        We generate data for nested cross-valated analysis in Table.5 in TBME2021
        
        The data will be stored at Pickles/Session_formants_people_vowel_feat
    
    '''
    dfFormantStatisticFractionpath=dfFormantStatisticpath+'/Session_formants_people_vowel_feat'
    if not os.path.exists(dfFormantStatisticFractionpath):
        raise FileExistsError('Directory not exist')
    df_phonation_statistic_77=pickle.load(open(dfFormantStatisticFractionpath+'/{prefix}_{suffix}.pkl'.format(\
                                                                         prefix=prefix,suffix=suffix),'rb'))
    return df_phonation_statistic_77

def TBMEB1Preparation_SaveForClassifyData(dfpath,\
                        df_SegLvl_features,suffix=''):
    '''
        
        We generate data for nested cross-valated analysis in Table.5 in TBME2021
        
        The data will be stored at Pickles/Session_formants_people_vowel_feat
    
    '''
    dfFormantStatisticFractionpath= dfpath + 'Session_formants_people_vowel_feat'
    if not os.path.exists(dfFormantStatisticFractionpath):
        os.makedirs(dfFormantStatisticFractionpath)
    pickle.dump(df_SegLvl_features,open(dfFormantStatisticFractionpath+'/df_SegLvl_features_{}.pkl'.format(suffix),'wb'))

# TBMEB1Preparation_SaveForClassifyData('Pickles/',df_phonation_kid,suffix=method)

df_formant_statistic_77=TBMEB1Preparation_LoadForFromOtherData(pklpath,prefix='Formant_AUI_tVSAFCRFvals',\
                                           suffix='KID_FromASD_DOCKID')
df_disvoice_prosody_energy=TBMEB1Preparation_LoadForFromOtherData('Pickles/',\
                                           prefix='df_SegLvl_features',\
                                           suffix='Disvoice_prosody_energy')
df_disvoice_phonation=TBMEB1Preparation_LoadForFromOtherData('Pickles/',\
                                                            prefix='df_SegLvl_features',\
                                                            suffix='Disvoice_phonation')
df_disvoice_prosodyF0=TBMEB1Preparation_LoadForFromOtherData('Pickles/',\
                                                            prefix='df_SegLvl_features',\
                                                            suffix='prosodyF0')    

# N=2
# Eval_med=Evaluation_method()
# Aaadf_spearmanr_table_prosody_energy=Eval_med.Calculate_correlation(label_correlation_choose_lst,\
#                                             pd.merge(df_disvoice_prosody_energy,df_formant_statistic_77,left_index=True, right_index=True),\
#                                             N,df_disvoice_prosody_energy.columns,constrain_sex=-1, constrain_module=-1,feature_type='Session_formant')
# Aaadf_spearmanr_table_phonation=Eval_med.Calculate_correlation(label_correlation_choose_lst,\
#                                             pd.merge(df_disvoice_phonation,df_formant_statistic_77,left_index=True, right_index=True),\
#                                             N,df_disvoice_phonation.columns,constrain_sex=-1, constrain_module=-1,feature_type='Session_formant')

    



df_phonation_kid_added=pd.merge(df_phonation_kid,df_formant_statistic_77,left_index=True, right_index=True).sort_index()
df_phonation_kid_added=df_phonation_kid_added.loc[:,~df_phonation_kid_added.columns.duplicated()].sort_index()

# df_phonation_kid_added=pd.merge(df_phonation_kid,df_formant_statistic_77,left_index=True, right_index=True)
# df_phonation_kid_added=df_phonation_kid_added.loc[:,~df_phonation_kid_added.columns.duplicated()]



# df_kid_ManualComb=pd.merge(df_formant_statistic_77,df_disvoice_phonation,left_index=True, right_index=True)
df_kid_ManualComb=pd.merge(df_disvoice_phonation,df_disvoice_prosody_energy,left_index=True, right_index=True)
df_kid_ManualComb=pd.merge(df_kid_ManualComb,df_disvoice_prosodyF0,left_index=True, right_index=True)
df_kid_ManualComb=pd.merge(df_kid_ManualComb,df_formant_statistic_77,left_index=True, right_index=True)
# df_kid_ManualComb=pd.merge(df_kid_ManualComb,df_disvoice_phonation,left_index=True, right_index=True)
# df_kid_ManualComb=df_kid_ManualComb.loc[:,~df_kid_ManualComb.columns.duplicated()]
df_kid_ManualComb=df_kid_ManualComb.loc[:,~df_kid_ManualComb.columns.duplicated()].sort_index()


# pickle.dump(df_kid_ManualComb,open('Pickles/df_SegLvl_features_prosodyF0PhonationEnergyLOC.pkl','wb'))
# =============================================================================
# 
# =============================================================================


additional_columns=df_formant_statistic_77.columns


# df_phonation_kid=Add_label(df_phonation_kid,Label,label_choose='ADOS_C')

columns=list(set(df_phonation_kid.columns) - set(additional_columns)) # Exclude added labels
# columns=columns+list(df_formant_statistic_77.columns)

label_correlation_choose_lst=['ADOS_C']
N=0
Eval_med=Evaluation_method()
Aaadf_spearmanr_table_NoLimit=Eval_med.Calculate_correlation(label_correlation_choose_lst,df_kid_ManualComb,N,columns,constrain_sex=-1, constrain_module=-1,feature_type='Session_formant')

# =============================================================================
''' 
    Multi feature prediction 

'''
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

''' cross validation prediction '''
# feature_chos_lst=['between_covariance_norm(A:,i:,u:)',
# 'sam_wilks_lin_norm(A:,i:,u:)',
# 'hotelling_lin_norm(A:,i:,u:)',
# 'pillai_lin_norm(A:,i:,u:)']

# feature_chos_lst_top=['between_covariance_norm(A:,i:,u:)']
# feature_chos_lst_top=['between_variance_norm(A:,i:,u:)']

# feature_chos_lst_top=['roys_root_lin_norm(A:,i:,u:)', 'Angles']
# feature_chos_lst_top=['Between_Within_Det_ratio_norm(A:,i:,u:)']
# feature_chos_lst=['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ai']
# feature_chos_lst=['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ua']
# feature_chos_lst=['pillai_lin_norm(A:,i:,u:)']
# feature_chos_lst=['ang_ai']
# feature_chos_lst=['ang_ua']
# feature_chos_lst=['FCR2']
# feature_chos_lst=['FCR2','ang_ai']
# feature_chos_lst=['FCR2','ang_ua']
# feature_chos_lst=['FCR2','ang_ai','ang_ua']
# feature_chos_lst_top=['between_covariance_norm(A:,i:,u:)','localabsoluteJitter_mean(A:,i:,u:)','dcorr_12']
# feature_chos_lst_top=['between_covariance_norm(A:,i:,u:)','localabsoluteJitter_mean(A:,i:,u:)']
# feature_chos_lst_top=list(df_phonation_kid.columns) 
# LOC_columns=[ 'between_covariance_norm(A:,i:,u:)',


# feature_chos_lst_top=list(df_phonation_kid.columns)
# feature_chos_lst_top=list(df_phonation_kid.columns) + LOC_columns
# feature_chos_lst_top=LOC_columns
# feature_chos_lst_top=list(df_disvoice_prosodyF0.columns)
# feature_chos_lst_top=list(df_disvoice_prosodyF0.columns) + LOC_columns
# feature_chos_lst_top=list(df_disvoice_phonation.columns)
# feature_chos_lst_top=list(df_disvoice_phonation.columns) + LOC_columns
# feature_chos_lst_top=list(df_disvoice_prosody_energy.columns) 
# feature_chos_lst_top=[
#     'avgEvoiced', 'stdEvoiced', 'skwEvoiced', 'kurtosisEvoiced',
#         'avgtiltEvoiced', 'stdtiltEvoiced', 'skwtiltEvoiced',
#         'kurtosistiltEvoiced', 'avgmseEvoiced', 'stdmseEvoiced',
#         'skwmseEvoiced', 'kurtosismseEvoiced', 
#         ]
New_prosodyF0=[
    'F0avg',
 'F0std',
 'F0max',
 'F0min',
 'F0skew',
 'F0kurt',
 'F0tiltavg',
 'F0mseavg',
 'F0tiltstd',
 'F0msestd',
 'F0tiltmax',
 'F0msemax',
 'F0tiltmin',
 'F0msemin',
 'F0tiltskw',
 'F0mseskw',
 'F0tiltku',
 'F0mseku',
    ]


New_VoiceQuality = ['avg Jitter',
 'avg Shimmer',
 # 'avg apq',
 # 'avg ppq',
 # 'avg logE',
 'std Jitter',
 'std Shimmer',
 # 'std apq',
 # 'std ppq',
 # 'std logE',
 'skewness Jitter',
 'skewness Shimmer',
 # 'skewness apq',
 # 'skewness ppq',
 # 'skewness logE',
 'kurtosis Jitter',
 'kurtosis Shimmer',
 # 'kurtosis apq',
 # 'kurtosis ppq',
 # 'kurtosis logE'
 ]

New_energy = [
    'avgEvoiced', 'stdEvoiced', 'skwEvoiced', 'kurtosisEvoiced',
        'avgtiltEvoiced', 'stdtiltEvoiced', 'skwtiltEvoiced',
        'kurtosistiltEvoiced', 'avgmseEvoiced', 'stdmseEvoiced',
        'skwmseEvoiced', 'kurtosismseEvoiced', 
        ]


# =============================================================================
# 
# =============================================================================
LOC_columns=[ 'between_covariance_norm(A:,i:,u:)',
        'between_variance_norm(A:,i:,u:)',
        'total_covariance_norm(A:,i:,u:)',
        'total_variance_norm(A:,i:,u:)', 
        'sam_wilks_lin_norm(A:,i:,u:)',
        'pillai_lin_norm(A:,i:,u:)', 
        'hotelling_lin_norm(A:,i:,u:)',
        'roys_root_lin_norm(A:,i:,u:)',
        'Between_Within_Det_ratio_norm(A:,i:,u:)',
        'Between_Within_Tr_ratio_norm(A:,i:,u:)',
       ]
DEP_columns=[
    'pear_12',
    'spear_12',
    'kendall_12',
    'dcorr_12'
    ]

# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)','Between_Within_Det_ratio_norm(A:,i:,u:)']]
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','dcorr_12']]
# X = df_formant_statistic[[ 'pillai_lin_norm(A:,i:,u:)','dcorr_12']]
# X = df_formant_statistic[['dcorr_12','dcov_12']]
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ai']]
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ua']] 
# X = df_formant_statistic[['between_covariance_norm(A:,i:,u:)', 'pillai_lin_norm(A:,i:,u:)','ang_ai','ang_ua']] 
# X = df_formant_statistic[['FCR2']] 
# X = df_kid_ManualComb[feature_chos_lst_top]
# y = df_kid_ManualComb[lab_chos_lst]
# ## fit a OLS model with intercept on TV and Radio
# X = sm.add_constant(X)
# est = sm.OLS(y, X).fit()
# est.summary()



# feature_chos_lst_top=[
#     'avgEvoiced', 'stdEvoiced', 'skwEvoiced', 'kurtosisEvoiced',
#         'avgtiltEvoiced', 'stdtiltEvoiced', 'skwtiltEvoiced',
#         'kurtosistiltEvoiced', 'avgmseEvoiced', 'stdmseEvoiced',
#         'skwmseEvoiced', 'kurtosismseEvoiced', 
#         # 'avg1Evoiced', 'std1Evoiced',
#         # 'max1Evoiced', 'min1Evoiced', 'skw1Evoiced', 'kurtosis1Evoiced',
#         # 'avglastEvoiced', 'stdlastEvoiced', 'maxlastEvoiced', 'minlastEvoiced',
#         # 'skwlastEvoiced', 'kurtosislastEvoiced'
# ]+ LOC_columns
# feature_chos_lst_top=list(df_disvoice_prosody_energy.columns)+ LOC_columns
# feature_chos_lst_top=list(df_disvoice_prosody_energy.columns) + list(df_disvoice_phonation.columns)
# feature_chos_lst_top=list(df_disvoice_prosody_energy.columns) + list(df_disvoice_phonation.columns) + LOC_columns
# feature_chos_lst_top=[
#     'avgEvoiced', 'stdEvoiced', 'skwEvoiced', 'kurtosisEvoiced',
#         'avgtiltEvoiced', 'stdtiltEvoiced', 'skwtiltEvoiced',
#         'kurtosistiltEvoiced', 'avgmseEvoiced', 'stdmseEvoiced',
#         'skwmseEvoiced', 'kurtosismseEvoiced', 
#         ] + list(df_disvoice_phonation.columns) + LOC_columns
# + LOC_columns
# 
# feature_chos_lst_top=list(df_disvoice_phonation.columns)


# feature_chos_lst_top=LOC_columns
lab_chos_lst=['ADOS_C']
# feature_chos_lst_top=['between_covariance_norm(A:,i:,u:)','dcorr_12']


# C_variable=np.array(np.arange(0.1,1.1,0.2))
epsilon=np.array(np.arange(0.1,1.5,0.1) )
# epsilon=np.array(np.arange(0.01,0.15,0.02))
# C_variable=np.array([0.001,0.01,10.0,50,100] + list(np.arange(0.1,1.5,0.1)))
# C_variable=np.array([0.001,0.01,10.0,50,100])
Classifier={}
loo=LeaveOneOut()
# CV_settings=loo
CV_settings=10
pca = PCA(n_components=1)

# =============================================================================
Classifier['SVR']={'model':sklearn.svm.SVR(),\
                  'parameters':{
                    'model__epsilon': epsilon,\
                    # 'model__C':C_variable,\
                    'model__kernel': ['rbf'],\
                    # 'gamma': ['auto'],\
                                }}
Classifier['EN']={'model':ElasticNet(random_state=0),\
                  'parameters':{'model__alpha':np.arange(0,1,0.25),\
                                'model__l1_ratio': np.arange(0,1,0.25)}} #Just a initial value will be changed by parameter tuning



    
clf=Classifier['SVR']
Comb=Dict()
Comb['LOC_columns']=LOC_columns
Comb['DEP_columns']=DEP_columns
Comb['LOC_columns+DEP_columns']=LOC_columns+DEP_columns

Comb['New_prosodyF0']=New_prosodyF0
Comb['New_prosodyF0+LOC_columns']=New_prosodyF0+LOC_columns
Comb['New_prosodyF0+DEP_columns']=New_prosodyF0+DEP_columns
Comb['New_prosodyF0+LOC_columns+DEP_columns']=New_prosodyF0+LOC_columns+DEP_columns

Comb['New_VoiceQuality']=New_VoiceQuality
Comb['New_VoiceQuality+LOC_columns']=New_VoiceQuality+LOC_columns
Comb['New_VoiceQuality+DEP_columns']=New_VoiceQuality+DEP_columns
Comb['New_VoiceQuality+LOC_columns+DEP_columns']=New_VoiceQuality+LOC_columns+DEP_columns

Comb['New_energy']=New_energy
Comb['New_energy+LOC_columns']=New_energy+LOC_columns 
Comb['New_energy+DEP_columns']=New_energy+DEP_columns 
Comb['New_energy+LOC_columns+DEP_columns']=New_energy+LOC_columns+DEP_columns

Comb['New_prosodyF0+New_energy']=New_prosodyF0+New_energy 
Comb['New_prosodyF0+New_energy+LOC_columns']=New_prosodyF0+New_energy+LOC_columns
Comb['New_prosodyF0+New_energy+DEP_columns']=New_prosodyF0+New_energy+DEP_columns
Comb['New_prosodyF0+New_energy+LOC_columns+DEP_columns']=New_prosodyF0+New_energy+LOC_columns+DEP_columns

Comb['New_VoiceQuality+New_energy']=New_VoiceQuality+New_energy 
Comb['New_VoiceQuality+New_energy+LOC_columns']=New_VoiceQuality+New_energy+LOC_columns
Comb['New_VoiceQuality+New_energy+DEP_columns']=New_VoiceQuality+New_energy+DEP_columns
Comb['New_VoiceQuality+New_energy+LOC_columns+DEP_columns']=New_VoiceQuality+New_energy+LOC_columns+DEP_columns

Comb['New_prosodyF0+New_VoiceQuality']=New_prosodyF0+New_VoiceQuality 
Comb['New_prosodyF0+New_VoiceQuality+LOC_columns']=New_prosodyF0+New_VoiceQuality+LOC_columns
Comb['New_prosodyF0+New_VoiceQuality+DEP_columns']=New_prosodyF0+New_VoiceQuality+DEP_columns
Comb['New_prosodyF0+New_VoiceQuality+LOC_columns+DEP_columns']=New_prosodyF0+New_VoiceQuality+LOC_columns+DEP_columns

Comb['New_prosodyF0+New_energy+New_VoiceQuality']=New_prosodyF0+New_energy+New_VoiceQuality
Comb['New_prosodyF0+New_energy+New_VoiceQuality+LOC_columns']=New_prosodyF0+New_energy+New_VoiceQuality+LOC_columns
Comb['New_prosodyF0+New_energy+New_VoiceQuality+DEP_columns']=New_prosodyF0+New_energy+New_VoiceQuality+DEP_columns
Comb['New_prosodyF0+New_energy+New_VoiceQuality+LOC_columns+DEP_columns']=New_prosodyF0+New_energy+New_VoiceQuality+ LOC_columns+ DEP_columns



 

# comb2 = combinations(feature_chos_lst_top, 2)
# comb3 = combinations(feature_chos_lst_top, 3)
# comb4 = combinations(feature_chos_lst_top, 4)
# combinations_lsts=list(comb2) + list(comb3)+ list(comb4)
# combinations_lsts=[feature_chos_lst_top]
# combinations_lsts=[comb0,comb1,comb2,comb3,comb4,comb5,comb6,comb7,\
#                    comb8,comb9,comb10,comb11,comb12,comb13]
combinations_lsts=[ Comb[k] for k in Comb.keys()]
combinations_keylsts=[ k for k in Comb.keys()]

RESULT_dict=Dict()
for key,feature_chos_tup in zip(combinations_keylsts,combinations_lsts):
    feature_chos_lst=list(feature_chos_tup)
    for feature_chooses in [feature_chos_lst]:
        pipe = Pipeline(steps=[('scalar',StandardScaler()),("model", clf['model'])])
        # pipe = Pipeline(steps=[ ("pca", pca), ("model", clf['model'])])
        p_grid=clf['parameters']

        Gclf = GridSearchCV(pipe, param_grid=p_grid, scoring='neg_mean_squared_error', cv=CV_settings, refit=True, n_jobs=-1)
        
        features=Dict()
        # features.X=df_formant_statistic[feature_chooses]
        # features.y=df_formant_statistic[lab_chos_lst]
        
        features.X=df_kid_ManualComb[feature_chooses]
        features.y=df_kid_ManualComb[lab_chos_lst]
        StandardScaler().fit_transform(features.X)
        
        # features.X=pd.merge(df_disvoice_phonation,df_formant_statistic_77,left_index=True, right_index=True)[feature_chooses]
        # features.y=pd.merge(df_disvoice_phonation,df_formant_statistic_77,left_index=True, right_index=True)[lab_chos_lst]
        # features.X=pd.concat([df_disvoice_phonation,df_formant_statistic_77],axis=1)[feature_chooses].dropna(axis=0)
        # features.y=pd.concat([df_disvoice_phonation,df_formant_statistic_77],axis=1)[lab_chos_lst].dropna(axis=0)
        
        # features.X=pd.concat([df_formant_statistic_77,df_disvoice_phonation],axis=1)[feature_chooses].dropna(axis=0)
        # features.y=pd.concat([df_formant_statistic_77,df_disvoice_phonation],axis=1)[lab_chos_lst].dropna(axis=0)
        
        
        # features.X=pd.merge(df_formant_statistic_77,df_disvoice_phonation,left_index=True, right_index=True)[feature_chooses]
        # features.y=pd.merge(df_formant_statistic_77,df_disvoice_phonation,left_index=True, right_index=True)[lab_chos_lst]
        
        # features.X=pd.merge(df_disvoice_phonation,df_formant_statistic_77)[feature_chooses]
        # features.X=pd.concat([df_disvoice_phonation,df_formant_statistic_77],axis=1)[feature_chooses]
        # features.y=pd.merge(df_disvoice_phonation,df_formant_statistic_77,left_index=True, right_index=True)[lab_chos_lst]
        
        # features.X=df_phonation_kid_added[feature_chooses]
        # features.y=df_phonation_kid_added[lab_chos_lst]

        
        # Gclf.fit(features.X, features.y)
        # print("The best score with scoring parameter: 'r2' is", Gclf.best_score_)
        # print("The best parameters are :", Gclf.best_params_)
        # Score=cross_val_score(Gclf, features.X, features.y, cv=10)
        CVpredict=cross_val_predict(Gclf, features.X, features.y.values.ravel(), cv=CV_settings)  
        r2=r2_score(features.y,CVpredict )
        n,p=features.X.shape
        r2_adj=1-(1-r2)*(n-1)/(n-p-1)
        MSE=sklearn.metrics.mean_squared_error(features.y.values.ravel(),CVpredict)
        pearson_result, pearson_p=pearsonr(features.y.values.ravel(),CVpredict )
        spear_result, spearman_p=spearmanr(features.y.values.ravel(),CVpredict )
        
        
        # feature_keys='+'.join(feature_chooses)
        feature_keys=key
        print('Feature {0}, MSE {1}, pearson_result {2} ,spear_result {3}'.format(feature_keys, MSE, pearson_result,spear_result))
        RESULT_dict[feature_keys]=[MSE,pearson_result,spear_result]


df_RESULT_list=pd.DataFrame.from_dict(RESULT_dict,orient='index')
df_RESULT_list.columns=['MSE','pear','spear']
print(df_RESULT_list)

Result_nice=Dict()
for measureOI in df_RESULT_list.columns:
    Result_nice[measureOI]=pd.DataFrame()
    for prosody in ['','New_prosodyF0', 'New_energy', 'New_VoiceQuality','New_prosodyF0 New_energy',\
                    'New_VoiceQuality New_energy','New_prosodyF0 New_VoiceQuality','New_prosodyF0 New_energy New_VoiceQuality']:
        for addition in ['','LOC_columns', 'DEP_columns','LOC_columns DEP_columns']:
            P=prosody.replace(" ","+")
            A=addition.replace(" ","+")
    
            if len(P)==0 and len(A)==0:
                continue
            elif len(A)==0:
                IDXquery='{0}'.format(P)
            elif len(P)==0:
                IDXquery='{0}'.format(A)
            else:
                IDXquery='{0}+{1}'.format(P,A)
            Result_nice[measureOI].loc[P,A]=df_RESULT_list.loc[IDXquery,measureOI]
            
        
            
del Gclf


# Aaa=pd.merge(df_disvoice_phonation,df_formant_statistic_77,left_index=True, right_index=True)[feature_chooses] -\
#     pd.merge(df_formant_statistic_77,df_disvoice_phonation,left_index=True, right_index=True)[feature_chooses]

aaa=ccc
