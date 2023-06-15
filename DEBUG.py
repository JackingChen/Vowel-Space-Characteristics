#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 20:26:33 2023

@author: jack
"""

import pickle


def read_pkl(path):
    # 指定pickle檔案的路徑
    pickle_file_path = path
    
    try:
        # 使用二進位模式打開pickle檔案
        with open(pickle_file_path, 'rb') as file:
            # 使用pickle模組的load函式來載入pickle檔案中的資料
            data = pickle.load(file)
            # 在這裡可以對載入的資料進行適當的處理或使用
    
            # 範例：列印載入的資料
            print(data)
    
    except FileNotFoundError:
        print("找不到指定的pickle檔案。")
    except pickle.UnpicklingError:
        print("無法解析pickle檔案中的資料。")
    except Exception as e:
        print("發生錯誤：", str(e))
    return data



# cmp1
# Feature_old_path='/media/jack/workspace/DisVoice/Features_BeforeTASLPreview120230607/artuculation_AUI/Interaction/Phonation/Syncrony_measure_of_variance_phonation_ASD_DOCKID.pkl'
# Feature_old=read_pkl(Feature_old_path)

# Feature_new_path='/media/jack/workspace/DisVoice/Features/artuculation_AUI/Interaction/Phonation/Syncrony_measure_of_variance_phonation_ASD_DOCKID.pkl'
# Feature_new=read_pkl(Feature_new_path)

Feature_old_path='/media/jack/workspace/DisVoice/Features_BeforeTASLPreview120230607/artuculation_AUI/Interaction/Formants/Syncrony_measure_of_variance_DKIndividual_ASD_DOCKID.pkl'
Feature_old=read_pkl(Feature_old_path)

Feature_new_path='/media/jack/workspace/DisVoice/Features/artuculation_AUI/Interaction/Formants/proposed/Syncrony_measure_of_variance_DKIndividual_ASD_DOCKID.pkl'
Feature_new=read_pkl(Feature_new_path)


# Feature_old_path='/media/jack/workspace/DisVoice/Session_level_all_class_old.pkl'
# Feature_old=read_pkl(Feature_old_path)

# Feature_new_path='/media/jack/workspace/DisVoice/Session_level_all_class_new.pkl'
# Feature_new=read_pkl(Feature_new_path)

import numpy as np
import pandas as pd
def criterion_filter(df_formant_statistic,N=10,\
                      constrain_sex=-1, constrain_module=-1,constrain_agemax=-1,constrain_ADOScate=-1,constrain_agemin=-1,\
                      evictNamelst=[],feature_type='Session_formant'):
    if feature_type == 'Session_formant':
        filter_bool=np.logical_and(df_formant_statistic['u_num']>N,df_formant_statistic['a_num']>N)
        filter_bool=np.logical_and(filter_bool,df_formant_statistic['i_num']>N)
    elif feature_type == 'Syncrony_formant':
        filter_bool=df_formant_statistic['timeSeries_len']>N
    else:
        filter_bool=pd.Series([True]*len(df_formant_statistic),index=df_formant_statistic.index)
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
        
    if len(evictNamelst)>0:
        for name in evictNamelst:
            filter_bool.loc[name]=False
    # get rid of nan values
    filter_bool=np.logical_and(filter_bool,~df_formant_statistic.isna().T.any())
    return df_formant_statistic[filter_bool]


df_syncrony_measurement=Feature_new
timeSeries_len_columns=[col  for col in df_syncrony_measurement.columns if 'timeSeries_len' in col]
df_syncrony_measurement['timeSeries_len']=df_syncrony_measurement[timeSeries_len_columns].min(axis=1)


feat_type='Syncrony_formant'
N=4
df_syncrony_measurement=criterion_filter(df_syncrony_measurement,N=N,evictNamelst=[],feature_type=feat_type)

assert all(item not in df_syncrony_measurement.index for item in ['2015_12_07_02_003', '2017_03_18_01_196_1'])

print(df_syncrony_measurement)
list1= list(set(df_syncrony_measurement.index) - set(Feature_old.index))
list2= list(set(Feature_old.index) - set(df_syncrony_measurement.index))
diff_list=list1+list2
print(diff_list)
# set(Feature_new['TD vs df_feature_high_CSS >> Phonation_Convergence_cols::ASDTD']['X'].index) - set(Feature_old['TD vs df_feature_high_CSS >> Phonation_Convergence_cols::ASDTD']['X'].index)


# {'2015_12_07_02_003', '2017_03_18_01_196_1'}
# cmp2
# Feature_old_path='/media/jack/workspace/DisVoice/Session_level_all_class_old.pkl'
# Feature_old=read_pkl(Feature_old_path)

# Feature_new_path='/media/jack/workspace/DisVoice/Session_level_all_class_new.pkl'
# Feature_new=read_pkl(Feature_new_path)

# cmp3
# Feature_old_path='/media/jack/workspace/DisVoice/Features_BeforeTASLPreview120230607/artuculation_AUI/Interaction/Phonation/Syncrony_measure_of_variance_phonation_ASD_DOCKID.pkl'
# Feature_old=read_pkl(Feature_old_path)

# Feature_new_path='/media/jack/workspace/DisVoice/Features/artuculation_AUI/Interaction/Phonation/Syncrony_measure_of_variance_phonation_ASD_DOCKID.pkl'
# Feature_new=read_pkl(Feature_new_path)

# cmp4
# Feature_old_path='/media/jack/workspace/DisVoice/Features/artuculation_AUI/Vowels/Formants/proposed/Formant_AUI_tVSAFCRFvals_KID_FromASD_DOCKID.pkl'
# Feature_old=read_pkl(Feature_old_path)

# Feature_new_path='/media/jack/workspace/DisVoice/Features_BeforeTASLPreview120230607/artuculation_AUI/Vowels/Formants/Formant_AUI_tVSAFCRFvals_KID_FromASD_DOCKID.pkl'
# Feature_new=read_pkl(Feature_new_path)