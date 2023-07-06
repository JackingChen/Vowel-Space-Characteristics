#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 22:31:32 2020

@author: jackchen
"""
import pandas as pd
import numpy as np
# =============================================================================
'''

    Label binary


'''
base_dir='/home/jack/Desktop/ADOS_label_workspace/'
# =============================================================================
label_Bin_all={}

label_path=base_dir+'ADOS_label20220525.xlsx'
label_raw=pd.read_excel(label_path)
Labelfile_TD='/media/jack/workspace/DisVoice/data/ADOS_TD_Label20220219.xlsx'
df_labels_TD=pd.read_excel(Labelfile_TD)
label_raw = pd.concat([label_raw, pd.DataFrame(df_labels_TD)], ignore_index=True)
# label_raw=label_raw.append(df_labels_TD)



#double check label and feature
#check_df=pd.DataFrame([],columns=['label','feature'])
#check_df['label2']=label_raw['name']
#check_df['label']=Session_level_tem_df.index
##check_df['label']=Session_level_tem_df.index
#check_df['feature']=Feature_spk_speed.index
#check_df['feature2']=pd.concat((Session_level_tem_df,Feature_spk_speed),axis=1)[:-1]

#label_choose=['BB1',	'BB2',	'BB3',	'BB4',	'BB5',\
#           'BB6',  	'BB7',	'BB8',	'BB9',	'BB10',\
#              'AA1',	'AA2',	'AA3',	'AA4',	'AA5',	'AA6',\
#             'AA7', 'AA8', 'AA9'
#]

#label_choose=['AA1',	'AA3',	'AA4',	'AA5',	'AA6',\
#             'AA7', 'AA8', 'AA9'] # labels are too biased
#label_choose=['AA5', 'AA8','BB4','BB5','BB6','BB7','BB9','BB10','ADOS_SC'] # labels are too biased
label_choose=['AA1','AA2', 'AA3', 'AA4','AA5','AA6','AA7', 'AA8', 'AA9','BB1','BB2','BB4','BB5','BB6','BB7','BB8','BB9','BB10','ADOS_C','ADOS_S','ADOS_SC','AD/ASD/normal'] # labels are too biased
# label_choose=['ADOS_C','ADOS_S','ADOS_SC']
indicators=["0.123"]
#include=['AA1_0.123',	'AA2_01.23',	'AA3_0.123',	'AA4_0.123',	\
#         'AA5_0.123',	'AA6_0.123',	'AA7_0.123',	'AA8_0.123',	'AA9_0.123',\
#         'BB1_0.123',	'BB10_0.123',	'BB2_0.123',	'BB4_0.123',\
#         'BB5_0.123',	'BB6_0.123',	'BB7_0.123',	'BB8_0.123',	'BB9_0.123']
include=['AA1_0.123',	'AA2_01.23',	'AA3_0.123',	'AA4_0.123',	\
         'AA5_0.123',	'AA6_0.123',	'AA7_0.123',	'AA8_0.123',	'AA9_0.123']
#for lab in label_choose:
#    for indicator in indicators:
for label_str  in include:
    lab=label_str.split("_")[0]
    indicator=label_str.split("_")[1]
    boundary_value=int(indicator.split(".")[0][-1])        
#    label_str='{0}_{1}'.format(lab,indicator)
    
    boundary_value=int(indicator.split(".")[0][-1])
    
    zeros_inst=label_raw[lab]<=boundary_value
    ones_inst=label_raw[lab]>boundary_value
    
    if len(np.where(zeros_inst == True)[0])<2 or len(np.where(ones_inst == True)[0])<2:
        continue
    else:
        label_Bin_all[label_str]=np.zeros(len(label_raw))
        label_Bin_all[label_str][label_raw[lab]<=boundary_value]=0
        label_Bin_all[label_str][label_raw[lab]>boundary_value]=1


'''Construct label full'''
label_full={}
for label_str  in label_choose:
    label_full[label_str]=np.array(label_raw[label_str])
    label_full['name']=label_raw['name']
    
    
''' ADD ADOS category '''
label_raw['ADOS_cate_C']=np.array([0]*len(label_raw))
label_raw.loc[label_raw['T_ADOS_C']<2,'ADOS_cate_C']=0
label_raw.loc[(label_raw['T_ADOS_C']<3) & (label_raw['T_ADOS_C']>=2),'ADOS_cate_C']=1
label_raw.loc[label_raw['T_ADOS_C']>=3,'ADOS_cate_C']=2

label_raw.loc[label_raw['T_ADOS_S']<4,'ADOS_cate_S']=0
label_raw.loc[(label_raw['T_ADOS_S']<6) & (label_raw['T_ADOS_S']>=4),'ADOS_cate_S']=1
label_raw.loc[label_raw['T_ADOS_S']>=6,'ADOS_cate_S']=2

label_raw.loc[label_raw['T_ADOS_SC']<7,'ADOS_cate_SC']=0
label_raw.loc[(label_raw['T_ADOS_SC']<10) & (label_raw['T_ADOS_SC']>=7),'ADOS_cate_SC']=1
label_raw.loc[label_raw['T_ADOS_SC']>=10,'ADOS_cate_SC']=2

label_raw.loc[label_raw['ADOS_CSS']<3,'ADOS_cate_CSS']=0
label_raw.loc[(label_raw['ADOS_CSS']<5) & (label_raw['ADOS_CSS']>=3),'ADOS_cate_CSS']=1
label_raw.loc[(label_raw['ADOS_CSS']<8) & (label_raw['ADOS_CSS']>=5),'ADOS_cate_CSS']=2
label_raw.loc[label_raw['ADOS_CSS']>=8,'ADOS_cate_CSS']=3