#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:45:57 2022

@author: jackchen
"""
import os
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import special, stats
from addict import Dict
from Syncrony import Syncrony
import seaborn as sns
from pylab import text
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing
from matplotlib.offsetbox import AnchoredText
import pandas as pd
from articulation.HYPERPARAM import phonewoprosody, Label
from utils_jack  import  Formant_utt2people_reshape, Gather_info_certainphones, \
                         FilterUttDictsByCriterion, GetValuelimit_IQR, \
                         Get_aligned_sequences, WER, Get_Vowels_AUI
from datetime import datetime as dt
import pathlib
from multiprocessing import Pool, current_process
import articulation.Multiprocess as Multiprocess
import articulation.HYPERPARAM.PaperNameMapping as PprNmeMp
from tqdm import tqdm
import seaborn as sns
def Swap2PaperName(feature_rawname,PprNmeMp):
    if feature_rawname in PprNmeMp.Paper_name_map.keys():
        featurename_paper=PprNmeMp.Paper_name_map[feature_rawname]
        feature_keys=featurename_paper
    else: 
        feature_keys=feature_rawname
    return feature_keys
def Process_IQRFiltering_Multi(Formants_utt_symb, limit_people_rule,\
                               outpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles',\
                               prefix='Formants_utt_symb',\
                               suffix='KID_FromASD_DOCKID'):
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
    
    Formants_utt_symb_limited=Dict()
    for load_file_tmp,_ in final_results:        
        for utt, df_utt in load_file_tmp.items():
            Formants_utt_symb_limited[utt]=df_utt
    
    pickle.dump(Formants_utt_symb_limited,open(outpath+"/[Analyzing]{0}_limited_{1}.pkl".format(prefix,suffix),"wb"))
    print('Formants_utt_symb saved to ',outpath+"/[Analyzing]{0}_limited_{1}.pkl".format(prefix,suffix))
    
def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--knn_weights', default='uniform',
                            help='uniform distance')
    parser.add_argument('--knn_neighbors', default=2,  type=int,
                            help='path of the base directory')
    parser.add_argument('--Reorder_type', default='DKIndividual',
                            help='[DKIndividual, DKcriteria]')
    parser.add_argument('--Result_path', default='./Result_Interaction',
                            help='')
    parser.add_argument('--Inspect_roles', default=['D','K'],
                            help='')
    parser.add_argument('--dataset_role', default='ASD_DOCKID',
                            help='ASD_DOCKID, TD_DOCKID')
    parser.add_argument('--Person_segment_df_path', default="articulation/Pickles/Session_formants_people_vowel_feat/",
                            help='')
    parser.add_argument('--SyncronyFeatures_root_path', default='Features/artuculation_AUI/Interaction/Syncrony_Knnparameters/',
                            help='')
    parser.add_argument('--Formants_utt_symb_path', default='articulation/Pickles',
                            help='')
    parser.add_argument('--poolMed', default='middle',
                            help='path of the base directory')
    parser.add_argument('--poolWindowSize', default=3,
                            help='path of the base directory')
    parser.add_argument('--Inspect_features', default=['F1','F2'],
                            help='')
    parser.add_argument('--reFilter', default=False,
                            help='')
    args = parser.parse_args()
    return args

args = get_args()
syncrony=Syncrony()
PhoneMapp_dict=phonewoprosody.PhoneMapp_dict
PhoneOfInterest=list(PhoneMapp_dict.keys())


knn_weights=args.knn_weights
knn_neighbors=args.knn_neighbors
Reorder_type=args.Reorder_type
dataset_role=args.dataset_role
Person_segment_df_path=args.Person_segment_df_path
SyncronyFeatures_root_path=args.SyncronyFeatures_root_path




# =============================================================================
'''
    
    畫conversational feautre的地方

'''
# Manual 準備formant部份的feature
if Reorder_type == 'DKIndividual':
    df_person_segment_feature_DKIndividual_dict=pickle.load(open(Person_segment_df_path+"df_person_segment_feature_{Reorder_type}_dict_{0}_{1}.pkl".format(dataset_role, 'formant',Reorder_type=Reorder_type),"rb"))
elif Reorder_type == 'DKcriteria':
    df_person_segment_feature_DKcriteria_dict=pickle.load(open(Person_segment_df_path+"df_person_segment_feature_{Reorder_type}_dict_{0}_{1}.pkl".format(dataset_role, 'formant',Reorder_type=Reorder_type),"rb"))

df_syncrony_measurement=pickle.load(open(SyncronyFeatures_root_path+"Syncrony_measure_of_variance_{knn_weights}_{knn_neighbors}_{Reorder_type}_{dataset_role}.pkl".format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,dataset_role=dataset_role,Reorder_type=Reorder_type),"rb"))

# Manual 準備phonation部份的feature
# if Reorder_type == 'DKIndividual':
#     df_POI_person_segment_DKIndividual_feature_dict=pickle.load(open(Person_segment_df_path+"df_POI_person_segment_DKIndividual_feature_dict_{0}_{1}.pkl".format(dataset_role, 'phonation'),"rb"))
#     df_person_segment_feature_DKIndividual_dict=df_POI_person_segment_DKIndividual_feature_dict['A:,i:,u:']['segment']
# elif Reorder_type == 'DKcriteria':
#     df_POI_person_segment_DKcriteria_feature_dict=pickle.load(open(Person_segment_df_path+"df_POI_person_segment_DKcriteria_feature_dict_{0}_{1}.pkl".format(dataset_role, 'phonation'),"rb"))
#     df_person_segment_feature_DKcriteria_dict=df_POI_person_segment_DKcriteria_feature_dict['A:,i:,u:']['segment']

# df_syncrony_measurement=pickle.load(open(SyncronyFeatures_root_path+"Syncrony_measure_of_variance_phonation_{knn_weights}_{knn_neighbors}_{Reorder_type}_{dataset_role}.pkl".format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,dataset_role=dataset_role,Reorder_type=Reorder_type),"rb"))
# phonation 跟 formant 只要擇一
def KDE_Filtering(df_vowel,THRESHOLD=10,scale_factor=100):
    # X=df_vowel[args.Inspect_features].values
    labels=df_vowel['vowel']
    
    df_vowel_calibrated=pd.DataFrame([])
    for phone in set(labels):
        
        df=df_vowel[df_vowel['vowel']==phone][args.Inspect_features]
        data_array=df_vowel[df_vowel['vowel']==phone][args.Inspect_features].values

        x=data_array[:,0]
        y=data_array[:,1]
        xmin = x.min()
        xmax = x.max()        
        ymin = y.min()
        ymax = y.max()
        
        image_num=1j
        X, Y = np.mgrid[xmin:xmax:image_num*scale_factor, ymin:ymax:image_num*scale_factor]
        
        positions = np.vstack([X.ravel(), Y.ravel()])
        
        values = np.vstack([x, y])
        
        kernel = stats.gaussian_kde(values)
                
        Z = np.reshape(kernel(positions).T, X.shape)
        normalized_z = preprocessing.normalize(Z)
        
        df['x_to_scale'] = (100*(x - np.min(x))/np.ptp(x)).astype(int) 
        df['y_to_scale'] = (100*(y - np.min(y))/np.ptp(y)).astype(int) 
        
        normalized_z=(100*(Z - np.min(Z.ravel()))/np.ptp(Z.ravel())).astype(int)
        to_delete = zip(*np.where((normalized_z<THRESHOLD) == True))
        
        # The indexes that are smaller than threshold
        deletepoints_bool=df.apply(lambda x: (x['x_to_scale'], x['y_to_scale']), axis=1).isin(to_delete)
        df_calibrated=df.loc[(deletepoints_bool==False).values]
        df_deleted_after_calibrated=df.loc[(deletepoints_bool==True).values]
        
        df_vowel_calibrated_tmp=df_calibrated.drop(columns=['x_to_scale','y_to_scale'])
        df_vowel_calibrated_tmp['vowel']=phone
        df_vowel_output=df_vowel_calibrated_tmp.copy()
        df_vowel_calibrated=df_vowel_calibrated.append(df_vowel_output)
        
        
        # Data prepare for plotting 
        
        # df_calibrated_tocombine=df_calibrated.copy()
        # df_calibrated_tocombine['cal']='calibrated'
        # df_deleted_after_calibrated['cal']='deleted'
        # df_calibratedcombined=df_calibrated_tocombine.append(df_deleted_after_calibrated)
        
        # #Plotting code
        # fig = plt.figure(figsize=(8,8))
        # ax = fig.gca()
        # ax.set_xlim(xmin, xmax)
        # ax.set_ylim(ymin, ymax)
        # # cfset = ax.contourf(X, Y, Z, cmap='coolwarm')
        # # ax.imshow(Z, cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
        # # cset = ax.contour(X, Y, Z, colors='k')
        # cfset = ax.contourf(X, Y, normalized_z, cmap='coolwarm')
        # ax.imshow(normalized_z, cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
        # cset = ax.contour(X, Y, normalized_z, colors='k')
        # ax.clabel(cset, inline=1, fontsize=10)
        # ax.set_xlabel('F1')
        # ax.set_ylabel('F2')
        # plt.title('KDE filtering')
        
        # sns.scatterplot(data=df_vowel[df_vowel['vowel']==phone], x="F1", y="F2")
        # sns.scatterplot(data=df_calibratedcombined, x="F1", y="F2",hue='cal')
    return df_vowel_calibrated
def PlotSyncronyFeatures(score_df,df_person_segment_feature_DKIndividual_dict,\
                         knn_weights,knn_neighbors,col,\
                         Inspect_people_lst=[],\
                         Inspect_role_lst=['D', 'K'],\
                         score_column='timeSeries_len[]',\
                         show_text=True,\
                         Ylim_parameter=None,\
                         st_col_str='IPU_st',ed_col_str='IPU_ed'):
    # Inputs: 
    # score_df=df_syncrony_measurement
    # df_person_segment_feature_DKIndividual_dict
    # knn_weights
    # knn_neighbors,col
    # col = 'intensity_mean_mean(A:,i:,u:)'
    # # col = 'between_variance_norm(A:,i:,u:)'
    # st_col_str='IPU_st'  #It is related to global info
    # ed_col_str='IPU_ed'  #It is related to global info
    if Ylim_parameter != None:
        [ylim_min,ylim_max]=Ylim_parameter
    
    MinNumTimeSeries=knn_neighbors+1
    # score_column=score_df.columns[0]
    score_cols=[score_column]
    features=[col]
    Knn_aggressive_mode=True
    for col in features:
        Col_continuous_function_DK=syncrony.KNNFitting(df_person_segment_feature_DKIndividual_dict,\
                    col, args.Inspect_roles,\
                    knn_weights=knn_weights,knn_neighbors=knn_neighbors,MinNumTimeSeries=MinNumTimeSeries,\
                    st_col_str='IPU_st', ed_col_str='IPU_ed', aggressive_mode=Knn_aggressive_mode)
        
        # df_syncrony_measurement_col=syncrony._calculate_features_col(Col_continuous_function_DK,col)
        # if df_syncrony_measurement_col.isna().any().any():
        #     print("The columns with Nan is ", col)
    functionDK_people=Col_continuous_function_DK
    Colormap_role_dict=Dict()
    Colormap_role_dict['D']='orange'
    Colormap_role_dict['K']='blue'

    PprNme_role_dict=Dict()
    PprNme_role_dict['D']='investigator'
    PprNme_role_dict['K']='participant'
    if len(Inspect_people_lst)>0:
        Inspect_people=Inspect_people_lst
    else:
        # All people and sort by certian score
        Inspect_people=list(score_df.sort_values(by=score_column).index)
    for people in Inspect_people:
        df_person_segment_feature_role_dict=df_person_segment_feature_DKIndividual_dict[people]
        fig, ax = plt.subplots()
        for role_choose in Inspect_role_lst:
            # print(role_choose,col)
            
            df_dynVals=df_person_segment_feature_role_dict[role_choose][col]
            # remove outlier that is falls over 3 times of the std
            # print(df_dynVals)
            # print(df_dynVals[(np.abs(stats.zscore(df_dynVals)) < 3)])
            df_dynVals_deleteOutlier=df_dynVals[(np.abs(stats.zscore(df_dynVals)) < 3)]
            df_stidx=df_person_segment_feature_role_dict[role_choose][st_col_str]
            df_edidx=df_person_segment_feature_role_dict[role_choose][ed_col_str]
            # recWidth=df_dynVals_deleteOutlier.min()/100
            recWidth=0.0005
            for x_1 , x_2, y in zip(df_stidx.values ,df_edidx.values,df_dynVals_deleteOutlier.values):            
                ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,recWidth,color=Colormap_role_dict[role_choose],alpha=0.5))
            
            # add an totally overlapped rectangle but it will show the label
            ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,recWidth,color=Colormap_role_dict[role_choose],label=PprNme_role_dict[role_choose],alpha=0.5))
            
            
            plt.plot(functionDK_people[people][role_choose],color=Colormap_role_dict[role_choose])
        ax.autoscale()
        if show_text==True:
            plt.title(col)
        plt.legend()
        
        score=score_df.loc[people,score_cols]
        info_arr=["{}: {}".format(idx,v) for idx, v in zip(score.index.values,np.round(score.values,3))]
        addtext='\n'.join(info_arr+[people])
        x0, xmax = plt.xlim()
        y0, ymax = plt.ylim()
        data_width = xmax - x0
        data_height = ymax - y0
        # text(x0/0.1 + data_width * 0.004, -data_height * 0.002, addtext, ha='center', va='center')
        if show_text==True:
            text(0, -0.1,addtext, ha='center', va='center', transform=ax.transAxes)
        
        if Ylim_parameter != None:
            plt.ylim(ylim_min,ylim_max)
        plt.show()
        fig.clf()

def PlotSyncronyFeatures_axes(ax,score_df,df_person_segment_feature_DKIndividual_dict,\
                         knn_weights,knn_neighbors,col,\
                         people='2017_01_20_01_243_1',\
                         Inspect_role_lst=['D', 'K'],\
                         score_column='timeSeries_len[]',\
                         show_text=True,\
                         Ylim_parameter=None,\
                         legend_bool=True,\
                         st_col_str='IPU_st',ed_col_str='IPU_ed'):
    # Inputs: 
    # score_df=df_syncrony_measurement
    # df_person_segment_feature_DKIndividual_dict
    # knn_weights
    # knn_neighbors,col
    # col = 'intensity_mean_mean(A:,i:,u:)'
    # # col = 'between_variance_norm(A:,i:,u:)'
    # st_col_str='IPU_st'  #It is related to global info
    # ed_col_str='IPU_ed'  #It is related to global info
    if Ylim_parameter != None:
        [ylim_min,ylim_max]=Ylim_parameter
    
    MinNumTimeSeries=knn_neighbors+1
    # score_column=score_df.columns[0]
    score_cols=[score_column]
    features=[col]
    Knn_aggressive_mode=True
    for col in features:
        Col_continuous_function_DK=syncrony.KNNFitting(df_person_segment_feature_DKIndividual_dict,\
                    col, args.Inspect_roles,\
                    knn_weights=knn_weights,knn_neighbors=knn_neighbors,MinNumTimeSeries=MinNumTimeSeries,\
                    st_col_str='IPU_st', ed_col_str='IPU_ed', aggressive_mode=Knn_aggressive_mode)
        
        # df_syncrony_measurement_col=syncrony._calculate_features_col(Col_continuous_function_DK,col)
        # if df_syncrony_measurement_col.isna().any().any():
        #     print("The columns with Nan is ", col)
    functionDK_people=Col_continuous_function_DK
    Colormap_role_dict=Dict()
    Colormap_role_dict['D']='orange'
    Colormap_role_dict['K']='blue'

    PprNme_role_dict=Dict()
    PprNme_role_dict['D']='investigator'
    PprNme_role_dict['K']='participant'

        
    df_person_segment_feature_role_dict=df_person_segment_feature_DKIndividual_dict[people]
    # fig, ax = plt.subplots()
    for role_choose in Inspect_role_lst:
        # print(role_choose,col)
        
        df_dynVals=df_person_segment_feature_role_dict[role_choose][col]
        # remove outlier that is falls over 3 times of the std
        # print(df_dynVals)
        # print(df_dynVals[(np.abs(stats.zscore(df_dynVals)) < 3)])
        df_dynVals_deleteOutlier=df_dynVals[(np.abs(stats.zscore(df_dynVals)) < 3)]
        df_stidx=df_person_segment_feature_role_dict[role_choose][st_col_str]
        df_edidx=df_person_segment_feature_role_dict[role_choose][ed_col_str]
        recWidth=df_dynVals_deleteOutlier.min()/100
        # recWidth=0.0005
        for x_1 , x_2, y in zip(df_stidx.values ,df_edidx.values,df_dynVals_deleteOutlier.values):            
            ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,recWidth,color=Colormap_role_dict[role_choose],alpha=0.5))
        
        # add an totally overlapped rectangle but it will show the label
        ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,recWidth,color=Colormap_role_dict[role_choose],label=PprNme_role_dict[role_choose],alpha=0.5))
        
        
        ax.plot(functionDK_people[people][role_choose],color=Colormap_role_dict[role_choose])
    ax.autoscale()
    if legend_bool == True:
        ax.legend()
    
    score=score_df.loc[people,score_cols]
    info_arr=["{}: {}".format(idx,v) for idx, v in zip(score.index.values,np.round(score.values,3))]
    addtext='\n'.join(info_arr+[people])
    # x0, xmax = ax.xlim()
    # y0, ymax = ax.ylim()
    # data_width = xmax - x0
    # data_height = ymax - y0
    # text(x0/0.1 + data_width * 0.004, -data_height * 0.002, addtext, ha='center', va='center')
    if show_text==True:
        text(0, -0.1,addtext, ha='center', va='center', transform=ax.transAxes)
    
    if Ylim_parameter != None:
        ax.axis(ymin=ylim_min,ymax=ylim_max)
    # plt.show()
    # fig.clf()
    return ax


''' 這邊實際 Call function '''
col = 'FCR2'
# col = 'between_variance_norm(A:,i:,u:)'
# col = 'between_covariance_norm(A:,i:,u:)'
# col = 'within_covariance_norm(A:,i:,u:)'
# col = 'within_variance_norm(A:,i:,u:)'
# col = 'spear_12'
# col = 'Between_Within_Det_ratio_norm(A:,i:,u:)'
# col = 'intensity_mean_mean(A:,i:,u:)'

# Moderate ASD 因為Trend[FCR2]_d 被判斷成ASD的個案
# Inspect_people=[
#     '2017_03_05_01_365_1',
#     '2017_07_24_01_217_1',
#     '2017_08_09_01_013',
#     '2017_08_15_01_413',
#     ]
Inspect_people=[
    # '2017_08_15_01_413',
    # '2016_12_04_01_234_1',
    # '2017_12_16_01_416',
    # '2017_10_18_01_309_4',
    '2017_12_02_01_380_1',
    '2017_12_20_01_510_1'
    ]



# Moderate ASD vs TD 時因為Trend[FCR2]_d 被判斷成TD的個案
# Inspect_people=[
#     '2018_05_19_5593_1_emotion',
#     ]

Demonstration_dict={}
Demonstration_dict['col']='Between_Within_Det_ratio_norm(A:,i:,u:)'

Demonstration_people=Dict()
Demonstration_people['Syncrony'].high='2017_12_20_01_510_1'
Demonstration_people['Convergence'].high='2017_01_25_01_225'
Demonstration_people['Proximity'].high='2017_01_20_243_1'
Demonstration_people['Trend_d'].high='2016_01_26_02_108_1'

Demonstration_people['Syncrony'].low='2017_04_08_01_256_1'
Demonstration_people['Convergence'].low='2016_01_26_02_108_1'
Demonstration_people['Proximity'].low='2017_08_11_01_300_1'
Demonstration_people['Trend_d'].low='2016_09_21_01_131_1'




# score_cols='Syncrony[{}]'.format(col)
# score_cols='Convergence[{}]'.format(col)
# score_cols='Proximity[{}]'.format(col)
score_cols='Trend[{}]_d'.format(col)
# feature_name='between_covariance_norm(A:,i:,u:)'
feature_name='FCR2'
# People_VowelSpace_inspect=Vowels_AUI.keys()
ASD_samples_bool=~Label.label_raw['ADOS_cate_CSS'].isna().values



# score_df=Label.label_raw.loc[ASD_samples_bool].sort_values(by=score_cols)[[score_cols,'name']]
# score_df=score_df.set_index('name')



feature_df=df_syncrony_measurement.copy()
Inspect_people=list(feature_df.sort_values(by=score_cols).index)
Inspect_people=[
    '2016_12_24_01_233',
    '2017_03_05_01_365_1',
    ]


Demonstration_people['FCR'].low='2016_12_24_01_233'
Demonstration_people['FCR'].high='2017_03_05_01_365_1'
# 直接畫圖
# PlotSyncronyFeatures(feature_df,df_person_segment_feature_DKIndividual_dict,\
#                           knn_weights,knn_neighbors,col,\
#                           Inspect_people_lst=Inspect_people,\
#                           Inspect_role_lst=args.Inspect_roles,\
#                           score_column=score_cols,\
#                           show_text=True,\
#                           Ylim_parameter=None,\
#                           st_col_str='IPU_st',ed_col_str='IPU_ed')
PlotSyncronyFeatures(feature_df,df_person_segment_feature_DKIndividual_dict,\
                          knn_weights,knn_neighbors,col,\
                          Inspect_people_lst=Inspect_people,\
                          Inspect_role_lst=['D'],\
                          score_column=score_cols,\
                          show_text=True,\
                          Ylim_parameter=None,\
                          st_col_str='IPU_st',ed_col_str='IPU_ed')
# PlotSyncronyFeatures(feature_df,df_person_segment_feature_DKIndividual_dict,\
#                           knn_weights,knn_neighbors,col,\
#                           Inspect_people_lst=Inspect_people,\
#                           score_column=score_cols,\
#                           show_text=False,\
#                           Ylim_parameter=None,\
#                           st_col_str='IPU_st',ed_col_str='IPU_ed')
aaa=ccc

Inspect_people=[
    '2017_12_20_01_510_1',
    '2017_01_25_01_225',
    '2017_01_20_01_243_1',
    '2016_01_26_02_108_1',
    '2017_04_08_01_256_1',
    '2016_01_26_02_108_1',
    '2017_08_11_01_300_1',
    '2016_09_21_01_191_1'
    ]
Titles=[
 'high Syncrony',
 'high Convergence',
 'high Proximity',
 'high GC',
 'low Syncrony',
 'low Convergence',
 'low Proximity',
 'low GC',
 ]
Ylim_parameters=[
    None,
    None,
    [0,120],
    None,
    None,
    None,
    [0,120],
    None,
    ]
Inspect_roles=[
    args.Inspect_roles,
    args.Inspect_roles,
    args.Inspect_roles,
    ['D'],
    args.Inspect_roles,
    args.Inspect_roles,
    args.Inspect_roles,
    ['D'],
    ]
Legend_bool=[
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    False,
    ]

# 把圖加進subplot
fig, axs = plt.subplots(2, 4, figsize=(10, 6), constrained_layout=True)
for ax, titls, people, ylim_p, roles, LB in zip(axs.flat, Titles,Inspect_people,Ylim_parameters,Inspect_roles,Legend_bool):
    ax.set_title(f'{titls}')
    ax=PlotSyncronyFeatures_axes(ax,feature_df,df_person_segment_feature_DKIndividual_dict,\
                              knn_weights,knn_neighbors,col,\
                              people=people,\
                              Inspect_role_lst=roles,\
                              score_column=score_cols,\
                              show_text=False,\
                              Ylim_parameter=ylim_p,\
                              legend_bool=LB,\
                              st_col_str='IPU_st',ed_col_str='IPU_ed')
    # ax.plot(x, y, 'o', ls='-', ms=4, markevery=markevery)

# =============================================================================
    
    
aaa=ccc    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

#%%
# =============================================================================
'''
    
    畫Vowel space的地方

'''
Formants_utt_symb=pickle.load(open(args.Formants_utt_symb_path+"/Formants_utt_symb_by{0}_window{1}_{2}.pkl".format(args.poolMed,args.poolWindowSize,args.dataset_role),'rb'))
print("Loading Formants_utt_symb from ", args.Formants_utt_symb_path+"/Formants_utt_symb_by{0}_window{1}_{2}.pkl".format(args.poolMed,args.poolWindowSize,args.dataset_role))


# 第一步：從Formants_utt_symb準備Vowel_AUI
Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
AUI_info=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)
limit_people_rule=GetValuelimit_IQR(AUI_info,PhoneMapp_dict,args.Inspect_features)

''' multi processing start '''
prefix,suffix = 'Formants_utt_symb', args.dataset_role
# date_now='{0}-{1}-{2} {3}'.format(dt.now().year,dt.now().month,dt.now().day,dt.now().hour)
date_now='{0}-{1}-{2}'.format(dt.now().year,dt.now().month,dt.now().day)
outpath='articulation/Pickles'
filepath=outpath+"/[Analyzing]{0}_limited_{1}.pkl".format(prefix,suffix)
if os.path.exists(filepath) and args.reFilter==False:
    fname = pathlib.Path(filepath)
    mtime = dt.fromtimestamp(fname.stat().st_mtime)
    # filemtime='{0}-{1}-{2} {3}'.format(mtime.year,mtime.month,mtime.day,mtime.hour)
    filemtime='{0}-{1}-{2}'.format(mtime.year,mtime.month,mtime.day)
    
    # If file last modify time is not now (precisions to the hours) than we create new one
    if filemtime != date_now:
        Process_IQRFiltering_Multi(Formants_utt_symb,limit_people_rule,\
                               outpath=outpath,\
                               prefix=prefix,\
                               suffix=suffix) # the results will be output as pkl file at outpath+"/[Analyzing]Formants_utt_symb_limited.pkl"
else:
    Process_IQRFiltering_Multi(Formants_utt_symb,limit_people_rule,\
                               outpath=outpath,\
                               prefix=prefix,\
                               suffix=suffix)
Formants_utt_symb_limited=pickle.load(open(filepath,"rb"))
''' multi processing end '''
if len(limit_people_rule) >0:
    Formants_utt_symb=Formants_utt_symb_limited

Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb,Align_OrinCmp=False)
AUI_info=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)
Vowels_AUI=Get_Vowels_AUI(AUI_info, args.Inspect_features,VUIsource="From__Formant_people_information")

# Get BCC, FCR, ... utterance level features
dfFormantStatisticpath=os.getcwd()
df_formant_statistic77_path=dfFormantStatisticpath+'/Features/ClassificationMerged_dfs/{knn_weights}_{knn_neighbors}_{Reorder_type}/ASD_DOCKID/static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation.pkl'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,Reorder_type=Reorder_type)
df_feature_ASD=pickle.load(open(df_formant_statistic77_path,'rb'))
# 第二步：準備 df_vowel_calibrated 或 df_vowel
# Play code for KDE filtering

THRESHOLD=40
# scale_factor=100
N=2
RESULT_DICTIONARY=Dict()
df_simulate=pd.DataFrame()
df_vowel_calibrated_dict=Dict()
for people in Vowels_AUI.keys():
    # plt.figure(count)
    F12_raw_dict=Vowels_AUI[people]
    df_vowel = pd.DataFrame()
    for keys in F12_raw_dict.keys():
        if len(df_vowel) == 0:
            df_vowel=F12_raw_dict[keys]
            df_vowel['vowel']=keys
        else:
            df_=F12_raw_dict[keys]
            df_['vowel']=keys
            df_vowel=df_vowel.append(df_)
    
    len_a=len(np.where(df_vowel['vowel']=='A:')[0])
    len_u=len(np.where(df_vowel['vowel']=='u:')[0])
    len_i=len(np.where(df_vowel['vowel']=='i:')[0])
    
    
    if len_a<=N or len_u<=N or len_i<=N:
        continue
    df_vowel_calibrated=KDE_Filtering(df_vowel,THRESHOLD=THRESHOLD,scale_factor=100)
    df_vowel_calibrated_dict[people]=df_vowel_calibrated


    
# Moderate ASD vs TD 時因為DEP feature 被判斷成ASD的個案


score_cols='ADOS_C'
# feature_name='between_covariance_norm(A:,i:,u:)'
feature_name='FCR2'
# People_VowelSpace_inspect=Vowels_AUI.keys()
ASD_samples_bool=~Label.label_raw['ADOS_cate_CSS'].isna().values

People_VowelSpace_inspect=list(Label.label_raw.loc[ASD_samples_bool].sort_values(by=score_cols)['name'])
People_VowelSpace_inspect=[
    # '2016_07_30_01_164',
    # '2017_07_05_01_310_1',
    '2017_12_23_01_407'
    ]


score_df=Label.label_raw.loc[ASD_samples_bool].sort_values(by=score_cols)[[score_cols,'name']]
score_df=score_df.set_index('name')

feature_df=df_feature_ASD.copy()

# Moderate ASD vs TD 時因為DEP feature 被判斷成TD的個案
# People_VowelSpace_inspect=[
# '2021_01_25_5833_1(醫生鏡頭模糊)_emotion',
# ]



count=0
for Pple in People_VowelSpace_inspect:
    # try:
    fig, ax = plt.subplots()
    # plt.figure(count)
    score_PprNme=feature_papername=Swap2PaperName(score_cols,PprNmeMp)
    score=score_df.loc[Pple,score_cols]
    # Title_str="{}: {} = {}".format(Pple, score_cols,score)
    Title_str="{} = {}".format(score_PprNme,score)
    sns.scatterplot(data=df_vowel_calibrated_dict[Pple], x="F1", y="F2", hue="vowel").set(title=Title_str)
    
    
    feature_value=feature_df.loc[Pple,feature_name]
    feature_papername=Swap2PaperName(feature_name,PprNmeMp)
    info_arr=["{}: {}".format(feature_papername,np.round(feature_value,3))]
    # info_arr=["{}: {}".format(idx,v) for idx, v in zip(score.index.values,np.round(score.values,3))]
    addtext='\n'.join(info_arr)
    x0, xmax = plt.xlim()
    y0, ymax = plt.ylim()
    data_width = xmax - x0
    data_height = ymax - y0
    # text(x0/0.1 + data_width * 0.004, -data_height * 0.002, addtext, ha='center', va='center')
    text(0, -0.1,addtext, ha='center', va='center', transform=ax.transAxes)
    
    plt.show()
    fig.clf()
    
    count+=1
    # except: 
    #     pass
#%%

# =============================================================================
'''

    Plot KDE filtering

'''
def PlotWithKDEFiltering(df_vowel,THRESHOLD=10,scale_factor=100,people='2017_12_23_01_465'):
    df_vowel_calibrated=pd.DataFrame([])
    
    df=df_vowel[args.Inspect_features+['vowel']]
    data_array=df_vowel[args.Inspect_features].values
    
    x=data_array[:,0]
    y=data_array[:,1]
    xmin = x.min()
    xmax = x.max()        
    ymin = y.min()
    ymax = y.max()
    
    image_num=1j
    X, Y = np.mgrid[xmin:xmax:image_num*scale_factor, ymin:ymax:image_num*scale_factor]
    
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    values = np.vstack([x, y])
    
    kernel = stats.gaussian_kde(values)
            
    Z = np.reshape(kernel(positions).T, X.shape)
    normalized_z = preprocessing.normalize(Z)
    
    df['x_to_scale'] = (100*(x - np.min(x))/np.ptp(x)).astype(int) 
    df['y_to_scale'] = (100*(y - np.min(y))/np.ptp(y)).astype(int) 
    
    normalized_z=(100*(Z - np.min(Z.ravel()))/np.ptp(Z.ravel())).astype(int)
    to_delete = zip(*np.where((normalized_z<THRESHOLD) == True))
    
    # The indexes that are smaller than threshold
    deletepoints_bool=df.apply(lambda x: (x['x_to_scale'], x['y_to_scale']), axis=1).isin(to_delete)
    df_calibrated=df.loc[(deletepoints_bool==False).values]
    df_deleted_after_calibrated=df.loc[(deletepoints_bool==True).values]
    
    
    df_vowel_calibrated_tmp=df_calibrated.drop(columns=['x_to_scale','y_to_scale'])
    df_vowel_calibrated_tmp['vowel']=df_calibrated.index
    df_vowel_output=df_vowel_calibrated_tmp.copy()
    df_vowel_calibrated=df_vowel_calibrated.append(df_vowel_output)
    
    
    # Data prepare for plotting 
    
    df_calibrated_tocombine=df_calibrated.copy()
    df_calibrated_tocombine['cal']='calibrated'
    # df_calibrated_tocombine['vowel']=df_calibrated.index
    df_deleted_after_calibrated['cal']='removed'
    df_calibratedcombined=df_calibrated_tocombine.append(df_deleted_after_calibrated)
    
    #Plotting code
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # cfset = ax.contourf(X, Y, Z, cmap='coolwarm')
    # ax.imshow(Z, cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
    # cset = ax.contour(X, Y, Z, colors='k')
    # cfset = ax.contourf(X, Y, normalized_z, cmap='coolwarm')
    cfset = ax.contourf(X, Y, normalized_z, colors='white')
    # ax.imshow(normalized_z, cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
    ax.imshow(normalized_z, extent=[xmin, xmax, ymin, ymax])
    cset = ax.contour(X, Y, normalized_z, colors='k')
    ax.clabel(cset, inline=1, fontsize='x-large')
    ax.set_xlabel('F1')
    ax.set_ylabel('F2')
    ax.set_aspect(0.25)
    # plt.title('KDE filtering ' + people)
    plt.title('KDE filtering')
    
    
    # sns.scatterplot(data=df_deleted_after_calibrated, x="F1", y="F2",marker='.',color='k')
    ax = sns.scatterplot(data=df_deleted_after_calibrated, x="F1", y="F2",marker='.',s=15,hue='cal',palette=['black'])
    sns.scatterplot(data=df_calibrated_tocombine, x="F1", y="F2",hue='vowel')
    handles, lables = ax.get_legend_handles_labels()
    handles[0].set_sizes([4.0])
    plt.legend(handles, lables)
    plt.show()
    # for h in handles:
    #     h.set_sizes([10])
    # plt.gca().legend(('y0','y1'))

selected_people='2017_12_23_01_407'
for people in Vowels_AUI.keys():
    # plt.figure(count)
    F12_raw_dict=Vowels_AUI[people]
    df_vowel = pd.DataFrame()
    for keys in F12_raw_dict.keys():
        if len(df_vowel) == 0:
            df_vowel=F12_raw_dict[keys]
            df_vowel['vowel']=keys
        else:
            df_=F12_raw_dict[keys]
            df_['vowel']=keys
            df_vowel=df_vowel.append(df_)

    if len_a<=N or len_u<=N or len_i<=N:
        continue
    if people == selected_people:
        PlotWithKDEFiltering(df_vowel,THRESHOLD=THRESHOLD,scale_factor=100,people=people)

# =============================================================================

# =============================================================================
