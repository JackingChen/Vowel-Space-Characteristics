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
from tqdm import tqdm
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

df_syncrony_measurement=pickle.load(open(SyncronyFeatures_root_path+"Syncrony_measure_of_variance_phonation_{knn_weights}_{knn_neighbors}_{Reorder_type}_{dataset_role}.pkl".format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,dataset_role=dataset_role,Reorder_type=Reorder_type),"rb"))
# phonation 跟 formant 只要擇一
def KDE_Filtering(df_vowel,THRESHOLD=10,scale_factor=100):
    X=df_vowel[args.Inspect_features].values
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
        # import seaborn as sns
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
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # plt.title('2D Gaussian Kernel density estimation')
        
        # sns.scatterplot(data=df_vowel[df_vowel['vowel']==phone], x="F1", y="F2")
        # sns.scatterplot(data=df_calibratedcombined, x="F1", y="F2",hue='cal')
    return df_vowel_calibrated
def PlotSyncronyFeatures(score_df,df_person_segment_feature_DKIndividual_dict,\
                         knn_weights,knn_neighbors,col,\
                         Inspect_people_lst=[],\
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
    
    MinNumTimeSeries=knn_neighbors+1
    score_column=score_df.columns[0]
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
    if len(Inspect_people_lst)>0:
        Inspect_people=Inspect_people_lst
    else:
        # All people and sort by certian score
        Inspect_people=list(score_df.sort_values(by=score_column).index)
    for people in Inspect_people:
        df_person_segment_feature_role_dict=df_person_segment_feature_DKIndividual_dict[people]
        fig, ax = plt.subplots()
        for role_choose in args.Inspect_roles:
            df_dynVals=df_person_segment_feature_role_dict[role_choose][col]
            # remove outlier that is falls over 3 times of the std
            df_dynVals_deleteOutlier=df_dynVals[(np.abs(stats.zscore(df_dynVals)) < 3)]
            df_stidx=df_person_segment_feature_role_dict[role_choose][st_col_str]
            df_edidx=df_person_segment_feature_role_dict[role_choose][ed_col_str]
            # recWidth=df_dynVals_deleteOutlier.min()/100
            recWidth=0.0005
            for x_1 , x_2, y in zip(df_stidx.values ,df_edidx.values,df_dynVals_deleteOutlier.values):            
                ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,recWidth,color=Colormap_role_dict[role_choose],alpha=0.5))
            
            # add an totally overlapped rectangle but it will show the label
            ax.add_patch(plt.Rectangle((x_1,y),x_2-x_1,recWidth,color=Colormap_role_dict[role_choose],label=role_choose,alpha=0.5))
            
            
            plt.plot(functionDK_people[people][role_choose],color=Colormap_role_dict[role_choose])
        ax.autoscale()
        plt.title(col)
        plt.legend()
        
        score=score_df.loc[people,score_cols]
        info_arr=["{}: {}".format(idx,v) for idx, v in zip(score.index.values,np.round(score.values,3))]
        addtext='\n'.join(info_arr)
        x0, xmax = plt.xlim()
        y0, ymax = plt.ylim()
        data_width = xmax - x0
        data_height = ymax - y0
        # text(x0/0.1 + data_width * 0.004, -data_height * 0.002, addtext, ha='center', va='center')
        text(0, -0.1,addtext, ha='center', va='center', transform=ax.transAxes)
        
        plt.show()
        fig.clf()

''' 這邊實際 Call function '''
col = 'FCR2'
# col = 'between_variance_norm(A:,i:,u:)'
# col = 'intensity_mean_mean(A:,i:,u:)'

# Moderate ASD 因為Trend[FCR2]_d 被判斷成ASD的個案
# Inspect_people=[
#     '2017_03_05_01_365_1',
#     '2017_07_24_01_217_1',
#     '2017_08_09_01_013',
#     '2017_08_15_01_413',
#     ]

# Moderate ASD vs TD 時因為Trend[FCR2]_d 被判斷成TD的個案
Inspect_people=[
    '2018_05_19_5593_1_emotion',
    ]
PlotSyncronyFeatures(df_syncrony_measurement,df_person_segment_feature_DKIndividual_dict,\
                          knn_weights,knn_neighbors,col,\
                          Inspect_people_lst=Inspect_people,\
                          st_col_str='IPU_st',ed_col_str='IPU_ed')
# =============================================================================
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

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
outpath='/homes/ssd1/jackchen/DisVoice/articulation/Pickles'
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

People_VowelSpace_inspect=[
    '2015_12_06_01_097',
    '2016_01_27_01_154_1',
    '2017_11_18_01_371',
    ]

# Moderate ASD vs TD 時因為DEP feature 被判斷成TD的個案
# People_VowelSpace_inspect=[
# '2021_01_25_5833_1(醫生鏡頭模糊)_emotion',
# ]
count=0
for Pple in People_VowelSpace_inspect:
    plt.figure(count)
    sns.scatterplot(data=df_vowel_calibrated_dict[Pple], x="F1", y="F2", hue="vowel").set(title=Pple)
    count+=1
# =============================================================================
