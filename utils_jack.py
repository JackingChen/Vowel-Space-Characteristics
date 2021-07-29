
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa
"""


import numpy as np
import scipy.stats as st
import kaldi_io
import re
import sys
path_app = '/mnt/sdd/jackchen/egs/formosa/s6/local'
sys.path.append(path_app)
from utils_wer.wer import  wer as WER
sys.path.remove(path_app)
from tqdm import tqdm


def dynamic2static(feat):

    me=np.mean(feat,0)

    std=np.std(feat,0)
    sk=st.skew(feat)
    ku=st.kurtosis(feat)

    return np.hstack((me,std,sk,ku))

def dynamic2statict(feat):

    me=[]
    std=[]
    sk=[]
    ku=[]
    for k in feat:
        me.append(np.mean(k,0))
        std.append(np.std(k,0))
        sk.append(st.skew(k))
        ku.append(st.kurtosis(k))
    return np.hstack((me,std,sk,ku))


def dynamic2statict_artic(feat):

    me=[]
    std=[]
    sk=[]
    ku=[]
    for k in feat:
        if k.shape[0]>1:
            me.append(np.mean(k,0))
            std.append(np.std(k,0))
            sk.append(st.skew(k))
            ku.append(st.kurtosis(k))
        elif k.shape[0]==1:
            me.append(k[0,:])
            std.append(np.zeros(k.shape[1]))
            sk.append(np.zeros(k.shape[1]))
            ku.append(np.zeros(k.shape[1]))
        else:
            me.append(np.zeros(k.shape[1]))
            std.append(np.zeros(k.shape[1]))
            sk.append(np.zeros(k.shape[1]))
            ku.append(np.zeros(k.shape[1]))

    return np.hstack((np.hstack(me),np.hstack(std),np.hstack(sk),np.hstack(ku)))

def dynamic2statict_artic_formant(feat):

    me=[]
    std=[]
    sk=[]
    sk_sign=[]
    sk_abs=[]
    ku=[]
    for k in feat:
        if k.shape[0]>1:
            me.append(np.mean(k,0))
            std.append(np.std(k,0))
            sk.append(st.skew(k))
            sk_sign.append(np.sign(st.skew(k)))
            sk_abs.append(np.abs(st.skew(k)))
            ku.append(st.kurtosis(k))
        elif k.shape[0]==1:
            me.append(k[0,:])
            std.append(np.zeros(k.shape[1]))
            sk.append(np.zeros(k.shape[1]))
            sk_sign.append(np.sign(st.skew(k)))
            sk_abs.append(np.abs(st.skew(k)))
            ku.append(np.zeros(k.shape[1]))
        else:
            me.append(np.zeros(k.shape[1]))
            std.append(np.zeros(k.shape[1]))
            sk.append(np.zeros(k.shape[1]))
            sk_sign.append(np.sign(st.skew(k)))
            sk_abs.append(np.abs(st.skew(k)))
            ku.append(np.zeros(k.shape[1]))

    return np.hstack((np.hstack(me),np.hstack(std),np.hstack(sk),np.hstack(sk_sign),np.hstack(sk_abs),np.hstack(ku)))


def get_dict(feat_mat, IDs):
    uniqueids=np.unique(IDs)
    df={}
    for k in uniqueids:
        p=np.where(IDs==k)[0]
        featid=feat_mat[p,:]
        df[str(k)]=featid
    return df

def save_dict_kaldimat(dict_feat, temp_file):
    ark_scp_output='ark:| copy-feats --compress=true ark:- ark,scp:'+temp_file+'.ark,'+temp_file+'.scp'
    with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
        for key,mat in dict_feat.items(): 
            kaldi_io.write_mat(f, mat, key=key)

def multi_find(s, r):
    s_len = len(s)
    r_len = len(r)
    _complete = []
    if s_len < r_len:
        n = -1
    else:
        for i in range(s_len):
            # search for r in s until not enough characters are left
            if s[i:i + r_len] == r:
                _complete.append(i)
            else:
                i = i + 1
    return(_complete)

def functional_method(data, method='middle', window=3):
    
    if method=="mean":
        ret=np.mean(data)
        
    elif method=="middle":
        from_ind=max(0,int(len(data)/2)-window)
        to_ind=min(len(data)-1,int(len(data)/2)+window)
        
        ret=np.mean(data[from_ind: to_ind])
    
    # print('data: ',data)
    # print('ret: ',ret)
    
    return ret
from sklearn.utils import (as_float_array, check_array, check_X_y, safe_sqr,
                     safe_mask)
import warnings
from scipy import special, stats
def f_classif(X, y):
    
    """Compute the ANOVA F-value for the provided sample.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} shape = [n_samples, n_features]
        The set of regressors that will be tested sequentially.

    y : array of shape(n_samples)
        The data matrix.

    Returns
    -------
    F : array, shape = [n_features,]
        The set of F values.

    pval : array, shape = [n_features,]
        The set of p-values.

    See also
    --------
    chi2: Chi-squared stats of non-negative features for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    """
    # X, y = check_X_y(X, y, ['csr', 'csc', 'coo'])
    args = [X[safe_mask(X, y == k)] for k in np.unique(y)]
    n_classes = len(args)
    args = [as_float_array(a) for a in args]
    n_samples_per_class = np.array([a.shape[0] for a in args])
    n_samples = np.sum(n_samples_per_class)
    ss_alldata = sum(safe_sqr(a).sum(axis=0) for a in args) #sum of square of all data [21001079.59736017]
    sums_args = [np.asarray(a.sum(axis=0)) for a in args] #sum of data in each group [3204.23910828, 7896.25971663, 5231.79595847]
    square_of_sums_alldata = sum(sums_args) ** 2  # square of summed data [2.66743853e+08]
    square_of_sums_args = [s ** 2 for s in sums_args]
    sstot = ss_alldata - square_of_sums_alldata / float(n_samples)
    ssbn = 0.
    for k, _ in enumerate(args):
        ssbn += square_of_sums_args[k] / n_samples_per_class[k]
    ssbn -= square_of_sums_alldata / float(n_samples)
    sswn = sstot - ssbn
    dfbn = n_classes - 1
    dfwn = n_samples - n_classes
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    constant_features_idx = np.where(msw == 0.)[0]
    if (np.nonzero(msb)[0].size != msb.size and constant_features_idx.size):
        warnings.warn("Features %s are constant." % constant_features_idx,
                      UserWarning)
    f = msb / msw
    # flatten matrix to vector in sparse case
    f = np.asarray(f).ravel()
    prob = special.fdtrc(dfbn, dfwn, f)
    return f, prob, msb, msw, ssbn


def Get_aligned_sequences(ref, hype, error_info):
    # utilize error_info from WER function. WER function will generate alignment
    utt_human_ali=pd.DataFrame()
    utt_hype_ali=pd.DataFrame()
    human_ali_idx=0
    hype_ali_idx=0
    for j,element in enumerate(error_info):
        if element=="e" or element=="s":
            utt_human_ali=utt_human_ali.append(ref.iloc[human_ali_idx])
            utt_hype_ali=utt_hype_ali.append(hype.iloc[hype_ali_idx])
            human_ali_idx+=1
            hype_ali_idx+=1
        elif element=="i":
            hype_ali_idx+=1
        elif element=="d":
            human_ali_idx+=1
    return utt_human_ali.reset_index(drop=True), utt_hype_ali.reset_index(drop=True)

def Formant_utt2people_reshape(Formants_utt_symb,Formants_utt_symb_cmp,Align_OrinCmp=True):
    # =============================================================================
    #     Formant_people_symb_total['cmp'][people] = DF
    #     DF.loc[phone] = [F1, F2, end, start, text, utt]
    # =============================================================================
    Formant_people_symb_total=Dict()
    Formant_people_symb_total['ori']=Dict()
    Formant_people_symb_total['cmp']=Dict()
    for keys, values in Formants_utt_symb_cmp.items():
        people=keys[:keys.find(re.findall("_[K|D]",keys)[0])]
        if people not in Formant_people_symb_total['cmp'].keys():
            Formant_people_symb_total['cmp'][people]=pd.DataFrame()
        if people not in Formant_people_symb_total['ori'].keys():
            Formant_people_symb_total['ori'][people]=pd.DataFrame()
        
        if Align_OrinCmp:
            # Align cmp, ori text sequences
            values['text']=values.index
            Formants_utt_symb[keys]['text']=Formants_utt_symb[keys].index
            r=values['text'].astype(str)
            h=Formants_utt_symb[keys]['text'].astype(str)
            
            error_info, WER_value=WER(r,h)
            utt_human_ali, utt_hype_ali=Get_aligned_sequences(ref=values, hype=Formants_utt_symb[keys],error_info=error_info)
            utt_human_ali=utt_human_ali.sort_values(by='start')
            utt_hype_ali=utt_hype_ali.sort_values(by='start')
            utt_human_ali['utt']=keys
            utt_hype_ali['utt']=keys
            utt_human=utt_human_ali
            utt_hype=utt_hype_ali
        else:
            utt_human=Formants_utt_symb_cmp[keys]
            utt_human['text']=Formants_utt_symb_cmp[keys].index
            utt_hype=Formants_utt_symb[keys]
            utt_hype['text']=Formants_utt_symb[keys].index
            
            utt_human['utt']=keys
            utt_hype['utt']=keys
        Formant_people_symb_total['cmp'][people]=Formant_people_symb_total['cmp'][people].append(utt_human)
        Formant_people_symb_total['ori'][people]=Formant_people_symb_total['ori'][people].append(utt_hype)
    return Formant_people_symb_total

def Gather_info_certainphones(Formant_people_symb_total,PhoneMapp_dict,PhoneOfInterest,):
    # =============================================================================
    #     AUI_dict[people][phone] = DF
    #     DF.loc[int] = [F1, F2, end, start, text, utt]
    # =============================================================================
    AUI_dict=Dict()
    for people in Formant_people_symb_total['cmp'].keys():
        df_people_phone_cmp=Formant_people_symb_total['cmp'][people]
        df_people_phone_ori=Formant_people_symb_total['ori'][people]
        for symb in PhoneOfInterest:
            data_aui_cmp_idx=np.array([xx for x in  PhoneMapp_dict[symb] for xx in np.where(df_people_phone_cmp['text']==x)[0]])
            df_aui_phone_cmp=df_people_phone_cmp.iloc[data_aui_cmp_idx,:]
            df_aui_phone_cmp.loc[:,'cmps']=np.array(['cmp']*len(df_aui_phone_cmp))
            
            data_aui_ori_idx=np.array([xx for x in  PhoneMapp_dict[symb] for xx in np.where(df_people_phone_ori['text']==x)[0]])
            df_aui_phone_ori=df_people_phone_ori.iloc[data_aui_ori_idx,:]
            df_aui_phone_ori.loc[:,'cmps']=np.array(['ori']*len(df_aui_phone_ori))
            AUI_dict[people][symb]=pd.concat([df_aui_phone_cmp,df_aui_phone_ori])
    return AUI_dict

def FilterUttDictsByCriterion(Formants_utt_symb,Formants_utt_symb_cmp,limit_people_rule):
    # Masks will be generated by setting criterion on Formants_utt_symb
    # and Formants_utt_symb_cmp will be masked by the same mask as Formants_utt_symb
    # we need to make sure two things:
    #   1. the length of Formants_utt_symb_cmp and Formants_utt_symb are the same
    #   2. the phone sequences are aligned correctly
    Formants_utt_symb_limited=Dict()
    Formants_utt_symb_cmp_limited=Dict()
    for utt in tqdm(Formants_utt_symb_cmp.keys()):
        people=utt[:utt.find(re.findall("_[K|D]",utt)[0])]
        df_ori=Formants_utt_symb[utt].sort_values(by="start")
        df_cmp=Formants_utt_symb_cmp[utt].sort_values(by="start")
        df_ori['text']=df_ori.index
        df_cmp['text']=df_cmp.index
        
        r=df_cmp.index.astype(str)
        h=df_ori.index.astype(str)
        
        error_info, WER_value=WER(r,h)
        utt_human_ali, utt_hype_ali=Get_aligned_sequences(ref=df_cmp, hype=df_ori ,error_info=error_info) # This step cannot gaurentee hype and human be exact the same string
                                                                                                          # because substitude error also counts when selecting the optimal 
                                                                                                          # matched string         
        utt_human_ali.index=utt_human_ali['text']
        utt_human_ali=utt_human_ali.drop(columns=["text"])
        utt_hype_ali.index=utt_hype_ali['text']
        utt_hype_ali=utt_hype_ali.drop(columns=["text"])
        
        assert len(utt_human_ali) == len(utt_hype_ali)
        limit_rule=limit_people_rule[people]
        SymbRuleChecked_bookkeep={}
        for symb_P in limit_rule.keys():
            values_limit=limit_rule[symb_P]
    
            # filter_bool=utt_hype_ali.index.str.contains(symb_P)
            filter_bool=utt_hype_ali.index==symb_P           #  1. select the phone with criterion
            filter_bool_inv=np.invert(filter_bool)           #  2. create a mask for unchoosed phones
                                                             #  we want to make sure that only 
                                                             #  the phones not match the criterion will be False
            for feat in values_limit.keys():
                feat_max_value=values_limit[feat]['max']
                filter_bool=np.logical_and(filter_bool , (utt_hype_ali[feat]<=feat_max_value))
                feat_min_value=values_limit[feat]['min']
                filter_bool=np.logical_and(filter_bool , (utt_hype_ali[feat]>=feat_min_value))
                
            filter_bool=np.logical_or(filter_bool_inv,filter_bool)
            
            # check & debug
            if not filter_bool.all():
                print(utt,filter_bool[filter_bool==False])
            
            SymbRuleChecked_bookkeep[symb_P]=filter_bool.to_frame()
        
        df_True=pd.DataFrame(np.array([True]*len(utt_hype_ali)))
        for keys, values in SymbRuleChecked_bookkeep.items():
            df_True=np.logical_and(values,df_True)
        
        Formants_utt_symb_limited[utt]=utt_hype_ali[df_True[0].values]
        Formants_utt_symb_cmp_limited[utt]=utt_human_ali[df_True[0].values]
    return Formants_utt_symb_limited,Formants_utt_symb_cmp_limited 

def GetValuelimit_IQR(AUI_info,PhoneMapp_dict,Inspect_features,maxFreq=5500,minFreq=0):
    limit_people_rule=Dict()
    for people in AUI_info.keys():
        for phoneRepresent in AUI_info[people].keys():
            df_values = AUI_info[people][phoneRepresent][AUI_info[people][phoneRepresent]['cmps'] == 'ori']
            for feat in Inspect_features:
                data=df_values[[feat]][df_values[feat]>0]
                q25, q75 = data.quantile(q=0.25).values[0], data.quantile(q=0.75).values[0]
                iqr=q75 - q25
                cut_off = iqr * 1.5
                lower, upper = q25 - cut_off, q75 + cut_off
                upper = min(maxFreq,upper)
                lower = max(minFreq,lower)
                for phone in PhoneMapp_dict[phoneRepresent]:
                    limit_people_rule[people][phone][feat].max=upper
                    limit_people_rule[people][phone][feat].min=lower
    return limit_people_rule




def Postprocess_dfformantstatistic(df_formant_statistic):
    ''' Remove person that has unsufficient data '''
    df_formant_statistic_bool=(df_formant_statistic['u_num']!=0) & (df_formant_statistic['a_num']!=0) & (df_formant_statistic['i_num']!=0)
    df_formant_statistic=df_formant_statistic[df_formant_statistic_bool]
    
    ''' ADD ADOS category '''
    df_formant_statistic['ADOS_cate']=np.array([0]*len(df_formant_statistic))
    df_formant_statistic['ADOS_cate'][df_formant_statistic['ADOS']<2]=0
    df_formant_statistic['ADOS_cate'][(df_formant_statistic['ADOS']<3) & (df_formant_statistic['ADOS']>=2)]=1
    df_formant_statistic['ADOS_cate'][df_formant_statistic['ADOS']>=3]=2
    return df_formant_statistic
# =============================================================================
'''

Get additional information

'''
import pandas as pd
# =============================================================================

Labelfile='/homes/ssd1/jackchen/gop_prediction/ADOS_label20210721.xlsx'
df_labels=pd.read_excel(Labelfile)
Info_name_sex=df_labels[['name','sex','age_year']].copy()
Info_name_sex.loc[Info_name_sex['sex']==1,'sex']='male'
Info_name_sex.loc[Info_name_sex['sex']==2,'sex']='female'

Labelfile='/homes/ssd1/jackchen/DisVoice/data/ADOS_TD_Label22.xlsx'
df_labels_TD=pd.read_excel(Labelfile)
Info_name_sex_TD=df_labels_TD[['name','sex','age_year']].copy()
# Info_name_sex_TD['sex'][Info_name_sex_TD['sex']==1]='male'
Info_name_sex_TD.loc[Info_name_sex_TD['sex']==1,'sex']='male'
Info_name_sex_TD.loc[Info_name_sex_TD['sex']==2,'sex']='female'

Info_name_sex=Info_name_sex.append(Info_name_sex_TD)
from addict import  Dict
''' codings of filename '''
Namecode_dict=Dict()
Namecode_dict['1st_pass']={'role':-3,'emotion':-1}
Namecode_dict['1st_pass']={'role':-3,'emotion':-1}
Namecode_dict['2nd_pass']={'role':-4,'emotion':-2}
''' min/max F0 for F1/F2 estimation '''
F0_parameter_dict=Dict()
F0_parameter_dict['male']={'f0_min':75,'f0_max':400}
F0_parameter_dict['female']={'f0_min':75,'f0_max':800}