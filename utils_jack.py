
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa
"""


import numpy as np
import scipy.stats as st
import kaldi_io



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
        ret=np.mean(data[int(len(data)/2)-window: int(len(data)/2)+window])
    
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
# =============================================================================
'''

Get additional information

'''
import pandas as pd
# =============================================================================

Labelfile='/homes/ssd1/jackchen/jackchen/ADOS_data/workplace/label_ADOS_87.xlsx'
df_labels=pd.read_excel(Labelfile)
Info_name_sex=df_labels[['name','sex']]
Info_name_sex.loc[Info_name_sex['sex']==1,'sex']='male'
Info_name_sex.loc[Info_name_sex['sex']==2,'sex']='female'

Labelfile='/homes/ssd1/jackchen/DisVoice/data/ADOS_TD_Label22.xlsx'
df_labels_TD=pd.read_excel(Labelfile)
Info_name_sex_TD=df_labels_TD[['name','sex']]
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