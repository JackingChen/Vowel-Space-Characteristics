#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:56:45 2021

@author: jackchen


    This is a branch of MainClassification that run specific experiments
    
    Include SHAP values in this script

"""

import os, sys
import pandas as pd
import numpy as np

import glob
import pickle

from scipy.stats import spearmanr,pearsonr 
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier

from addict import Dict
# import functions
import argparse
from scipy.stats import zscore

from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.linear_model import ElasticNet
import sklearn.svm
import torch
import re
import matplotlib.pyplot as plt
from sklearn import preprocessing

from articulation.HYPERPARAM import phonewoprosody, Label
from articulation.HYPERPARAM.PeopleSelect import SellectP_define
import articulation.HYPERPARAM.FeatureSelect as FeatSel

import articulation.articulation
from sklearn.metrics import f1_score,recall_score,roc_auc_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from itertools import combinations
import shap
import articulation.HYPERPARAM.PaperNameMapping as PprNmeMp
import inspect
import seaborn as sns
from scipy.stats import gaussian_kde
def Find_Experiment_actualindex(Total_FeatComb_idxs,search_string):
    # usage:
    # e.x. :
    # search_string='Phonation_Trend_K_cols+Phonation_Syncrony_cols+Phonation_Trend_D_cols'
    # Total_FeatComb_idxs=FeatSel.Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation'].keys()
    # Find_Experiment_actualindex(Total_FeatComb_idxs,search_string)
    for FC_str in Total_FeatComb_idxs:
        if ''.join(sorted(FC_str.split("+"))) == ''.join(sorted(search_string.split("+"))):
            print(FC_str)



def Assert_labelfeature(feat_name,lab_name):
    # =============================================================================
    #     To check if the label match with feature
    # =============================================================================
    for i,n in enumerate(feat_name):
        assert feat_name[i] == lab_name[i]

def FilterFile_withinManualName(files,Manual_choosen_feature):
    files_manualChoosen=[f  for f in files if os.path.basename(f).split(".")[0]  in Manual_choosen_feature]
    return files_manualChoosen

def Merge_dfs(df_1, df_2):
    return pd.merge(df_1,df_2,left_index=True, right_index=True)

def Add_label(df_formant_statistic,Label,label_choose='ADOS_cate_C'):
    for people in df_formant_statistic.index:
        bool_ind=Label.label_raw['name']==people
        df_formant_statistic.loc[people,label_choose]=Label.label_raw.loc[bool_ind,label_choose].values
    return df_formant_statistic
def Swap2PaperName(feature_rawname,PprNmeMp):
    if feature_rawname in PprNmeMp.Paper_name_map.keys():
        featurename_paper=PprNmeMp.Paper_name_map[feature_rawname]
        feature_keys=featurename_paper
    else: 
        feature_keys=feature_rawname
    return feature_keys

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def Return_ALigned_dfXtestndfShapValues(Feature_SHAP_info_dict,logit_number=1):
    #Inputs
    # logit_numberit_number=1
    # Feature_SHAP_info_dict=Proposed_changed_info_dict
    ###################################
    df_XTest_stacked=pd.DataFrame()
    df_ShapValues_stacked=pd.DataFrame()
    for people in Feature_SHAP_info_dict.keys():
        df_XTest=Feature_SHAP_info_dict[people]['XTest']
        df_ShapValues=Feature_SHAP_info_dict[people]['shap_values'].loc[logit_number]
        df_ShapValues.name=df_XTest.name
        df_XTest_stacked=pd.concat([df_XTest_stacked,df_XTest],axis=1)
        df_ShapValues_stacked=pd.concat([df_ShapValues_stacked,df_ShapValues],axis=1)
    return df_XTest_stacked,df_ShapValues_stacked

def Calculate_XTestShape_correlation(Feature_SHAP_info_dict,logit_number=1):
    df_XTest_stacked,df_ShapValues_stacked=Return_ALigned_dfXtestndfShapValues(Feature_SHAP_info_dict,logit_number=logit_number)
    Correlation_XtestnShap={}
    for features in df_XTest_stacked.index:
        r,p=pearsonr(df_XTest_stacked.loc[features],df_ShapValues_stacked.loc[features])
        Correlation_XtestnShap[features]=r
    df_Correlation_XtestnShap=pd.DataFrame.from_dict(Correlation_XtestnShap,orient='index')
    df_Correlation_XtestnShap.columns=['correlation w logit:{}'.format(logit_number)]
    return df_Correlation_XtestnShap

def Prepare_data_for_summaryPlot(SHAPval_info_dict, feature_columns=None, PprNmeMp=None):
    keys_bag=[]
    XTest_dict={}
    shap_values_0_bag=[]
    shap_values_1_bag=[]
    for people in sorted(SHAPval_info_dict.keys()):
        keys_bag.append(people)
        if not feature_columns == None:
            df_=SHAPval_info_dict[people]['XTest'][feature_columns]
            df_shape_value=SHAPval_info_dict[people]['shap_values'][feature_columns]
        else:
            df_=SHAPval_info_dict[people]['XTest']
            df_shape_value=SHAPval_info_dict[people]['shap_values']
        # if not SumCategorical_feats == None:
        #     for k,values in SumCategorical_feats.items():
        #         df_[k]=df_.loc[values].sum()
        XTest_dict[people]=df_        
        shap_values_0_bag.append(df_shape_value.loc[0].values)
        shap_values_1_bag.append(df_shape_value.loc[1].values)
    shap_values_0_array=np.vstack(shap_values_0_bag)
    shap_values_1_array=np.vstack(shap_values_1_bag)
    shap_values=[shap_values_0_array,shap_values_1_array]
    df_XTest=pd.DataFrame.from_dict(XTest_dict,orient='index')
    if PprNmeMp!=None:
        df_XTest.columns=[Swap2PaperName(k,PprNmeMp) for k in df_XTest.columns]
    return shap_values, df_XTest, keys_bag
def shorten_text(text, length_limit):
    if len(text) > length_limit:
        return text[:length_limit - 3] + "..."
    else:
        return text
    
    
def summary_legacy(df_shap_values, features=None, feature_names=None, max_display=None, plot_type=None,
                 color=None, axis_color="#333333", title=None, alpha=1, show=True, sort=True,
                 color_bar=True, plot_size="auto", layered_violin_max_num_bins=20, class_names=None,
                 class_inds=None,
                 color_bar_label=labels["FEATURE_VALUE"],
                 cmap=colors.red_blue,
                 font_size=13,\
                 dot_size=10,\
                 # depreciated
                 auto_size_plot=None,
                 use_log_scale=False):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : numpy.array
        For single output explanations this is a matrix of SHAP values (# samples x # features).
        For multi-output explanations this is a list of such matrices of SHAP values.

    features : numpy.array or pandas.DataFrame or list
        Matrix of feature values (# samples x # features) or a feature_names list as shorthand

    feature_names : list
        Names of the features (length # features)

    max_display : int
        How many top features to include in the plot (default is 20, or 7 for interaction plots)

    plot_type : "dot" (default for single output), "bar" (default for multi-output), "violin",
        or "compact_dot".
        What type of summary plot to produce. Note that "compact_dot" is only used for
        SHAP interaction values.

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default the size is auto-scaled based on the number of
        features that are being displayed. Passing a single float will cause each row to be that 
        many inches high. Passing a pair of floats will scale the plot by that
        number of inches. If None is passed then the size of the current figure will be left
        unchanged.
    """
    
    
    assert type(df_shap_values) == pd.core.frame.DataFrame
    
    # TODO: 把marker加進來
    Marker_dict={
        'O':'<',
        'X':'X',
        
        
        }
    markers_bag=list(df_shap_values.index)
    shap_values=df_shap_values.values

    # support passing an explanation object
    # if str(type(shap_values)).endswith("Explanation'>"):
    #     shap_exp = shap_values
    #     base_value = shap_exp.base_values
    #     shap_values = shap_exp.values
    #     if features is None:
    #         features = shap_exp.data
    #     if feature_names is None:
    #         feature_names = shap_exp.feature_names


    # deprecation warnings
    # if auto_size_plot is not None:
    #     warnings.warn("auto_size_plot=False is deprecated and is now ignored! Use plot_size=None instead.")

    multi_class = False
    if isinstance(shap_values, list):
        multi_class = True
        if plot_type is None:
            plot_type = "bar" # default for multi-output explanations
        assert plot_type == "bar", "Only plot_type = 'bar' is supported for multi-output explanations!"
    else:
        if plot_type is None:
            plot_type = "dot" # default for single output explanations
        assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."

    # default color:
    if color is None:
        if plot_type == 'layered_violin':
            color = "coolwarm"
        elif multi_class:
            color = lambda i: colors.red_blue_circle(i/len(shap_values))
        else:
            color = colors.blue_rgb

    idx2cat = None
    # convert from a DataFrame or other types
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = features.columns
        # feature index to category flag
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = (shap_values[0].shape[1] if multi_class else shap_values.shape[1])

    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the " \
                    "provided data matrix."
        if num_features - 1 == features.shape[1]:
            assert False, shape_msg + " Perhaps the extra column in the shap_values matrix is the " \
                          "constant offset? Of so just pass shap_values[:,:-1]."
        else:
            assert num_features == features.shape[1], shape_msg

    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

    if use_log_scale:
        plt.xscale('symlog')

    

    if max_display is None:
        max_display = 20

    if sort:
        # order features by the sum of their effect magnitudes
        if multi_class:
            feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
        else:
            feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)):]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    row_height = 0.4
    if plot_size == "auto":
        plt.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    elif type(plot_size) in (list, tuple):
        plt.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        plt.gcf().set_size_inches(8, len(feature_order) * plot_size + 1.5)
    plt.axvline(x=0, color="#999999", zorder=-1)

    if plot_type == "dot":
        for pos, i in enumerate(feature_order):
            plt.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
            shaps = shap_values[:, i]  # shap_values: (feature_num, people)
            values = None if features is None else features[:, i]
            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            if values is not None:
                values = values[inds]
            shaps = shaps[inds]
            colored_feature = True
            try:
                if idx2cat is not None and idx2cat[i]: # check categorical feature
                    colored_feature = False
                else:
                    values = np.array(values, dtype=np.float64)  # make sure this can be numeric
            except:
                colored_feature = False
            N = len(shaps)
            # hspacing = (np.max(shaps) - np.min(shaps)) / 200
            # curr_bin = []
            nbins = 100
            quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(N) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * (row_height / np.max(ys + 1))

            if features is not None and colored_feature:
                # trim the color range, but prevent the color range from collapsing
                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)
                if vmin > vmax: # fixes rare numerical precision issues
                    vmin = vmax

                assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

                # plot the nan values in the interaction feature as grey
                nan_mask = np.isnan(values)
                # 這個在把nan的值畫成灰色，不重要
                plt.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                           vmax=vmax, s=16, alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)

                # plot the non-nan values colored by the trimmed feature value
                cvals = values[np.invert(nan_mask)].astype(np.float64)
                cvals_imp = cvals.copy()
                cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                cvals[cvals_imp > vmax] = vmax
                cvals[cvals_imp < vmin] = vmin
                
                
                # 這個是主要話資料點的function
                # 
                # plt.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                #            cmap=cmap, vmin=vmin, vmax=vmax, s=16,
                #            marker=Marker_dict[markers_bag[pos]],
                #            c=cvals, alpha=alpha, linewidth=0,
                #            zorder=3, rasterized=len(shaps) > 500)
                position_correct_bool=np.array(markers_bag) == 'O'
                position_Incorrect_bool=np.array(markers_bag) == 'X'
                
                shap_position_correct_x_array=shaps[np.invert(nan_mask)][position_correct_bool]
                shap_position_correct_y_array=(pos + ys[np.invert(nan_mask)])[position_correct_bool]
                
                shap_position_Incorrect_x_array=shaps[np.invert(nan_mask)][position_Incorrect_bool]
                shap_position_Incorrect_y_array=(pos + ys[np.invert(nan_mask)])[position_Incorrect_bool]
                
                cvals_correct=cvals[position_correct_bool]
                cvals_Incorrect=cvals[position_Incorrect_bool]
                
                
                if len(shap_position_correct_x_array)>0:
                    plt.scatter(shap_position_correct_x_array, shap_position_correct_y_array,
                               cmap=cmap, vmin=vmin, vmax=vmax, s=dot_size,
                               marker=Marker_dict['O'],
                               c=cvals_correct, alpha=alpha, linewidth=0,
                               zorder=3, rasterized=len(shaps) > 500)
                if len(shap_position_Incorrect_x_array)>0:
                    plt.scatter(shap_position_Incorrect_x_array, shap_position_Incorrect_y_array,
                               cmap=cmap, vmin=vmin, vmax=vmax, s=dot_size,
                               marker=Marker_dict['X'],
                               c=cvals_Incorrect, alpha=alpha, linewidth=0,
                               zorder=3, rasterized=len(shaps) > 500)
                
            else:
                plt.scatter(shaps, pos + ys, s=16, alpha=alpha, linewidth=0, zorder=3,
                           color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)
    # draw the color bar
    if color_bar and features is not None and plot_type != "bar" and \
            (plot_type != "layered_violin" or color in plt.cm.datad):
        import matplotlib.cm as cm
        # from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
        # aspect = 20
        # pad_fraction = 0.5
        m = cm.ScalarMappable(cmap=cmap if plot_type != "layered_violin" else plt.get_cmap(color))
        m.set_array([0, 50])
        
        cb = plt.colorbar(m, ticks=[0, 50], aspect=1000)
        cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.9) * 20)
        # cb.draw_all()

    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().tick_params(color=axis_color, labelcolor=axis_color)
    # Get or set the current tick locations and labels of the y-axis.
    # 這邊用來把feature 名稱放在y軸
    plt.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=font_size)
    if plot_type != "bar":
        plt.gca().tick_params('y', length=20, width=0.5, which='major')
    plt.gca().tick_params('x', labelsize=11)
    plt.ylim(-1, len(feature_order))
    if plot_type == "bar":
        plt.xlabel(labels['GLOBAL_VALUE'], fontsize=font_size)
    else:
        plt.xlabel(labels['VALUE'], fontsize=font_size)
    if show:
        plt.show()    
    
def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser()
    parser.add_argument('--Feature_mode', default='Customized_feature',
                        help='what kind of data you want to get')
    parser.add_argument('--preprocess', default=True,
                        help='')
    parser.add_argument('--start_point', default=-1,
                        help='In case that the program stop at certain point, we can resume the progress by setting this variable')
    parser.add_argument('--experiment', default='gop_exp_ADOShappyDAAIKidallDeceiptformosaCSRC',
                        help='If the mode is set to Session_phone_phf, you may need to determine the experiment used to generate the gop feature')
    parser.add_argument('--pseudo', default=False,
                        help='what kind of data you want to get')
    parser.add_argument('--suffix', default="",
                        help='what kind of data you want to get')
    parser.add_argument('--FS_method_str', default=None,
                        help='Feature selection')
    parser.add_argument('--Print_Analysis_grp_Manual_select', default=True,
                        help='')
    parser.add_argument('--Plot', default=True,
                        help='')
    parser.add_argument('--selectModelScoring', default='recall_macro',
                        help='[recall_macro,accuracy]')
    parser.add_argument('--Mergefeatures', default=False,
                        help='')
    parser.add_argument('--logit_number', default=1,
                        help='')
    parser.add_argument('--knn_weights', default='uniform',
                            help='path of the base directory')
    parser.add_argument('--knn_neighbors', default=2,  type=int,
                            help='path of the base directory')
    parser.add_argument('--Reorder_type', default='DKIndividual',
                            help='[DKIndividual, DKcriteria]')
    parser.add_argument('--FeatureComb_mode', default='Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation',
                            help='[Add_UttLvl_feature, feat_comb3, feat_comb5, feat_comb6,feat_comb7, baselineFeats,Comb_dynPhonation,Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation]')
    args = parser.parse_args()
    return args
args = get_args()
start_point=args.start_point
experiment=args.experiment
knn_weights=args.knn_weights
knn_neighbors=args.knn_neighbors
Reorder_type=args.Reorder_type
logit_number=args.logit_number
#%%

Session_level_all=Dict()
label_choose=['ADOS_C']

df_formant_statistics_CtxPhone_collect_dict=Dict()

# =============================================================================

class ADOSdataset():
    def __init__(self,knn_weights,knn_neighbors,Reorder_type,FeatureComb_mode):
        self.featurepath='Features'            
        self.N=2
        self.LabelType=Dict()
        self.LabelType['ADOS_C']='regression'
        self.LabelType['ADOS_cate_C']='classification'
        self.LabelType['ASDTD']='classification'
        self.Fractionfeatures_str='Features/artuculation_AUI/Vowels/Fraction/*.pkl'    
        self.FeatureComb_mode=FeatureComb_mode
        if self.FeatureComb_mode == 'Add_UttLvl_feature':
            self.File_root_path='Features/ClassificationMerged_dfs/ADDed_UttFeat/{knn_weights}_{knn_neighbors}_{Reorder_type}/'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,Reorder_type=Reorder_type)
            self.Merge_feature_path=self.File_root_path+'{dataset_role}/*.pkl'.format(dataset_role='ASD_DOCKID')
        else:
            self.File_root_path='Features/ClassificationMerged_dfs/{knn_weights}_{knn_neighbors}_{Reorder_type}/'.format(knn_weights=knn_weights,knn_neighbors=knn_neighbors,Reorder_type=Reorder_type)
            self.Merge_feature_path=self.File_root_path+'{dataset_role}/*.pkl'.format(dataset_role='ASD_DOCKID')
        self.Merge_feature_path='Features/ClassificationMerged_dfs/distance_2_DKIndividual/{dataset_role}/*.pkl'.format(dataset_role='ASD_DOCKID')
        
        self.Top_ModuleColumn_mapping_dict={}
        self.Top_ModuleColumn_mapping_dict['Add_UttLvl_feature']=FeatSel.Columns_comb2.copy()
        self.Top_ModuleColumn_mapping_dict['feat_comb']=FeatSel.Columns_comb.copy()
        self.Top_ModuleColumn_mapping_dict['feat_comb3']=FeatSel.Columns_comb3.copy()
        self.Top_ModuleColumn_mapping_dict['feat_comb5']=FeatSel.Columns_comb5.copy()
        self.Top_ModuleColumn_mapping_dict['feat_comb6']=FeatSel.Columns_comb6.copy()
        self.Top_ModuleColumn_mapping_dict['feat_comb7']=FeatSel.Columns_comb7.copy()
        self.Top_ModuleColumn_mapping_dict['Comb_dynPhonation']=FeatSel.Comb_dynPhonation.copy()
        self.Top_ModuleColumn_mapping_dict['Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation']=FeatSel.Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation.copy()
        
        self.Top_ModuleColumn_mapping_dict['baselineFeats']=FeatSel.Baseline_comb.copy()
        
        self.FeatureCombs_manual=Dict()
        self.FeatureCombs_manual['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_kid']=['df_formant_statistic_TD_normal_kid', 'df_formant_statistic_agesexmatch_ASDSevereGrp_kid']
        self.FeatureCombs_manual['TD_normal vs ASDMild_agesexmatch >> FeatLOC_kid']=['df_formant_statistic_TD_normal_kid', 'df_formant_statistic_agesexmatch_ASDMildGrp_kid']
        self.FeatureCombs_manual['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_doc']=['df_formant_statistic_TD_normal_doc', 'df_formant_statistic_agesexmatch_ASDSevereGrp_doc']
        self.FeatureCombs_manual['TD_normal vs ASDMild_agesexmatch >> FeatLOC_doc']=['df_formant_statistic_TD_normal_doc', 'df_formant_statistic_agesexmatch_ASDMildGrp_doc']
        self.FeatureCombs_manual['TD_normal vs ASDSevere_agesexmatch >> FeatLOC_DKRatio']=['df_formant_statistic_TD_normalGrp_DKRatio', 'df_formant_statistic_agesexmatch_ASDSevereGrp_DKRatio']
        self.FeatureCombs_manual['TD_normal vs ASDMild_agesexmatch >> FeatLOC_DKRatio']=['df_formant_statistic_TD_normalGrp_DKRatio', 'df_formant_statistic_agesexmatch_ASDMildGrp_DKRatio']
        
        self.FeatureCombs_manual['TD_normal vs ASDSevere_agesexmatch >> FeatCoor']=['df_syncrony_statistic_TD_normalGrp', 'df_syncrony_statistic_agesexmatch_ASDSevereGrp']
        self.FeatureCombs_manual['TD_normal vs ASDMild_agesexmatch >> FeatCoor']=['df_syncrony_statistic_TD_normalGrp', 'df_syncrony_statistic_agesexmatch_ASDMildGrp']
        # self.FeatureCombs_manual['Notautism vs ASD']=['df_formant_statistic_77_Notautism', 'df_formant_statistic_77_ASD']
        # self.FeatureCombs_manual['ASD vs Autism']=['df_formant_statistic_77_ASD', 'df_formant_statistic_77_Autism']
        # self.FeatureCombs_manual['Notautism vs Autism']=['df_formant_statistic_77_Notautism', 'df_formant_statistic_77_Autism']
    
        # self._FeatureBuild_single()
        self._FeatureBuild_Module()
    def Get_FormantAUI_feat(self,label_choose,pickle_path,featuresOfInterest=['MSB_f1','MSB_f2','MSB_mix'],filterbyNum=True,**kwargs):
        arti=articulation.articulation.Articulation()
        
        
        #如果path有放的話字串的話，就使用path的字串，不然就使用「feat_」等於的東西，在function裡面會以kwargs的形式出現
        
        if not kwargs and len(pickle_path)>0:
            df_tmp=pickle.load(open(pickle_path,"rb")).sort_index()
        elif len(kwargs)>0: # usage Get_FormantAUI_feat(...,key1=values1):
            for k, v in kwargs.items(): #there will be only one element
                df_tmp=kwargs[k].sort_index()

        if filterbyNum:
            df_tmp=arti.BasicFilter_byNum(df_tmp,N=self.N)

        if label_choose not in df_tmp.columns:
            for people in df_tmp.index:
                lab=Label.label_raw[label_choose][Label.label_raw['name']==people]
                df_tmp.loc[people,'ADOS']=lab.values
            df_y=df_tmp['ADOS'] #Still keep the form of dataframe
        else:
            df_y=df_tmp[label_choose] #Still keep the form of dataframe
        
        
        feature_array=df_tmp[featuresOfInterest]
        
            
        LabType=self.LabelType[label_choose]
        return feature_array, df_y, LabType
    def _FeatureBuild_single(self):
        Features=Dict()
        Features_comb=Dict()
        files = glob.glob(self.Fractionfeatures_str)
        for file in files:
            feat_name=os.path.basename(file).replace(".pkl","")
            df_tmp=pickle.load(open(file,"rb")).sort_index()
            Features[feat_name]=df_tmp
        for keys in self.FeatureCombs_manual.keys():
            combF=[Features[k] for k in self.FeatureCombs_manual[keys]]
            Features_comb[keys]=pd.concat(combF)
        
        self.Features_comb_single=Features_comb
    def _FeatureBuild_Module(self):
        Labels_add=['ASDTD']
        ModuledFeatureCombination=self.Top_ModuleColumn_mapping_dict[self.FeatureComb_mode]
        
        sellect_people_define=SellectP_define()
        #Loading features from ASD        
        Features_comb=Dict()
        IterateFilesFullPaths = glob.glob(self.Merge_feature_path)
        

        if self.FeatureComb_mode in ['feat_comb3','feat_comb5','feat_comb6','feat_comb7','Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation']:
            DfCombFilenames=['static_feautre_LOC+dynamic_feature_LOC+dynamic_feature_phonation.pkl']
        if self.FeatureComb_mode == 'Comb_dynPhonation':
            DfCombFilenames=['dynamic_feature_phonation.pkl']
        elif self.FeatureComb_mode == 'baselineFeats':
             DfCombFilenames=['{}.pkl'.format(Dataname) for Dataname in ModuledFeatureCombination.keys()]
        else:
            DfCombFilenames=[os.path.basename(f) for f in IterateFilesFullPaths]
        File_ASD_paths=[self.File_root_path+"ASD_DOCKID/"+f for f in DfCombFilenames]
        File_TD_paths=[self.File_root_path+"TD_DOCKID/"+f for f in DfCombFilenames]
        
        
        df_Top_Check_length=pd.DataFrame()
        for file_ASD, file_TD in zip(File_ASD_paths,File_TD_paths):
            if not os.path.exists(file_ASD) or not os.path.exists(file_TD):
                raise FileExistsError()
            
            assert os.path.basename(file_ASD) == os.path.basename(file_TD)
            filename=os.path.basename(file_ASD)
            k_FeatTypeLayer1=filename.replace(".pkl","")
            df_feature_ASD=pickle.load(open(file_ASD,"rb")).sort_index()
            df_feature_TD=pickle.load(open(file_TD,"rb")).sort_index()
            df_feature_ASD['ASDTD']=sellect_people_define.ASDTD_label['ASD']
            df_feature_TD['ASDTD']=sellect_people_define.ASDTD_label['TD']
            
            # ADD label
            df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='ADOS_cate_CSS')
            df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='ADOS_cate_C')
            df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='ADOS_cate_S')
            df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='ADOS_cate_SC')
            df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='age_year')
            df_feature_ASD=Add_label(df_feature_ASD,Label,label_choose='sex')
            # create different ASD cohort
            filter_Minimal_TCSS=df_feature_ASD['ADOS_cate_CSS']==0
            filter_low_TCSS=df_feature_ASD['ADOS_cate_CSS']==1
            filter_moderate_TCSS=df_feature_ASD['ADOS_cate_CSS']==2
            filter_high_TCSS=df_feature_ASD['ADOS_cate_CSS']==3
            
            filter_Notautism_TC=df_feature_ASD['ADOS_cate_C']==0
            filter_ASD_TC=df_feature_ASD['ADOS_cate_C']==1
            filter_Autism_TC=df_feature_ASD['ADOS_cate_C']==2
            
            filter_Notautism_TS=df_feature_ASD['ADOS_cate_S']==0
            filter_ASD_TS=df_feature_ASD['ADOS_cate_S']==1
            filter_Autism_TS=df_feature_ASD['ADOS_cate_S']==2
            
            filter_Notautism_TSC=df_feature_ASD['ADOS_cate_SC']==0
            filter_ASD_TSC=df_feature_ASD['ADOS_cate_SC']==1
            filter_Autism_TSC=df_feature_ASD['ADOS_cate_SC']==2
            
            df_feauture_ASDgrp_dict={}
            df_feauture_ASDgrp_dict['df_feature_ASD']=df_feature_ASD
            
            # df_feauture_ASDgrp_dict['df_feature_Minimal_CSS']=df_feature_ASD[filter_Minimal_TCSS]
            # df_feauture_ASDgrp_dict['df_feature_low_CSS']=df_feature_ASD[filter_low_TCSS]
            # df_feauture_ASDgrp_dict['df_feature_moderate_CSS']=df_feature_ASD[filter_moderate_TCSS]
            # df_feauture_ASDgrp_dict['df_feature_high_CSS']=df_feature_ASD[filter_high_TCSS]
            df_feauture_ASDgrp_dict['df_feature_lowMinimal_CSS']=df_feature_ASD[filter_low_TCSS | filter_Minimal_TCSS]
            df_feauture_ASDgrp_dict['df_feature_moderate_CSS']=df_feature_ASD[filter_moderate_TCSS]
            df_feauture_ASDgrp_dict['df_feature_high_CSS']=df_feature_ASD[filter_high_TCSS]
            
            # df_feauture_ASDgrp_dict['df_feature_Notautism_TC']=df_feature_ASD[filter_Notautism_TC]
            # df_feauture_ASDgrp_dict['df_feature_ASD_TC']=df_feature_ASD[filter_ASD_TC]
            # df_feauture_ASDgrp_dict['df_feature_NotautismandASD_TC']=df_feature_ASD[filter_Notautism_TC | filter_ASD_TC]
            # df_feauture_ASDgrp_dict['df_feature_Autism_TC']=df_feature_ASD[filter_Autism_TC]
            
            # df_feauture_ASDgrp_dict['df_feature_Notautism_TS']=df_feature_ASD[filter_Notautism_TS]
            # df_feauture_ASDgrp_dict['df_feature_ASD_TS']=df_feature_ASD[filter_ASD_TS]
            # df_feauture_ASDgrp_dict['df_feature_NotautismandASD_TS']=df_feature_ASD[filter_Notautism_TS | filter_ASD_TS]
            # df_feauture_ASDgrp_dict['df_feature_Autism_TS']=df_feature_ASD[filter_Autism_TS]
            
            # df_feauture_ASDgrp_dict['df_feature_Notautism_TSC']=df_feature_ASD[filter_Notautism_TSC]
            # df_feauture_ASDgrp_dict['df_feature_ASD_TSC']=df_feature_ASD[filter_ASD_TSC]
            # df_feauture_ASDgrp_dict['df_feature_NotautismandASD_TSC']=df_feature_ASD[filter_Notautism_TSC | filter_ASD_TSC]
            # df_feauture_ASDgrp_dict['df_feature_Autism_TSC']=df_feature_ASD[filter_Autism_TSC]
            
            #Check the length of each paired comparison, should be stored on the top of for loop
            
            Tmp_Numcmp_dict={}
            for key in df_feauture_ASDgrp_dict.keys():
                Numcmp_str='ASD({0}) vs TD({1})'.format(len(df_feauture_ASDgrp_dict[key]),len(df_feature_TD))
                Tmp_Numcmp_dict[key]=Numcmp_str
                
            
            df_Tmp_Numcmp_list=pd.DataFrame.from_dict(Tmp_Numcmp_dict,orient='index')
            df_Tmp_Numcmp_list.columns=[k_FeatTypeLayer1]

            if len(df_Top_Check_length)==0:
                df_Top_Check_length=df_Tmp_Numcmp_list
            else:
                df_Top_Check_length=Merge_dfs(df_Top_Check_length,df_Tmp_Numcmp_list)
            # 手動執行到這邊，從for 上面
            
            
            for k_FeatTypeLayer2 in ModuledFeatureCombination[k_FeatTypeLayer1].keys():
                colums_sel=ModuledFeatureCombination[k_FeatTypeLayer1][k_FeatTypeLayer2]
                
                # 1. Set ASD vs TD experiment
                for k_ASDgrp in df_feauture_ASDgrp_dict.keys():
                    df_ASD_subgrp=df_feauture_ASDgrp_dict[k_ASDgrp].copy()[colums_sel+Labels_add]
                    df_TD_subgrp=df_feature_TD.copy()[colums_sel+Labels_add]
                    
                    experiment_str="{TD_name} vs {ASD_name} >> {feature_type}".format(TD_name='TD',ASD_name=k_ASDgrp,feature_type=k_FeatTypeLayer2)
                    Features_comb[experiment_str]=pd.concat([df_ASD_subgrp,df_TD_subgrp],axis=0)
                # 2. Set ASDsevere vs ASDmild experiment
                # experiment_str="{ASDsevere_name} vs {ASDmild_name} >> {feature_type}".format(ASDsevere_name='df_feature_moderatehigh_CSS',ASDmild_name='df_feature_lowMinimal_CSS',feature_type=k_FeatTypeLayer2)
                # df_ASDsevere_subgrp=df_feauture_ASDgrp_dict['df_feature_moderatehigh_CSS'].copy()
                # df_ASDmild_subgrp=df_feauture_ASDgrp_dict['df_feature_lowMinimal_CSS'].copy()
                # df_ASDsevere_subgrp['ASDsevereMild']=sellect_people_define.ASDsevereMild_label['ASDsevere']
                # df_ASDmild_subgrp['ASDsevereMild']=sellect_people_define.ASDsevereMild_label['ASDmild']
                # Features_comb[experiment_str]=pd.concat([df_ASDsevere_subgrp,df_ASDmild_subgrp],axis=0)
        self.Features_comb_multi=Features_comb

# =============================================================================
'''

    Feature merging function
    
    Ths slice of code provide user to manually make functions to combine df_XXX_infos

'''
# =============================================================================

ados_ds=ADOSdataset(knn_weights,knn_neighbors,Reorder_type,FeatureComb_mode=args.FeatureComb_mode)
ErrorFeat_bookeep=Dict()


FeatureLabelMatch_manual=[

    ['TD vs df_feature_lowMinimal_CSS >> LOC_columns+Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols', 'ASDTD'],
    # ['TD vs df_feature_lowMinimal_CSS >> LOC_columns+Phonation_Trend_K_cols+Phonation_Syncrony_cols', 'ASDTD'],
    # ['TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols', 'ASDTD'],
    ['TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_K_cols+Phonation_Syncrony_cols', 'ASDTD'],


    ['TD vs df_feature_moderate_CSS >> LOCDEP_Trend_D_cols+Phonation_Proximity_cols', 'ASDTD'],
    ['TD vs df_feature_moderate_CSS >> Phonation_Proximity_cols', 'ASDTD'],
    

    # ['TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_K_cols+Phonation_Syncrony_cols', 'ASDTD'],
    ['TD vs df_feature_high_CSS >> DEP_columns+Phonation_Trend_D_cols+Phonation_Proximity_cols', 'ASDTD'],
    # ['TD vs df_feature_high_CSS >> Phonation_Trend_D_cols+Phonation_Proximity_cols', 'ASDTD'],
    # ['TD vs df_feature_high_CSS >> DEP_columns+Phonation_Proximity_cols', 'ASDTD'],
    ['TD vs df_feature_high_CSS >> Phonation_Proximity_cols', 'ASDTD'],


 ]
    
# FeatSel 掌管該出現的columns
# ados_ds.Features_comb_multi 掌管load進來的data


Top_ModuleColumn_mapping_dict={}
Top_ModuleColumn_mapping_dict['Add_UttLvl_feature']={ e2_str:FeatSel.Columns_comb2[e_str][e2_str] for e_str in FeatSel.Columns_comb2.keys() for e2_str in FeatSel.Columns_comb2[e_str].keys()}
Top_ModuleColumn_mapping_dict['feat_comb3']=ModuleColumn_mapping={ e2_str:FeatSel.Columns_comb3[e_str][e2_str] for e_str in FeatSel.Columns_comb3.keys() for e2_str in FeatSel.Columns_comb3[e_str].keys()}
Top_ModuleColumn_mapping_dict['feat_comb5']=ModuleColumn_mapping={ e2_str:FeatSel.Columns_comb5[e_str][e2_str] for e_str in FeatSel.Columns_comb5.keys() for e2_str in FeatSel.Columns_comb5[e_str].keys()}
Top_ModuleColumn_mapping_dict['feat_comb6']=ModuleColumn_mapping={ e2_str:FeatSel.Columns_comb6[e_str][e2_str] for e_str in FeatSel.Columns_comb6.keys() for e2_str in FeatSel.Columns_comb6[e_str].keys()}
Top_ModuleColumn_mapping_dict['feat_comb7']=ModuleColumn_mapping={ e2_str:FeatSel.Columns_comb7[e_str][e2_str] for e_str in FeatSel.Columns_comb7.keys() for e2_str in FeatSel.Columns_comb7[e_str].keys()}
Top_ModuleColumn_mapping_dict['Comb_dynPhonation']=ModuleColumn_mapping={ e2_str:FeatSel.Comb_dynPhonation[e_str][e2_str] for e_str in FeatSel.Comb_dynPhonation.keys() for e2_str in FeatSel.Comb_dynPhonation[e_str].keys()}
Top_ModuleColumn_mapping_dict['Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation']=ModuleColumn_mapping={ e2_str:FeatSel.Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation[e_str][e2_str] for e_str in FeatSel.Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation.keys() for e2_str in FeatSel.Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation[e_str].keys()}
Top_ModuleColumn_mapping_dict['feat_comb']=ModuleColumn_mapping={ e2_str:FeatSel.Columns_comb[e_str][e2_str] for e_str in FeatSel.Columns_comb.keys() for e2_str in FeatSel.Columns_comb[e_str].keys()}


ModuleColumn_mapping=Top_ModuleColumn_mapping_dict[args.FeatureComb_mode]
# =============================================================================
'''

    Here starts to load features to Session_level_all dict

'''
# =============================================================================
for exp_str, lab_ in FeatureLabelMatch_manual:
    comparison_pair=exp_str.split(" >> ")[0]
    ModuleColumn_str=exp_str.split(" >> ")[-1]
    featuresOfInterest=[ModuleColumn_mapping[ModuleColumn_str]]
    # feat_=key
    for feat_col in featuresOfInterest:
        feat_col_ = list(feat_col) # ex: ['MSB_f1']
        if len(feat_col) > 144: # 144 is the limit of the filename
            key=feat_col_
        else:
            key=[feat_col_]
        # X,y, featType=ados_ds.Get_FormantAUI_feat(\
        #     label_choose=lab_,pickle_path='',featuresOfInterest=feat_col_,filterbyNum=False,\
        #     feat_=ados_ds.Features_comb_single[feat_])
        
        X,y, featType=ados_ds.Get_FormantAUI_feat(\
            label_choose=lab_,pickle_path='',featuresOfInterest=feat_col,filterbyNum=False,\
            feat_=ados_ds.Features_comb_multi[exp_str])
        
        if X.isnull().values.any() or y.isnull().values.any():
            print("Feat: ",key,'Contains nan')
            ErrorFeat_bookeep['{0} {1} {2}'.format(exp_str,lab_,key)].X=X
            ErrorFeat_bookeep['{0} {1} {2}'.format(exp_str,lab_,key)].y=y
            continue
        
        Item_name="{feat}::{lab}".format(feat=' >> '.join([comparison_pair,ModuleColumn_str]),lab=lab_)
        Session_level_all[Item_name].X, \
            Session_level_all[Item_name].y, \
                Session_level_all[Item_name].feattype = X,y, featType

# =============================================================================
# Model parameters
# =============================================================================
# C_variable=np.array([0.0001, 0.01, 0.1,0.5,1.0,10.0, 50.0, 100.0, 1000.0])
# C_variable=np.array([0.01, 0.1,0.5,1.0,10.0, 50.0, 100.0])
# C_variable=np.array(np.arange(0.1,1.5,0.1))
C_variable=np.array([0.001,0.01,0.1,1,5,10.0,25,50,75,100])
# C_variable=np.array([0.001,0.01,10.0,50,100] + list(np.arange(0.1,1.5,0.2))  )
# C_variable=np.array([0.01, 0.1,0.5,1.0, 5.0])
n_estimator=[ 32, 50, 64, 100 ,128, 256]

'''

    Classifier

'''

# This is the closest 
Classifier={}
Classifier['SVC']={'model':sklearn.svm.SVC(),\
                  'parameters':{'model__random_state':[1],\
                      'model__C':C_variable,\
                    'model__kernel': ['rbf'],\
                      # 'model__gamma':['auto'],\
                    'model__probability':[True],\
                                }}

   

loo=LeaveOneOut()
# CV_settings=loo
CV_settings=10

# =============================================================================
# Outputs
Best_predict_optimize={}

df_best_result_r2=pd.DataFrame([])
df_best_result_pear=pd.DataFrame([])
df_best_result_spear=pd.DataFrame([])
df_best_cross_score=pd.DataFrame([])
df_best_result_UAR=pd.DataFrame([])
df_best_result_AUC=pd.DataFrame([])
df_best_result_f1=pd.DataFrame([])
df_best_result_allThreeClassifiers=pd.DataFrame([])
# =============================================================================
Result_path="RESULTS/"
if not os.path.exists(Result_path):
    os.makedirs(Result_path)
final_result_file="_ADOS_{}.xlsx".format(args.suffix)

import warnings
warnings.filterwarnings("ignore")
count=0
OutFeature_dict=Dict()
Best_param_dict=Dict()
sellect_people_define=SellectP_define()

# ''' 要手動執行一次從Incorrect2Correct_indexes和Correct2Incorrect_indexes決定哪些indexes 需要算shap value 再在這邊指定哪些fold需要停下來算SHAP value '''
SHAP_inspect_idxs_manual=None # None means calculate SHAP value of all people
# SHAP_inspect_idxs_manual=[] # empty list means we do not execute shap function
# SHAP_inspect_idxs_manual=sorted(list(set([14, 21]+[]+[24, 28, 30, 31, 39, 41, 45]+[22, 23, 27, 47, 58]+[6, 13, 19, 23, 24, 25]+[28, 35, 38, 45])))

for clf_keys, clf in Classifier.items(): #Iterate among different classifiers 
    writer_clf = pd.ExcelWriter(Result_path+"/"+clf_keys+"_"+args.Feature_mode+"_"+final_result_file, engine = 'xlsxwriter')
    for feature_lab_str, features in Session_level_all.items():

        feature_keys, label_keys= feature_lab_str.split("::")
        feature_rawname=feature_keys[feature_keys.find('-')+1:]
        # if feature_rawname in paper_name_map.keys():
        #     featurename_paper=paper_name_map[feature_rawname]
        #     feature_keys=feature_keys.replace(feature_rawname,featurename_paper)
        
        if SHAP_inspect_idxs_manual != None:
            SHAP_inspect_idxs=SHAP_inspect_idxs_manual
        else:
            SHAP_inspect_idxs=range(len(features.y))
        
        Labels = Session_level_all.X[feature_keys]
        print("=====================Cross validation start==================")
        pipe = Pipeline(steps=[('scalar',StandardScaler()),("model", clf['model'])])
        p_grid=clf['parameters']
        Gclf = GridSearchCV(estimator=pipe, param_grid=p_grid, scoring=args.selectModelScoring, cv=CV_settings, refit=True, n_jobs=-1)
        Gclf_manual = GridSearchCV(estimator=pipe, param_grid=p_grid, scoring=args.selectModelScoring, cv=CV_settings, refit=True, n_jobs=-1)
        
        CVpredict=cross_val_predict(Gclf, features.X, features.y, cv=CV_settings)  
        
        # The cv is as the one in cross_val_predict function
        cv = sklearn.model_selection.check_cv(CV_settings,features.y,classifier=sklearn.base.is_classifier(Gclf))
        splits = list(cv.split(features.X, features.y, groups=None))
        test_indices = np.concatenate([test for _, test in splits])

        
        CVpredict_manual=np.zeros(len(features.y))
        for i, (train_index, test_index) in enumerate(splits):
            X_train, X_test = features.X.iloc[train_index], features.X.iloc[test_index]
            y_train, y_test = features.y.iloc[train_index], features.y.iloc[test_index]
            Gclf_manual.fit(X_train,y_train)
            result_bestmodel=Gclf_manual.predict(X_test)
            
            CVpredict_manual[test_index]=result_bestmodel
            
            # result_bestmodel_fitted_again=best_model_fittedagain.predict(X_test_encoded)
            CVpred_fromFunction=CVpredict[test_index]
            
            # SHAP value generating
            # logit_number=0
            # inspect_sample=0
            # If the indexes we want to examine are in that fold, store the whole fold
            # 先把整個fold記錄下來然後在analysis area再拆解
            SHAP_exam_lst=[i for i in test_index if i in SHAP_inspect_idxs]
            if len(SHAP_exam_lst) != 0:
                explainer = shap.KernelExplainer(Gclf_manual.predict_proba, X_train)
                shap_values = explainer.shap_values(X_test)
                Session_level_all[feature_lab_str]['SHAP_info']['_'.join(test_index.astype(str))].explainer_expected_value=explainer.expected_value
                Session_level_all[feature_lab_str]['SHAP_info']['_'.join(test_index.astype(str))].shap_values=shap_values # shap_values= [logit, index, feature]
                Session_level_all[feature_lab_str]['SHAP_info']['_'.join(test_index.astype(str))].XTest=X_test
                Session_level_all[feature_lab_str]['Logit{}_predictproba'.format(logit_number)]['_'.join(test_index.astype(str))].predictproba=Gclf_manual.predict_proba(X_test)[:,logit_number]
                Session_level_all[feature_lab_str]['Logit{}_predictproba'.format(logit_number)]['_'.join(test_index.astype(str))].decisionfunc=Gclf_manual.decision_function(X_test)
                # Session_level_all[feature_lab_str]['SHAP_info']['_'.join(test_index)].testIndex=test_index
            # shap.force_plot(explainer.expected_value[logit_number], shap_values[logit_number][inspect_sample,:], X_test.iloc[inspect_sample,:], matplotlib=True,show=False)
            
            
            # assert (result_bestmodel==result_bestmodel_fitted_again).all()
            assert (result_bestmodel==CVpred_fromFunction).all()
        
            
        assert (CVpredict_manual==CVpredict).all()
        Session_level_all[feature_lab_str]['y_pred']=CVpredict_manual
        Session_level_all[feature_lab_str]['y_true']=features.y
        
        Gclf.fit(features.X,features.y)
        if clf_keys == "EN":
            print('The coefficient of best estimator is: ',Gclf.best_estimator_.coef_)
        
        print("The best score with scoring parameter: 'r2' is", Gclf.best_score_)
        print("The best parameters are :", Gclf.best_params_)
        best_parameters=Gclf.best_params_
        best_score=Gclf.best_score_
        best_parameters.update({'best_score':best_score})
        Best_param_dict[feature_lab_str]=best_parameters
        cv_results_info=Gclf.cv_results_

        num_ASD=len(np.where(features.y==sellect_people_define.ASDTD_label['ASD'])[0])
        num_TD=len(np.where(features.y==sellect_people_define.ASDTD_label['TD'])[0])
        
        if features.feattype == 'regression':
            r2=r2_score(features.y,CVpredict )
            n,p=features.X.shape
            r2_adj=1-(1-r2)*(n-1)/(n-p-1)
            pearson_result, pearson_p=pearsonr(features.y,CVpredict )
            spear_result, spearman_p=spearmanr(features.y,CVpredict )
            print('Feature {0}, label {1} ,spear_result {2}'.format(feature_keys, label_keys,spear_result))
        elif features.feattype == 'classification':
            n,p=features.X.shape
            CM=confusion_matrix(features.y, CVpredict)
            Session_level_all[feature_lab_str]['Confusion_matrix']=pd.DataFrame(CM,\
                                                                    index=['y_true_{}'.format(ii) for ii in range(CM.shape[0])],\
                                                                    columns=['y_pred_{}'.format(ii) for ii in range(CM.shape[1])])
            UAR=recall_score(features.y, CVpredict, average='macro')
            AUC=roc_auc_score(features.y, CVpredict)
            f1Score=f1_score(features.y, CVpredict, average='macro')
            print('Feature {0}, label {1} ,UAR {2}'.format(feature_keys, label_keys,UAR))
            
        if args.Plot and p <2:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 10), sharey=True)
            kernel_label = [clf_keys]
            model_color = ['m']
            # axes.plot((features.X - min(features.X) )/ max(features.X), Gclf.best_estimator_.fit(features.X,features.y).predict(features.X), color=model_color[0],
            #               label='CV Predict')
            axes.scatter((features.X.values - min(features.X.values) )/ max(features.X.values), CVpredict, 
                         facecolor="none", edgecolor="k", s=150,
                         label='{}'.format(feature_lab_str)
                         )
            axes.scatter((features.X.values - min(features.X.values) )/ max(features.X.values), features.y, 
                         facecolor="none", edgecolor="r", s=50,
                         label='Real Y')
            axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)
            
            Plot_path='./Plot/LinearRegress/'
            if not os.path.exists(Plot_path):
                os.makedirs(Plot_path)
            plot_file=Plot_path+"/{0}_{1}.png".format(clf_keys,feature_lab_str)
            plt.savefig(plot_file, dpi=200) 
        
        # =============================================================================
        '''
            Inspect the best result
        '''
        # =============================================================================
        Best_predict_optimize[label_keys]=pd.DataFrame(np.vstack((CVpredict,features.y)).T,columns=['y_pred','y'])
        excel_path='./Statistics/prediction_result'
        if not os.path.exists(excel_path):
            os.makedirs(excel_path)
        excel_file=excel_path+"/{0}_{1}.xlsx"
        writer = pd.ExcelWriter(excel_file.format(clf_keys,feature_keys.replace(":","")), engine = 'xlsxwriter')
        for label_name in  Best_predict_optimize.keys():
            Best_predict_optimize[label_name].to_excel(writer,sheet_name=label_name.replace("/","_"))
        writer.save()
                                
        # ================================================      =============================
        if features.feattype == 'regression':
            df_best_result_r2.loc[feature_keys,label_keys]='{0}/{1}'.format(np.round(r2_adj,3),np.round(np.nan,6))
            df_best_result_pear.loc[feature_keys,label_keys]='{0}/{1}'.format(np.round(pearson_result,3),np.round(pearson_p,6))
            df_best_result_spear.loc[feature_keys,label_keys]='{0}/{1}'.format(np.round(spear_result,3),np.round(spearman_p,6))
            df_best_result_spear.loc[feature_keys,'de-zero_num']=len(features.X)
            # df_best_cross_score.loc[feature_keys,label_keys]=Score.mean()
            df_best_result_allThreeClassifiers.loc[feature_keys,'{0}/{1} (R2adj/pear/spear)'.format(label_keys,clf_keys)]\
                        ='{0}/{1}/{2}'.format(np.round(r2_adj,3),np.round(pearson_result,3),np.round(spear_result,3))

        elif features.feattype == 'classification':
            df_best_result_UAR.loc[feature_keys,label_keys]='{0}'.format(UAR)
            df_best_result_AUC.loc[feature_keys,label_keys]='{0}'.format(AUC)
            df_best_result_f1.loc[feature_keys,label_keys]='{0}'.format(f1Score)
            # df_best_result_allThreeClassifiers.loc[feature_keys,'{0}/{1} (UAR/AUC/f1score)'.format(label_keys,clf_keys)]\
            #             ='{0}/{1}/{2}'.format(np.round(UAR,3),np.round(AUC,3),np.round(f1Score,3))
            df_best_result_allThreeClassifiers.loc[feature_keys,'{0}/{1}'.format(label_keys,clf_keys)]\
                        ='{0}'.format(np.round(UAR,3))
            df_best_result_allThreeClassifiers.loc[feature_keys,'num_ASD']\
                        ='{0}'.format(num_ASD)
            df_best_result_allThreeClassifiers.loc[feature_keys,'num_TD']\
                        ='{0}'.format(num_TD)
        count+=1
    if features.feattype == 'regression':
        df_best_result_r2.to_excel(writer_clf,sheet_name="R2_adj")
        df_best_result_pear.to_excel(writer_clf,sheet_name="pear")
        df_best_result_spear.to_excel(writer_clf,sheet_name="spear")
        df_best_result_spear.to_csv(Result_path+"/"+clf_keys+"_"+args.Feature_mode+"_spearman.csv")
    elif features.feattype == 'classification':
        df_best_result_UAR.to_excel(writer_clf,sheet_name="UAR")
        df_best_result_AUC.to_excel(writer_clf,sheet_name="AUC")
        df_best_result_f1.to_excel(writer_clf,sheet_name="f1")

writer_clf.save()
df_best_result_allThreeClassifiers.to_excel(Result_path+"/"+"Classification_"+args.Feature_mode+"_3clsferRESULT.xlsx")
print(df_best_result_allThreeClassifiers)

# Change to paper name
df_allThreeClassifiers_paperName=df_best_result_allThreeClassifiers.copy()
index_bag=[]
for exp_str in df_best_result_allThreeClassifiers.index:
    experiment_name, feature_name=exp_str.split(" >> ")
    paper_idx='+'.join([Swap2PaperName(n, PprNmeMp) for n in feature_name.split("+")])
    index_bag.append(paper_idx)
df_allThreeClassifiers_paperName.index=index_bag

#%%
# =============================================================================
'''

    Analysis part

'''
def Organize_Needed_SHAP_info(Incorrect2Correct_indexes, Session_level_all, proposed_expstr):
    Incorrect2Correct_info_dict=Dict()
    for tst_idx in Incorrect2Correct_indexes:
        for key, values in Session_level_all[proposed_expstr]['SHAP_info'].items():
            test_fold_idx=[int(k) for k in key.split("_")]
            for i,ii in enumerate(test_fold_idx): #ii is the index of the sample, i is the position of this sample in this test fold
                if tst_idx == ii:
                    Incorrect2Correct_info_dict[tst_idx]['XTest']=values['XTest'].iloc[i,:]
                    Incorrect2Correct_info_dict[tst_idx]['explainer_expected_value']=values['explainer_expected_value']
                    shap_values_array=[array[i,:] for array in values['shap_values']]
                    df_shap_values=pd.DataFrame(shap_values_array,columns=Incorrect2Correct_info_dict[tst_idx]['XTest'].index)
                    Incorrect2Correct_info_dict[tst_idx]['shap_values']=df_shap_values
                    # print("testing sample ", ii, "is in the ", i, "position of test fold", key)
                    assert (Incorrect2Correct_info_dict[tst_idx]['XTest'] == Session_level_all[proposed_expstr]['X'].iloc[tst_idx]).all()
                    # print("It's feature value captured is", Incorrect2Correct_info_dict[tst_idx]['XTest'])
                    # print("It's original X value is", Session_level_all[proposed_expstr]['X'].iloc[tst_idx])
                    # print("See if they match")
    return Incorrect2Correct_info_dict
def Get_Model_Type12Errors(model_str='baseline', tureLab_str='y_true'):
    # Positive = ASD
    Type1Err= ( df_Y_pred[model_str]  == sellect_people_define.ASDTD_label['ASD']) & ( df_Y_pred[tureLab_str] == sellect_people_define.ASDTD_label['TD']  )
    Type2Err= ( df_Y_pred[model_str]  == sellect_people_define.ASDTD_label['TD'] ) & ( df_Y_pred[tureLab_str] == sellect_people_define.ASDTD_label['ASD']  )
    return Type1Err, Type2Err

def Organize_Needed_decisionProb(Incorrect2Correct_indexes, Session_level_all, proposed_expstr):
    Incorrect2Correct_info_dict=Dict()
    for tst_idx in Incorrect2Correct_indexes:
        for key, values in Session_level_all[proposed_expstr]['Logit{}_predictproba'.format(logit_number)].items():
            test_fold_idx=[int(k) for k in key.split("_")]
            for i,ii in enumerate(test_fold_idx): #ii is the index of the sample, i is the position of this sample in this test fold
                if tst_idx == ii:
                    Incorrect2Correct_info_dict[tst_idx]['predictproba']=values['predictproba'][i]
                    Incorrect2Correct_info_dict[tst_idx]['decisionfunc']=values['decisionfunc'][i]

    return Incorrect2Correct_info_dict


# =============================================================================

'''

    Part 1: Check incorrect to correct and correct to incorrect

'''
ASDTD2Logit_map={
    'TD': sellect_people_define.ASDTD_label['TD']-1,
    'ASD': sellect_people_define.ASDTD_label['ASD']-1,
    }

############################################################
# Low Minimal
proposed_expstr='TD vs df_feature_lowMinimal_CSS >> LOC_columns+Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD'
baseline_expstr='TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD'
[14, 21]
[]

# proposed_expstr='TD vs df_feature_lowMinimal_CSS >> LOC_columns+Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD'
# baseline_expstr='TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD'
# [1, 14, 15, 21]
# []

# proposed_expstr='TD vs df_feature_lowMinimal_CSS >> LOC_columns+Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD'
# baseline_expstr='TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD'
[12, 21]
[4 , 11]

# proposed_expstr='TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD'
# baseline_expstr='TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD'
# []
# [1, 15]
############################################################
# Moderate
# proposed_expstr='TD vs df_feature_moderate_CSS >> LOCDEP_Trend_D_cols+Phonation_Proximity_cols::ASDTD'
# baseline_expstr='TD vs df_feature_moderate_CSS >> Phonation_Proximity_cols::ASDTD'
[24, 28, 30, 31, 39, 41, 45]
[22, 23, 27, 47, 58]


############################################################
# high
# proposed_expstr='TD vs df_feature_high_CSS >> DEP_columns+Phonation_Trend_D_cols+Phonation_Proximity_cols::ASDTD'
# baseline_expstr='TD vs df_feature_high_CSS >> Phonation_Proximity_cols::ASDTD'
[6, 13, 19, 23, 24, 25]
[28, 35, 38, 45]

# proposed_expstr='TD vs df_feature_high_CSS >> DEP_columns+Phonation_Trend_D_cols+Phonation_Proximity_cols::ASDTD'
# proposed_expstr='TD vs df_feature_high_CSS >> Phonation_Trend_D_cols+Phonation_Proximity_cols::ASDTD'

# proposed_expstr='TD vs df_feature_high_CSS >> Phonation_Trend_D_cols+Phonation_Proximity_cols::ASDTD'
# baseline_expstr='TD vs df_feature_high_CSS >> Phonation_Proximity_cols::ASDTD'
[13, 19, 23, 24, 25]
[2, 3, 28, 29, 35, 36, 38, 44]

# proposed_expstr='TD vs df_feature_high_CSS >> DEP_columns+Phonation_Proximity_cols::ASDTD'
# baseline_expstr='TD vs df_feature_high_CSS >> Phonation_Proximity_cols::ASDTD'
experiment_title=baseline_expstr[re.search("df_feature_",baseline_expstr).end():re.search("_CSS >> ",baseline_expstr).start()]

# =============================================================================
# Error type analyses
# =============================================================================
Y_pred_lst=[
Session_level_all[proposed_expstr]['y_pred'],
Session_level_all[baseline_expstr]['y_pred'],
Session_level_all[proposed_expstr]['y_true'],
Session_level_all[proposed_expstr]['y_true'].index,
]
assert (Session_level_all[proposed_expstr]['y_true'] == Session_level_all[baseline_expstr]['y_true']).all()

df_Y_pred=pd.DataFrame(Y_pred_lst[:-1],index=['proposed','baseline','y_true']).T
# df_Y_pred_withName=pd.DataFrame(Y_pred_lst,index=['proposed','baseline','y_true','name']).T
# df_Index2Name_mapping=df_Y_pred_withName['name']


Incorrect=df_Y_pred['baseline'] != df_Y_pred['y_true']
Correct=df_Y_pred['proposed'] == df_Y_pred['y_true']
Incorrect2Correct= Correct & Incorrect


Incorrect=df_Y_pred['baseline'] == df_Y_pred['y_true']
Correct=df_Y_pred['proposed'] != df_Y_pred['y_true']
Correct2Incorrect= Correct & Incorrect

# Incorrect2Correct_indexes=list(df_Y_pred[Incorrect2Correct].index)
# Correct2Incorrect_indexes=list(df_Y_pred[Correct2Incorrect].index)
# print('Incorrect2Correct_indexes: ', Incorrect2Correct_indexes)
# print('Correct2Incorrect_indexes: ', Correct2Incorrect_indexes)


Ones=df_Y_pred['baseline'] ==sellect_people_define.ASDTD_label['ASD']
Twos=df_Y_pred['proposed'] ==sellect_people_define.ASDTD_label['TD']
Ones2Twos=  Ones & Twos

Twos=df_Y_pred['baseline'] ==sellect_people_define.ASDTD_label['TD']
Ones=df_Y_pred['proposed'] ==sellect_people_define.ASDTD_label['ASD']
Twos2Ones=  Ones & Twos

Ones2Twos_indexes=list(df_Y_pred[Ones2Twos].index)
Twos2Ones_indexes=list(df_Y_pred[Twos2Ones].index)



# Type1Err_dict, Type2Err_dict={}, {}
# for model_str in ['baseline', 'proposed']:
#     Type1Err_dict[model_str], Type2Err_dict[model_str] = Get_Model_Type12Errors(model_str=model_str, tureLab_str='y_true')

# model_str='baseline'
# Type1Err_baseline_indexes=list(df_Y_pred[Type1Err_dict[model_str]].index)
# Type2Err_baseline_indexes=list(df_Y_pred[Type2Err_dict[model_str]].index)
# model_str='proposed'
# Type1Err_proposed_indexes=list(df_Y_pred[Type1Err_dict[model_str]].index)
# Type2Err_proposed_indexes=list(df_Y_pred[Type2Err_dict[model_str]].index)


# All_indexes=list(set(Type1Err_baseline_indexes+Type2Err_baseline_indexes+Type1Err_proposed_indexes+Type2Err_proposed_indexes))
#%%
# For Loop 畫炫炮的錯誤型態分析加上SHAP summary plot (Changed smaples的logit 1 decision function 的移動)
'''

    Part 2: Check the SHAP values based on indexes in part 1
    
    先紀錄，再執行分析和畫圖

'''

from shap.plots._labels import labels
from shap.plots import colors
#%%
##############################################
def Plot_CoolErrorAnal( 
                       selected_idxs,
                       Baseline_changed_decision_info_dict,
                       Proposed_changed_decision_info_dict,
                       Baseline_total_decision_info_dict,
                       Proposed_total_decision_info_dict,
                       Incorrect2Correct_bool,
                       Correct2Incorrect_bool,
                       df_Y_pred,
                       show_dots_max=1.1,
                       font_size=12,
                       ):

    df_Proposed_changed_decision_info_dict=pd.DataFrame.from_dict(Proposed_changed_decision_info_dict,orient='index')
    df_Baseline_changed_decision_info_dict=pd.DataFrame.from_dict(Baseline_changed_decision_info_dict,orient='index')
    df_Proposed_total_decision_info_dict=pd.DataFrame.from_dict(Proposed_total_decision_info_dict,orient='index')
    df_Baseline_total_decision_info_dict=pd.DataFrame.from_dict(Baseline_total_decision_info_dict,orient='index')
    Sample_idxs_array=df_Baseline_changed_decision_info_dict.index.values
    
    # df_Y_true=df_Y_pred.loc[df_Baseline_changed_decision_info_dict.index]['y_true']
    # df_Y_true_ASD_bool=df_Y_true==sellect_people_define.ASDTD_label['ASD']
    # df_Y_true_TD_bool=df_Y_true==sellect_people_define.ASDTD_label['TD']
    Incorrect_baseline=df_Y_pred['baseline'] != df_Y_pred['y_true']
    Incorrect_proposed=df_Y_pred['proposed'] != df_Y_pred['y_true']
    Correct_baseline=df_Y_pred['baseline'] == df_Y_pred['y_true']
    Correct_proposed=df_Y_pred['proposed'] == df_Y_pred['y_true']
    
    
    # decision function 負的表示predict logit 0, 正的表示logit 1
    Baseline_x= df_Baseline_changed_decision_info_dict['predictproba'].values
    Baseline_y_decisionfunc= df_Baseline_changed_decision_info_dict['decisionfunc'].abs().values.copy()
    Baseline_y= df_Baseline_changed_decision_info_dict['decisionfunc'].abs().values.copy()
    # Baseline_total_x= df_Baseline_total_decision_info_dict['predictproba'].values
    Baseline_total_y_decisionfunc= df_Baseline_total_decision_info_dict['decisionfunc'].abs().values.copy()
    Baseline_total_y= df_Baseline_total_decision_info_dict['decisionfunc'].abs().values.copy()
    # 如果是TD decision function是正的y軸就是正的，decision function是負的y軸就是負的
    # 如果是ASD decision function是正的y軸就是負的，decision function是負的y軸就是正的
    Baseline_y[Incorrect2Correct_bool.loc[df_Baseline_changed_decision_info_dict.index]]\
        =-Baseline_y_decisionfunc[Incorrect2Correct_bool.loc[df_Baseline_changed_decision_info_dict.index]]
    Baseline_y[Correct2Incorrect_bool.loc[df_Baseline_changed_decision_info_dict.index]]\
        =Baseline_y_decisionfunc[Correct2Incorrect_bool.loc[df_Baseline_changed_decision_info_dict.index]]
    
    df_Baseline_total_decision_info_dict.loc[Incorrect_baseline]
    
    Baseline_total_y[Incorrect_baseline.values]=-Baseline_total_y_decisionfunc[Incorrect_baseline.values]
    Baseline_total_y[Correct_baseline.values]=Baseline_total_y_decisionfunc[Correct_baseline.values]
    
    
    
    Proposed_x= df_Proposed_changed_decision_info_dict['predictproba'].values
    Proposed_y_decisionfunc= df_Proposed_changed_decision_info_dict['decisionfunc'].abs().values.copy()
    Proposed_y= df_Proposed_changed_decision_info_dict['decisionfunc'].abs().values.copy()
    # Proposed_total_x= df_Proposed_total_decision_info_dict['predictproba'].values
    Proposed_total_y_decisionfunc= df_Proposed_total_decision_info_dict['decisionfunc'].abs().values.copy()
    Proposed_total_y= df_Proposed_total_decision_info_dict['decisionfunc'].abs().values.copy()
    
    Proposed_y[Incorrect2Correct_bool.loc[df_Baseline_changed_decision_info_dict.index]]=\
        Proposed_y_decisionfunc[Incorrect2Correct_bool.loc[df_Baseline_changed_decision_info_dict.index]]
    Proposed_y[Correct2Incorrect_bool.loc[df_Baseline_changed_decision_info_dict.index]]=\
        -Proposed_y_decisionfunc[Correct2Incorrect_bool.loc[df_Baseline_changed_decision_info_dict.index]]
    
    Proposed_total_y[Incorrect_proposed.values]=-Proposed_total_y_decisionfunc[Incorrect_proposed.values]
    Proposed_total_y[Correct_proposed.values]=Proposed_total_y_decisionfunc[Correct_proposed.values]
    
    Total_y=list(Baseline_y)+list(Proposed_y)
    Total_x=list(Baseline_x)+list(Proposed_x)
    
    # y_max=np.max(Total_y)
    # y_min=np.min(Total_y)
    y_scale=.9
    y_max=y_scale
    y_min=-y_scale
    
    x_max=np.max(Total_x)
    x_min=np.min(Total_x)
    x_middle=(x_max+x_min)/2
    y_middle=(y_max+y_min)/2
    
    # ax.annotate("", xy=(起點x, 起點y), xytext=(終點x, 終點y),arrowprops=dict(arrowstyle="->"))
    for B_x, B_y, P_x, P_y,idx in zip(Baseline_x,Baseline_y,Proposed_x,Proposed_y,Sample_idxs_array):
        plt.annotate("", xy=(B_x, B_y), xytext=(P_x, P_y),arrowprops=dict(arrowstyle="<-",alpha=.4))
    
    plt.scatter(Baseline_x, Baseline_y, c='b', alpha=1)
    plt.scatter(Proposed_x, Proposed_y, c='r', alpha=1)
    
    # 去掉那些背景但是大於1.1的點
    BackgroundDots_baseline=df_Baseline_total_decision_info_dict[df_Baseline_total_decision_info_dict['predictproba'].abs()<show_dots_max]
    BackgroundDots_proposed=df_Proposed_total_decision_info_dict[df_Proposed_total_decision_info_dict['predictproba'].abs()<show_dots_max]
    
    plt.scatter(df_Baseline_total_decision_info_dict.predictproba, Baseline_total_y, c='b', alpha=.05)
    plt.scatter(df_Proposed_total_decision_info_dict.predictproba, Proposed_total_y, c='r', alpha=.05)
    
    plt.annotate('',xy=(0, 0), xytext=(1, 0),arrowprops=dict(arrowstyle="<->",alpha=1,))                                                                     
    plt.annotate('',xy=(0.5, y_min), xytext=(0.5, y_max),arrowprops=dict(arrowstyle="<->",alpha=1,))
    margin_y=(y_max-y_min)/10
    margin_x=(1-0)/20
    
    # plt.text(0, y_middle-margin_y, 'ASD', fontsize=font_size)
    # plt.text(1-margin_x, y_middle-margin_y, 'TD', fontsize=font_size)
    plt.text(0, y_middle, 'ASD', fontsize=font_size)
    plt.text(1-margin_x, y_middle, 'TD', fontsize=font_size)
        
    plt.text(0.5, y_min, 'Incorrect', fontsize=font_size)
    plt.text(0.5, y_max-margin_y, 'Correct', fontsize=font_size)   
    # fig.patch.set_visible(True)
    plt.axis('off')
    # plt.ylim(-1.5,1.5)
    # plt.xlim(-0,1.1)
    # plt.title(experiment_title)
    # plt.show()
    
    plt.xlim(-0,1.1)
    plt.ylim(-show_dots_max,show_dots_max)
    # plt.title(experiment_title)
    # plt.show()

# TODO: remove unused title argument / use title argument
# TODO: Add support for hclustering based explanations where we sort the leaf order by magnitude and then show the dendrogram to the left
def summary_legacy_jack(df_shap_values, features=None, feature_names=None, max_display=None, plot_type=None,
                  color=None, axis_color="#333333", title=None, alpha=1, show=True, sort=True,
                  color_bar=True, plot_size="auto", layered_violin_max_num_bins=20, class_names=None,
                  class_inds=None,
                  color_bar_label=labels["FEATURE_VALUE"],
                  cmap=colors.red_blue,
                  font_size=13,\
                  dot_size=10,\
                  tick_sizes=20,\
                  colorbar_aspect=100,\
                  # depreciated
                  auto_size_plot=None,
                  use_log_scale=False):
    assert type(df_shap_values) == pd.core.frame.DataFrame
    Marker_dict={
        'O':'<',
        'X':'X',
        }
    markers_bag=list(df_shap_values.index)
    shap_values=df_shap_values.values

    multi_class = False
    if isinstance(shap_values, list):
        multi_class = True
        if plot_type is None:
            plot_type = "bar" # default for multi-output explanations
        assert plot_type == "bar", "Only plot_type = 'bar' is supported for multi-output explanations!"
    else:
        if plot_type is None:
            plot_type = "dot" # default for single output explanations
        assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."

    # default color:
    if color is None:
        if plot_type == 'layered_violin':
            color = "coolwarm"
        elif multi_class:
            color = lambda i: colors.red_blue_circle(i/len(shap_values))
        else:
            color = colors.blue_rgb

    idx2cat = None
    # convert from a DataFrame or other types
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = features.columns
        # feature index to category flag
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = (shap_values[0].shape[1] if multi_class else shap_values.shape[1])

    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the " \
                    "provided data matrix."
        if num_features - 1 == features.shape[1]:
            assert False, shape_msg + " Perhaps the extra column in the shap_values matrix is the " \
                          "constant offset? Of so just pass shap_values[:,:-1]."
        else:
            assert num_features == features.shape[1], shape_msg

    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

    if use_log_scale:
        plt.xscale('symlog')
    if max_display is None:
        max_display = 20
    if sort:
        # order features by the sum of their effect magnitudes
        if multi_class:
            feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
        else:
            feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)):]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    row_height = 0.4
    if plot_size == "auto":
        plt.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    elif type(plot_size) in (list, tuple):
        plt.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        plt.gcf().set_size_inches(8, len(feature_order) * plot_size + 1.5)
    plt.axvline(x=0, color="#999999", zorder=-1)

    for pos, i in enumerate(feature_order):
        plt.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = shap_values[:, i]  # shap_values: (feature_num, people)
        values = None if features is None else features[:, i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        if values is not None:
            values = values[inds]
        shaps = shaps[inds]
        colored_feature = True
        try:
            if idx2cat is not None and idx2cat[i]: # check categorical feature
                colored_feature = False
            else:
                values = np.array(values, dtype=np.float64)  # make sure this can be numeric
        except:
            colored_feature = False
        N = len(shaps)
        # hspacing = (np.max(shaps) - np.min(shaps)) / 200
        # curr_bin = []
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        if features is not None and colored_feature:
            # trim the color range, but prevent the color range from collapsing
            vmin = np.nanpercentile(values, 5)
            vmax = np.nanpercentile(values, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(values, 1)
                vmax = np.nanpercentile(values, 99)
                if vmin == vmax:
                    vmin = np.min(values)
                    vmax = np.max(values)
            if vmin > vmax: # fixes rare numerical precision issues
                vmin = vmax

            assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

            # plot the nan values in the interaction feature as grey
            nan_mask = np.isnan(values)
            # 這個在把nan的值畫成灰色，不重要
            plt.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                       vmax=vmax, s=16, alpha=alpha, linewidth=0,
                       zorder=3, rasterized=len(shaps) > 500)

            # plot the non-nan values colored by the trimmed feature value
            cvals = values[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            
            
            # 這個是主要話資料點的function
            # 
            # plt.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
            #            cmap=cmap, vmin=vmin, vmax=vmax, s=16,
            #            marker=Marker_dict[markers_bag[pos]],
            #            c=cvals, alpha=alpha, linewidth=0,
            #            zorder=3, rasterized=len(shaps) > 500)
            position_correct_bool=np.array(markers_bag) == 'O'
            position_Incorrect_bool=np.array(markers_bag) == 'X'
            
            shap_position_correct_x_array=shaps[np.invert(nan_mask)][position_correct_bool]
            shap_position_correct_y_array=(pos + ys[np.invert(nan_mask)])[position_correct_bool]
            
            shap_position_Incorrect_x_array=shaps[np.invert(nan_mask)][position_Incorrect_bool]
            shap_position_Incorrect_y_array=(pos + ys[np.invert(nan_mask)])[position_Incorrect_bool]
            
            cvals_correct=cvals[position_correct_bool]
            cvals_Incorrect=cvals[position_Incorrect_bool]
            
            
            if len(shap_position_correct_x_array)>0:
                plt.scatter(shap_position_correct_x_array, shap_position_correct_y_array,
                           cmap=cmap, vmin=vmin, vmax=vmax, s=dot_size,
                           marker=Marker_dict['O'],
                           c=cvals_correct, alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)
            if len(shap_position_Incorrect_x_array)>0:
                plt.scatter(shap_position_Incorrect_x_array, shap_position_Incorrect_y_array,
                           cmap=cmap, vmin=vmin, vmax=vmax, s=dot_size,
                           marker=Marker_dict['X'],
                           c=cvals_Incorrect, alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)
            
        else:
            plt.scatter(shaps, pos + ys, s=16, alpha=alpha, linewidth=0, zorder=3,
                       color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)
    # draw the color bar
    if color_bar and features is not None and plot_type != "bar" and \
            (plot_type != "layered_violin" or color in plt.cm.datad):

        import matplotlib.cm as cm
        # from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
        # aspect = 20
        # pad_fraction = 0.5
        m = cm.ScalarMappable(cmap=cmap if plot_type != "layered_violin" else plt.get_cmap(color))
        m.set_array([0, tick_sizes])
        
        cb = plt.colorbar(m, ticks=[0, tick_sizes], aspect=colorbar_aspect)
        cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.9) * 20)
        # cb.draw_all()

    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().tick_params(color=axis_color, labelcolor=axis_color)
    # Get or set the current tick locations and labels of the y-axis.
    # 這邊用來把feature 名稱放在y軸
    plt.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=font_size)
    if plot_type != "bar":
        plt.gca().tick_params('y', length=20, width=0.5, which='major')
    plt.gca().tick_params('x', labelsize=11)
    plt.ylim(-1, len(feature_order))
    if plot_type == "bar":
        plt.xlabel(labels['GLOBAL_VALUE'], fontsize=font_size)
    else:
        plt.xlabel(labels['VALUE'], fontsize=font_size)
    if show:
        plt.show()

proposed_expstr_lst=[
    'TD vs df_feature_lowMinimal_CSS >> LOC_columns+Phonation_Trend_D_cols+Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD',
    'TD vs df_feature_moderate_CSS >> LOCDEP_Trend_D_cols+Phonation_Proximity_cols::ASDTD',
    'TD vs df_feature_high_CSS >> DEP_columns+Phonation_Trend_D_cols+Phonation_Proximity_cols::ASDTD'
    ]
baseline_expstr_lst=[
    'TD vs df_feature_lowMinimal_CSS >> Phonation_Trend_K_cols+Phonation_Syncrony_cols::ASDTD',
    'TD vs df_feature_moderate_CSS >> Phonation_Proximity_cols::ASDTD',
    'TD vs df_feature_high_CSS >> Phonation_Proximity_cols::ASDTD'
    ]

count=0
plt_unit=2
Subplot_columns=3*plt_unit
Subplot_rows=2
ErrPlt_base_idx=2
SummaryPlt_base_idx=8
global_font_size=12
plt.figure(figsize=(15,8))

ExperimentTitleMap_dict={
    'lowMinimal':'low-symptom',
    'moderate':'moderate-symptom',
    'high':'high-symptom',
    
    }

# plt.suptitle(experiment_title, fontsize=16)
# for iiii in [1]:
for baseline_expstr, proposed_expstr in zip(baseline_expstr_lst,proposed_expstr_lst):    
    # print(iiii)
    #Data prepare
    #//////////////////////////////////////////////////////////////////////////
    Y_pred_lst=[
    Session_level_all[proposed_expstr]['y_pred'],
    Session_level_all[baseline_expstr]['y_pred'],
    Session_level_all[proposed_expstr]['y_true'],
    Session_level_all[proposed_expstr]['y_true'].index,
    ]
    assert (Session_level_all[proposed_expstr]['y_true'] == Session_level_all[baseline_expstr]['y_true']).all()

    df_Y_pred=pd.DataFrame(Y_pred_lst[:-1],index=['proposed','baseline','y_true']).T
    
    Incorrect=df_Y_pred['baseline'] != df_Y_pred['y_true']
    Correct=df_Y_pred['proposed'] == df_Y_pred['y_true']
    Incorrect2Correct= Correct & Incorrect


    Incorrect=df_Y_pred['baseline'] == df_Y_pred['y_true']
    Correct=df_Y_pred['proposed'] != df_Y_pred['y_true']
    Correct2Incorrect= Correct & Incorrect
    
    Incorrect2Correct_indexes=list(df_Y_pred[Incorrect2Correct].index)
    Correct2Incorrect_indexes=list(df_Y_pred[Correct2Incorrect].index)
    
    Ones=df_Y_pred['baseline'] ==sellect_people_define.ASDTD_label['ASD']
    Twos=df_Y_pred['proposed'] ==sellect_people_define.ASDTD_label['TD']
    Ones2Twos=  Ones & Twos

    Twos=df_Y_pred['baseline'] ==sellect_people_define.ASDTD_label['TD']
    Ones=df_Y_pred['proposed'] ==sellect_people_define.ASDTD_label['ASD']
    Twos2Ones=  Ones & Twos

    Ones2Twos_indexes=list(df_Y_pred[Ones2Twos].index)
    Twos2Ones_indexes=list(df_Y_pred[Twos2Ones].index)
    
    
    experiment_title=baseline_expstr[re.search("df_feature_",baseline_expstr).end():re.search("_CSS >> ",baseline_expstr).start()]
    #//////////////////////////////////////////////////////////////////////////
    
    
    
    
    plt.subplot(Subplot_rows,Subplot_columns,count*plt_unit+ErrPlt_base_idx)
    # print('Error_plt: ',Subplot_rows,Subplot_columns,count*plt_unit+ErrPlt_base_idx)
    
    # step 1: prepare data
    selected_idxs=Ones2Twos_indexes+Twos2Ones_indexes
    Baseline_changed_decision_info_dict=Organize_Needed_decisionProb(selected_idxs, Session_level_all, baseline_expstr)
    Proposed_changed_decision_info_dict=Organize_Needed_decisionProb(selected_idxs, Session_level_all, proposed_expstr)
    Baseline_total_decision_info_dict=Organize_Needed_decisionProb(df_Y_pred.index, Session_level_all, baseline_expstr)
    Proposed_total_decision_info_dict=Organize_Needed_decisionProb(df_Y_pred.index, Session_level_all, proposed_expstr)
    
    
    
    Plot_CoolErrorAnal(selected_idxs,Baseline_changed_decision_info_dict,
                       Proposed_changed_decision_info_dict,
                       Baseline_total_decision_info_dict,
                       Proposed_total_decision_info_dict,
                       Incorrect2Correct_bool=Incorrect2Correct,
                       Correct2Incorrect_bool=Correct2Incorrect,
                       df_Y_pred=df_Y_pred,
                       show_dots_max=1.1,
                       font_size=global_font_size,
                       )
    plt.gca().set_title(ExperimentTitleMap_dict[experiment_title], fontsize=global_font_size+10)
    
    SummaryPltPosition=count*plt_unit+SummaryPlt_base_idx
    plt.subplot(Subplot_rows,Subplot_columns,SummaryPltPosition)
    print('Summary_lt: ',Subplot_rows,Subplot_columns,SummaryPltPosition)
    
    Proposed_changed_info_dict=Organize_Needed_SHAP_info(selected_idxs, Session_level_all, proposed_expstr)
    shap_values_proposedchanged, df_XTest, keys=Prepare_data_for_summaryPlot(Proposed_changed_info_dict,\
                                                              feature_columns=None,\
                                                              PprNmeMp=PprNmeMp) 
    
    Modified_columns=[]    
    for c in df_XTest.columns: 
        if c.startswith('$') and c.endswith('$'):
            Modified_columns.append(c.replace("[$","[").replace("$]","]"))
        else:
            Modified_columns.append(c)
    df_XTest.columns=Modified_columns
    
        
    # df_shap_values DataFrame 
    df_shap_values_proposedchanged=pd.DataFrame(shap_values_proposedchanged[1],columns=df_XTest.columns,index=df_XTest.index)
    
    df_shap_values_proposedchanged_corrIncorrMarkers=df_shap_values_proposedchanged.copy()
    new_idx_bag=[]
    for idx in df_shap_values_proposedchanged.index:
        if idx in Incorrect2Correct_indexes:
            new_idx_bag.append('O')
        elif idx in Correct2Incorrect_indexes:
            new_idx_bag.append('X')
        else:
            raise KeyError()
    df_shap_values_proposedchanged_corrIncorrMarkers.index=new_idx_bag
    
    max_display=5
    # shap.summary_plot(shap_values[1], df_XTest,feature_names=df_XTest.columns, plot_size=None,show=False, max_display=8)
    
    # summary_legacy_jack(df_shap_values_proposedchanged_corrIncorrMarkers, df_XTest,feature_names=df_XTest.columns,\
    #                plot_size=None,show=False, max_display=max_display,title=None,\
    #                font_size=8,dot_size=15)
    
    if SummaryPltPosition==12:
        color_bar=True
    else:
        color_bar=False
    
    summary_legacy_jack(df_shap_values_proposedchanged_corrIncorrMarkers, df_XTest,feature_names=df_XTest.columns,\
                    plot_size=None,show=False, max_display=max_display,title=None,\
                    color_bar=color_bar,\
                    tick_sizes=10,\
                    font_size=global_font_size,\
                    dot_size=30,\
                    colorbar_aspect=100)
    
    # plt.tight_layout()

    count+=1
# plt.tight_layout()
# plt.subplots_adjust(wspace=1.0)
plt.show()
print(f'Size: {plt.gcf().get_size_inches()}')
#%%


summary_legacy_jack(df_shap_values_proposedchanged_corrIncorrMarkers, df_XTest,feature_names=df_XTest.columns,\
                plot_size=None,show=False, max_display=max_display,title=None,\
                tick_sizes=15,\
                font_size=8,dot_size=15)
#%%    
# =============================================================================
# summary_legacy_jack
# =============================================================================
SummaryPlt_base_idx=1
plt.figure(figsize=(13,5))
for count in [0,1,2]:
    plt.subplot(1,6,count*plt_unit+SummaryPlt_base_idx)
    
    df_shap_values, features, feature_names=\
        df_shap_values_proposedchanged_corrIncorrMarkers,df_XTest, df_XTest.columns
    max_display=max_display
    plot_type=None
    color=None
    axis_color="#333333"
    title=None
    alpha=1
    show=False
    sort=True
    color_bar=True
    plot_size=None
    layered_violin_max_num_bins=20
    class_names=None
    color_bar_label=labels["FEATURE_VALUE"]
    cmap=colors.red_blue
    font_size=10
    dot_size=15
    # depreciated
    auto_size_plot=None
    use_log_scale=False
    
    assert type(df_shap_values) == pd.core.frame.DataFrame
    Marker_dict={
        'O':'<',
        'X':'X',
        }
    markers_bag=list(df_shap_values.index)
    shap_values=df_shap_values.values
    
    multi_class = False
    if isinstance(shap_values, list):
        multi_class = True
        if plot_type is None:
            plot_type = "bar" # default for multi-output explanations
        assert plot_type == "bar", "Only plot_type = 'bar' is supported for multi-output explanations!"
    else:
        if plot_type is None:
            plot_type = "dot" # default for single output explanations
        assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."
    
    # default color:
    if color is None:
        if plot_type == 'layered_violin':
            color = "coolwarm"
        elif multi_class:
            color = lambda i: colors.red_blue_circle(i/len(shap_values))
        else:
            color = colors.blue_rgb
    
    idx2cat = None
    # convert from a DataFrame or other types
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = features.columns
        # feature index to category flag
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None
    
    num_features = (shap_values[0].shape[1] if multi_class else shap_values.shape[1])
    
    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the " \
                    "provided data matrix."
        if num_features - 1 == features.shape[1]:
            assert False, shape_msg + " Perhaps the extra column in the shap_values matrix is the " \
                          "constant offset? Of so just pass shap_values[:,:-1]."
        else:
            assert num_features == features.shape[1], shape_msg
    
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])
    
    if use_log_scale:
        plt.xscale('symlog')
    
    
    
    if max_display is None:
        max_display = 20
    
    if sort:
        # order features by the sum of their effect magnitudes
        if multi_class:
            feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
        else:
            feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)):]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)
    
    row_height = 0.4
    if plot_size == "auto":
        plt.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    elif type(plot_size) in (list, tuple):
        plt.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        plt.gcf().set_size_inches(8, len(feature_order) * plot_size + 1.5)
    plt.axvline(x=0, color="#999999", zorder=-1)
    
    for pos, i in enumerate(feature_order):
        plt.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = shap_values[:, i]  # shap_values: (feature_num, people)
        values = None if features is None else features[:, i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        if values is not None:
            values = values[inds]
        shaps = shaps[inds]
        colored_feature = True
        try:
            if idx2cat is not None and idx2cat[i]: # check categorical feature
                colored_feature = False
            else:
                values = np.array(values, dtype=np.float64)  # make sure this can be numeric
        except:
            colored_feature = False
        N = len(shaps)
        # hspacing = (np.max(shaps) - np.min(shaps)) / 200
        # curr_bin = []
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))
    
        if features is not None and colored_feature:
            # trim the color range, but prevent the color range from collapsing
            vmin = np.nanpercentile(values, 5)
            vmax = np.nanpercentile(values, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(values, 1)
                vmax = np.nanpercentile(values, 99)
                if vmin == vmax:
                    vmin = np.min(values)
                    vmax = np.max(values)
            if vmin > vmax: # fixes rare numerical precision issues
                vmin = vmax
    
            assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"
    
            # plot the nan values in the interaction feature as grey
            nan_mask = np.isnan(values)
            # 這個在把nan的值畫成灰色，不重要
            plt.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                       vmax=vmax, s=16, alpha=alpha, linewidth=0,
                       zorder=3, rasterized=len(shaps) > 500)
    
            # plot the non-nan values colored by the trimmed feature value
            cvals = values[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            
            
            # 這個是主要話資料點的function
            # 
            # plt.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
            #            cmap=cmap, vmin=vmin, vmax=vmax, s=16,
            #            marker=Marker_dict[markers_bag[pos]],
            #            c=cvals, alpha=alpha, linewidth=0,
            #            zorder=3, rasterized=len(shaps) > 500)
            position_correct_bool=np.array(markers_bag) == 'O'
            position_Incorrect_bool=np.array(markers_bag) == 'X'
            
            shap_position_correct_x_array=shaps[np.invert(nan_mask)][position_correct_bool]
            shap_position_correct_y_array=(pos + ys[np.invert(nan_mask)])[position_correct_bool]
            
            shap_position_Incorrect_x_array=shaps[np.invert(nan_mask)][position_Incorrect_bool]
            shap_position_Incorrect_y_array=(pos + ys[np.invert(nan_mask)])[position_Incorrect_bool]
            
            cvals_correct=cvals[position_correct_bool]
            cvals_Incorrect=cvals[position_Incorrect_bool]
            
            
            if len(shap_position_correct_x_array)>0:
                plt.scatter(shap_position_correct_x_array, shap_position_correct_y_array,
                           cmap=cmap, vmin=vmin, vmax=vmax, s=dot_size,
                           marker=Marker_dict['O'],
                           c=cvals_correct, alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)
            if len(shap_position_Incorrect_x_array)>0:
                plt.scatter(shap_position_Incorrect_x_array, shap_position_Incorrect_y_array,
                           cmap=cmap, vmin=vmin, vmax=vmax, s=dot_size,
                           marker=Marker_dict['X'],
                           c=cvals_Incorrect, alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)
            
        else:
            plt.scatter(shaps, pos + ys, s=16, alpha=alpha, linewidth=0, zorder=3,
                       color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)
    # draw the color bar
    tick_sizes=20
    if color_bar and features is not None and plot_type != "bar" and \
            (plot_type != "layered_violin" or color in plt.cm.datad):
    
        import matplotlib.cm as cm
        # from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
        # aspect = 20
        # pad_fraction = 0.5
        m = cm.ScalarMappable(cmap=cmap if plot_type != "layered_violin" else plt.get_cmap(color))
        m.set_array([0, tick_sizes])
        
        cb = plt.colorbar(m, ticks=[0, tick_sizes], aspect=1000)
        cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.9) * 20)
        # cb.draw_all()
    
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().tick_params(color=axis_color, labelcolor=axis_color)
    # Get or set the current tick locations and labels of the y-axis.
    # 這邊用來把feature 名稱放在y軸
    plt.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=font_size)
    if plot_type != "bar":
        plt.gca().tick_params('y', length=20, width=0.5, which='major')
    plt.gca().tick_params('x', labelsize=11)
    plt.ylim(-1, len(feature_order))
    if plot_type == "bar":
        plt.xlabel(labels['GLOBAL_VALUE'], fontsize=font_size)
    else:
        plt.xlabel(labels['VALUE'], fontsize=font_size)
    if show:
        plt.show()


# summary_legacy_jack(df_shap_values_proposedchanged_corrIncorrMarkers, df_XTest,feature_names=df_XTest.columns,\
#                 plot_size=None,show=False, max_display=max_display,title=None,\
#                 font_size=8,dot_size=15)