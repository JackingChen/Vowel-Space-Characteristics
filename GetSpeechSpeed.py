#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 21:18:09 2021

@author: jackchen
"""

import os, glob
import pandas as pd
import argparse
import re
import pickle
from transformers import BertTokenizer
from Syncrony import Syncrony
from multiprocessing import Pool, current_process

def Files2BookeepUttDict(trnfiles_str,global_info, namestr_end_position):
    files=glob.glob(trnfiles_str+'/*.txt')
        
    DurSpeed_utt_dict=Dict()
    Bookeep_utt_dict=Dict()
    for file in tqdm(files):
        uttname=os.path.basename(file).replace(suffix,'')
        name=uttname[:re.search(namestr_end_position,uttname).start()]
        #Initialize here
        # if name not in DurSpeed_utt_dict.keys(): 
            # DurSpeed_utt_dict[name].phonedf=pd.DataFrame()
            # DurSpeed_utt_dict[name].speeddf=pd.DataFrame()
            # Bookeep_utt_dict[name]=list()
        
        df_uttinfo=global_info[global_info['utt'] == uttname]
        dur_utt=(df_uttinfo['ed'] - df_uttinfo['st']).values
        df_phoneinfo=pd.read_csv(file, sep='\t',header=None)
        df_phoneinfo.columns=['st','ed','txt']
        # Switch combined phone like ueI to two splitted phones w-eI
        regexp = re.compile(r'u(A|aI|ax|eI|O)_') #['uA_', 'uaI_', 'uax_', 'ueI_', 'uO_']
        u_Biphoneset = [symbbset for symbbset in phonewoprosody.Phoneme_sets.keys() if regexp.search(symbbset)]
        def ScanBiphone(df_phoneinfo,Phoneme_sets,Biphoneset):
            df_bool=df_phoneinfo.copy()
            for idx in df_phoneinfo.index:
                Biphone_lst= [symb for symbset in u_Biphoneset for symb in Phoneme_sets[symbset]]
                if df_phoneinfo.loc[idx,'txt'] in Biphone_lst:
                    df_bool.loc[idx,'txt']=True
                else:
                    df_bool.loc[idx,'txt']=False
                
            return df_bool
        
        df_u_BiphonBool=ScanBiphone(df_phoneinfo,phonewoprosody.Phoneme_sets,u_Biphoneset)
        
        for ind in df_phoneinfo[df_u_BiphonBool['txt']].index:
            st,ed,symb = df_phoneinfo.loc[ind]
            old_idx=df_phoneinfo.loc[ind].name
            dur_distrubute=(ed - st)/2
            
            left_symb=symb[:re.search('u',symb).end()]
            right_symb=symb[re.search('u',symb).end():]
            df_phoneinfo=df_phoneinfo.append(pd.DataFrame({'st': st, 'ed': st + dur_distrubute, \
                                                       'txt': '{0}'.format(phonewoprosody.LeftSymbMapp[left_symb])}, index=[old_idx+len(df_phoneinfo)]))
            df_phoneinfo=df_phoneinfo.append(pd.DataFrame({'st': st + dur_distrubute, 'ed': ed, \
                                                       'txt': '{0}-{1}'.format(phonewoprosody.LeftSymbMapp[left_symb],right_symb)}, index=[old_idx+len(df_phoneinfo)]))
            df_phoneinfo=df_phoneinfo.drop(index=old_idx)
        
        
        df_phoneinfo_DeSIL=df_phoneinfo[df_phoneinfo['txt'] !='SIL']
        
        NumOfphones=len(df_phoneinfo_DeSIL)
        if NumOfphones==0:
            if df_phoneinfo['txt'].values.all() == 'SIL':
                NumOfphones=len(df_phoneinfo)
            else:
                print("Error happens and the df  values is ", df_phoneinfo)
                raise ValueError()
        df_phoneinfo_DeSIL['dur']=df_phoneinfo_DeSIL['ed'] - df_phoneinfo_DeSIL['st']
        # utterance_speechspeed=pd.DataFrame(dur_utt/NumOfphones,columns=['speed'])
        
        # DurSpeed_utt_dict[name].phonedf=pd.concat([DurSpeed_utt_dict[name].phonedf,df_phoneinfo_DeSIL])
        # DurSpeed_utt_dict[name].speeddf=pd.concat([DurSpeed_utt_dict[name].speeddf,utterance_speechspeed])
        
        
        # Get context phone variety
        
        df_Utt_phfContextDep=pd.DataFrame([],columns=df_phoneinfo.columns)
        
        values = df_phoneinfo
        phoneSeq=list(values['txt'].astype(str))    
        if len(phoneSeq) == 1:
            df_ctxdepP=values.iloc[[0],:]
            df_ctxdepP.index=['{0}-{1}+{2}'.format('[s]',phoneSeq[0],'[\s]')]
            df_Utt_phfContextDep=df_Utt_phfContextDep.append(df_ctxdepP)
        else:    
            for i in range(len(phoneSeq)): # df_Utt_phfContextDep append each word
                
                df_ctxdepP=values.iloc[[i],:]
                critical_P=FindCentralPhone(phoneSeq[i])
                if i==0:
                    right_critical_P=FindCentralPhone(phoneSeq[i+1])
                    df_ctxdepP.index=['{0}-{1}+{2}'.format('[s]',critical_P,right_critical_P)]
                    # df_ctxdepP=pd.DataFrame(,index=['{0}-{1}+{2}'.format('[s]',phoneSeq[i],phoneSeq[i+1])],columns=values.columns)
                elif i==len(phoneSeq)-1:
                    left_critical_P=FindCentralPhone(phoneSeq[i-1])
                    df_ctxdepP.index=['{0}-{1}+{2}'.format(left_critical_P,critical_P,'[\s]')]
                else:
                    left_critical_P=FindCentralPhone(phoneSeq[i-1])
                    right_critical_P=FindCentralPhone(phoneSeq[i+1])
                    df_ctxdepP.index=['{0}-{1}+{2}'.format(left_critical_P,critical_P,right_critical_P)]
                df_Utt_phfContextDep=df_Utt_phfContextDep.append(df_ctxdepP)
        
        assert len(df_Utt_phfContextDep) == len(phoneSeq) #check if the appended df: 'df_Utt_phfContextDep' ,atches phone sequence
        
        df_Utt_phfContextDep_reindex=df_Utt_phfContextDep.copy()
        df_Utt_phfContextDep_reindex['ctxP']=df_Utt_phfContextDep.index
        df_Utt_phfContextDep_reindex.index=df_Utt_phfContextDep['txt']
        Bookeep_utt_dict[uttname]=df_Utt_phfContextDep_reindex
    return Bookeep_utt_dict

def Files2BookeepUttDict_map(keys,global_info, namestr_end_position):
    # files=glob.glob(trnfiles_str+'/*.txt')
        
    DurSpeed_utt_dict=Dict()
    Bookeep_utt_dict=Dict()
    for file in tqdm(keys):
        uttname=os.path.basename(file).replace(suffix,'')
        name=uttname[:re.search(namestr_end_position,uttname).start()]
        #Initialize here
        # if name not in DurSpeed_utt_dict.keys(): 
            # DurSpeed_utt_dict[name].phonedf=pd.DataFrame()
            # DurSpeed_utt_dict[name].speeddf=pd.DataFrame()
            # Bookeep_utt_dict[name]=list()
        
        df_uttinfo=global_info[global_info['utt'] == uttname]
        dur_utt=(df_uttinfo['ed'] - df_uttinfo['st']).values
        df_phoneinfo=pd.read_csv(file, sep='\t',header=None)
        df_phoneinfo.columns=['st','ed','txt']
        # Switch combined phone like ueI to two splitted phones w-eI
        regexp = re.compile(r'u(A|aI|ax|eI|O)_') #['uA_', 'uaI_', 'uax_', 'ueI_', 'uO_']
        u_Biphoneset = [symbbset for symbbset in phonewoprosody.Phoneme_sets.keys() if regexp.search(symbbset)]
        def ScanBiphone(df_phoneinfo,Phoneme_sets,Biphoneset):
            df_bool=df_phoneinfo.copy()
            for idx in df_phoneinfo.index:
                Biphone_lst= [symb for symbset in u_Biphoneset for symb in Phoneme_sets[symbset]]
                if df_phoneinfo.loc[idx,'txt'] in Biphone_lst:
                    df_bool.loc[idx,'txt']=True
                else:
                    df_bool.loc[idx,'txt']=False
                
            return df_bool
        
        df_u_BiphonBool=ScanBiphone(df_phoneinfo,phonewoprosody.Phoneme_sets,u_Biphoneset)
        
        for ind in df_phoneinfo[df_u_BiphonBool['txt']].index:
            st,ed,symb = df_phoneinfo.loc[ind]
            old_idx=df_phoneinfo.loc[ind].name
            dur_distrubute=(ed - st)/2
            
            left_symb=symb[:re.search('u',symb).end()]
            right_symb=symb[re.search('u',symb).end():]
            df_phoneinfo=df_phoneinfo.append(pd.DataFrame({'st': st, 'ed': st + dur_distrubute, \
                                                       'txt': '{0}'.format(phonewoprosody.LeftSymbMapp[left_symb])}, index=[old_idx+len(df_phoneinfo)]))
            df_phoneinfo=df_phoneinfo.append(pd.DataFrame({'st': st + dur_distrubute, 'ed': ed, \
                                                       'txt': '{0}-{1}'.format(phonewoprosody.LeftSymbMapp[left_symb],right_symb)}, index=[old_idx+len(df_phoneinfo)]))
            df_phoneinfo=df_phoneinfo.drop(index=old_idx)
        
        
        df_phoneinfo_DeSIL=df_phoneinfo[df_phoneinfo['txt'] !='SIL']
        
        NumOfphones=len(df_phoneinfo_DeSIL)
        if NumOfphones==0:
            if df_phoneinfo['txt'].values.all() == 'SIL':
                NumOfphones=len(df_phoneinfo)
            else:
                print("Error happens and the df  values is ", df_phoneinfo)
                raise ValueError()
        df_phoneinfo_DeSIL['dur']=df_phoneinfo_DeSIL['ed'] - df_phoneinfo_DeSIL['st']
        # utterance_speechspeed=pd.DataFrame(dur_utt/NumOfphones,columns=['speed'])
        
        # DurSpeed_utt_dict[name].phonedf=pd.concat([DurSpeed_utt_dict[name].phonedf,df_phoneinfo_DeSIL])
        # DurSpeed_utt_dict[name].speeddf=pd.concat([DurSpeed_utt_dict[name].speeddf,utterance_speechspeed])
        
        
        # Get context phone variety
        
        df_Utt_phfContextDep=pd.DataFrame([],columns=df_phoneinfo.columns)
        
        values = df_phoneinfo
        phoneSeq=list(values['txt'].astype(str))    
        if len(phoneSeq) == 1:
            df_ctxdepP=values.iloc[[0],:]
            df_ctxdepP.index=['{0}-{1}+{2}'.format('[s]',phoneSeq[0],'[\s]')]
            df_Utt_phfContextDep=df_Utt_phfContextDep.append(df_ctxdepP)
        else:    
            for i in range(len(phoneSeq)): # df_Utt_phfContextDep append each word
                
                df_ctxdepP=values.iloc[[i],:]
                critical_P=FindCentralPhone(phoneSeq[i])
                if i==0:
                    right_critical_P=FindCentralPhone(phoneSeq[i+1])
                    df_ctxdepP.index=['{0}-{1}+{2}'.format('[s]',critical_P,right_critical_P)]
                    # df_ctxdepP=pd.DataFrame(,index=['{0}-{1}+{2}'.format('[s]',phoneSeq[i],phoneSeq[i+1])],columns=values.columns)
                elif i==len(phoneSeq)-1:
                    left_critical_P=FindCentralPhone(phoneSeq[i-1])
                    df_ctxdepP.index=['{0}-{1}+{2}'.format(left_critical_P,critical_P,'[\s]')]
                else:
                    left_critical_P=FindCentralPhone(phoneSeq[i-1])
                    right_critical_P=FindCentralPhone(phoneSeq[i+1])
                    df_ctxdepP.index=['{0}-{1}+{2}'.format(left_critical_P,critical_P,right_critical_P)]
                df_Utt_phfContextDep=df_Utt_phfContextDep.append(df_ctxdepP)
        
        assert len(df_Utt_phfContextDep) == len(phoneSeq) #check if the appended df: 'df_Utt_phfContextDep' ,atches phone sequence
        
        df_Utt_phfContextDep_reindex=df_Utt_phfContextDep.copy()
        df_Utt_phfContextDep_reindex['ctxP']=df_Utt_phfContextDep.index
        df_Utt_phfContextDep_reindex.index=df_Utt_phfContextDep['txt']
        Bookeep_utt_dict[uttname]=df_Utt_phfContextDep_reindex
    return Bookeep_utt_dict

def GetPersonalSegmentFeature_map(keys_people, Formants_people_segment_role_utt_dict,\
                                  PhoneMapp_dict, PhoneOfInterest ,\
                                  Inspect_roles, In_Segments_order,\
                                  vowel_min_num, global_info):
    # keys_people, Formants_people_segment_role_utt_dict,\
    #                               PhoneMapp_dict, PhoneOfInterest ,\
    #                               Inspect_roles, In_Segments_order,\
    #                               vowel_min_num, global_info =keys_people, Basicfeature_people_half_role_utt_dict,\
    #                               PhoneMapp_dict, PhoneOfInterest ,\
    #                               args.Inspect_roles, list(HalfDesider.keys()),\
    #                               args.MinPhoneNum, global_info
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    Segments_order=In_Segments_order
    df_person_segment_feature_dict=Dict()
    MissingSegment_bag=[]
    for people in tqdm(keys_people):
        Formants_segment_role_utt_dict=Formants_people_segment_role_utt_dict[people]
        if len(In_Segments_order) ==0 :
            Segments_order=sorted(list(Formants_segment_role_utt_dict.keys()))
            
        for role in Inspect_roles:
            df_person_segment_feature=pd.DataFrame([])
            for segment in Segments_order:
                Result_dict=Dict()
                Formants_utt_symb_SegmentRole=Formants_segment_role_utt_dict[segment][role]
                if len(Formants_utt_symb_SegmentRole)==0:
                    MissingSegment_bag.append([people,role,segment])
                
                # Fillin missing values
                # AUI_info_filled = Fill_n_Create_AUIInfo(Formants_utt_symb_SegmentRole, People_data_distrib, Inspect_features ,PhoneMapp_dict, PhoneOfInterest ,people, vowel_min_num)
                
                ''' calculate features '''
                # phone related features
                utterances=sorted(list(Formants_utt_symb_SegmentRole.keys()))
                Formant_people_information=Formant_utt2people_reshape(Formants_utt_symb_SegmentRole,Formants_utt_symb_SegmentRole,Align_OrinCmp=False)
                AUI_info=Gather_info_certainphones(Formant_people_information,PhoneMapp_dict,PhoneOfInterest)
                
                phoneinfo=pd.DataFrame()
                for phone in AUI_info[people].keys():
                    phoneinfo=phoneinfo.append(AUI_info[people][phone][AUI_info[people][phone]['cmps']=='ori'])
                phoneinfo['dur'] = phoneinfo['ed'] - phoneinfo['st']
                Result_dict['Average_POIduration']=phoneinfo['dur'].mean()
                
                personal_segment_glbinfo=global_info[global_info['utt'].isin(utterances)].copy()
                personal_segment_glbinfo.index=personal_segment_glbinfo['utt']
                
                personal_segment_glbinfo['dur']=personal_segment_glbinfo['ed'] - personal_segment_glbinfo['st']
                Result_dict['Average_uttduration']=personal_segment_glbinfo['dur'].mean()
                personal_segment_glbinfo['NumOfWords']=personal_segment_glbinfo['txt'].str.len()
                personal_segment_glbinfo['Utt_wordSpeed']=personal_segment_glbinfo['NumOfWords'] / personal_segment_glbinfo['dur']
                Result_dict['Average_uttwordSpeed']=personal_segment_glbinfo['Utt_wordSpeed'].mean()
                Result_dict['Average_uttwordlengeth']=personal_segment_glbinfo['NumOfWords'].mean()
                
                #phone speed
                for utt in utterances:
                    personal_segment_glbinfo.loc[utt,'NumberOfPhones']=len(Formants_utt_symb_SegmentRole[utt])
                Result_dict['Average_NumberOfPhones']=personal_segment_glbinfo['NumberOfPhones'].mean()
                personal_segment_glbinfo['Utt_phoneSpeed']=personal_segment_glbinfo['NumberOfPhones'] / personal_segment_glbinfo['dur']
                Result_dict['Average_uttphoneSpeed']=personal_segment_glbinfo['Utt_phoneSpeed'].mean()
                words=personal_segment_glbinfo['txt'].dropna()
                
                word_lsts=[s for strs in words for s in tokenizer.tokenize(strs)]
                word_variety=set(word_lsts)
                Result_dict['Segment_wordvariety']=len(word_variety)
                Result_dict['Segment_CtxPvariety']=len(set(phoneinfo['ctxP']))
                
                df_basicfeat=pd.DataFrame.from_dict(Result_dict,orient='index').T
                df_basicfeat.index=[people]
                
                # if len(PhoneOfInterest) >= 3:
                #     df_formant_statistic=Eval_med._Postprocess_dfformantstatistic(df_formant_statistic)
                # assert len(df_formant_statistic.columns) > 10 #check if df_formant_statistic is empty DF
                if len(df_person_segment_feature) == 0:
                    df_person_segment_feature=pd.DataFrame([],columns=df_basicfeat.columns)
                df_person_segment_feature.loc[segment]=df_basicfeat.loc[people]
            df_person_segment_feature_dict[people][role]=df_person_segment_feature
    return df_person_segment_feature_dict

def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Select utterances with entropy values that are close to disribution of target domain data",
        )
    parser.add_argument('--ASR_path', default='/mnt/sdd/jackchen/egs/formosa/s6',
                        help='path of the base directory', dest='ASR_path')
    parser.add_argument('--trnpath', default='/mnt/sdd/jackchen/egs/formosa/s6/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/kid_TD/ADOS_tdnn_fold_transfer',
                        help='/mnt/sdd/jackchen/egs/formosa/s6/{Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/{kid88|kid_TD|ASD_DOCKID|ASD_DOCKID_emotion|TD_DOCKID_emotion}/ADOS_tdnn_fold_transfer | Alignment_human/kid/Audacity_phone|')
    parser.add_argument('--columns', default=['utt','st','ed','txt','spk'],
                        help='path of the base directory')
    parser.add_argument('--Inspect_roles', default=['D','K'],
                            help='')
    parser.add_argument('--out_path', default='Features/Other',
                        help='path of the base directory')
    parser.add_argument('--MinPhoneNum', default=3,
                            help='path of the base directory')
    args = parser.parse_args()
    return args


args = get_args()
ASR_path=args.ASR_path
trnpath=args.trnpath


TD_info_path=ASR_path + '/Segments_info_ADOS_TD.txt' 
ASD_info_path=ASR_path + '/Segments_info_ADOS_ASD.txt' 
df_ASD_info=pd.read_csv(ASD_info_path, sep='\t',header=None)
df_ASD_info.columns=args.columns
df_ASD_info_kid=df_ASD_info[df_ASD_info['spk'].str.contains("_K")]


df_TD_info=pd.read_csv(TD_info_path, sep='\t',header=None)
df_TD_info.columns=args.columns
df_TD_info_kid=df_TD_info[df_TD_info['spk'].str.contains("_K")]

def GetSpeechSpeed(df_TD_info_kid, namestr_end_position="_emotion"):
    people_set=set(df_TD_info_kid['spk'])
    df_speed=pd.DataFrame()
    for people in people_set:
        df_TD_info_kid_utt=df_TD_info_kid[df_TD_info_kid['spk']==people]
        name=people[:re.search(namestr_end_position,people).start()]
        
        dur = df_TD_info_kid_utt['ed'] - df_TD_info_kid_utt['st']
        Strlength = df_TD_info_kid_utt['txt'].str.len()
        speed=dur/Strlength
        
        df_speed.loc[name,'dur']=dur.mean()
        df_speed.loc[name,'strlen']=Strlength.mean()
        df_speed.loc[name,'speed']=speed.mean()
        df_speed.loc[name,'totalword']=Strlength.sum()
    return df_speed

df_speed_TDkid= GetSpeechSpeed(df_TD_info_kid, namestr_end_position="_[D|K]")
df_speed_ASDkid= GetSpeechSpeed(df_ASD_info_kid, namestr_end_position="_[D|K]")

# =============================================================================
'''

    Get Phone level Infos
    
    Want to get :
        1. mean phone duration len (should be able to inspect certain phones)
        2. speech speed (utt duration / number of phones)

'''
def FindCentralPhone(symb):
    regexp = re.compile(r'.*[-+].*')
    if regexp.search(symb):
        if '-' in symb:
            critical_P=symb[symb.find('-')+1:]
        elif '+' in symb:
            critical_P=symb[:symb.find('+')]
        else:
            raise ValueError
    else:
        critical_P=symb
    return critical_P

from addict import Dict
from articulation.HYPERPARAM import phonewoprosody, Label
from tqdm import tqdm
PhoneMapp_dict=phonewoprosody.PhoneMapp_dict
PhoneOfInterest=sorted(list(PhoneMapp_dict.keys()))
suffix='.txt'
# =============================================================================
ASDKid88_path=ASR_path + '/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/kid88/ADOS_tdnn_fold_transfer'
TD_path=ASR_path + '/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/kid_TD/ADOS_tdnn_fold_transfer'

def GetPhonticBasicInfo(trnfiles_str=TD_path,global_info=df_TD_info,namestr_end_position="_emotion"):
    '''
    Example inputs:
        TD_path=
         /mnt/sdd/jackchen/egs/formosa/s6/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/kid_TD/ADOS_tdnn_fold_transfer
    
        df_TD_info=
        1843  2021_03_15_5874_1(醫生鏡頭模糊，醫生聲音雜訊大)_emotion_D_62  ...  2021_03_15_5874_1(醫生鏡頭模糊，醫生聲音雜訊大)_emotion_D
        1844  2021_03_15_5874_1(醫生鏡頭模糊，醫生聲音雜訊大)_emotion_D_63  ...  2021_03_15_5874_1(醫生鏡頭模糊，醫生聲音雜訊大)_emotion_D
    
    '''
    files=glob.glob(trnfiles_str+'/*.txt')
    
    DurSpeed_utt_dict=Dict()
    Bookeep_utt_dict=Dict()
    for file in tqdm(files):
        uttname=os.path.basename(file).replace(suffix,'')
        name=uttname[:re.search(namestr_end_position,uttname).start()]
        #Initialize here
        if name not in DurSpeed_utt_dict.keys(): 
            DurSpeed_utt_dict[name].phonedf=pd.DataFrame()
            DurSpeed_utt_dict[name].speeddf=pd.DataFrame()
            DurSpeed_utt_dict[name].CtxPhoneSeq=list()
        
        df_uttinfo=global_info[global_info['utt'] == uttname]
        dur_utt=(df_uttinfo['ed'] - df_uttinfo['st']).values
        df_phoneinfo=pd.read_csv(file, sep='\t',header=None)
        df_phoneinfo.columns=['st','ed','txt']
        # Switch combined phone like ueI to two splitted phones w-eI
        regexp = re.compile(r'u(A|aI|ax|eI|O)_') #['uA_', 'uaI_', 'uax_', 'ueI_', 'uO_']
        u_Biphoneset = [symbbset for symbbset in phonewoprosody.Phoneme_sets.keys() if regexp.search(symbbset)]
        def ScanBiphone(df_phoneinfo,Phoneme_sets,Biphoneset):
            df_bool=df_phoneinfo.copy()
            for idx in df_phoneinfo.index:
                Biphone_lst= [symb for symbset in u_Biphoneset for symb in Phoneme_sets[symbset]]
                if df_phoneinfo.loc[idx,'txt'] in Biphone_lst:
                    df_bool.loc[idx,'txt']=True
                else:
                    df_bool.loc[idx,'txt']=False
                
            return df_bool
        
        df_u_BiphonBool=ScanBiphone(df_phoneinfo,phonewoprosody.Phoneme_sets,u_Biphoneset)
        
        for ind in df_phoneinfo[df_u_BiphonBool['txt']].index:
            st,ed,symb = df_phoneinfo.loc[ind]
            old_idx=df_phoneinfo.loc[ind].name
            dur_distrubute=(ed - st)/2
            
            left_symb=symb[:re.search('u',symb).end()]
            right_symb=symb[re.search('u',symb).end():]
            df_phoneinfo=df_phoneinfo.append(pd.DataFrame({'st': st, 'ed': st + dur_distrubute, \
                                                       'txt': '{0}'.format(phonewoprosody.LeftSymbMapp[left_symb])}, index=[old_idx+len(df_phoneinfo)]))
            df_phoneinfo=df_phoneinfo.append(pd.DataFrame({'st': st + dur_distrubute, 'ed': ed, \
                                                       'txt': '{0}-{1}'.format(phonewoprosody.LeftSymbMapp[left_symb],right_symb)}, index=[old_idx+len(df_phoneinfo)]))
            df_phoneinfo=df_phoneinfo.drop(index=old_idx)
        
        
        df_phoneinfo_DeSIL=df_phoneinfo[df_phoneinfo['txt'] !='SIL']
        
        NumOfphones=len(df_phoneinfo_DeSIL)
        if NumOfphones==0:
            if df_phoneinfo['txt'].values.all() == 'SIL':
                NumOfphones=len(df_phoneinfo)
            else:
                print("Error happens and the df  values is ", df_phoneinfo)
                raise ValueError()
        df_phoneinfo_DeSIL['dur']=df_phoneinfo_DeSIL['ed'] - df_phoneinfo_DeSIL['st']
        utterance_speechspeed=pd.DataFrame(dur_utt/NumOfphones,columns=['speed'])
        
        DurSpeed_utt_dict[name].phonedf=pd.concat([DurSpeed_utt_dict[name].phonedf,df_phoneinfo_DeSIL])
        DurSpeed_utt_dict[name].speeddf=pd.concat([DurSpeed_utt_dict[name].speeddf,utterance_speechspeed])
        
        
        # Get context phone variety
        
        df_Utt_phfContextDep=pd.DataFrame([],columns=df_phoneinfo.columns)
        
        values = df_phoneinfo
        phoneSeq=list(values['txt'].astype(str))    
        if len(phoneSeq) == 1:
            df_ctxdepP=values.iloc[[0],:]
            df_ctxdepP.index=['{0}-{1}+{2}'.format('[s]',phoneSeq[0],'[\s]')]
            df_Utt_phfContextDep=df_Utt_phfContextDep.append(df_ctxdepP)
        else:    
            for i in range(len(phoneSeq)): # df_Utt_phfContextDep append each word
                
                df_ctxdepP=values.iloc[[i],:]
                critical_P=FindCentralPhone(phoneSeq[i])
                if i==0:
                    right_critical_P=FindCentralPhone(phoneSeq[i+1])
                    df_ctxdepP.index=['{0}-{1}+{2}'.format('[s]',critical_P,right_critical_P)]
                    # df_ctxdepP=pd.DataFrame(,index=['{0}-{1}+{2}'.format('[s]',phoneSeq[i],phoneSeq[i+1])],columns=values.columns)
                elif i==len(phoneSeq)-1:
                    left_critical_P=FindCentralPhone(phoneSeq[i-1])
                    df_ctxdepP.index=['{0}-{1}+{2}'.format(left_critical_P,critical_P,'[\s]')]
                else:
                    left_critical_P=FindCentralPhone(phoneSeq[i-1])
                    right_critical_P=FindCentralPhone(phoneSeq[i+1])
                    df_ctxdepP.index=['{0}-{1}+{2}'.format(left_critical_P,critical_P,right_critical_P)]
                df_Utt_phfContextDep=df_Utt_phfContextDep.append(df_ctxdepP)
        assert len(df_Utt_phfContextDep) == len(phoneSeq) #check if the appended df: 'df_Utt_phfContextDep' ,atches phone sequence
        Bookeep_utt_dict[uttname]=df_Utt_phfContextDep
        DurSpeed_utt_dict[name].CtxPhoneSeq.append(list(df_Utt_phfContextDep.index))
    
    
    df_Statistics_PhoneDurSpeed=pd.DataFrame()
    for people in DurSpeed_utt_dict.keys():
        Result_dict={}
        df_phone_info_people=DurSpeed_utt_dict[people].phonedf
        df_SpeedPerUtt_people=DurSpeed_utt_dict[people].speeddf
        
        phones2Capture=[x for symb in PhoneMapp_dict.keys() for x in  PhoneMapp_dict[symb]]
        df_phone_info_people_POI=df_phone_info_people[df_phone_info_people['txt'].isin(phones2Capture)]
        
        Result_dict['Average_POIDur']=df_phone_info_people_POI['dur'].mean()
        Result_dict['Average_phoneSpeed']=df_SpeedPerUtt_people['speed'].mean()
        Result_dict['PhoneVariety']=len(set([e for lst in DurSpeed_utt_dict[people].CtxPhoneSeq for e in lst ]))
        
        df_PhoneticBaseInfo=pd.DataFrame.from_dict(Result_dict,orient='index').T
        df_PhoneticBaseInfo.index=[people]
        df_Statistics_PhoneDurSpeed=pd.concat([df_Statistics_PhoneDurSpeed,df_PhoneticBaseInfo])
    
    return df_Statistics_PhoneDurSpeed
# df_PhoneBasicInfo_TDkid= GetPhonticBasicInfo(TD_path,df_TD_info_kid, namestr_end_position="_[D|K]")
# df_PhoneBasicInfo_ASDkid= GetPhonticBasicInfo(ASDKid88_path,df_ASD_info_kid, namestr_end_position="_[D|K]")


# =============================================================================
'''

Get Basic info timeseries

'''

from SlidingWindow import Reorder2_PER_utt
from utils_jack  import  Formant_utt2people_reshape, Gather_info_certainphones





excluded_people=['2015_12_06_01_056_1',
                 '2015_12_07_02_003',
'2016_01_27_01_154_1',
'2016_12_04_01_234_1',
'2017_01_14_01_222_1',
'2017_03_18_01_196_1',
'2017_11_18_01_371']  
# =============================================================================
DOCKID_ASD_path=ASR_path + '/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/ASD_DOCKID/ADOS_tdnn_fold_transfer'
DOCKID_TD_path=ASR_path + '/Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain/new_system/TD_DOCKID/ADOS_tdnn_fold_transfer'

files_DOCKID_ASD=glob.glob(DOCKID_ASD_path+'/*.txt')
files_DOCKID_TD=glob.glob(DOCKID_TD_path+'/*.txt')
# trnfiles_str=DOCKID_TD_path
# global_info=df_TD_info
# namestr_end_position="_emotion"

# trnfiles_str=DOCKID_ASD_path
# global_info=df_ASD_info
# namestr_end_position="_[D|K]"
# excluded_people=excluded_people
import numpy as np
def GetPhonticDynamicBasicInfo(files,global_info,namestr_end_position,\
                               excluded_people):

    
    people_set=set([os.path.basename(f)[:re.search("_[K|D]_",os.path.basename(f)).start()] for f in files])
    # Stage1. Create utt_dict
    keys=[]
    for p in people_set:
        keys.append([file for file in files if p in file])
    flat_keys=[item for sublist in keys for item in sublist]
    assert len(flat_keys) == len(files)
    
    pool = Pool(os.cpu_count())
    
    # Bookeep_utt_dict=Files2BookeepUttDict(trnfiles_str,global_info, namestr_end_position)

    final_result = pool.starmap(Files2BookeepUttDict_map, [(key,global_info, namestr_end_position) for key in tqdm(keys)])
    print('Files2BookeepUttDict_map done !!!')
    Bookeep_utt_dict=Dict()
    for d in final_result:
        Bookeep_utt_dict.update(d)

    
    
    # Stage2. Reorder to segment:values format===
    Basicfeature_people_segment_role_utt_dict=Reorder2_PER_utt(Bookeep_utt_dict,PhoneMapp_dict,\
                                                               PhoneOfInterest,args.Inspect_roles,\
                                                               MinNum=args.MinPhoneNum)    
    
    Basicfeature_people_half_role_utt_dict=Dict()
    for people in Basicfeature_people_segment_role_utt_dict.keys():
        split_num=len(Basicfeature_people_segment_role_utt_dict[people])//2
        for segment in Basicfeature_people_segment_role_utt_dict[people].keys():
            for role in Basicfeature_people_segment_role_utt_dict[people][segment].keys():
                if segment <= split_num:
                    Basicfeature_people_half_role_utt_dict[people]['first_half'][role].update(Basicfeature_people_segment_role_utt_dict[people][segment][role])
                else:
                    Basicfeature_people_half_role_utt_dict[people]['last_half'][role].update(Basicfeature_people_segment_role_utt_dict[people][segment][role])
    #end ==============================    
        
    HalfDesider={'first_half':["happy","afraid"],
                 'last_half':["angry","sad"]}
    
      
    keys_people=[p for p in Basicfeature_people_segment_role_utt_dict.keys() if p not in excluded_people]
    
    keys=[]
    interval=5
    for i in range(0,len(keys_people),interval):
        # print(list(Utt_ctxdepP_dict.keys())[i:i+interval])
        keys.append(list(keys_people)[i:i+interval])
    flat_keys=[item for sublist in keys for item in sublist]
    assert len(flat_keys) == len(keys_people)
    

    final_result = pool.starmap(GetPersonalSegmentFeature_map, [(key, Basicfeature_people_segment_role_utt_dict,\
                                      PhoneMapp_dict, PhoneOfInterest ,\
                                      args.Inspect_roles, [],\
                                      args.MinPhoneNum, global_info) for key in tqdm(keys)])
    print('GetPersonalSegmentFeature_map segment done !!!')
    df_person_segment_feature_dict=Dict()
    for d in final_result:
        for spk in d.keys():
            df_person_segment_feature_dict[spk]=d[spk]
    
    final_result = pool.starmap(GetPersonalSegmentFeature_map, [(key, Basicfeature_people_half_role_utt_dict,\
                                      PhoneMapp_dict, PhoneOfInterest ,\
                                      args.Inspect_roles, list(HalfDesider.keys()),\
                                      args.MinPhoneNum, global_info) for key in tqdm(keys)])
    
    print('GetPersonalSegmentFeature_map half done !!!')
    df_person_half_feature_dict=Dict()
    for d in final_result:
        for spk in d.keys():
            df_person_half_feature_dict[spk]=d[spk]
    # df_person_segment_feature_dict=GetPersonalSegmentFeature(keys_people, Basicfeature_people_segment_role_utt_dict,\
    #                                   PhoneMapp_dict, PhoneOfInterest ,\
    #                                   args.Inspect_roles, [],\
    #                                   args.MinPhoneNum, global_info)
    
    
        
    # df_person_half_feature_dict=GetPersonalSegmentFeature(keys_people, Basicfeature_people_half_role_utt_dict,\
    #                                   PhoneMapp_dict, PhoneOfInterest ,\
    #                                   args.Inspect_roles, list(HalfDesider.keys()),\
    #                                   args.MinPhoneNum, global_info)
    # =============================================================================
    '''
    
        Calculate syncrony features
    
    '''
    
    example_cols=df_person_segment_feature_dict[keys_people[0]][args.Inspect_roles[0]].columns
    # features=[
    #         'Average_POIduration', 'Average_uttduration', 'Average_uttwordSpeed',
    #        'Average_uttwordlengeth', 'Average_uttphoneSpeed',
    #        'Segment_wordvariety', 'Segment_CtxPvariety']
    features=example_cols
    exclude_cols=[]   # covariance of only two classes are easily to be zero
    FilteredFeatures = [c for c in features if c not in exclude_cols]
    # =============================================================================
    
    syncrony=Syncrony()
    PhoneOfInterest_str=''
    df_syncrony_measurement=syncrony.calculate_features(df_person_segment_feature_dict, df_person_half_feature_dict,\
                                   FilteredFeatures,PhoneOfInterest_str,\
                                   args.Inspect_roles, Label,\
                                   MinNumTimeSeries=2, label_choose_lst=['ADOS_C'])        
    
    for col in df_syncrony_measurement.columns:
        if     df_syncrony_measurement[col].isnull().values.any():
            print(col,' contains nan')

    return df_syncrony_measurement, df_person_segment_feature_dict
df_syncronyBasic_TD, df_person_segment_feature_dict_TD=GetPhonticDynamicBasicInfo(files_DOCKID_TD,df_TD_info,namestr_end_position="_[D|K]",\
                               excluded_people=excluded_people)


df_syncronyBasic_ASD, df_person_segment_feature_dict_ASD=GetPhonticDynamicBasicInfo(files_DOCKID_ASD,df_ASD_info,namestr_end_position="_[D|K]",\
                               excluded_people=excluded_people)

# =============================================================================
'''

    Data Output

'''
# =============================================================================
# df_speedlenBasicInfo_TD=pd.concat([df_speed_TDkid,df_PhoneBasicInfo_TDkid],axis=1)
# df_speedlenBasicInfo_ASD=pd.concat([df_speed_ASDkid,df_PhoneBasicInfo_ASDkid],axis=1)

# df_speedlenBasicInfo_TD=pd.concat([df_speed_TDkid,df_syncronyBasic_TD],axis=1)
# df_speedlenBasicInfo_ASD=pd.concat([df_speed_ASDkid,df_syncronyBasic_ASD],axis=1)

# if not os.path.exists(args.out_path):
#     os.makedirs(args.out_path)
# pickle.dump(df_speedlenBasicInfo_TD, open(args.out_path + "/df_speedlenBasicInfo_{role}.pkl".format(role='TD'),"wb"))
# pickle.dump(df_speedlenBasicInfo_ASD, open(args.out_path + "/df_speedlenBasicInfo_{role}.pkl".format(role='ASD'),"wb"))


if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)
pickle.dump(df_syncronyBasic_TD, open(args.out_path + "/df_syncronyBasicInfo_{role}.pkl".format(role='TD'),"wb"))
pickle.dump(df_syncronyBasic_ASD, open(args.out_path + "/df_syncronyBasicInfo_{role}.pkl".format(role='ASD'),"wb"))

pickle.dump(df_person_segment_feature_dict_TD,open(args.out_path+"/df_person_segment_feature_dict_{0}_{1}.pkl".format('TD', 'syncronyBasicInfo'),"wb"))
pickle.dump(df_person_segment_feature_dict_ASD,open(args.out_path+"/df_person_segment_feature_dict_{0}_{1}.pkl".format('ASD', 'syncronyBasicInfo'),"wb"))