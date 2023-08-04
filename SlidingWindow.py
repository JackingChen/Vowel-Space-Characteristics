#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 15:33:51 2021

@author: jackchen
"""
import pandas as pd
import re
from addict import Dict
from tqdm import tqdm


class slidingwindow:
    def __init__(self):
        self.null = 0
    
    def Reorder2_PER_utt_formants(self,Formants_utt_symb,PhoneMapp_dict,\
                     PhonesOfInterest=['u:', 'i:', 'A:'],Inspect_roles=['D','K'],\
                     MinNum=3):
        # =============================================================================
        '''
        
            Keep in mind that last frame may have unenough  vowels, so sometimes the 
            latter code will fill arbitrary data
        
        '''
        def Initialize_VowelStack(PhoneMapp_dict, PhonesOfInterest):
            Vowel_stack={k:pd.DataFrame([]) for k in PhonesOfInterest }
            return Vowel_stack
        
        
        def Initialize_VowelStackPair(PhoneMapp_dict, Inspect_roles, PhonesOfInterest):
            Vowel_stack_DK=Dict()
            for R in Inspect_roles:
                Vowel_stack=Initialize_VowelStack(PhoneMapp_dict, PhonesOfInterest)
                Vowel_stack_DK[R]=Vowel_stack
            return Vowel_stack_DK
        
        def PhoneIn_check(phone, PhoneMapp_dict, PhonesOfInterest):
            for k in PhonesOfInterest:
                for p in PhoneMapp_dict[k]:
                    if phone in p:
                        return k
            return -1
        
        def Fillin_VowelStack(Vowel_stack, values, PhoneMapp_dict, PhonesOfInterest):
            # put target vowels in stack
            for i,ind in enumerate(list(values.index)):
                if PhoneIn_check(ind, PhoneMapp_dict, PhonesOfInterest) != -1:
                    # print("DEBUG: values.iloc[i] ", values.iloc[i])
                    # print("DEBUG: Vowel_stack ", Vowel_stack)
                    # Vowel_stack[PhoneIn_check(ind, PhoneMapp_dict, PhonesOfInterest)]=\
                    #     Vowel_stack[PhoneIn_check(ind, PhoneMapp_dict, PhonesOfInterest)].append(values.iloc[i])
                    Vowel_stack[PhoneIn_check(ind, PhoneMapp_dict, PhonesOfInterest)]=\
                        pd.concat([Vowel_stack[PhoneIn_check(ind, PhoneMapp_dict, PhonesOfInterest)],pd.DataFrame(values.iloc[i]).T])
                        
            return Vowel_stack
        def Check_VowelStackFull(Vowel_stack, minNum=3):
            cond= True
            for p_symb in Vowel_stack.keys():
                cond = cond and len(Vowel_stack[p_symb]) >= minNum
            return cond
        
        def Refill_VowelStack_condition(Vowel_stack_DK,minNum=3):
            cond_top=True
            for R in Vowel_stack_DK.keys():
                cond=Check_VowelStackFull(Vowel_stack_DK[R], minNum=minNum)
                cond_top = cond_top and cond
            return cond_top
        def Check_nextperson(nextname, name):
            if nextname != name:
                return True
            else:
                return False
        # =============================================================================
        
        
        
        Vowel_stack_DK=Initialize_VowelStackPair(PhoneMapp_dict, Inspect_roles, PhonesOfInterest)
        Formants_people_segment_role_utt_dict=Dict()
        
        FormantsUttKeys_numberOrder=sorted(list(Formants_utt_symb.keys()),key=lambda x: (x[:re.search("_[K|D]_",x).start()], int(x.split("_")[-1])))

        segment_Num=0
        print(" =====*****  Reorder2_PER_utt start  *****=====")
        for i, keys in tqdm(enumerate(FormantsUttKeys_numberOrder)):
            values= Formants_utt_symb[keys]
            name=keys[:re.search("_[K|D]_",keys).start()]
            if i < len(FormantsUttKeys_numberOrder)-1:
                nextname=FormantsUttKeys_numberOrder[i+1][:re.search("_[K|D]_",FormantsUttKeys_numberOrder[i+1]).start()]
            
            
            res_key_str=keys[re.search("_[K|D]_",keys).start()+1:]
            res_key = res_key_str.split("_")
            if len(res_key) != 2:
                raise ValueError("not using emotion data, and Perhaps using the worng Alignments")
            role, turn_number=res_key
    
    
            Vowel_stack = Vowel_stack_DK[role].copy()  # We use Vowel_stack to bookeep a u i vowels
            Vowel_stack = Fillin_VowelStack(Vowel_stack, values, PhoneMapp_dict, PhonesOfInterest)# function: Fillin_VowelStack, fills PhoneOfInterests from value to the Vowel stack
            Vowel_stack_DK[role] = Vowel_stack # Vowel_stack for both Doctor and kid
            cond = Refill_VowelStack_condition(Vowel_stack_DK, minNum=MinNum) # The condition to renew the Vowel_stack is that both Doc and Kid should have enough PhoneOfInterests
            
            # cond=Check_VowelStackFull(Vowel_stack_DK[role], minNum=2)
            Formants_people_segment_role_utt_dict[name][segment_Num][role][keys]=values
            if Check_nextperson(nextname, name) or i == len(FormantsUttKeys_numberOrder)-1: 
                                                  # If next utterance is from another person (or last person)
                                                  # Reinitialize Vowel_stack, set segment_Num to 0 
                                                  # and delete the incomplete segment
                if not cond: # The number of vowels does not satisfy to renew Vowel_stack_DK
                    del Formants_people_segment_role_utt_dict[name][segment_Num]
                segment_Num = 0
                Vowel_stack_DK=Initialize_VowelStackPair(PhoneMapp_dict, Inspect_roles, PhonesOfInterest)
                continue
            
            if cond:
                
                Vowel_stack_DK=Initialize_VowelStackPair(PhoneMapp_dict, Inspect_roles, PhonesOfInterest)
                segment_Num+=1 #next segment_Num will be different
    
        return Formants_people_segment_role_utt_dict
    def Reorder2_PER_utt_phonation(self,Formants_utt_symb,PhoneMapp_dict,\
                                   PhonesOfInterest=['u:', 'i:', 'A:'],Inspect_roles=['D','K'],\
                                   MinNum=3, check=True):
        # =============================================================================
        '''
        
            Keep in mind that last frame may have unenough  vowels, so sometimes the 
            latter code will fill arbitrary data
        
        '''
        def Initialize_VowelStack(PhoneMapp_dict, PhonesOfInterest):
            Vowel_stack={k:pd.DataFrame([]) for k in PhonesOfInterest }
            return Vowel_stack
        
        
        def Initialize_VowelStackPair(PhoneMapp_dict, Inspect_roles, PhonesOfInterest):
            Vowel_stack_DK=Dict()
            for R in Inspect_roles:
                Vowel_stack=Initialize_VowelStack(PhoneMapp_dict, PhonesOfInterest)
                Vowel_stack_DK[R]=Vowel_stack
            return Vowel_stack_DK
        
        def PhoneIn_check(phone, PhoneMapp_dict, PhonesOfInterest):
            for k in PhonesOfInterest:
                for p in PhoneMapp_dict[k]:
                    if phone in p:
                        return k
            return -1
        
        def Fillin_VowelStack(Vowel_stack, values, PhoneMapp_dict, PhonesOfInterest):
            # put target vowels in stack
            for i,ind in enumerate(list(values.index)):
                if PhoneIn_check(ind, PhoneMapp_dict, PhonesOfInterest) != -1:
                    Vowel_stack[PhoneIn_check(ind, PhoneMapp_dict, PhonesOfInterest)]=\
                        Vowel_stack[PhoneIn_check(ind, PhoneMapp_dict, PhonesOfInterest)].append(values.iloc[i])
            return Vowel_stack
        def Check_VowelStackFull(Vowel_stack, minNum=3):
            cond= True
            for p_symb in Vowel_stack.keys():
                cond = cond and len(Vowel_stack[p_symb]) >= minNum
            return cond
        
        def Refill_VowelStack_condition(Vowel_stack_DK,minNum=3):
            cond_top=True
            for R in Vowel_stack_DK.keys():
                cond=Check_VowelStackFull(Vowel_stack_DK[R], minNum=minNum)
                cond_top = cond_top and cond
            return cond_top
        def Check_nextperson(nextname, name):
            if nextname != name:
                return True
            else:
                return False
        # =============================================================================
        Vowel_stack_DK=Initialize_VowelStackPair(PhoneMapp_dict, Inspect_roles, PhonesOfInterest)
        Formants_people_segment_role_utt_dict=Dict()
        
        FormantsUttKeys_numberOrder=sorted(list(Formants_utt_symb.keys()),key=lambda x: (x[:re.search("_[K|D]_",x).start()], int(x.split("_")[-1])))
        
        # Inspect_certain_people=[e for e in FormantsUttKeys_numberOrder if '2015_12_06_01_097' in e] # For debugging 
        
        
        NumberOrder_checkdict=Dict()
        segment_Num=0
        # values_bag=[]
        # for i, keys in enumerate(Inspect_certain_people):
        for i, keys in enumerate(FormantsUttKeys_numberOrder):
            values= Formants_utt_symb[keys]
            # values_bag.append(values['text'].values.tolist())
            name=keys[:re.search("_[K|D]_",keys).start()]
            if i < len(FormantsUttKeys_numberOrder)-1:
                nextname=FormantsUttKeys_numberOrder[i+1][:re.search("_[K|D]_",FormantsUttKeys_numberOrder[i+1]).start()]
            
            
            res_key_str=keys[re.search("_[K|D]_",keys).start()+1:]
            res_key = res_key_str.split("_")
            if len(res_key) != 2:
                raise ValueError("not using emotion data, and Perhaps using the worng Alignments")
            role, turn_number=res_key
            if check:
                if name not in NumberOrder_checkdict.keys():
                    NumberOrder_checkdict[name]=[]
                NumberOrder_checkdict[name].append(int(turn_number))
    
            Vowel_stack = Vowel_stack_DK[role].copy()  # We use Vowel_stack to bookeep a u i vowels
            Vowel_stack = Fillin_VowelStack(Vowel_stack, values, PhoneMapp_dict, PhonesOfInterest)# function: Fillin_VowelStack, fills PhoneOfInterests from value to the Vowel stack
            Vowel_stack_DK[role] = Vowel_stack # Vowel_stack for both Doctor and kid
            cond = Refill_VowelStack_condition(Vowel_stack_DK, minNum=MinNum) # The condition to renew the Vowel_stack is that both Doc and Kid should have enough PhoneOfInterests
            # cond=Check_VowelStackFull(Vowel_stack_DK[role], minNum=2)
            Formants_people_segment_role_utt_dict[name][segment_Num][role][keys]=values
            # print ("Now the vowel stack is ",Vowel_stack_DK)
            if Check_nextperson(nextname, name) or i == len(FormantsUttKeys_numberOrder)-1: 
                                                  # If next utterance is from another person (or last person)
                                                  # Reinitialize Vowel_stack, set segment_Num to 0 
                                                  # and delete the incomplete segment
                if not cond: # The number of vowels does not satisfy to renew Vowel_stack_DK
                    del Formants_people_segment_role_utt_dict[name][segment_Num]
                segment_Num = 0
                Vowel_stack_DK=Initialize_VowelStackPair(PhoneMapp_dict, Inspect_roles, PhonesOfInterest)
                continue
            
            if cond:
                Vowel_stack_DK=Initialize_VowelStackPair(PhoneMapp_dict, Inspect_roles, PhonesOfInterest)
                segment_Num+=1 #next segment_Num will be different
    
        return Formants_people_segment_role_utt_dict