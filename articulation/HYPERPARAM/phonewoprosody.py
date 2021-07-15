#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:13:47 2020

@author: jackchen
"""

import collections


# Phoneme_sets=collections.OrderedDict()
# Phoneme_sets['A_']=['A:', 'A:1', 'A:2', 'A:3', 'A:4', 'A:5']
# Phoneme_sets['aI']=['aI1', 'aI2','aI3', 'aI4', 'aI5']
# Phoneme_sets['aU']=['aU1', 'aU2', 'aU3', 'aU4', 'aU5']
# Phoneme_sets['ax']=['ax','ax1', 'ax2', 'ax3', 'ax4', 'ax5']
# Phoneme_sets['axr']=['axr1', 'axr2', 'axr3', 'axr4','axr5']
# Phoneme_sets['E']=['E1', 'E2', 'E3', 'E4', 'E5']
# Phoneme_sets['eI']=['eI1', 'eI2', 'eI3','eI4', 'eI5', 'eI7']
# Phoneme_sets['f']=['f']
# Phoneme_sets['H']=['H']
# Phoneme_sets['i_']=['i:1', 'i:2', 'i:3', 'i:4', 'i:5','i:7']
# Phoneme_sets['j']=['j']
# Phoneme_sets['k']=['k']
# Phoneme_sets['k_h']=['k_h']
# Phoneme_sets['l']=['l']
# Phoneme_sets['m']=['m']
# Phoneme_sets['n']=['n', 'n1','n2','n3','n4','n5']
# Phoneme_sets['N']=['N1','N2','N3','N4','N5']
# Phoneme_sets['O_']=['O:1', 'O:2', 'O:3', 'O:4','O:5']
# Phoneme_sets['oU']=['oU1', 'oU2', 'oU3', 'oU4', 'oU5']
# Phoneme_sets['p']=['p']
# Phoneme_sets['p_h']=['p_h']
# Phoneme_sets['s']=['s','s1','s2', 's3', 's4', 's5']
# Phoneme_sets['s6']=['s6']
# Phoneme_sets['ss']=['ss', 'ss1', 'ss2', 'ss3', 'ss4','ss5']
# Phoneme_sets['t']=['t']
# Phoneme_sets['t_h']=['t_h']
# Phoneme_sets['ts']=['ts', 'ts1', 'ts2', 'ts3', 'ts4', 'ts5']
# Phoneme_sets['ts6']=['ts6']
# Phoneme_sets['ts6_h']=['ts6_h']
# Phoneme_sets['ts_h']=['ts_h', 'ts_h1', 'ts_h2', 'ts_h3', 'ts_h4']
# Phoneme_sets['ttss']=['ttss','ttss1', 'ttss2', 'ttss3', 'ttss4', 'ttss5']
# Phoneme_sets['ttss_h']=['ttss_h', 'ttss_h1','ttss_h2', 'ttss_h3', 'ttss_h4', 'ttss_h5']
# Phoneme_sets['u_']=['u:1', 'u:2', 'u:3','u:4', 'u:5', 'u:7']
# Phoneme_sets['w']=['w']
# Phoneme_sets['x']=['x']
# Phoneme_sets['y']=['y1', 'y2', 'y3', 'y4', 'y5']
# Phoneme_sets['zz']=['zz','zz4', 'zz5']


Phoneme_sets=collections.OrderedDict()
Phoneme_sets['A_']=['A:', 'A:1', 'A:2', 'A:3', 'A:4', 'A:5']
Phoneme_sets['aI']=['aI1', 'aI2','aI3', 'aI4', 'aI5']
Phoneme_sets['aU']=['aU1', 'aU2', 'aU3', 'aU4', 'aU5']
Phoneme_sets['ax']=['ax','ax1', 'ax2', 'ax3', 'ax4', 'ax5']
Phoneme_sets['axr']=['axr1', 'axr2', 'axr3', 'axr4','axr5']
Phoneme_sets['b']=['b']
Phoneme_sets['E']=['E1', 'E2', 'E3', 'E4', 'E5']
Phoneme_sets['eI']=['eI1', 'eI2', 'eI3','eI4', 'eI5', 'eI7']
Phoneme_sets['f']=['f']
Phoneme_sets['H']=['H']
Phoneme_sets['i_']=['i:1', 'i:2', 'i:3', 'i:4', 'i:5','i:7']
Phoneme_sets['j']=['j']
Phoneme_sets['k']=['k']
Phoneme_sets['k_h']=['k_h']
Phoneme_sets['l']=['l']
Phoneme_sets['m']=['m']
Phoneme_sets['n']=['n', 'n1','n2','n3','n4','n5']
Phoneme_sets['N']=['N1','N2','N3','N4','N5']
Phoneme_sets['O_']=['O:1', 'O:2', 'O:3', 'O:4','O:5']
Phoneme_sets['oU']=['oU1', 'oU2', 'oU3', 'oU4', 'oU5']
Phoneme_sets['p']=['p']
Phoneme_sets['p_h']=['p_h']
Phoneme_sets['s']=['s','s1','s2', 's3', 's4', 's5']
Phoneme_sets['s6']=['s6']
Phoneme_sets['ss']=['ss', 'ss1', 'ss2', 'ss3', 'ss4','ss5']
Phoneme_sets['t']=['t']
Phoneme_sets['t_h']=['t_h']
Phoneme_sets['ts']=['ts', 'ts1', 'ts2', 'ts3', 'ts4', 'ts5']
Phoneme_sets['ts6']=['ts6']
Phoneme_sets['ts6_h']=['ts6_h']
Phoneme_sets['ts_h']=['ts_h', 'ts_h1', 'ts_h2', 'ts_h3', 'ts_h4']
Phoneme_sets['ttss']=['ttss','ttss1', 'ttss2', 'ttss3', 'ttss4', 'ttss5']
Phoneme_sets['ttss_h']=['ttss_h', 'ttss_h1','ttss_h2', 'ttss_h3', 'ttss_h4', 'ttss_h5']
Phoneme_sets['u_']=['u:1', 'u:2', 'u:3','u:4', 'u:5', 'u:7']
Phoneme_sets['uA_']=['uA:','uA:1','uA:2','uA:3','uA:4','uA:5']
Phoneme_sets['uaI']=['uaI','uaI1','uaI2','uaI3','uaI4','uaI5']
Phoneme_sets['uax']=['uax']
Phoneme_sets['ueI']=['ueI','ueI1','ueI2','ueI3','ueI4','ueI5']
Phoneme_sets['uO_']=['uO:','uO:1','uO:2','uO:3','uO:4','uO:5']
Phoneme_sets['x']=['x']
Phoneme_sets['y']=['y1', 'y2', 'y3', 'y4', 'y5']
Phoneme_sets['zz']=['zz','zz4', 'zz5']




Phoneme01_sets=collections.OrderedDict()
Phoneme01_sets['A_']=['A:', 'A:1']
Phoneme01_sets['aI']=['aI1']
Phoneme01_sets['aU']=['aU1']
Phoneme01_sets['ax']=['ax','ax1']
Phoneme01_sets['axr']=['axr1']
Phoneme01_sets['E']=['E1']
Phoneme01_sets['eI']=['eI1']
Phoneme01_sets['f']=['f']
Phoneme01_sets['H']=['H']
Phoneme01_sets['i_']=['i:1']
Phoneme01_sets['j']=['j']
Phoneme01_sets['k']=['k']
Phoneme01_sets['k_h']=['k_h']
Phoneme01_sets['l']=['l']
Phoneme01_sets['m']=['m']
Phoneme01_sets['n']=['n', 'n1']
Phoneme01_sets['N']=['N1']
Phoneme01_sets['O_']=['O:1']
Phoneme01_sets['oU']=['oU1']
Phoneme01_sets['p']=['p']
Phoneme01_sets['p_h']=['p_h']
Phoneme01_sets['s']=['s','s1']
Phoneme01_sets['s6']=['s6']
Phoneme01_sets['ss']=['ss', 'ss1']
Phoneme01_sets['t']=['t']
Phoneme01_sets['t_h']=['t_h']
Phoneme01_sets['ts']=['ts', 'ts1']
Phoneme01_sets['ts6']=['ts6']
Phoneme01_sets['ts6_h']=['ts6_h']
Phoneme01_sets['ts_h']=['ts_h', 'ts_h1']
Phoneme01_sets['ttss']=['ttss','ttss1']
Phoneme01_sets['ttss_h']=['ttss_h', 'ttss_h1']
Phoneme01_sets['u_']=['u:1']
Phoneme01_sets['w']=['w']
Phoneme01_sets['x']=['x']
Phoneme01_sets['y']=['y1']
Phoneme01_sets['zz']=['zz']


Place_sets=collections.OrderedDict()
Place_sets['bilabial']=Phoneme_sets['p'] + Phoneme_sets['p_h'] + Phoneme_sets['m']
Place_sets['labio_dental']=Phoneme_sets['f']
Place_sets['alveol_palatal']=Phoneme_sets['ts6'] + Phoneme_sets['ts6_h'] + Phoneme_sets['s6']
Place_sets['alveolar']=Phoneme_sets['t'] + Phoneme_sets['t_h'] + Phoneme_sets['n'] + \
                            Phoneme_sets['ts'] + Phoneme_sets['ts_h'] + Phoneme_sets['s'] + Phoneme_sets['l']
Place_sets['velar']=Phoneme_sets['k'] + Phoneme_sets['k_h'] + Phoneme_sets['n'] + Phoneme_sets['x'] 
Place_sets['retroflex']=Phoneme_sets['ttss'] + Phoneme_sets['ttss_h'] + Phoneme_sets['ss'] + Phoneme_sets['zz']
Place_sets['palatal']=Phoneme_sets['j'] + Phoneme_sets['H']
Place_sets['rounded']= Phoneme_sets['H']
Place_sets['Vowel']=Phoneme_sets['A_'] + Phoneme_sets['O_'] + Phoneme_sets['ax'] + Phoneme_sets['E'] +\
                    Phoneme_sets['aI'] + Phoneme_sets['eI'] + Phoneme_sets['aU'] + Phoneme_sets['oU'] +\
                    Phoneme_sets['axr'] + Phoneme_sets['i_'] + Phoneme_sets['u_'] + Phoneme_sets['y'] 
Place_sets['labial']=Place_sets['bilabial'] + Place_sets['labio_dental'] + Place_sets['rounded']
Place_sets['coronal']=Place_sets['alveolar'] + Place_sets['retroflex'] + Place_sets['alveol_palatal']
Place_sets['dorsal']=Place_sets['velar'] + Place_sets['palatal']

Place_sets_simple1=collections.OrderedDict()
Place_sets_simple1['bilabial']=Phoneme_sets['p'] + Phoneme_sets['p_h'] + Phoneme_sets['m']
Place_sets_simple1['labio_dental']=Phoneme_sets['f']
Place_sets_simple1['alveol_palatal']=Phoneme_sets['ts6'] + Phoneme_sets['ts6_h'] + Phoneme_sets['s6']
Place_sets_simple1['alveolar']=Phoneme_sets['t'] + Phoneme_sets['t_h'] + Phoneme_sets['n'] + \
                            Phoneme_sets['ts'] + Phoneme_sets['ts_h'] + Phoneme_sets['s'] + Phoneme_sets['l']
Place_sets_simple1['velar']=Phoneme_sets['k'] + Phoneme_sets['k_h'] + Phoneme_sets['n'] + Phoneme_sets['x'] 
Place_sets_simple1['retroflex']=Phoneme_sets['ttss'] + Phoneme_sets['ttss_h'] + Phoneme_sets['ss'] + Phoneme_sets['zz']
Place_sets_simple1['palatal']=Phoneme_sets['j'] + Phoneme_sets['H']
Place_sets_simple1['rounded']= Phoneme_sets['H']
Place_sets_simple1['Vowel']=Phoneme_sets['A_'] + Phoneme_sets['O_'] + Phoneme_sets['ax'] + Phoneme_sets['E'] +\
                    Phoneme_sets['aI'] + Phoneme_sets['eI'] + Phoneme_sets['aU'] + Phoneme_sets['oU'] +\
                    Phoneme_sets['axr'] + Phoneme_sets['i_'] + Phoneme_sets['u_'] + Phoneme_sets['y'] 

Place_sets_simple2=collections.OrderedDict()
Place_sets_simple2['labial']=Phoneme_sets['p'] + Phoneme_sets['p_h'] + Phoneme_sets['m'] +\
                             Phoneme_sets['f']  + Phoneme_sets['H']
Place_sets_simple2['coronal']=Phoneme_sets['t'] + Phoneme_sets['t_h'] + Phoneme_sets['n'] + \
                            Phoneme_sets['ts'] + Phoneme_sets['ts_h'] + Phoneme_sets['s'] + Phoneme_sets['l'] +\
                            Phoneme_sets['ttss'] + Phoneme_sets['ttss_h'] + Phoneme_sets['ss'] + Phoneme_sets['zz'] +\
                            Phoneme_sets['ts6'] + Phoneme_sets['ts6_h'] + Phoneme_sets['s6']
Place_sets_simple2['dorsal']=Phoneme_sets['k'] + Phoneme_sets['k_h'] + Phoneme_sets['n'] + Phoneme_sets['x']  +\
                             Phoneme_sets['j'] + Phoneme_sets['H']
Place_sets_simple2['Vowel']=Phoneme_sets['A_'] + Phoneme_sets['O_'] + Phoneme_sets['ax'] + Phoneme_sets['E'] +\
                    Phoneme_sets['aI'] + Phoneme_sets['eI'] + Phoneme_sets['aU'] + Phoneme_sets['oU'] +\
                    Phoneme_sets['axr'] + Phoneme_sets['i_'] + Phoneme_sets['u_'] + Phoneme_sets['y'] 


Manner_sets=collections.OrderedDict()
Manner_sets['plosive_aspirated']=Phoneme_sets['p'] + Phoneme_sets['t'] + Phoneme_sets['k']
Manner_sets['plosive_unaspirated']=Phoneme_sets['p_h'] + Phoneme_sets['t_h'] + Phoneme_sets['k_h']
Manner_sets['nasal'] = Phoneme_sets['m'] + Phoneme_sets['n']
Manner_sets['affricate_unaspirated'] = Phoneme_sets['ts'] + Phoneme_sets['ttss'] + Phoneme_sets['ts6']
Manner_sets['affricate_aspirated'] = Phoneme_sets['ts_h'] + Phoneme_sets['ttss_h'] + Phoneme_sets['ts6_h'] 
Manner_sets['fricative_sibilant'] = Phoneme_sets['s'] + Phoneme_sets['ss'] + Phoneme_sets['s6'] 
Manner_sets['fricative_unsibilant'] = Phoneme_sets['f'] + Phoneme_sets['x']
Manner_sets['liquid'] = Phoneme_sets['l'] + Phoneme_sets['zz']
Manner_sets['glide'] =  Phoneme_sets['j'] + Phoneme_sets['y']
Manner_sets['Vowel']=Phoneme_sets['A_'] + Phoneme_sets['O_'] + Phoneme_sets['ax'] + Phoneme_sets['E'] +\
                    Phoneme_sets['aI'] + Phoneme_sets['eI'] + Phoneme_sets['aU'] + Phoneme_sets['oU'] +\
                    Phoneme_sets['axr'] + Phoneme_sets['i_'] + Phoneme_sets['u_'] + Phoneme_sets['y'] 
Manner_sets['plosive']=Manner_sets['plosive_aspirated'] + Manner_sets['plosive_unaspirated']
Manner_sets['affricate'] = Manner_sets['affricate_unaspirated'] + Manner_sets['affricate_aspirated']
Manner_sets['fricative'] = Manner_sets['fricative_sibilant'] + Manner_sets['fricative_unsibilant']


Manner_sets_simple1=collections.OrderedDict()
Manner_sets_simple1['plosive_aspirated']=Phoneme_sets['p'] + Phoneme_sets['t'] + Phoneme_sets['k']
Manner_sets_simple1['plosive_unaspirated']=Phoneme_sets['p_h'] + Phoneme_sets['t_h'] + Phoneme_sets['k_h']
Manner_sets_simple1['nasal'] = Phoneme_sets['m'] + Phoneme_sets['n']
Manner_sets_simple1['affricate_unaspirated'] = Phoneme_sets['ts'] + Phoneme_sets['ttss'] + Phoneme_sets['ts6']
Manner_sets_simple1['affricate_aspirated'] = Phoneme_sets['ts_h'] + Phoneme_sets['ttss_h'] + Phoneme_sets['ts6_h'] 
Manner_sets_simple1['fricative_sibilant'] = Phoneme_sets['s'] + Phoneme_sets['ss'] + Phoneme_sets['s6'] 
Manner_sets_simple1['fricative_unsibilant'] = Phoneme_sets['f'] + Phoneme_sets['x']
Manner_sets_simple1['liquid'] = Phoneme_sets['l'] + Phoneme_sets['zz']
Manner_sets_simple1['glide'] = Phoneme_sets['j']  + Phoneme_sets['y']
Manner_sets_simple1['Vowel']=Phoneme_sets['A_'] + Phoneme_sets['O_'] + Phoneme_sets['ax'] + Phoneme_sets['E'] +\
                    Phoneme_sets['aI'] + Phoneme_sets['eI'] + Phoneme_sets['aU'] + Phoneme_sets['oU'] +\
                    Phoneme_sets['axr'] + Phoneme_sets['i_'] + Phoneme_sets['u_'] + Phoneme_sets['y'] 

Manner_sets_simple2=collections.OrderedDict()
Manner_sets_simple2['nasal'] = Phoneme_sets['m'] + Phoneme_sets['n']
Manner_sets_simple2['liquid'] = Phoneme_sets['l'] + Phoneme_sets['zz']
Manner_sets_simple2['glide'] =  Phoneme_sets['j'] + Phoneme_sets['y']
Manner_sets_simple2['plosive']=Phoneme_sets['p'] + Phoneme_sets['t'] + Phoneme_sets['k'] +\
                       Phoneme_sets['p_h'] + Phoneme_sets['t_h'] + Phoneme_sets['k_h']
Manner_sets_simple2['affricate'] = Phoneme_sets['ts'] + Phoneme_sets['ttss'] + Phoneme_sets['ts6'] +\
                           Phoneme_sets['ts_h'] + Phoneme_sets['ttss_h'] + Phoneme_sets['ts6_h'] 
Manner_sets_simple2['fricative'] = Phoneme_sets['s'] + Phoneme_sets['ss'] + Phoneme_sets['s6']  +\
                           Phoneme_sets['f'] + Phoneme_sets['x']
Manner_sets_simple2['Vowel']=Phoneme_sets['A_'] + Phoneme_sets['O_'] + Phoneme_sets['ax'] + Phoneme_sets['E'] +\
                    Phoneme_sets['aI'] + Phoneme_sets['eI'] + Phoneme_sets['aU'] + Phoneme_sets['oU'] +\
                    Phoneme_sets['axr'] + Phoneme_sets['i_'] + Phoneme_sets['u_'] + Phoneme_sets['y'] 
                    
# =============================================================================
'''

    

'''             
# =============================================================================

PhoneMapp_dict={'u:':Phoneme_sets['u_'],\
                'i:':Phoneme_sets['i_']+['j'],\
                'A:':Phoneme_sets['A_']}
# PhoneMapp_dict={'u:':Phoneme_sets['u_'],\
#                 'i:':Phoneme_sets['i_'],\
#                 'A:':Phoneme_sets['A_']}
    
