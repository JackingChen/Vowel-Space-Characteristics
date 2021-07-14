Guide:

這邊紀錄一下做Formant分析報告的步驟
![](https://i.imgur.com/mCsDdr8.png)

* 首先： 利用畫圖的code (Analyze_F1F2.py, Analyze_F1F2_VowelFormantFeatures.py, Analyze_F1F2_tVSA_FCR.py)來畫出主要分析圖（scatter or histogram）
* 第二： 畫出特定feature的相關性圖，對角線上的點都是比較有可能有相關的點
* 第三： 截圖並手動輸入那些feature數值
![](https://i.imgur.com/a3keCPF.png)



test_articulation.sh: 沒啥屁用

Get_F1F2.py: 更改過articulation.py 的function 拿來算F1, F2

get_F1F2_feature.py: 將算好的F1, F2變成Session level feature的形式

Analyze_F1F2_tVSA_FCR.py: 主要拿來分析tVSA_FCR和F-value的腳本Analyze_F1F2_VowelFormantFeatures.py: 主要拿來分析F1 F2的mean, std, skew, kurtosis的腳本
Analyze_F1F2.py: 主要拿來分析各個vowel的F1 F2的散佈情況(畫圖和做檢定)


畫圖與檢驗的coding介紹

# Analyze_F1F2.py:

## Catagorize ADOS cases in three groups and do t-test
```
groups=[[np.array([0]), np.array([2, 3, 4, 5, 6, 7, 8])],\
        [np.array([0]), np.array([3, 4, 5, 6, 7, 8])]]
for group in groups:
    print(" processing under assumption of group", group)
    df_data_all=pd.DataFrame([],columns=['F1','F2','ADOS'])
    for people in Formants_people_symb.keys():
        for phone, values in Formants_people_symb[people].items():
            if phone not in phonewoprosody.Phoneme_sets[phoneme]:
                continue
            F1F2_vals=np.vstack(Formants_people_symb[people][phone])
            ASDlab=Label.label_raw[label_choose][Label.label_raw['name']==people].values
            ADOS_severity=find_group(ASDlab,group)
            ADOS_catagory_array=np.repeat(ADOS_severity,len(F1F2_vals)).reshape(-1,1).astype(int)
            
            df_data=pd.DataFrame(np.hstack((F1F2_vals,ADOS_catagory_array)),columns=['F1','F2','ADOS'])
            df_data_all=pd.concat([df_data_all,df_data])

    ''' t-test '''
    sets={}
    for i in range(len(set(df_data_all['ADOS']))):
        sets[i]=df_data_all[df_data_all['ADOS']==i]
    set0=df_data_all[df_data_all['ADOS']==0]
    set1=df_data_all[df_data_all['ADOS']==1]
    set2=df_data_all[df_data_all['ADOS']==2]
    print(stats.ttest_ind(set0, set1, equal_var=False))
    print(stats.ttest_ind(set1, set2, equal_var=False))
    print(stats.ttest_ind(set0, set2, equal_var=False))
    
    #Data Visualization
    
    hisglow_sets=pd.concat([set0,set1])
    
    x = hisglow_sets['F1'].values
    y = hisglow_sets['F2'].values
    
    colors = hisglow_sets['ADOS'].values
    area=np.repeat(1,len(x))
    
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.show()
    
    
    set0F1_array=set0['F1'].values
    set1F1_array=set1['F1'].values
    set2F1_array=set2['F1'].values
    
    set0F2_array=set0['F2'].values
    set1F2_array=set1['F2'].values
    set2F2_array=set2['F2'].values
    
   
    plt.hist(set0F1_array ,alpha=0.3, label='set0')
    plt.hist(set1F1_array ,alpha=0.5, label='set1')
    plt.hist(set2F1_array ,alpha=0.7, label='set2')
    
    plt.legend(loc='upper right')
    plt.show()
    
    plt.hist(set0F2_array ,alpha=0.3, label='set0')
    plt.hist(set1F2_array ,alpha=0.5, label='set1')
    plt.hist(set2F2_array ,alpha=0.7, label='set2')
    plt.legend(loc='upper right')
    plt.show()
    
    
    print("Mean value of  Set 0 = ")
    print(set0.mean())
    print("Mean value of  Set 1 = ")
    print(set1.mean())
    print("Mean value of Set 2 = ")
    print(set2.mean())
''''############################################################################'''
```


## Manova and one-tailed ANOVA-test
```
''''############################################################################'''

three_grps=[g for g in groups if len(g)>2]
highlow_grps=[[sublist[0],sublist[-1]]  for sublist in three_grps ]
groups=highlow_grps
''' testing area '''
for group in groups:
    print(" processing under assumption of group", group)
    df_data_all=pd.DataFrame([],columns=['F1','F2','ADOS'])
    for people in Formants_people_symb.keys():
        for phone, values in Formants_people_symb[people].items():
            if phone not in phonewoprosody.Phoneme_sets[phoneme]:
                continue
            F1F2_vals=np.vstack(Formants_people_symb[people][phone])
            ASDlab=Label.label_raw[label_choose][Label.label_raw['name']==people].values
            assert ASDlab in list(set(ADOS_label))
            ADOS_severity=find_group(ASDlab,group)
            if ADOS_severity == -1:
                continue
            ADOS_catagory_array=np.repeat(ADOS_severity,len(F1F2_vals)).reshape(-1,1).astype(int)
            
            df_data=pd.DataFrame(np.hstack((F1F2_vals,ADOS_catagory_array)),columns=['F1','F2','ADOS'])
            df_data_all=pd.concat([df_data_all,df_data])
    
    '''  MANOVA test '''
    maov = MANOVA.from_formula('F1 + F2   ~ ADOS', data=df_data_all)
    
    # print(maov.mv_test())
    
    '''  ANOVA test '''
    moore_lm = ols('ADOS ~ F1 + F2 ',data=df_data_all).fit()
    print("utt number of group 0 = {0}, utt number of group 1 = {1}".format(len(df_data_all[df_data_all['ADOS']==0]),len(df_data_all[df_data_all['ADOS']==1])))
    print(sm.stats.anova_lm(moore_lm, typ=2))
''''############################################################################'''
```

## Analyze_F1F2_VowelFormantFeatures.py
畫二維度散佈圖：
```
=============================================================================
'''

Plotting area

'''

pid=list(sorted(Formants_people_symb.keys()))
pid_dict={}
for i,pi in enumerate(pid):
    pid_dict[pi]=i

phoneme_color_map={'A':'tab:blue','w':'tab:orange','j':'tab:green'}
# =============================================================================

Plotout_path="Plots/"

if not os.path.exists(Plotout_path):
    os.makedirs(Plotout_path)
if not os.path.exists('Hist'):
    os.makedirs('Hist')

phone='E2'
for people in Vowels_phonenprosody.keys():
    df_formant_statistic=formants_df_people_vowel[phone].astype(float)
    formant_info=df_formant_statistic.loc[people]
    ASDlab=Label.label_raw[label_choose][Label.label_raw['name']==people].values
    
    
    values=Vowels_phonenprosody[people][phone]
    if len(values) == 0:
        continue
    
    x,y=np.vstack(values)[:,0],np.vstack(values)[:,1]
    # 

    fig, ax = plt.subplots()
    area=np.repeat(10,len(x))
    # cms=np.repeat(phoneme_color_map[phone],len(x))
    
    
    plt.scatter(x, y, s=area,  label=phone)
    plt.title('{0}_ADOS{1}'.format(pid_dict[people],ASDlab[0]))
        
    additional_infoF1="mean={mean}, std={std}, skew={skew}, kurtosis={kurtosis}".format(\
        mean=formant_info['F1+mean'],std=formant_info['F1+std'],skew=formant_info['F1+skew'],kurtosis=formant_info['F1+kurtosis'])
    additional_infoF2="mean={mean}, std={std}, skew={skew}, kurtosis={kurtosis}".format(\
        mean=formant_info['F2+mean'],std=formant_info['F2+std'],skew=formant_info['F2+skew'],kurtosis=formant_info['F2+kurtosis'])
    plt.ylim(0, 5000)
    plt.xlim(0, 5000)
    ax.legend()
    plt.figtext(0,0.02,additional_infoF1)
    plt.figtext(0,-0.05,additional_infoF2)
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.savefig(Plotout_path+'{0}_ADOS{1}'.format(pid_dict[people],ASDlab[0]),dpi=300, bbox_inches = "tight")
    plt.show()
```

畫單維度散佈圖：
```
=============================================================================
    ''' Debug figure '''
    ''' plot histogram '''
    # F1F2Mapdict={'F1':x,'F2':y}
    # inspect_dim='F2'
    # #''' plot histogram '''
    # figure = plt.figure()
    # plt.hist(F1F2Mapdict[inspect_dim])
    # plt.title('{0}_ADOS{1}'.format(pid_dict[people],ASDlab[0]))
    # additional_infoF2="mean={mean}, std={std}, skew={skew}, kurtosis={kurtosis}".format(\
    #     mean=formant_info[inspect_dim+'+mean'],std=formant_info[inspect_dim+'+std'],\
    #         skew=formant_info[inspect_dim+'+skew'],kurtosis=formant_info[inspect_dim+'+kurtosis'])
    # plt.legend()
    # plt.xlim(0, 5000)
    # plt.figtext(0,-0.05,additional_infoF2)
    # plt.savefig("Hist/"+'{0}_ADOS{1}_{2}'.format(pid_dict[people],ASDlab[0],inspect_dim),dpi=300, bbox_inches = "tight")
    # =============================================================================
```

畫圖找哪些點比較有相關性（在x = y 上的點都比較有相關性, x=1-y上的點比較有負相關性）。這邊這個圖的功用在於找幾個例子佐證

```
=============================================================================
    '''

    Plot x = desired value, y= ADOS score. We want to sample critical samples for further inspection
    
    '''
    num_spoken_phone=pd.DataFrame.from_dict(Vowels_sampNum[phone],orient='index',columns=['num_spoken_phone'])
    N=10
    # =============================================================================
    fig, ax = plt.subplots()
    feature_str='E2_F2+skew'
    phone=feature_str.split("_")[0]
    Inspect_column='_'.join(feature_str.split("_")[1:])
    df_formant_statistic=formants_df_people_vowel[phone].astype(float)[Inspect_column]
    label_ADOSC=Label.label_raw[['name','ADOS_C']].set_index("name")
    df_formant_statistic=pd.concat([df_formant_statistic,label_ADOSC],axis=1)
    df_formant_statistic=pd.concat([df_formant_statistic,num_spoken_phone],axis=1)
    df_formant_statistic_qualified=df_formant_statistic[df_formant_statistic['num_spoken_phone']>N]
    
    area=np.repeat(10,len(df_formant_statistic_qualified))
    cms=range(len(df_formant_statistic_qualified))
    
    
    x, y=df_formant_statistic_qualified.iloc[:,0], df_formant_statistic_qualified.iloc[:,1]
    plt.scatter(x,y, c=cms, s=area)
    for xi, yi, pidi in zip(x,y,df_formant_statistic_qualified.index):
        ax.annotate(str(pid_dict[pidi]), xy=(xi,yi))
    plt.title(feature_str)
    plt.xlabel("feature")
    plt.ylabel("ADOS")
    ax.legend()
    plt.savefig(Plotout_path+'Acorrelation{0}.png'.format(feature_str),dpi=300, bbox_inches = "tight")
    plt.show()
```

## Analyze_F1F2_tVSA_FCR.py:
計算tVSA，FCR，和F-value

```
# =============================================================================
'''

Calculating FCR
FCR=(F2u+F2a+F1i+F1u)/(F2i+F1a)
VSA1=ABS((F1i*(F2a –F2u)+F1a *(F2u–F2i)+F1u*(F2i–F2a))/2)
VSA2=sqrt(S*(S-EDiu)(S-EDia)(S-EDau))
LnVSA=sqrt(LnS*(LnS-LnEDiu)(LnS-LnEDia)(LnS-LnEDau))

where,
u=F12_val_dict['w']
a=F12_val_dict['A']
i=F12_val_dict['j']

EDiu=sqrt((F2u–F2i)^2+(F1u–F1i)^2)
EDia=sqrt((F2a–F2i)^2+(F1a–F1i)^2)
EDau=sqrt((F2u–F2a)^2+(F1u–F1a)^2)
S=(EDiu+EDia+EDau)/2
'''
# =============================================================================
import statistics 

df_formant_statistic=pd.DataFrame([],columns=['FCR','VSA1','VSA2','LnVSA','ADOS','u_num','a_num','i_num',\
                                              'F_vals_f1', 'F_vals_f2', 'F_val_mix', 'criterion_score'])
for people in Vowels_AUI_mean.keys():
    F12_val_dict=Vowels_AUI_mean[people]
    u_num, a_num, i_num=Vowels_AUI_sampNum[people]['w'],Vowels_AUI_sampNum[people]['A'],Vowels_AUI_sampNum[people]['j']
    
    u=F12_val_dict['w']
    a=F12_val_dict['A']
    i=F12_val_dict['j']
    if len(u)==0 or len(a)==0 or len(i)==0:
        continue
    
    numerator=u[1] + a[1] + i[0] + u[0]
    demominator=i[1] + a[0]
    FCR=np.float(numerator/demominator)
    # assert FCR <=2
    
    VSA1=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
    
    LnVSA=np.abs((i[0]*(a[1]-u[1]) + a[0]*(u[1]-i[1]) + u[0]*(i[1]-a[1]) )/2)
    
    EDiu=np.sqrt((u[1]-i[1])**2+(u[0]-i[0])**2)
    EDia=np.sqrt((a[1]-i[1])**2+(a[0]-i[0])**2)
    EDau=np.sqrt((u[1]-a[1])**2+(u[0]-a[0])**2)
    S=(EDiu+EDia+EDau)/2
    VSA2=np.sqrt(S*(S-EDiu)*(S-EDia)*(S-EDau))
    
    LnVSA=np.sqrt(np.log(S)*(np.log(S)-np.log(EDiu))*(np.log(S)-np.log(EDia))*(np.log(S)-np.log(EDau)))
    
    ASDlab=Label.label_raw[label_choose][Label.label_raw['name']==people].values
    
    
    
    # =============================================================================
    ''' F-value, Valid Formant measure '''
    
    # =============================================================================
    # Get data
    F12_raw_dict=Vowels_AUI[people]
    u=F12_raw_dict['w']
    a=F12_raw_dict['A']
    i=F12_raw_dict['j']
    df_vowel = pd.DataFrame(np.vstack([u,a,i]),columns=['F1','F2'])
    df_vowel['vowel'] = np.hstack([np.repeat('u',len(u)),np.repeat('a',len(a)),np.repeat('i',len(i))])
    df_vowel['target']=pd.Categorical(df_vowel['vowel'])
    df_vowel['target']=df_vowel['target'].cat.codes
    # F-test
    moore_lm = ols('target ~ F1 + F2 ',data=df_vowel).fit()
    print("utt number of group u = {0}, utt number of group i = {1}, utt number of group A = {2}".format(\
        len(u),len(a),len(i)))
    F_vals=sm.stats.anova_lm(moore_lm, typ=2)
    F_vals_f1=F_vals.loc['F1','F']
    F_vals_f2=F_vals.loc['F2','F']
    F_val_mix=F_vals_f1 + F_vals_f2

    
    # =============================================================================
    # criterion
    # F1u < F1a
    # F2u < F2a
    # F2u < F2i
    # F1i < F1a
    # F2a < F2i
    # =============================================================================
    u_mean=F12_val_dict['w']
    a_mean=F12_val_dict['A']
    i_mean=F12_val_dict['j']
    
    F1u, F2u=u_mean[0], u_mean[1]
    F1a, F2a=a_mean[0], a_mean[1]
    F1i, F2i=i_mean[0], i_mean[1]
    
    filt1 = [1 if F1u < F1a else 0]
    filt2 = [1 if F2u < F2a else 0]
    filt3 = [1 if F2u < F2i else 0]
    filt4 = [1 if F1i < F1a else 0]
    filt5 = [1 if F2a < F2i else 0]
    criterion_score=np.sum([filt1,filt2,filt3,filt4,filt5])
    
    
    df_formant_statistic.loc[people]=[np.round(FCR,3), np.round(VSA1,3),\
                                      np.round(VSA2,3), np.round(LnVSA,3), ASDlab[0],\
                                      u_num, a_num, i_num,\
                                      np.round(F_vals_f1,3), np.round(F_vals_f2,3), np.round(F_val_mix,3),\
                                      np.round(criterion_score,3)]
```

## CalculatenPlotVSA_bydiagnosis.py:
---
這個腳本拿來畫圖支援IS2021的paper，

![Analysis](figures/Analysis.png)




## CalculatenPlotSampleScatter_bydiagnosis.py

##  InspectF1F2Details.py 
---
這個腳本拿來產生praat檔，目的是用來檢查某個人的F1 F2 derive的結果怎樣。
範例圖：

![praatPhoneStich](figures/phone_stitch_fig.png)

流程：
1. 根據上圖看出大略critical phone （AUI） 的狀況，至於這個phone在某個utterance發生了什麼狀況就要檢查Inspect/fileOfInterest/{utterance name}了
![praatPhoneStich](figures/utteranceFormantPlot.png)

---
