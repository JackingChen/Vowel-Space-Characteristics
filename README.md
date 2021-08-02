# Acoustic_Biomarker
---

### miscellaneous.py

暫且拿來做一些分析上需要用的雜工

目前可以拿來比較CtxPhone 跟 original phone 在Vowel space分佈上的差別

![](https://i.imgur.com/dYV0dgF.png)


# Statistical_tests.py

拿來進行統計分析的script，一般會做的統計分析有兩種：correlation和t-test(or U-test, Anova)， 統計對象可以發生在各族群間(ASD/TD, male/female, AD/AS/HFA)
還有做regression (OLS) 來檢查有沒有compounds


`Input: df_formant_statistics`


# get_Context_dependant_phone_ver2.py

從Formant_utt_symb製造各種context dependendt phone的script (會需要時間)，運作方式是
可以從CtxDepPhone_merger.py 融合規則，從phonwopros編寫phone群組規則

# Analyze_Context_dependant_phone.py


Filtering_n_FeatureExtracting.py

Main_regression.py

utils_jack.py




./CtxDepPhone_merger.py


./Try_Context_dependant_phone_multithread.py

./Analyze_features.py


./metric.py
./GetSpeechSpeed.py