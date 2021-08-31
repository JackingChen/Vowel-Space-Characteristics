# Acoustic_Biomarker
---

## 本Repo的架構
---

所有的feature 計算執行腳本在 acoustic_biomarker/ 之下。計算feature的function module會放在feature資料夾裡面（ex: articulation）

參數儲存
├── articulation
│   ├── HYPERPARAM

執行feature計算腳本：
├── Analyze_DOCKID_syncrony_formant.py (dynamic feature)
├── Analyze_DOCKID_syncrony_phonation.py (dynamic feature)
├── Analyze_F0_phonation.py (static feature)
├── Analyze_F1F2_tVSA_FCR.py (static feature)

├── articulation
│   ├── Get_F1F2.py
│   ├── Multiprocess.py

特定功能執行腳本：
├── Main_regression.py (Leave one out prediction 實驗)
├── Statistical_tests.py (做各種統計分析的腳本)
├── GetSpeechSpeed.py (算一些trivial feature的腳本（字數、語速）)

Context dependent phone 執行腳本：
├── get_Context_dependant_phone_ver2.py
├── Try_Context_dependant_phone_multithread.py
├── Analyze_Context_dependant_phone.py


Module:

├── CtxDepPhone_merger.py
├── Filtering_n_FeatureExtracting.py
├── metric.py
├── utils_jack.py

├── phonation
│   ├── phonation.py
├── articulation
│   ├── articulation.py


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
