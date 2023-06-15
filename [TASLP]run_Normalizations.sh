#!/usr/bin/env bash

set -x

# 默认值
stage=0
normalize_way="proposed"

# 解析命名參數
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -s|--stage)
        stage="$2"
        shift
        shift
        ;;
        -n|--Normalize_way)
        normalize_way="$2"
        shift
        shift
        ;;
        *)
        echo "未知的參數: $1"
        exit 1
        ;;
    esac
done

# 檢查必要參數是否提供
if [[ -z $stage ]]; then
    echo "請提供 stage 參數。"
    echo "使用方法: $0 -s|--stage stage_number [-n|--Normalize_way way]"
    exit 1
fi

# 定義執行 Python 腳本的函數
run_python_script() {
    script_name="$1"
    shift
    echo "執行腳本 $script_name"
    python "$script_name" "$@"
    if [[ $? -ne 0 ]]; then
        echo "腳本 $script_name 執行時出現錯誤。"
        exit 1
    fi
}




if [[ $stage -le 1 ]]; then
    run_python_script "[TASLP]3.LevelOfClustering_LOC.py" "--dataset_role" "KID_FromTD_DOCKID" "--Normalize_way" "$normalize_way"
    run_python_script "[TASLP]3.LevelOfClustering_LOC.py" "--dataset_role" "KID_FromASD_DOCKID" "--Normalize_way" "$normalize_way"
fi

if [[ $stage -le 2 ]]; then
    run_python_script "[TASLP]4-1.1.SynchronyVSC(Full).py"  "--Normalize_way" "$normalize_way"
fi

if [[ $stage -le 3 ]]; then
    run_python_script "[TASLP]4-2.1.SynchronyPhonation(proprocess).py"
    pass
fi
if [[ $stage -le 4 ]]; then
    run_python_script "[TASLP]4-2.2.SynchronyPhonation(Feat).py"
    pass
fi
if [[ $stage -le 5 ]]; then
    run_python_script "[TASLP]ClassificationExpFusion_script.py" "--Normalize_way" "$normalize_way"
    run_python_script "[TASLP]ClassificationExpBaseFeat_script.py" "--Normalize_way" "$normalize_way"
fi

if [[ $stage -le 6 ]]; then
    run_python_script "[TASLP]RegressionExpFusion_script.py" "--Normalize_way" "$normalize_way"
    run_python_script "[TASLP]RegressionExpBaseFeat_script.py" "--Normalize_way" "$normalize_way"
    pass
fi

if [[ $stage -le 7 ]]; then
    run_python_script "[TASLP]OrganizeResult.py" "--Normalize_way" "$normalize_way"
fi
exit 0;