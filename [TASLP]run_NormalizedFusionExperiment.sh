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
    run_python_script "[TASLP]ClassificationExpFusion_Experiment.py" "--Normalize_way" "$normalize_way"
fi

if [[ $stage -le 2 ]]; then
    run_python_script "[TASLP]RegressionExpFusion_Experiment.py" "--Normalize_way" "$normalize_way"
fi

exit 0;