#!/bin/sh
# 実験A03 の予約: 板の間隔 z ごとの正解率（理想の回折）と、全域通過フィルタで近似した場合。CPU 2 スレッド、1 つずつ。
cd "$(dirname "$0")"
python3 exp_a03_allpass.py /home/claude/Sin-CNN2 0:0 2:-1 4:-1 8:-1 16:-1 32:-1 64:-1 8:8 16:16 64:48 > result_a03.txt 2>&1
