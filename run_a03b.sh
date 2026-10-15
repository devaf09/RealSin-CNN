#!/bin/sh
# 実験A03b の予約: FPAA（OK2902A, CAB 56 個）に載る規模。段数を減らす・層を減らす。CPU 2 スレッド。
cd "$(dirname "$0")"
python3 exp_a03_allpass.py /home/claude/Sin-CNN2 8:4 8:2 4:2 2:1 8:8:4 4:2:4 8:4:4 2:1:4 0:0:4 > result_a03b.txt 2>&1
