#!/bin/sh
# 実験A01 の予約: mid → full の順に、CPU（2 スレッド）で 1 つずつ実行する
cd "$(dirname "$0")"
for c in mid full; do
  python3 exp_a01_nonideal.py /home/claude/Sin-CNN2 $c 1500 > result_a01_$c.txt 2>&1
done
echo done > result_a01.done
