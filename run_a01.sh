#!/bin/sh
# Sin-CNN2 のフォルダは環境変数 SIN_CNN2 で指定できる（既定は隣のフォルダ ../Sin-CNN2）。
SIN_CNN2=${SIN_CNN2:-../Sin-CNN2}
# 実験A01 の予約: mid → full の順に、CPU（2 スレッド）で 1 つずつ実行する
cd "$(dirname "$0")"
for c in mid full; do
  python3 exp_a01_nonideal.py "$SIN_CNN2" $c 1500 > result_a01_$c.txt 2>&1
done
echo done > result_a01.done
