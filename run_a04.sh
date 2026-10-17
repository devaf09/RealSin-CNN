#!/bin/sh
# 実験A04 の予約: 板の間隔 z=2 と z=64 で、簡略化の効き目を測る。CPU 2 スレッド、1 つずつ。
# MKL がネットワーク上のフォルダからだと読み込みに失敗することがあるので、作業場所は一時フォルダにする。
D=$(cd "$(dirname "$0")" && pwd)
# Sin-CNN2 のフォルダは環境変数 SIN_CNN2 で指定できる（既定は隣のフォルダ ../Sin-CNN2）。
SIN_CNN2=${SIN_CNN2:-$D/../Sin-CNN2}
cd /tmp
for z in 2 64; do
  python3 $D/exp_a04_simplify.py "$SIN_CNN2" $z 600 > $D/result_a04_z$z.txt 2>&1
done
