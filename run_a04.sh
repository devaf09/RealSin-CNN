#!/bin/sh
# 実験A04 の予約: 板の間隔 z=2 と z=64 で、簡略化の効き目を測る。CPU 2 スレッド、1 つずつ。
# MKL がネットワーク上のフォルダからだと読み込みに失敗することがあるので、作業場所は一時フォルダにする。
D=/mnt/project-files/analog-study
cd /tmp
for z in 2 64; do
  python3 $D/exp_a04_simplify.py /home/claude/Sin-CNN2 $z 600 > $D/result_a04_z$z.txt 2>&1
done
