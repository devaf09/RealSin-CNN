"""Sin-CNN2 (sinwave_minimal.py) の 1 回の順伝播・逆伝播で、どの部品に時間がかかるかを CPU で測る。
Sin-CNN2 は読むだけ（import のみ）。GPU は使わない。
使い方: python3 profile_ops.py /path/to/Sin-CNN2
"""
import sys, time, math
import torch
sys.path.insert(0, sys.argv[1])
import sinwave_minimal as sw
torch.set_num_threads(2)

def timeit(f, n=5):
    f(); t = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - t) / n

N, L, P = 64, 2048, 8
u = torch.polar(torch.ones(N, P, L), torch.rand(N, P, L) * 6.28)
net = sw.SinWaveNet(n_slots=4, patterns=P)
ph = torch.rand(L)
H = net.H
parts = {
    "Sin の板（位相をかける: sin と exp）": lambda: u * torch.polar(torch.ones(L), net.A[0, net.patch] * torch.sin(net.w[0, net.patch] * net.t + net.phi[0, net.patch])),
    "回折（FFT → H を掛ける → 逆FFT）": lambda: torch.fft.ifft(torch.fft.fft(u) * H),
    "増幅（平均の明るさを 1 に）": lambda: u / (u.abs().pow(2).mean(-1, keepdim=True).sqrt() + 1e-12),
}
print(f"N={N} 文字列, 照明 {P} 通り, L={L}, 1 層あたり (ms)")
tot = 0
for k, f in parts.items():
    t = timeit(f) * 1e3; tot += t
    print(f"  {k:40s} {t:8.2f}")
x = torch.polar(torch.ones(N, L), torch.rand(N, L) * 6.28)
for pat in (0, 8):
    net = sw.SinWaveNet(n_slots=4, patterns=pat)
    tf = timeit(lambda: net(x), 3) * 1e3
    def fb():
        net.zero_grad(); net(x).sum().backward()
    tb = timeit(fb, 3) * 1e3
    print(f"patterns={pat}: 順伝播 {tf:.1f} ms, 順+逆伝播 {tb:.1f} ms（{N} 文字列）")
# FLOP の目安（複素 FFT ≈ 5 L log2 L 実数演算）
fft = 5 * L * math.log2(L)
per_layer = 2 * fft + 6 * L + 8 * L + 4 * L
print(f"1 文字列・1 照明・16 層の順伝播 ≈ {16 * per_layer / 1e6:.2f} M 実数演算（うち FFT {16 * 2 * fft / (16 * per_layer):.0%}）")
