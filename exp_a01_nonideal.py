"""実験A01: アナログ回路の不完全さ（雑音・部品のずれ・設定値の粗さ）に Sin-CNN2 がどれだけ耐えるか。

sinwave_minimal.py を import して読むだけ（書き換えない）。CPU のみ（GPU は使わない）。
1. デモと同じ合成 4 クラス課題で学習（標準構成と、FPAA でも載りそうな小さい構成）
2. 学習済みの板の値を使い、順伝播にアナログ回路で起きる誤差を入れて正解率を測る
使い方: python3 exp_a01_nonideal.py /path/to/Sin-CNN2 [small|full] [steps]
"""
import sys, math, random, time
import torch
sys.path.insert(0, sys.argv[1])
import sinwave_minimal as sw
torch.set_num_threads(2)
size = sys.argv[2] if len(sys.argv) > 2 else "small"
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1500

# --- デモと同じ合成データ（sinwave_minimal.py の __main__ と同じ作り方）---
keys = {0: "あいうえ", 1: "かきくけ", 2: "さしすせ", 3: "たちつて"}
pool = "なにぬねのはひふへほまみむめもやゆよらりるれろわを"
def sample(r, y):
    return "".join(r.choice(keys[y]) if r.random() < 0.6 else r.choice(pool) for _ in range(r.randint(4, 8)))
r = random.Random(0)
tr = [(sample(r, y), y) for y in keys for _ in range(300)]
te = [(sample(r, y), y) for y in keys for _ in range(250)]

cfg = dict(full=dict(L=2048, S=256, layers=16, patches=64),
           mid=dict(L=512, S=64, layers=8, patches=32),
           small=dict(L=512, S=64, layers=4, patches=16),
           tiny=dict(L=256, S=32, layers=4, patches=16))[size]
net = sw.SinWaveNet(n_slots=4, **cfg)
t0 = time.time()
sw.train(net, [t for t, _ in tr], [y for _, y in tr], steps=steps)
print(f"[{size}] {cfg} 学習 {steps} 回: {time.time() - t0:.0f} 秒, 学習する値 {sum(p.numel() for p in net.parameters())} 個")
X = net.encode([t for t, _ in te]); Y = torch.tensor([y for _, y in te])

@torch.no_grad()
def run(phase_sigma=0.0, phase_bits=0, z_err=0.0, h_ripple=0.0, snr_db=None, det_snr_db=None,
        norm=True, seed=1):
    """net.intensity と同じ計算に、アナログ回路の誤差を入れたもの。"""
    g = torch.Generator().manual_seed(seed)
    u = X[:, None] * net.illum[None]
    u = (torch.roll(u, -net.S, -1) + u + torch.roll(u, net.S, -1)) / 3
    L = net.L
    kx = 2 * math.pi * torch.fft.fftfreq(L)
    for d in range(net.A.shape[0]):
        p = net.patch
        A, phi = net.A[d, p], net.phi[d, p]
        if phase_bits:  # 振幅 A と初期位相 φ を DAC（数値→電圧の変換器）の段数に丸める
            q = 2 ** phase_bits
            A = torch.round(A / 2 * q) / q * 2               # A は 0〜2 の範囲を想定
            phi = torch.round(phi / (2 * math.pi) * q) / q * 2 * math.pi
        phase = A * torch.sin(net.w[d, p] * net.t + phi)
        if phase_sigma:  # 位相変調器の揺らぎ（標本ごと）
            phase = phase + phase_sigma * torch.randn(phase.shape, generator=g)
        u = u * torch.polar(torch.ones_like(phase), phase)
        # 回折 = 分散フィルタ。z_err: 遅延のずれ（部品の値のずれ）、h_ripple: 周波数ごとの位相の波打ち
        kz = torch.sqrt((2 * math.pi) ** 2 - kx ** 2) * 64.0 * (1 + z_err)
        if h_ripple:
            kz = kz + h_ripple * torch.randn(L, generator=g)
        u = torch.fft.ifft(torch.fft.fft(u) * torch.polar(torch.ones(L), kz))
        if snr_db is not None:  # 各段で加わる白色雑音（信号の明るさ 1 に対して）
            s = 10 ** (-snr_db / 20) / math.sqrt(2)
            u = u + s * torch.complex(torch.randn(u.shape, generator=g), torch.randn(u.shape, generator=g))
        if norm:
            u = u / (u.abs().pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
    u = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)
    inten = u.abs().pow(2).mean(1)
    inten = inten / (inten.mean(-1, keepdim=True) + 1e-12)
    sc = inten @ net.det.T
    if det_snr_db is not None:  # 検出器の雑音（窓の明るさの平均に対して）
        sc = sc + sc.mean() * 10 ** (-det_snr_db / 20) * torch.randn(sc.shape, generator=g)
    sc = net.gain * sc + net.bias
    return (sc.argmax(1) == Y).float().mean().item()

with torch.no_grad():
    ref = (net(X).argmax(1) == Y).float().mean().item()
print(f"誤差なし: {ref:.3f}（sinwave_minimal の順伝播そのまま）  再実装: {run():.3f}  増幅（正規化）なし: {run(norm=False):.3f}")
tests = [("位相変調の揺らぎ σ [rad]", "phase_sigma", [0.01, 0.03, 0.1, 0.3, 1.0]),
         ("A, φ の DAC 段数 [bit]", "phase_bits", [10, 8, 6, 4, 3, 2]),
         ("分散フィルタの遅延のずれ（割合）", "z_err", [0.001, 0.01, 0.03, 0.1, 0.3]),
         ("分散フィルタの位相の波打ち σ [rad]", "h_ripple", [0.03, 0.1, 0.3, 1.0]),
         ("各段の雑音 SNR [dB]", "snr_db", [40, 30, 20, 10, 0]),
         ("検出器の雑音 SNR [dB]", "det_snr_db", [40, 30, 20, 10, 0])]
for name, key, vals in tests:
    res = []
    for v in vals:
        a = sum(run(**{key: v}, seed=s) for s in (1, 2, 3)) / 3
        res.append(f"{v}: {a:.3f}")
    print(f"{name:32s} " + "  ".join(res))
