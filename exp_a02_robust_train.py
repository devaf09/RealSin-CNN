"""実験A02: 誤差を入れながら学習すると、分散フィルタのずれや雑音に強くなるか。

実験A01 で一番弱かった「分散フィルタ（回折）の遅延のずれ」に対して、
学習中に毎回ランダムなずれ・雑音を入れる（誤差を前提にした学習）と、どこまで保つかを測る。
sinwave_minimal.py を import して読むだけ。CPU のみ。
使い方: python3 exp_a02_robust_train.py /path/to/Sin-CNN2 [z_train] [snr_train]
"""
import sys, math, random, time
import torch
import torch.nn.functional as F
sys.path.insert(0, sys.argv[1])
import sinwave_minimal as sw
torch.set_num_threads(2)
z_train = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
snr_train = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0

keys = {0: "あいうえ", 1: "かきくけ", 2: "さしすせ", 3: "たちつて"}
pool = "なにぬねのはひふへほまみむめもやゆよらりるれろわを"
def sample(r, y):
    return "".join(r.choice(keys[y]) if r.random() < 0.6 else r.choice(pool) for _ in range(r.randint(4, 8)))
r = random.Random(0)
tr = [(sample(r, y), y) for y in keys for _ in range(300)]
te = [(sample(r, y), y) for y in keys for _ in range(250)]
cfg = dict(L=512, S=64, layers=8, patches=32)

def forward(net, X, z_err=0.0, snr_db=None, phase_sigma=0.0, g=None, per_sample_z=False):
    L = net.L
    kx = 2 * math.pi * torch.fft.fftfreq(L)
    u = X[:, None] * net.illum[None]
    u = (torch.roll(u, -net.S, -1) + u + torch.roll(u, net.S, -1)) / 3
    for d in range(net.A.shape[0]):
        p = net.patch
        phase = net.A[d, p] * torch.sin(net.w[d, p] * net.t + net.phi[d, p])
        if phase_sigma:
            phase = phase + phase_sigma * torch.randn(phase.shape, generator=g)
        u = u * torch.polar(torch.ones_like(phase), phase)
        if per_sample_z:   # 学習時: 文字列ごと・層ごとに違うずれ（-z_err〜+z_err の一様乱数）
            e = (2 * torch.rand(u.shape[0], 1, 1, generator=g) - 1) * z_err
        else:
            e = torch.tensor(z_err)
        kz = torch.sqrt((2 * math.pi) ** 2 - kx ** 2) * 64.0 * (1 + e)
        u = torch.fft.ifft(torch.fft.fft(u) * torch.polar(torch.ones_like(kz), kz))
        if snr_db is not None:
            s = 10 ** (-snr_db / 20) / math.sqrt(2)
            u = u + s * torch.complex(torch.randn(u.shape, generator=g), torch.randn(u.shape, generator=g))
        u = u / (u.abs().pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
    u = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)
    inten = u.abs().pow(2).mean(1)
    inten = inten / (inten.mean(-1, keepdim=True) + 1e-12)
    return net.gain * (inten @ net.det.T) + net.bias

def train(net, robust, steps=1500, seed=0):
    X = net.encode([t for t, _ in tr]); Y = torch.tensor([y for _, y in tr])
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(Y), (256,), generator=g)
        kw = dict(z_err=z_train, snr_db=snr_train, per_sample_z=True, g=g) if robust else {}
        loss = F.cross_entropy(forward(net, X[i], **kw), Y[i])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net

Xt = None
for robust in (False, True):
    net = sw.SinWaveNet(n_slots=4, **cfg)
    t0 = time.time(); train(net, robust)
    Xt = net.encode([t for t, _ in te]); Yt = torch.tensor([y for _, y in te])
    name = f"誤差を入れて学習 (ずれ ±{z_train:.0%}, SNR {snr_train:g} dB)" if robust else "普通に学習"
    print(f"--- {name}（{time.time() - t0:.0f} 秒）")
    with torch.no_grad():
        acc = lambda **kw: sum((forward(net, Xt, g=torch.Generator().manual_seed(s), **kw).argmax(1) == Yt)
                               .float().mean().item() for s in (1, 2, 3)) / 3
        print("  誤差なし:", f"{acc():.3f}")
        print("  分散フィルタの遅延のずれ:", "  ".join(f"{v:+}: {acc(z_err=v):.3f}" for v in (-0.05, -0.03, -0.01, 0.01, 0.03, 0.05)))
        print("  各段の雑音 SNR [dB]:", "  ".join(f"{v}: {acc(snr_db=v):.3f}" for v in (30, 20, 15, 10)))
        print("  位相変調の揺らぎ σ:", "  ".join(f"{v}: {acc(phase_sigma=v):.3f}" for v in (0.1, 0.3)))
