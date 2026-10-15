"""実験A03: 回折（分散フィルタ）を、アナログの 2 次全域通過フィルタ N 段で近似しても学習できるか。

搬送波に乗せた実数信号を想定する。標本速度 fs = 1、搬送波 fc = 1 とし、信号は fc±0.5 の帯域にある。
2 次全域通過フィルタ（振幅は変えず位相だけ回すフィルタ）1 段の位相: -2·atan2(w·w0/Q, w0² − w²)
N 段の (w0, Q) を、Sin-CNN2 の回折の位相に合うよう勾配法で合わせる。
一定の位相と傾き（= 単なる遅れ）は答えに影響しない（遅れは次の層の変調信号をずらして合わせる想定）ので除いて比べる。
その近似フィルタを回折の代わりに入れて学習し直し（実機の特性で学習する = 較正 を想定）、正解率を測る。
sinwave_minimal.py を import するだけ。CPU のみ。
使い方: python3 exp_a03_allpass.py /path/to/Sin-CNN2 z:N[:層数] [...]
    z = 板の間隔（Sin-CNN2 の既定は 64）、N = 全域通過フィルタの段数（0 は回折なし、-1 は理想の回折）
必要な段数の目安: 周波数による遅れの差（±z·π/kz(π) 標本）× 帯域 で決まり、z=64 では約 40 段。
"""
import sys, math, random, time
import torch
sys.path.insert(0, sys.argv[1])
import sinwave_minimal as sw
torch.set_num_threads(2)

keys = {0: "あいうえ", 1: "かきくけ", 2: "さしすせ", 3: "たちつて"}
pool = "なにぬねのはひふへほまみむめもやゆよらりるれろわを"
def sample(r, y):
    return "".join(r.choice(keys[y]) if r.random() < 0.6 else r.choice(pool) for _ in range(r.randint(4, 8)))
r = random.Random(0)
tr = [(sample(r, y), y) for y in keys for _ in range(300)]
te = [(sample(r, y), y) for y in keys for _ in range(250)]
cfg = dict(L=512, S=64, layers=8, patches=32)
L = cfg["L"]

k = 2 * math.pi * torch.fft.fftfreq(L, dtype=torch.float64)       # 基底帯域の周波数（rad/標本）
def target_of(z):
    return torch.sqrt((2 * math.pi) ** 2 - k ** 2) * z            # 理想の回折の位相
w = 2 * math.pi * 1.0 + k                                         # 搬送波に乗せたときの角周波数

def detrend(ph):
    """一定の位相と傾き（遅れ）を最小二乗で除く。"""
    Amat = torch.stack([torch.ones_like(k), k], 1)
    c = torch.linalg.lstsq(Amat, ph[:, None]).solution
    return ph - (Amat @ c)[:, 0]

def allpass_phase(lw0, lq, w):
    w0, Q = torch.exp(lw0)[:, None], torch.exp(lq)[:, None]
    return (-2 * torch.atan2(w * w0 / Q, w0 ** 2 - w ** 2)).sum(0)

def fit(N, z, iters=3000):
    """群遅延（周波数ごとの遅れ）の形に合わせて各段の中心を置いてから、位相の誤差を勾配法で小さくする。"""
    target = target_of(z)
    ks, order = torch.sort(k)
    tau = -torch.gradient(target[order], spacing=(ks,))[0]        # 群遅延（標本）
    need = tau - tau.min() + 1e-3                                 # 遅れは正でしかありえないので底上げ
    cdf = torch.cumsum(need, 0); cdf = cdf / cdf[-1]
    best = None
    for rs in range(3):
        q = (torch.arange(N, dtype=torch.float64) + 0.5) / N
        idx = torch.searchsorted(cdf, q).clamp(max=L - 1)
        w0i = 2 * math.pi + ks[idx]
        spacing = (2 * math.pi) / max(N, 1) * (0.7 + 0.3 * rs)
        lw0 = torch.log(w0i).clone().requires_grad_()
        lq = torch.log((w0i / spacing).clamp(min=0.6)).clone().requires_grad_()
        opt = torch.optim.Adam([lw0, lq], lr=0.02)
        for _ in range(iters):
            e = detrend(allpass_phase(lw0, lq, w)) - detrend(target)
            loss = e.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if best is None or loss.item() < best[0]:
            best = (loss.item(), lw0.detach(), lq.detach())
    ph = detrend(allpass_phase(best[1], best[2], w))
    return ph, best[0] ** 0.5, torch.exp(best[2])

for arg in sys.argv[2:]:
    parts = arg.split(":")
    z, N = float(parts[0]), int(parts[1])
    layers = int(parts[2]) if len(parts) > 2 else cfg["layers"]         # 層の数（省略時 8）
    dyn = detrend(target_of(z))
    head = f"z={z:g}（位相の振れ RMS {dyn.pow(2).mean().sqrt():.2f} rad）"
    if N == -1:
        H, desc = torch.polar(torch.ones(L), target_of(z).float()), f"{head} 理想の回折"
    elif N == 0:
        H, desc = torch.ones(L, dtype=torch.complex64), "回折なし（H=1）"
    else:
        ph, err, Q = fit(N, z)
        H = torch.polar(torch.ones(L), ph.float())
        desc = f"{head} 全域通過 {N} 段（位相の誤差 RMS {err:.2f} rad、Q の最大 {Q.max():.1f}）"
    accs = []
    for seed in (0, 1):
        net = sw.SinWaveNet(n_slots=4, seed=seed, **dict(cfg, layers=layers))
        net.H = H.clone()
        t0 = time.time()
        sw.train(net, [t for t, _ in tr], [y for _, y in tr], seed=seed)
        accs.append(sum(p == y for p, (_, y) in zip(sw.predict(net, [t for t, _ in te]), te)) / len(te))
    print(f"{desc}、{layers} 層: 正解率 {accs[0]:.3f} / {accs[1]:.3f}（seed 0 / 1、1 回 {time.time() - t0:.0f} 秒）", flush=True)
