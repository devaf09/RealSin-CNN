"""実験A04: Sin-CNN2 1.3.0（既定: 文字の光がレイヤ全体を照らす wave 入力 + 虹、L=1024）を
ハードで作るときに、どこまで計算を簡略化できるか（推論のみ。CPU のみ）。

調べる簡略化:
  - 数値の形式: float16 / bfloat16 / 固定小数点 N bit（光の場の実部・虚部を各段の後で丸める）
  - 位相の表（sin・exp を表引きにする）の段数
  - 学習した値 A, φ の段数
  - 「明るさを 1 に戻す」処理を外す
  - 回折を短い畳み込み（K タップ）にする。板の間隔 z=64（既定）と z=2（実験A03）
sinwave_minimal.py を import するだけ（書き換えない）。
使い方: python3 exp_a04_simplify.py /path/to/Sin-CNN2 [z] [steps]
"""
import sys, math, random, time
import torch
sys.path.insert(0, sys.argv[1])
import sinwave_minimal as sw
torch.set_num_threads(2)
z = float(sys.argv[2]) if len(sys.argv) > 2 else 64.0
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 600

keys = {0: "あいうえ", 1: "かきくけ", 2: "さしすせ", 3: "たちつて"}
pool = "なにぬねのはひふへほまみむめもやゆよらりるれろわを"
def sample(r, y):
    return "".join(r.choice(keys[y]) if r.random() < 0.6 else r.choice(pool) for _ in range(r.randint(4, 8)))
r = random.Random(0)
tr = [(sample(r, y), y) for y in keys for _ in range(300)]
te = [(sample(r, y), y) for y in keys for _ in range(250)]

net = sw.SinWaveNet(n_slots=4, size="small")          # 板 16 枚 × 区画 64、L=1024、wave + 虹（T=8）
L = net.L
kx = 2 * math.pi * torch.fft.fftfreq(L)
H = torch.polar(torch.ones(L), torch.sqrt((2 * math.pi) ** 2 - kx ** 2) * z)
net.H = H
t0 = time.time()
sw.train(net, [t for t, _ in tr], [y for _, y in tr], steps=steps, batch_size=128)
print(f"z={z:g}、{sw.summary(net)}、学習 {steps} 回 {time.time() - t0:.0f} 秒", flush=True)
X = net.encode([t for t, _ in te]); Y = torch.tensor([y for _, y in te])

def q(u, fmt):
    """複素数の場を指定の形式に丸める（実部・虚部それぞれ）。"""
    if fmt == "fp32":
        return u
    if fmt in ("fp16", "bf16"):
        dt = torch.float16 if fmt == "fp16" else torch.bfloat16
        return torch.complex(u.real.to(dt).float(), u.imag.to(dt).float())
    bits = int(fmt[3:])                                   # fixN: ±R を N bit で表す
    R = 4.0                                               # 平均の明るさ 1 なので、振幅 4 までで十分（はみ出しは切る）
    s = 2 * R / 2 ** bits
    f = lambda x: (torch.round(x.clamp(-R, R - s) / s) * s)
    return torch.complex(f(u.real), f(u.imag))

def taps_H(K):
    """回折のフィルタを、中心から ±K タップの畳み込みに切り詰めたもの。"""
    h = torch.fft.ifft(H)
    keep = torch.zeros(L, dtype=torch.bool)
    keep[:K + 1] = True; keep[L - K:] = True
    return torch.fft.fft(torch.where(keep, h, torch.zeros_like(h)))

@torch.no_grad()
def run(fmt="fp32", phase_bits=0, param_bits=0, norm=True, K=None):
    A, phi = net.A.clone(), net.phi.clone()
    if param_bits:
        n = 2 ** param_bits
        A = torch.round(A.clamp(0, 2) / 2 * (n - 1)) / (n - 1) * 2
        phi = torch.round((phi % (2 * math.pi)) / (2 * math.pi) * n) / n * 2 * math.pi
    Hk = H if K is None else taps_H(K)
    u = q(X, fmt)
    for d in range(A.shape[0]):
        p = net.patch
        phase = A[d, p] * torch.sin(net.w[d, p] * net.t + phi[d, p])
        if phase_bits:
            n = 2 ** phase_bits
            phase = torch.round(phase / (2 * math.pi) * n) / n * 2 * math.pi
        u = q(u * q(torch.polar(torch.ones_like(phase), phase), fmt), fmt)
        u = q(torch.fft.ifft(torch.fft.fft(u) * Hk), fmt)
        if norm:
            u = q(u / (u.abs().pow(2).mean(-1, keepdim=True) + 1e-24).sqrt(), fmt)
    u = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)
    inten = u.abs().pow(2).mean(1)
    inten = inten / (inten.mean(-1, keepdim=True) + 1e-12)
    sc = net.gain * (inten @ net.det.T) + net.bias
    return (sc.argmax(1) == Y).float().mean().item()

with torch.no_grad():
    print(f"そのまま: {(net(X).argmax(1) == Y).float().mean().item():.3f}  再実装: {run():.3f}  正規化なし: {run(norm=False):.3f}", flush=True)
print("数値の形式:", "  ".join(f"{f}: {run(f):.3f}" for f in ("fp16", "bf16", "fix16", "fix12", "fix10", "fix8", "fix6")), flush=True)
print("正規化なし + 数値の形式:", "  ".join(f"{f}: {run(f, norm=False):.3f}" for f in ("fp16", "fix12", "fix10", "fix8")), flush=True)
print("位相の表の段数 [bit]:", "  ".join(f"{b}: {run(phase_bits=b):.3f}" for b in (10, 8, 6, 5, 4)), flush=True)
print("A, φ の段数 [bit]:", "  ".join(f"{b}: {run(param_bits=b):.3f}" for b in (8, 6, 4, 3)), flush=True)
hk = torch.fft.ifft(H).abs() ** 2
print("回折の短い畳み込み（タップ ±K）:", "  ".join(f"{K}: {run(K=K):.3f}" for K in (1, 2, 4, 8, 16, 32, 64)), flush=True)
print("まとめて (fix10, 位相表 6bit, A φ 4bit, 正規化なし):", f"{run('fix10', 6, 4, False):.3f}",
      " (fix8, 6, 4, なし):", f"{run('fix8', 6, 4, False):.3f}", " (fp16, 8, 4, なし):", f"{run('fp16', 8, 4, False):.3f}", flush=True)
