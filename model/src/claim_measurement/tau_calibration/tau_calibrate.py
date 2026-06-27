"""Front-4 tau calibration (#101): at what |d| does the signed measurement best detect
HUMAN-perceived anomalies (composite-rating tails) without false-alarming on normal
performances? Youden's-J-optimal tau + bootstrap CI + per-direction tau (which exposes
the flip+ < flip- asymmetry). No LLM in any label.

  mode=dynamics : d = amt_vel - 51.5,            label = composite dynamics (gb_amt_velocity_gate.json)
  mode=pedaling : d = amt_on_fraction - 0.4623,  label = composite pedaling (tau_cal_pedaling.json)

Signed-anomaly framing: a clip is a + anomaly if label>Q_hi, - anomaly if label<Q_lo,
else normal. The verifier fires +1 if d>tau, -1 if d<-tau, else 0. A detection is correct
only if it fires with the RIGHT sign on a tail clip; a false alarm is any firing on a
normal clip. tau* = argmax (TPR - FPR).

Caveat (honest): the composite label is a perceptual QUALITY rating, only a noisy proxy
for the signed level/density the statistic measures (rho ~0.5), so the combined symmetric
tau is weakly determined (wide CI); the DIRECTIONAL medians + per-direction tau are the
robust signal. Cross-check the locked value against the G-A flip sweep.

Run:  uv run python tau_calibrate.py {dynamics|pedaling} [QLO QHI]
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[4]
MODE = sys.argv[1] if len(sys.argv) > 1 else "dynamics"
QLO = int(sys.argv[2]) if len(sys.argv) > 2 else 25
QHI = int(sys.argv[3]) if len(sys.argv) > 3 else 75

if MODE == "dynamics":
    rows = json.loads((REPO / "model/data/results/gb_amt_velocity_gate.json").read_text())["rows"]
    d = np.array([r["amt_vel"] - 51.5 for r in rows])
    label = np.array([r["dyn"] for r in rows])
    unit, cur = "midi_velocity", "8.0"
elif MODE == "pedaling":
    data = json.loads((REPO / "model/data/results/tau_cal_pedaling.json").read_text())
    d = np.array([v["amt_on_fraction"] - 0.4623 for v in data.values()])
    label = np.array([v["composite_pedaling"] for v in data.values()])
    unit, cur = "fraction", "0.25"
else:
    raise SystemExit(f"unknown mode {MODE}")

n = len(d)
lo, hi = np.percentile(label, [QLO, QHI])
pos, neg = label > hi, label < lo
normal = ~pos & ~neg
print(f"=== {MODE} tau calibration (n={n}, unit={unit}, tails Q{QLO}/Q{QHI}) ===")
print(f"label tails: Q{QLO}={lo:.3f} Q{QHI}={hi:.3f}  | +anomaly={pos.sum()} -anomaly={neg.sum()} normal={normal.sum()}")
print(f"d range [{d.min():+.3f}, {d.max():+.3f}]  median +anom={np.median(d[pos]):+.3f} -anom={np.median(d[neg]):+.3f} normal={np.median(d[normal]):+.3f}")

taus = np.linspace(0.0, float(np.abs(d).max()), 400)


def youden(dvec, taus, pos_m, neg_m, norm_m):
    n_anom, n_norm = pos_m.sum() + neg_m.sum(), norm_m.sum()
    best_t, best_j = 0.0, -1.0
    for t in taus:
        tp = int(np.sum((dvec > t) & pos_m) + np.sum((dvec < -t) & neg_m))
        fp = int(np.sum((np.abs(dvec) > t) & norm_m))
        j = (tp / n_anom if n_anom else 0.0) - (fp / n_norm if n_norm else 0.0)
        if j > best_j:
            best_j, best_t = j, t
    return best_t, best_j


tau_star, j_star = youden(d, taus, pos, neg, normal)
tp = int(np.sum((d > tau_star) & pos) + np.sum((d < -tau_star) & neg))
fp = int(np.sum((np.abs(d) > tau_star) & normal))
print(f"\n[Youden tau*] = {tau_star:.3f} {unit}   J={j_star:.3f}  "
      f"(TPR={tp/(pos.sum()+neg.sum()):.2f} FPR={fp/max(normal.sum(),1):.2f})")


def dir_tau(dvec, anom_m, norm_m, sign):
    best_t, best_j = 0.0, -1.0
    for t in taus:
        tpr = int(np.sum((sign * dvec > t) & anom_m)) / max(anom_m.sum(), 1)
        fpr = int(np.sum((sign * dvec > t) & norm_m)) / max(norm_m.sum(), 1)
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, t
    return best_t, best_j


tp_pos, j_pos = dir_tau(d, pos, normal, +1)
tp_neg, j_neg = dir_tau(d, neg, normal, -1)
print(f"[per-direction] '+' tau={tp_pos:.3f} (J={j_pos:.2f})   '-' tau={tp_neg:.3f} (J={j_neg:.2f})")

rng = np.random.default_rng(7)
boots = []
for _ in range(2000):
    s = rng.integers(0, n, n)
    ds, ls = d[s], label[s]
    lo_b, hi_b = np.percentile(ls, [QLO, QHI])
    t, _ = youden(ds, taus, ls > hi_b, ls < lo_b, (ls <= hi_b) & (ls >= lo_b))
    boots.append(t)
clo, chi = np.percentile(boots, [2.5, 97.5])
print(f"[bootstrap 95% CI] tau* in [{clo:.3f}, {chi:.3f}]  half-width={(chi-clo)/2:.3f}  median={np.median(boots):.3f}")
print(f"\nCURRENT taxonomy tau: {cur} -> Youden point estimate {tau_star:.2f} {unit} (cross-check vs G-A flip before locking)")
