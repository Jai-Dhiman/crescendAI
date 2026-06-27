"""Front-4 (#101) pedaling G-B check ON THE AMT SUBSTRATE: does AMT-transcribed
sustain on-fraction (what the verifier actually reads) track perceived pedaling the way
the MIDI-native pedal_frac did (GATE-2 partial-rho 0.478)?

Answer (n=180 natural renders): NO. AMT halo-controlled partial-rho is ~0.181 [boot 95%
CI 0.03-0.33], vs ~0.386 for true MIDI on-fraction on the same clips. The AMT pedal head
recovers true on-fraction at only spearman ~0.39 (regression-to-the-middle: hallucinates
pedal on dry clips, saturates on wet) -- whereas the AMT velocity head recovers GT at 0.965
on the same renders. So pedaling is NOT perceptually validated on AMT; front-3's
"0.478 inherited" was MIDI-native and does not transfer. No LLM in any label.

Requires model/data/results/tau_cal_pedaling.json (tau_pedaling_render.py output).
Run:  uv run python gb_pedaling_amt_check.py
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata

REPO = Path(__file__).resolve().parents[4]
ALL = ["timing", "dynamics", "pedaling", "articulation", "phrasing", "interpretation"]


def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    D = np.vstack([rz, np.ones_like(rz)]).T
    bx, *_ = np.linalg.lstsq(D, rx, rcond=None)
    by, *_ = np.linalg.lstsq(D, ry, rcond=None)
    return float(np.corrcoef(rx - D @ bx, ry - D @ by)[0, 1])


def main() -> int:
    d = json.loads((REPO / "model/data/results/tau_cal_pedaling.json").read_text())
    comp = json.loads((REPO / "model/data/labels/composite/composite_labels.json").read_text())
    stems = [s for s in d if s in comp]
    amt = np.array([d[s]["amt_on_fraction"] for s in stems])
    true = np.array([d[s]["true_midi_frac"] for s in stems])
    ped = np.array([comp[s]["pedaling"] for s in stems])
    ctrl = np.array([np.mean([comp[s][k] for k in ALL if k != "pedaling"]) for s in stems])
    n = len(stems)
    print(f"=== pedaling G-B on AMT substrate (n={n}) ===")
    print(f"AMT  on-frac vs perceived: raw={spearmanr(amt, ped)[0]:+.3f} partial(halo)={partial(amt, ped, ctrl):+.3f}")
    print(f"TRUE midi    vs perceived: raw={spearmanr(true, ped)[0]:+.3f} partial(halo)={partial(true, ped, ctrl):+.3f}  (GATE-2 MIDI was +0.478)")
    rng = np.random.default_rng(7)
    boots = []
    for _ in range(2000):
        s = rng.integers(0, n, n)
        try:
            boots.append(partial(amt[s], ped[s], ctrl[s]))
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"AMT partial 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"AMT recovery of TRUE on-fraction: spearman={spearmanr(amt, true)[0]:+.3f}  "
          f"(contrast: AMT velocity head recovers GT at 0.965)")
    print("AMT regression-to-the-middle bias by TRUE bin:")
    for a, b in [(0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)]:
        m = (true >= a) & (true < b)
        if m.sum():
            print(f"  TRUE[{a:.1f},{b:.1f}) n={int(m.sum()):3d}  AMT mean={amt[m].mean():.3f} bias={amt[m].mean()-true[m].mean():+.3f}")
    bar = (hi - lo) / 2
    verdict = "PASS" if (partial(amt, ped, ctrl) - bar) >= 0.5 else "FAIL"
    print(f"\nG-B (AMT substrate) for pedaling: {verdict} vs ~0.5 ceiling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
