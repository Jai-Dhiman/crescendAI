"""#168 label ceiling: separate latent piece difficulty from grading-board bias.

The eval target `ps` is (measurably) the rounded MEAN of whatever boards graded a
piece -- r=0.981, 77% exact -- and for 722 of the 900 eval pieces exactly ONE
board graded it. So the "consensus" label is usually one examination board's
opinion carrying that board's full idiosyncratic noise, and any board-vs-board
agreement statistic computed against it is partly circular (`ps_rating`
aggregates the very grades being predicted).

This module removes the circularity by fitting a Graded Response Model to the
raw per-board judgements in `related_entries`:

    P(grade_ij >= k) = sigmoid(a_j * (theta_i - b_jk))

theta_i is the latent difficulty of piece i; board j gets its OWN discrimination
a_j and its OWN ordered thresholds b_j1..b_j,K_j-1. Per-board thresholds are not
a refinement -- they are required, because the boards do not share a scale:
ANZCA only ever issues grades 5..8 (4 categories) while RCM spans 0..12 (13).
A board that CANNOT say "grade 2" looks like it disagrees with one that can, so
raw board-vs-board tau-c confounds range restriction with real disagreement.

The decisive output is a ceiling, not a fit. A perfect model outputs theta
exactly; it is still scored against a one-board `ps` label, so it cannot reach
tau-c 1.0. `simulate_ceiling` draws fresh grades from the fitted boards and
reports the tau-c such a perfect model would score. Comparing our measured
0.8430 against that number is what decides whether more accuracy work has value.

    cd model && uv run python -m claim_measurement.difficulty.label_ceiling \\
        --arm-dir data/results/phase1_lora/fold_embeddings_7665
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import optimize

# 'null' is a single malformed board entry in new_clean_data.json, not a board.
NON_BOARD = {"null", ""}
QUAD_NODES = 61
QUAD_LIMIT = 4.0
EPS = 1e-9


@dataclass(frozen=True)
class Judgement:
    piece: str
    board: str
    grade: int


def load_judgements(labels: dict) -> tuple[list[Judgement], dict]:
    """Pure: the PSyllabus label dict -> per-board judgements + a loss report.

    `related_entries` is a board->grade map on all but two entries, where it is
    a list of dicts instead; those are reported, never silently coerced.
    """
    out: list[Judgement] = []
    skipped_shape = 0
    skipped_board = 0
    skipped_grade = 0
    for key, meta in labels.items():
        related = meta.get("related_entries")
        if not isinstance(related, dict):
            skipped_shape += 1
            continue
        for board, grade in related.items():
            if str(board).strip() in NON_BOARD:
                skipped_board += 1
                continue
            if not isinstance(grade, int) or isinstance(grade, bool):
                skipped_grade += 1
                continue
            out.append(Judgement(piece=key, board=str(board), grade=grade))
    report = {"n_judgements": len(out), "n_pieces_seen": len(labels),
              "skipped_bad_shape": skipped_shape,
              "skipped_non_board": skipped_board,
              "skipped_bad_grade": skipped_grade}
    return out, report


def co_grading_graph(judgements: list[Judgement]) -> tuple[dict, list[set[str]]]:
    """Board-pair shared-piece counts, and the connected components of the graph
    they induce. MFRM/GRM severities are only comparable on one scale within a
    connected component -- two board clusters that never co-grade a piece cannot
    be linked, and the fit would silently report incomparable numbers.
    """
    by_piece: dict[str, list[str]] = defaultdict(list)
    for j in judgements:
        by_piece[j.piece].append(j.board)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for boards in by_piece.values():
        uniq = sorted(set(boards))
        for a_i in range(len(uniq)):
            for b_i in range(a_i + 1, len(uniq)):
                counts[(uniq[a_i], uniq[b_i])] += 1
    boards = sorted({j.board for j in judgements})
    adj: dict[str, set[str]] = {b: set() for b in boards}
    for (a_b, b_b) in counts:
        adj[a_b].add(b_b)
        adj[b_b].add(a_b)
    seen: set[str] = set()
    components: list[set[str]] = []
    for b in boards:
        if b in seen:
            continue
        stack, comp = [b], set()
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            stack.extend(adj[cur] - comp)
        seen |= comp
        components.append(comp)
    return dict(counts), components


def _tau_c(x, y) -> float:
    """The MIREX metric. NOT comparable across targets with different numbers of
    distinct values: tau-c divides by (m-1)/m where m is the smaller of the two
    variables' distinct-value counts, so the same predictions score ~11/10 higher
    against an 11-grade label than against a continuous latent trait. Use
    `_tau_b` whenever the two targets have different granularity.
    """
    from scipy import stats
    return float(stats.kendalltau(x, y, variant="c").statistic)


def _tau_b(x, y) -> float:
    """Granularity-independent rank correlation, for comparing a prediction
    against targets of different resolution (an 11-grade label vs continuous
    theta)."""
    from scipy import stats
    return float(stats.kendalltau(x, y, variant="b").statistic)


def board_pair_agreement(judgements: list[Judgement], min_n: int = 50) -> dict:
    """Raw board-vs-board tau-c on co-graded pieces -- the naive 'human ceiling'.

    Reported so the GRM's answer can be read against it, NOT as the ceiling: a
    rank correlation between two noisy observers is attenuated by BOTH observers'
    noise, and board scale-compression enters it as if it were disagreement.
    """
    by_board: dict[str, dict[str, int]] = defaultdict(dict)
    for j in judgements:
        by_board[j.board][j.piece] = j.grade
    boards = sorted(by_board)
    pairs = []
    for a_i in range(len(boards)):
        for b_i in range(a_i + 1, len(boards)):
            a_b, b_b = boards[a_i], boards[b_i]
            shared = sorted(set(by_board[a_b]) & set(by_board[b_b]))
            if len(shared) < min_n:
                continue
            xs = [by_board[a_b][p] for p in shared]
            ys = [by_board[b_b][p] for p in shared]
            pairs.append({"a": a_b, "b": b_b, "n": len(shared),
                          "tau_c": _tau_c(xs, ys)})
    if not pairs:
        return {"pairs": [], "n_weighted": None, "unweighted": None}
    taus = np.array([p["tau_c"] for p in pairs])
    ns = np.array([p["n"] for p in pairs], dtype=float)
    return {"pairs": sorted(pairs, key=lambda p: p["tau_c"]),
            "n_weighted": float((taus * ns).sum() / ns.sum()),
            "unweighted": float(taus.mean()),
            "median": float(np.median(taus))}


# --------------------------------------------------------------------------
# Graded Response Model, marginal-maximum-likelihood EM over a fixed quadrature
# --------------------------------------------------------------------------

@dataclass
class GrmFit:
    pieces: list[str]
    boards: list[str]
    theta: np.ndarray          # EAP latent difficulty, one per piece
    theta_sd: np.ndarray       # posterior SD, one per piece
    n_obs: np.ndarray          # judgements per piece
    a: dict[str, float]        # per-board discrimination
    b: dict[str, np.ndarray]   # per-board ordered thresholds (K_j - 1 of them)
    categories: dict[str, list[int]]   # per-board observed grades, ascending
    observed: list[tuple[str, str]]    # (piece, board) of every judgement fit
    loglik: float
    n_iter: int


def _unpack(params: np.ndarray) -> tuple[float, np.ndarray]:
    """(log a, first threshold, log gaps) -> (a, ascending thresholds).

    Thresholds are built as a cumulative sum of exp(gap) so ordering is
    structural; an unconstrained optimizer cannot produce a crossed set.
    """
    a = float(np.exp(np.clip(params[0], -4.0, 4.0)))
    first = params[1]
    gaps = np.exp(np.clip(params[2:], -6.0, 3.0))
    return a, np.concatenate([[first], first + np.cumsum(gaps)])


def _category_probs(a: float, b: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """P(X = k | theta) at each quadrature node; shape (K, n_nodes)."""
    z = a * (nodes[None, :] - b[:, None])
    ge = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))   # P(X >= k), k=1..K-1
    top = np.ones((1, nodes.size))
    bot = np.zeros((1, nodes.size))
    upper = np.concatenate([top, ge])       # P(X >= k)   for k = 0..K-1
    lower = np.concatenate([ge, bot])       # P(X >= k+1) for k = 0..K-1
    return np.clip(upper - lower, EPS, 1.0)


def _quadrature() -> tuple[np.ndarray, np.ndarray]:
    nodes = np.linspace(-QUAD_LIMIT, QUAD_LIMIT, QUAD_NODES)
    prior = np.exp(-0.5 * nodes ** 2)
    return nodes, prior / prior.sum()


def _marginal_loglik(flat: np.ndarray, layout: list[tuple[str, int, int]],
                     obs: dict, n_pieces: int, nodes: np.ndarray,
                     log_prior: np.ndarray) -> tuple[float, np.ndarray]:
    """log P(all judgements | board params), marginalizing theta over the
    quadrature, plus the per-piece posterior over nodes."""
    log_lik = np.tile(log_prior, (n_pieces, 1))
    for board, lo, hi in layout:
        a, b = _unpack(flat[lo:hi])
        rows, cats = obs[board]
        np.add.at(log_lik, rows, np.log(_category_probs(a, b, nodes))[cats])
    mx = log_lik.max(axis=1, keepdims=True)
    w = np.exp(log_lik - mx)
    marginal = w.sum(axis=1, keepdims=True)
    return float((np.log(marginal) + mx).sum()), w / marginal


def fit_grm(judgements: list[Judgement], max_iter: int = 3000,
            tol: float = 1e-8) -> GrmFit:
    """Marginal maximum likelihood for a per-board GRM: theta is integrated out
    over a fixed quadrature and all board parameters are optimized JOINTLY.

    Deliberately not EM. The likelihood surface has a long flat ridge -- a
    board's discrimination trades off against the spread of its thresholds
    almost losslessly -- and EM crawls along it: on synthetic data with known
    discrimination 2.5, EM stopped at 0.642 after 8 iterations having bought the
    last 1.85 nats out of 11,624, and had still not arrived after 2,000. A
    converged likelihood is not a converged fit. Joint L-BFGS moves along the
    ridge in one step instead of stepping across it.

    theta ~ N(0,1) fixes location and scale, so no further identifiability
    constraint on the board parameters is needed.
    """
    if not judgements:
        raise ValueError("no judgements to fit")
    pieces = sorted({j.piece for j in judgements})
    boards = sorted({j.board for j in judgements})
    p_idx = {p: i for i, p in enumerate(pieces)}
    categories = {b: sorted({j.grade for j in judgements if j.board == b})
                  for b in boards}
    for b in boards:
        if len(categories[b]) < 2:
            raise ValueError(f"board {b!r} has one grade category; it carries "
                             f"no ordinal information and must be excluded")

    nodes, prior = _quadrature()
    log_prior = np.log(prior)

    obs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for b in boards:
        cat_idx = {g: k for k, g in enumerate(categories[b])}
        rows = [(p_idx[j.piece], cat_idx[j.grade])
                for j in judgements if j.board == b]
        obs[b] = (np.array([r for r, _ in rows]), np.array([k for _, k in rows]))

    layout: list[tuple[str, int, int]] = []
    init: list[np.ndarray] = []
    cursor = 0
    for b in boards:
        width = len(categories[b])          # log a, first threshold, K-2 gaps
        layout.append((b, cursor, cursor + width))
        init.append(np.concatenate([[0.0], [-1.0],
                                    np.zeros(len(categories[b]) - 2)]))
        cursor += width
    x0 = np.concatenate(init)
    n_pieces = len(pieces)

    def nll(flat):
        return -_marginal_loglik(flat, layout, obs, n_pieces, nodes,
                                 log_prior)[0]

    res = optimize.minimize(nll, x0, method="L-BFGS-B",
                            options={"maxiter": max_iter, "ftol": tol,
                                     "gtol": tol})
    loglik, post = _marginal_loglik(res.x, layout, obs, n_pieces, nodes,
                                    log_prior)

    theta = post @ nodes
    theta_sd = np.sqrt(np.maximum(post @ (nodes ** 2) - theta ** 2, 0.0))
    n_obs = np.zeros(n_pieces)
    for b in boards:
        np.add.at(n_obs, obs[b][0], 1.0)

    unpacked = {b: _unpack(res.x[lo:hi]) for b, lo, hi in layout}
    return GrmFit(pieces=pieces, boards=boards, theta=theta, theta_sd=theta_sd,
                  n_obs=n_obs,
                  a={b: unpacked[b][0] for b in boards},
                  b={b: unpacked[b][1] for b in boards},
                  categories=categories,
                  observed=[(j.piece, j.board) for j in judgements],
                  loglik=loglik, n_iter=int(res.nit))


def posterior_predictive_agreement(fit: GrmFit, seed: int = 0) -> dict:
    """Regenerate every observed judgement from the fit and re-measure
    board-vs-board agreement. This is what licenses the ceiling: a fit whose
    boards agree with each other far MORE than the real ones do would imply a
    cleaner label than exists, and would overstate the ceiling.

    Deliberately conditions on theta's point estimate rather than drawing from
    the posterior, which biases the check TOWARD too much agreement -- so
    simulated agreement at or below observed is the conservative outcome.
    """
    rng = np.random.default_rng(seed)
    nodes, _ = _quadrature()
    idx = {p: i for i, p in enumerate(fit.pieces)}
    cum = {b: np.cumsum(_category_probs(fit.a[b], fit.b[b], nodes), axis=0)
           for b in fit.boards}
    node_of = np.abs(fit.theta[:, None] - nodes[None, :]).argmin(axis=1)
    simulated = []
    for piece, board in fit.observed:
        q = node_of[idx[piece]]
        k = min(int(np.searchsorted(cum[board][:, q], rng.random())),
                len(fit.categories[board]) - 1)
        simulated.append(Judgement(piece, board, fit.categories[board][k]))
    return board_pair_agreement(simulated)


def simulate_ceiling(fit: GrmFit, eval_pieces: list[str],
                     boards_per_piece: dict[str, list[str]],
                     n_draws: int = 200, seed: int = 2026) -> dict:
    """tau-c a PERFECT model (one that outputs theta exactly) would score against
    the observed-style label, given the label is a fresh draw from the boards
    that actually graded each eval piece.

    Circularity is broken by drawing the grades afresh from the fitted boards
    rather than reusing the grades that produced theta. The label is rebuilt the
    way `ps` is actually built -- the rounded mean of its boards' grades.
    """
    rng = np.random.default_rng(seed)
    nodes = np.linspace(-QUAD_LIMIT, QUAD_LIMIT, QUAD_NODES)
    idx = {p: i for i, p in enumerate(fit.pieces)}
    rows = [idx[p] for p in eval_pieces]
    theta_hat = fit.theta[rows]
    sd = fit.theta_sd[rows]

    cum = {}
    for b in fit.boards:
        probs = _category_probs(fit.a[b], fit.b[b], nodes)      # (K, n_nodes)
        cum[b] = np.cumsum(probs / probs.sum(axis=0, keepdims=True), axis=0)

    taus = []
    for _ in range(n_draws):
        theta_true = theta_hat + sd * rng.standard_normal(len(rows))
        node_of = np.abs(theta_true[:, None] - nodes[None, :]).argmin(axis=1)
        label = np.empty(len(rows))
        for i, piece in enumerate(eval_pieces):
            grades = []
            for b in boards_per_piece[piece]:
                u = rng.random()
                k = int(np.searchsorted(cum[b][:, node_of[i]], u))
                k = min(k, len(fit.categories[b]) - 1)
                grades.append(fit.categories[b][k])
            label[i] = np.round(np.mean(grades))
        taus.append(_tau_c(theta_true, label))
    taus = np.array(taus)
    return {"mean": float(taus.mean()), "sd": float(taus.std()),
            "lo": float(np.percentile(taus, 2.5)),
            "hi": float(np.percentile(taus, 97.5)), "n_draws": n_draws}


# --------------------------------------------------------------------------
# CLI: fit the GRM, then read our measured arm against the latent trait
# --------------------------------------------------------------------------

def _eval_join(labels: dict, manifest: list, seg_ids: list[str]):
    """seg_id -> PSyllabus key -> the boards that graded it."""
    seg_to_key = {m["seg_id"]: m["key"] for m in manifest}
    missing = [s for s in seg_ids if s not in seg_to_key]
    if missing:
        raise SystemExit(f"{len(missing)} eval seg_ids absent from the manifest, "
                         f"e.g. {missing[:3]} -- the join would silently drop rows")
    keys = [seg_to_key[s] for s in seg_ids]
    boards = {}
    for key in keys:
        related = labels.get(key, {}).get("related_entries")
        boards[key] = sorted(b for b in related
                             if str(b).strip() not in NON_BOARD) \
            if isinstance(related, dict) else []
    return keys, boards


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--arm-dir", type=Path, default=None,
                    help="dir of emb_fold{0..4}.npz to score against the latent "
                         "trait; omit to report the GRM fit alone")
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args(argv)

    from claim_measurement.difficulty.bakeoff_paths import (
        resolve_paths,
    )
    paths = resolve_paths(args.data_root)
    labels = json.loads(paths.labels.read_text())

    judgements, report = load_judgements(labels)
    print(f"judgements {report['n_judgements']} over "
          f"{len({j.piece for j in judgements})} pieces, "
          f"{len({j.board for j in judgements})} boards "
          f"(skipped: shape {report['skipped_bad_shape']}, "
          f"non-board {report['skipped_non_board']}, "
          f"grade {report['skipped_bad_grade']})")

    _counts, components = co_grading_graph(judgements)
    if len(components) > 1:
        raise SystemExit(f"board co-grading graph has {len(components)} "
                         f"components {components}; severities across "
                         f"components are not on one scale")
    n_multi = sum(1 for _, v in _by_piece(judgements).items() if len(v) >= 2)
    n_pairs = sum(len(v) * (len(v) - 1) // 2
                  for v in _by_piece(judgements).values())
    print(f"co-graded pieces {n_multi}, board-pair judgements {n_pairs}, "
          f"graph connected over {len(components[0])} boards")

    agree = board_pair_agreement(judgements)
    print(f"\nboard-vs-board tau-c (naive, attenuated by BOTH boards' noise): "
          f"n-weighted {agree['n_weighted']:.4f}, "
          f"unweighted {agree['unweighted']:.4f}, "
          f"median {agree['median']:.4f}, {len(agree['pairs'])} pairs n>=50")
    for p in agree["pairs"][:3] + agree["pairs"][-3:]:
        print(f"    {p['a']:>10} vs {p['b']:<10} n={p['n']:<5} "
              f"tau-c {p['tau_c']:.4f}")

    fit = fit_grm(judgements)
    print(f"\nGRM fit: loglik {fit.loglik:.1f}, {fit.n_iter} L-BFGS iterations")
    print(f"{'board':<12} {'n_cat':>5} {'discrim':>8} {'severity':>9}  range")
    for b in sorted(fit.boards, key=lambda x: -fit.a[x]):
        print(f"{b:<12} {len(fit.categories[b]):>5} {fit.a[b]:>8.3f} "
              f"{float(np.mean(fit.b[b])):>9.3f}  "
              f"[{fit.b[b][0]:+.2f},{fit.b[b][-1]:+.2f}]")

    ppc = posterior_predictive_agreement(fit)
    verdict = ("CONSERVATIVE" if ppc["n_weighted"] <= agree["n_weighted"]
               else "OVERSTATES AGREEMENT -- the ceiling is inflated")
    print(f"\nposterior predictive: boards regenerated from the fit agree at "
          f"tau-c {ppc['n_weighted']:.4f} vs {agree['n_weighted']:.4f} "
          f"observed -- {verdict}")

    out = {"report": report, "board_agreement": agree,
           "posterior_predictive": ppc,
           "grm": {"loglik": fit.loglik, "n_iter": fit.n_iter,
                   "a": fit.a, "b": {k: v.tolist() for k, v in fit.b.items()}}}

    if args.arm_dir is not None:
        from claim_measurement.difficulty.adapter_ensemble import (
            prediction_mean_oof,
            single_adapter_oofs,
        )
        from claim_measurement.difficulty.ft_eval import (
            N_FOLDS,
            SEED,
            _load_features37,
            oof_tau_per_fold,
        )
        from claim_measurement.difficulty.window_sweep import load_arm

        _, y, composers, seg_ids = _load_features37(paths.emb_root)
        manifest = json.loads(paths.manifest.read_text())
        keys, boards_per_piece = _eval_join(labels, manifest, seg_ids)

        idx = {p: i for i, p in enumerate(fit.pieces)}
        theta = np.array([fit.theta[idx[k]] for k in keys])
        n_boards = np.array([len(boards_per_piece[k]) for k in keys])

        arm = load_arm(Path(args.arm_dir), seg_ids, N_FOLDS)
        diag = oof_tau_per_fold(arm, y, composers, N_FOLDS, SEED)
        ens = prediction_mean_oof(
            single_adapter_oofs(arm, y, composers, N_FOLDS, SEED))

        ceiling = simulate_ceiling(fit, keys, boards_per_piece,
                                   n_draws=args.n_draws, seed=args.seed)
        print(f"\neval n={len(y)}; single-board {int((n_boards == 1).sum())}, "
              f"multi-board {int((n_boards >= 2).sum())}")
        print(f"{'model (diagonal)  vs ps label':<38} tau-c {_tau_c(diag, y):.4f}")
        print(f"{'model (pred-mean) vs ps label':<38} tau-c {_tau_c(ens, y):.4f}")
        # tau-B for anything scored against theta: tau-c is not comparable
        # across targets of different granularity.
        print(f"{'model (diagonal)  vs latent theta':<38} tau-b "
              f"{_tau_b(diag, theta):.4f}")
        print(f"{'ps label          vs latent theta':<38} tau-b "
              f"{_tau_b(y, theta):.4f}   (upward biased: theta is fit to the "
              f"grades ps averages)")
        print(f"{'PERFECT model     vs ps label':<38} tau-c "
              f"{ceiling['mean']:.4f} CI95[{ceiling['lo']:.4f},"
              f"{ceiling['hi']:.4f}]  <-- the ceiling")
        headroom = ceiling["mean"] - _tau_c(diag, y)
        print(f"\nheadroom against the label: {headroom:+.4f} "
              f"(ceiling {ceiling['mean']:.4f} - measured "
              f"{_tau_c(diag, y):.4f})")
        out["eval"] = {"n": len(y), "n_single_board": int((n_boards == 1).sum()),
                       "diag_vs_label": _tau_c(diag, y),
                       "ens_vs_label": _tau_c(ens, y),
                       "diag_vs_theta_taub": _tau_b(diag, theta),
                       "label_vs_theta_taub": _tau_b(y, theta),
                       "ceiling": ceiling, "headroom": headroom}

    if args.out_json:
        args.out_json.write_text(json.dumps(out, indent=1, default=float))
        print(f"\nwrote {args.out_json}")
    return 0


def _by_piece(judgements: list[Judgement]) -> dict[str, list[str]]:
    by: dict[str, list[str]] = defaultdict(list)
    for j in judgements:
        by[j.piece].append(j.board)
    return by


if __name__ == "__main__":
    raise SystemExit(main())
