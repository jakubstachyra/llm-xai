#!/usr/bin/env python
"""
Analiza wyników RQ1 (stabilność wyjaśnień) – kontrola jakości, rekonstrukcja czasu
obliczeń, statystyki, testy istotności i wykresy do pracy.

Użycie:
    python scripts/analyze_rq1_results.py
    python scripts/analyze_rq1_results.py --input-dir results/rq1_stability_Qwen2.5-7B \
        --compare-dir results/rq1_stability --compare-label "Phi-2" --label "Qwen2.5-7B"

Wyniki trafiają do <input-dir>/analysis/:
    rq1_report.md                         – raport z wnioskami (generowany automatycznie)
    session_reconstruction.csv            – odtworzone sesje obliczeniowe (który zakres próbek, ile czasu)
    hardware_session_log_reconstructed.csv – log sprzętowy z doszacowaną brakującą częścią
    table_*.csv / table_*.tex             – tabele
    plots/*.png|pdf                       – wykresy
"""

import os
import sys
import argparse
import json
import warnings
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplcache"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

METRICS = ["spearman", "kendall", "pearson", "jaccard_k1", "jaccard_k3", "jaccard_k5"]
METRIC_LABELS = {
    "spearman": "Spearman ρ",
    "kendall": "Kendall τ",
    "pearson": "Pearson r",
    "jaccard_k1": "Jaccard@1",
    "jaccard_k3": "Jaccard@3",
    "jaccard_k5": "Jaccard@5",
}
CONTROL_CATEGORY = "control"
NOISE_FLOOR_NAME = "reseed_same_text"
EXPLAINER_ORDER = ["Occlusion", "GradientXInput", "IntegratedGradients", "LIME", "SHAP"]
CATEGORY_ORDER = ["surface", "lexical", "prefix", "suffix"]
COLORS = {
    "Occlusion": "#1f77b4",
    "GradientXInput": "#ff7f0e",
    "IntegratedGradients": "#2ca02c",
    "LIME": "#d62728",
    "SHAP": "#9467bd",
}


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------
def ordered(values, order):
    vals = list(dict.fromkeys(values))
    return [v for v in order if v in vals] + sorted(v for v in vals if v not in order)


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def save_fig(fig, plots_dir: str, name: str):
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f"{name}.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(plots_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def fmt_s(seconds: float) -> str:
    if seconds is None or np.isnan(seconds):
        return "n/a"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m:02d}m {s:04.1f}s" if h else f"{m}m {s:04.1f}s"


def df_to_md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# 1. Loading & quality control
# ----------------------------------------------------------------------------------------
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["sample_id", "explainer_name", "perturbation_name"])
    df["is_control"] = df["perturbation_category"].eq(CONTROL_CATEGORY)
    df["is_identity"] = (df["original_text"] == df["perturbed_text"]) & ~df["is_control"]
    return df


def quality_control(df: pd.DataFrame) -> Dict:
    qc: Dict = {}
    samples = df.drop_duplicates("sample_id")
    qc["n_rows"] = len(df)
    qc["n_samples"] = int(samples["sample_id"].nunique())
    ids = np.sort(samples["sample_id"].unique())
    qc["sample_id_range"] = (int(ids.min()), int(ids.max()))
    qc["missing_ids"] = sorted(set(range(ids.min(), ids.max() + 1)) - set(ids.tolist()))
    qc["explainers"] = ordered(df["explainer_name"].unique(), EXPLAINER_ORDER)
    qc["devices"] = df["device_name"].unique().tolist() if "device_name" in df else ["?"]
    qc["nan_metrics"] = {m: int(df[m].isna().sum()) for m in METRICS if m in df}
    qc["out_of_range"] = {
        m: int(((df[m] < -1.0001) | (df[m] > 1.0001)).sum()) for m in METRICS if m in df
    }

    # Classifier behaviour on the original sentences
    qc["accuracy"] = float((samples["orig_pred"] == samples["true_label"]).mean())
    qc["share_pred_positive"] = float(samples["orig_pred"].mean())
    qc["share_true_positive"] = float(samples["true_label"].mean())
    qc["confusion"] = pd.crosstab(samples["true_label"], samples["orig_pred"]).to_dict()
    qc["mean_p_orig"] = float(samples["p_orig"].mean())
    tp = ((samples.true_label == 1) & (samples.orig_pred == 1)).sum()
    tn = ((samples.true_label == 0) & (samples.orig_pred == 0)).sum()
    qc["recall_pos"] = float(tp / max(1, (samples.true_label == 1).sum()))
    qc["recall_neg"] = float(tn / max(1, (samples.true_label == 0).sum()))
    qc["balanced_accuracy"] = (qc["recall_pos"] + qc["recall_neg"]) / 2

    # Identity pairs (perturbation did not change the text)
    ident = df[df["is_identity"]]
    qc["identity_rows"] = len(ident)
    qc["identity_by_pert"] = ident.groupby("perturbation_name").size().to_dict()
    qc["fully_identity_perts"] = [
        p for p, g in df[~df["is_control"]].groupby("perturbation_name") if g["is_identity"].all()
    ]
    # Determinism sanity check: deterministic methods on identical text must give rho = 1
    det = ident[~ident["explainer_name"].isin(["LIME", "SHAP"])]
    qc["identity_det_min_spearman"] = (
        det.groupby("explainer_name")["spearman"].min().to_dict() if len(det) else {}
    )
    return qc


def decision_flip_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Rows exist only for decision-preserved pairs -> missing (sample, pert) == decision flip."""
    pert = df[~df["is_control"]]
    ref = pert["explainer_name"].value_counts().idxmax()
    ref_df = pert[pert["explainer_name"] == ref]
    n_samples = df["sample_id"].nunique()
    cat = ref_df.drop_duplicates("perturbation_name").set_index("perturbation_name")["perturbation_category"]
    kept = ref_df.groupby("perturbation_name")["sample_id"].nunique()
    out = pd.DataFrame({
        "perturbation_category": cat,
        "n_preserved": kept,
        "flip_rate": 1 - kept / n_samples,
    }).reset_index().rename(columns={"index": "perturbation_name"})
    return out.sort_values(["perturbation_category", "flip_rate"], ascending=[True, False])


# ----------------------------------------------------------------------------------------
# 2. Reconstruction of compute sessions (time got overwritten between batches)
# ----------------------------------------------------------------------------------------
def per_sample_cost(df: pd.DataFrame) -> pd.DataFrame:
    origs = df.drop_duplicates(["sample_id", "explainer_name"])
    o = origs.groupby("sample_id")[["orig_forward_passes", "orig_backward_passes", "orig_time_s"]].sum()
    p = df.groupby("sample_id")[["pert_forward_passes", "pert_backward_passes", "pert_time_s"]].sum()
    n_pert_names = df.loc[~df["is_control"], "perturbation_name"].nunique()
    per_expl = df[~df["is_control"]].groupby(["sample_id", "explainer_name"]).size().groupby("sample_id").max()
    cost = o.join(p)
    cost["n_perts_total"] = n_pert_names
    cost["has_flip"] = per_expl.reindex(cost.index).fillna(0) < n_pert_names
    # 1 prediction of original + 1 per perturbation (decision-preservation check)
    cost["fwd_new"] = cost["orig_forward_passes"] + cost["pert_forward_passes"] + 1 + n_pert_names
    cost["bwd_new"] = cost["orig_backward_passes"] + cost["pert_backward_passes"]
    cost["time_new_s"] = cost["orig_time_s"] + cost["pert_time_s"]
    # Cost of re-visiting a finished sample on resume (bug: samples with flipped pairs never
    # count as "done", so every resumed session re-predicts + re-explains their originals)
    cost["fwd_revisit"] = np.where(cost["has_flip"], cost["orig_forward_passes"] + 1 + n_pert_names, 0)
    cost["bwd_revisit"] = np.where(cost["has_flip"], cost["orig_backward_passes"], 0)
    cost["time_revisit_s"] = np.where(cost["has_flip"], cost["orig_time_s"], 0.0)
    return cost.sort_index()


def load_hw_sessions(input_dir: str) -> pd.DataFrame:
    """Final snapshot of every session found in hardware_timeline.csv / session logs."""
    frames = []
    for name in ["hardware_timeline.csv", "hardware_session_log.csv"]:
        p = os.path.join(input_dir, name)
        if os.path.exists(p):
            t = pd.read_csv(p)
            t["source"] = name
            frames.append(t)
    snap = os.path.join(input_dir, "hardware_live_snapshot.json")
    if os.path.exists(snap):
        with open(snap, encoding="utf-8") as f:
            t = pd.DataFrame([json.load(f)])
        t["source"] = "hardware_live_snapshot.json"
        frames.append(t)
    if not frames:
        return pd.DataFrame()
    hw = pd.concat(frames, ignore_index=True)
    hw = hw.sort_values(["timestamp", "duration_s"]).drop_duplicates(
        subset=["forward_passes", "backward_passes"], keep="last")
    hw = hw.sort_values("timestamp").reset_index(drop=True)
    # A new session starts whenever the cumulative counters go down
    session, last_fwd = 0, -1
    sess = []
    for _, r in hw.iterrows():
        if r["forward_passes"] <= last_fwd:
            session += 1
        sess.append(session)
        last_fwd = r["forward_passes"]
    hw["session"] = sess
    return hw


def reconstruct_sessions(df: pd.DataFrame, input_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    cost = per_sample_cost(df)
    hw = load_hw_sessions(input_dir)
    info: Dict = {"total_clean_time_s": float(cost["time_new_s"].sum())}
    if hw.empty:
        return pd.DataFrame(), cost, info

    ids = cost.index.to_numpy()
    cum_new = np.concatenate([[0], np.cumsum(cost["fwd_new"].to_numpy())])
    cum_rev = np.concatenate([[0], np.cumsum(cost["fwd_revisit"].to_numpy())])
    cum_t_new = np.concatenate([[0], np.cumsum(cost["time_new_s"].to_numpy())])
    cum_t_rev = np.concatenate([[0], np.cumsum(cost["time_revisit_s"].to_numpy())])

    rows = []
    for s_id, g in hw.groupby("session"):
        first = g.iloc[0]
        last = g.iloc[-1]
        n_first = int(first["num_samples"])
        # Find start index i: revisit(0..i) + new(i..i+n_first) == logged fwd of first snapshot
        best = None
        for i in range(0, len(ids) - n_first + 1):
            pred = (cum_rev[i] - cum_rev[0]) + (cum_new[i + n_first] - cum_new[i])
            err = abs(pred - first["forward_passes"])
            if best is None or err < best[1]:
                best = (i, err)
        i0, err = best
        n_last = int(last["num_samples"])
        i1 = min(i0 + n_last, len(ids))
        t_new = cum_t_new[i1] - cum_t_new[i0]
        t_rev = cum_t_rev[i0]
        rows.append({
            "session": int(s_id) + 1,
            "logged_at": last["timestamp"],
            "sample_from": int(ids[i0]),
            "sample_to": int(ids[i1 - 1]),
            "n_new_samples": n_last,
            "logged_duration_s": float(last["duration_s"]),
            "logged_fwd": int(last["forward_passes"]),
            "match_error_fwd": int(err),
            "explain_time_new_s": float(t_new),
            "revisit_time_s": float(t_rev),
            "overhead_ratio": float(last["duration_s"] / max(1e-9, t_new + t_rev)),
        })
    sessions = pd.DataFrame(rows)
    covered = set()
    for _, r in sessions.iterrows():
        covered.update(range(r["sample_from"], r["sample_to"] + 1))
    unlogged = cost.loc[~cost.index.isin(covered)]
    ratio = float(sessions["logged_duration_s"].sum() /
                  (sessions["explain_time_new_s"].sum() + sessions["revisit_time_s"].sum()))
    info.update({
        "wall_over_explain_ratio": ratio,
        "unlogged_n": len(unlogged),
        "unlogged_range": (int(unlogged.index.min()), int(unlogged.index.max())) if len(unlogged) else None,
        "unlogged_explain_time_s": float(unlogged["time_new_s"].sum()),
        "unlogged_est_wall_s": float(unlogged["time_new_s"].sum() * ratio),
        "logged_wall_s": float(sessions["logged_duration_s"].sum()),
        "est_total_wall_clean_s": float(cost["time_new_s"].sum() * ratio),
        "revisit_waste_s": float(sessions["revisit_time_s"].sum()),
        "n_samples_with_flip": int(cost["has_flip"].sum()),
    })
    return sessions, cost, info


def build_reconstructed_log(input_dir: str, sessions: pd.DataFrame, info: Dict, df: pd.DataFrame, cost: pd.DataFrame) -> pd.DataFrame:
    hw = load_hw_sessions(input_dir)
    if hw.empty or sessions.empty:
        return pd.DataFrame()
    last = hw.groupby("session").tail(1).copy()
    last["status"] = "logged"
    last["sample_range"] = [f"{a}-{b}" for a, b in zip(sessions["sample_from"], sessions["sample_to"])]
    rows = [last]
    if info.get("unlogged_n"):
        a, b = info["unlogged_range"]
        un = cost.loc[a:b]
        tmpl = last.iloc[0].copy()
        tmpl["timestamp"] = "(estimated)"
        tmpl["num_samples"] = info["unlogged_n"]
        tmpl["duration_s"] = round(info["unlogged_est_wall_s"], 2)
        tmpl["duration_formatted"] = fmt_s(info["unlogged_est_wall_s"])
        tmpl["forward_passes"] = int(un["fwd_new"].sum())
        tmpl["backward_passes"] = int(un["bwd_new"].sum())
        tmpl["forward_tokens"] = np.nan
        for c in ["peak_power_w", "avg_power_w", "gpu_util_pct", "avg_gpu_util_pct", "peak_vram_mb", "vram_reserved_mb"]:
            if c in tmpl:
                tmpl[c] = np.nan
        tmpl["fwd_per_second"] = round(tmpl["forward_passes"] / max(1, tmpl["duration_s"]), 2)
        tmpl["status"] = "estimated_from_raw_timings"
        tmpl["sample_range"] = f"{a}-{b}"
        rows.append(pd.DataFrame([tmpl]))
    out = pd.concat(rows, ignore_index=True).drop(columns=["source", "session"], errors="ignore")
    return out


# ----------------------------------------------------------------------------------------
# 3. Stability statistics
# ----------------------------------------------------------------------------------------
def main_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["is_control"] & ~df["is_identity"]]


def per_sample_means(df: pd.DataFrame, metric: str, by: List[str]) -> pd.DataFrame:
    """Cluster-aware aggregation: average over perturbations within a sample first."""
    return df.groupby(by + ["sample_id"])[metric].mean().reset_index()


def overall_table(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    sub = main_subset(df)
    rows = []
    for e in ordered(sub["explainer_name"].unique(), EXPLAINER_ORDER):
        g = sub[sub["explainer_name"] == e]
        row = {"explainer": e, "n_pairs": len(g), "n_samples": g["sample_id"].nunique()}
        for m in ["spearman", "kendall", "jaccard_k3", "jaccard_k5"]:
            ps = g.groupby("sample_id")[m].mean().to_numpy()
            lo, hi = bootstrap_ci(ps, n_boot)
            row[f"{m}_mean"] = float(np.mean(ps))
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        row["spearman_median_pair"] = float(g["spearman"].median())
        row["share_pairs_rho_lt_0.5"] = float((g["spearman"] < 0.5).mean())
        ctrl = df[(df["explainer_name"] == e) & (df["perturbation_name"] == NOISE_FLOOR_NAME)]
        row["noise_floor_spearman"] = float(ctrl["spearman"].mean()) if len(ctrl) else np.nan
        row["time_per_expl_s"] = float(g["pert_time_s"].mean())
        row["fwd_per_expl"] = float(g["pert_forward_passes"].mean())
        row["bwd_per_expl"] = float(g["pert_backward_passes"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def category_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = main_subset(df)
    ps = per_sample_means(sub, metric, ["explainer_name", "perturbation_category"])
    t = ps.groupby(["explainer_name", "perturbation_category"])[metric].mean().unstack()
    t = t.reindex(index=ordered(t.index, EXPLAINER_ORDER), columns=ordered(t.columns, CATEGORY_ORDER))
    return t


def perturbation_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = main_subset(df)
    t = sub.groupby(["perturbation_category", "perturbation_name", "explainer_name"])[metric].mean().unstack()
    t = t[ordered(t.columns, EXPLAINER_ORDER)]
    t["mean_all"] = t.mean(axis=1)
    return t.sort_values("mean_all", ascending=False)


def method_tests(df: pd.DataFrame, metric: str) -> Tuple[Dict, pd.DataFrame]:
    sub = main_subset(df)
    wide = sub.groupby(["sample_id", "explainer_name"])[metric].mean().unstack().dropna()
    wide = wide[ordered(wide.columns, EXPLAINER_ORDER)]
    res: Dict = {"n_samples": len(wide)}
    if wide.shape[1] >= 3 and len(wide) > 5:
        chi2, p = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
        k, n = wide.shape[1], len(wide)
        res.update({"friedman_chi2": float(chi2), "friedman_p": float(p), "kendall_W": float(chi2 / (n * (k - 1)))})
        res["mean_ranks"] = wide.rank(axis=1, ascending=False).mean().sort_values().to_dict()
    pairs = []
    cols = list(wide.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = wide[cols[i]], wide[cols[j]]
            d = a - b
            try:
                w, p = stats.wilcoxon(a, b)
            except ValueError:
                w, p = np.nan, 1.0
            # matched-pairs rank-biserial correlation
            r = d[d != 0]
            ranks = stats.rankdata(np.abs(r))
            rbc = (ranks[r > 0].sum() - ranks[r < 0].sum()) / ranks.sum() if len(r) else 0.0
            pairs.append({"A": cols[i], "B": cols[j], "mean_A": a.mean(), "mean_B": b.mean(),
                          "diff": d.mean(), "wilcoxon_p": p, "rank_biserial": rbc})
    pw = pd.DataFrame(pairs)
    if len(pw):
        pw["p_holm"] = holm(pw["wilcoxon_p"].tolist())
    return res, pw


def category_tests(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = main_subset(df)
    rows = []
    for e, g in sub.groupby("explainer_name"):
        wide = g.groupby(["sample_id", "perturbation_category"])[metric].mean().unstack().dropna()
        if wide.shape[1] >= 3 and len(wide) > 5:
            chi2, p = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
            rows.append({"explainer": e, "n_samples": len(wide), "friedman_chi2": chi2, "p": p,
                         "kendall_W": chi2 / (len(wide) * (wide.shape[1] - 1)),
                         "most_stable": wide.mean().idxmax(), "least_stable": wide.mean().idxmin()})
    return pd.DataFrame(rows).set_index("explainer").reindex(ordered([r["explainer"] for r in rows], EXPLAINER_ORDER)).reset_index() if rows else pd.DataFrame()


def noise_floor_table(df: pd.DataFrame, metric: str, n_boot: int) -> pd.DataFrame:
    sub = main_subset(df)
    rows = []
    for e in ordered(df.loc[df["perturbation_name"] == NOISE_FLOOR_NAME, "explainer_name"].unique(), EXPLAINER_ORDER):
        ctrl = df[(df["explainer_name"] == e) & (df["perturbation_name"] == NOISE_FLOOR_NAME)].set_index("sample_id")[metric]
        pert = sub[sub["explainer_name"] == e].groupby("sample_id")[metric].mean()
        common = ctrl.index.intersection(pert.index)
        diff = (ctrl.loc[common] - pert.loc[common]).to_numpy()
        lo, hi = bootstrap_ci(diff, n_boot)
        try:
            p = stats.wilcoxon(ctrl.loc[common], pert.loc[common]).pvalue
        except ValueError:
            p = np.nan
        rows.append({"explainer": e, "noise_floor": ctrl.mean(), "under_perturbation": pert.loc[common].mean(),
                     "excess_instability": diff.mean(), "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p,
                     "share_of_drop_explained_by_noise": (1 - ctrl.mean()) / max(1e-9, 1 - pert.loc[common].mean())})
    return pd.DataFrame(rows)


def covariate_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = main_subset(df)
    rows = []
    for e, g in sub.groupby("explainer_name"):
        ps = g.groupby("sample_id").agg(m=(metric, "mean"), p_orig=("p_orig", "first"),
                                        n_tok=("orig_num_tokens", "first"),
                                        correct=("true_label", "first"), pred=("orig_pred", "first"))
        ps["correct"] = ps["correct"] == ps["pred"]
        r_conf = stats.spearmanr(ps["m"], ps["p_orig"])
        r_len = stats.spearmanr(ps["m"], ps["n_tok"])
        a, b = ps.loc[ps["correct"], "m"], ps.loc[~ps["correct"], "m"]
        p_corr = stats.mannwhitneyu(a, b).pvalue if len(a) > 3 and len(b) > 3 else np.nan
        rows.append({"explainer": e, "rho_vs_confidence": r_conf.statistic, "p_conf": r_conf.pvalue,
                     "rho_vs_length": r_len.statistic, "p_len": r_len.pvalue,
                     "mean_correct": a.mean(), "mean_incorrect": b.mean(), "p_correct_vs_incorrect": p_corr})
    t = pd.DataFrame(rows).set_index("explainer")
    return t.reindex(ordered(t.index, EXPLAINER_ORDER)).reset_index()


def compare_models(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str, metric: str) -> pd.DataFrame:
    rows = []
    common_e = [e for e in EXPLAINER_ORDER if e in set(df_a.explainer_name) & set(df_b.explainer_name)]
    common_p = set(df_a.perturbation_name) & set(df_b.perturbation_name)
    for e in common_e:
        for cat in CATEGORY_ORDER + ["ALL"]:
            vals = []
            for d in (df_a, df_b):
                s = main_subset(d)
                s = s[(s.explainer_name == e) & s.perturbation_name.isin(common_p)]
                if cat != "ALL":
                    s = s[s.perturbation_category == cat]
                vals.append(s.groupby("sample_id")[metric].mean())
            if len(vals[0]) < 3 or len(vals[1]) < 3:
                continue
            p = stats.mannwhitneyu(vals[0], vals[1]).pvalue
            rows.append({"explainer": e, "category": cat, f"{label_a}": vals[0].mean(), f"{label_b}": vals[1].mean(),
                         "diff": vals[0].mean() - vals[1].mean(), "n_a": len(vals[0]), "n_b": len(vals[1]),
                         "mannwhitney_p": p})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# 4. Plots
# ----------------------------------------------------------------------------------------
def plot_heatmap(pt: pd.DataFrame, metric: str, plots_dir: str, label: str):
    data = pt.drop(columns=["mean_all"])
    fig, ax = plt.subplots(figsize=(1.6 * data.shape[1] + 3, 0.42 * data.shape[0] + 1.5))
    im = ax.imshow(data.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(data.shape[1]), data.columns, rotation=30, ha="right")
    ax.set_yticks(range(data.shape[0]), [f"{n} ({c})" for c, n in data.index])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.iat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=METRIC_LABELS[metric])
    ax.set_title(f"{label}: średnia {METRIC_LABELS[metric]} wg perturbacji i metody")
    save_fig(fig, plots_dir, f"heatmap_{metric}")


def plot_box_by_category(df: pd.DataFrame, metric: str, plots_dir: str, label: str):
    sub = main_subset(df)
    ps = per_sample_means(sub, metric, ["explainer_name", "perturbation_category"])
    expl = ordered(ps.explainer_name.unique(), EXPLAINER_ORDER)
    cats = ordered(ps.perturbation_category.unique(), CATEGORY_ORDER)
    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.8 / len(expl)
    for k, e in enumerate(expl):
        data = [ps[(ps.explainer_name == e) & (ps.perturbation_category == c)][metric].to_numpy() for c in cats]
        pos = np.arange(len(cats)) + (k - (len(expl) - 1) / 2) * w
        bp = ax.boxplot(data, positions=pos, widths=w * 0.9, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor(COLORS.get(e, "grey"))
            b.set_alpha(0.7)
        ax.plot([], [], "s", color=COLORS.get(e, "grey"), label=e)
    ctrl = df[df.perturbation_name == NOISE_FLOOR_NAME].groupby("explainer_name")[metric].mean()
    for e, v in ctrl.items():
        ax.axhline(v, ls="--", lw=1, color=COLORS.get(e, "grey"), alpha=0.8)
    ax.set_xticks(range(len(cats)), cats)
    ax.set_ylabel(f"{METRIC_LABELS[metric]} (średnia na próbkę)")
    ax.set_title(f"{label}: stabilność wg kategorii perturbacji (linie przerywane = noise floor LIME/SHAP)")
    ax.legend(ncol=len(expl), loc="lower center", bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, plots_dir, f"box_category_{metric}")


def plot_jaccard_curves(df: pd.DataFrame, plots_dir: str, label: str):
    sub = main_subset(df)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ks = [1, 3, 5]
    for e in ordered(sub.explainer_name.unique(), EXPLAINER_ORDER):
        g = sub[sub.explainer_name == e]
        means, los, his = [], [], []
        for k in ks:
            ps = g.groupby("sample_id")[f"jaccard_k{k}"].mean().to_numpy()
            lo, hi = bootstrap_ci(ps, 1000)
            means.append(ps.mean()); los.append(lo); his.append(hi)
        ax.plot(ks, means, "o-", color=COLORS.get(e), label=e)
        ax.fill_between(ks, los, his, color=COLORS.get(e), alpha=0.15)
    ax.set_xticks(ks)
    ax.set_xlabel("K")
    ax.set_ylabel("Jaccard top-K (95% CI)")
    ax.set_title(f"{label}: zgodność top-K tokenów")
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, plots_dir, "jaccard_topk")


def plot_noise_floor(nf: pd.DataFrame, metric: str, plots_dir: str, label: str):
    if nf.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(nf))
    ax.bar(x - 0.2, nf["noise_floor"], 0.4, label="Ten sam tekst, inne ziarno (noise floor)", color="#bbbbbb")
    ax.bar(x + 0.2, nf["under_perturbation"], 0.4, label="Perturbacja tekstu",
           color=[COLORS.get(e) for e in nf["explainer"]])
    ax.set_xticks(x, nf["explainer"])
    ax.set_ylim(0, 1)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"{label}: szum własny vs wpływ perturbacji")
    ax.legend(fontsize=8)
    save_fig(fig, plots_dir, f"noise_floor_{metric}")


def plot_flip_rates(fr: pd.DataFrame, plots_dir: str, label: str):
    fr = fr.sort_values("flip_rate")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(fr) + 1.2))
    cmap = {c: plt.cm.tab10(i) for i, c in enumerate(ordered(fr.perturbation_category.unique(), CATEGORY_ORDER))}
    ax.barh(fr["perturbation_name"], fr["flip_rate"] * 100, color=[cmap[c] for c in fr.perturbation_category])
    for c, col in cmap.items():
        ax.barh([], [], color=col, label=c)
    ax.set_xlabel("Odsetek zmienionych decyzji [%]")
    ax.set_title(f"{label}: flip rate (pary wykluczone z analizy stabilności)")
    ax.legend()
    save_fig(fig, plots_dir, "decision_flip_rate")


def plot_cost_vs_stability(ot: pd.DataFrame, plots_dir: str, label: str):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for _, r in ot.iterrows():
        e = r["explainer"]
        ax.errorbar(r["time_per_expl_s"], r["spearman_mean"],
                    yerr=[[r["spearman_mean"] - r["spearman_ci_lo"]], [r["spearman_ci_hi"] - r["spearman_mean"]]],
                    fmt="o", ms=9, color=COLORS.get(e), capsize=3)
        ax.annotate(f"{e}\n({r['fwd_per_expl']:.0f} fwd, {r['bwd_per_expl']:.0f} bwd)",
                    (r["time_per_expl_s"], r["spearman_mean"]), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Czas jednego wyjaśnienia [s] (log)")
    ax.set_ylabel("Spearman ρ (średnia, 95% CI)")
    ax.set_title(f"{label}: koszt vs stabilność")
    ax.grid(alpha=0.3, which="both")
    save_fig(fig, plots_dir, "cost_vs_stability")


def plot_covariates(df: pd.DataFrame, metric: str, plots_dir: str, label: str):
    sub = main_subset(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for e in ordered(sub.explainer_name.unique(), EXPLAINER_ORDER):
        ps = sub[sub.explainer_name == e].groupby("sample_id").agg(
            m=(metric, "mean"), p=("p_orig", "first"), n=("orig_num_tokens", "first"))
        for ax, col, bins in [(axes[0], "p", np.linspace(0.5, 1.0, 6)), (axes[1], "n", [0, 15, 20, 25, 30, 35, 40, 100])]:
            b = pd.cut(ps[col], bins, include_lowest=True)
            g = ps.groupby(b, observed=True)["m"].agg(["mean", "sem", "size"])
            centers = [iv.mid if np.isfinite(iv.right) and iv.right < 100 else iv.left + 5 for iv in g.index]
            ax.errorbar(centers, g["mean"], yerr=1.96 * g["sem"], fmt="o-", color=COLORS.get(e), label=e, capsize=2)
    axes[0].set_xlabel("Pewność modelu p(klasa przewidziana)")
    axes[1].set_xlabel("Długość zdania [tokeny]")
    for ax in axes:
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{label}: stabilność a pewność modelu i długość wejścia")
    save_fig(fig, plots_dir, f"covariates_{metric}")


def plot_model_comparison(cmp: pd.DataFrame, la: str, lb: str, metric: str, plots_dir: str):
    if cmp.empty:
        return
    c = cmp[cmp.category != "ALL"]
    expl = ordered(c.explainer.unique(), EXPLAINER_ORDER)
    cats = ordered(c.category.unique(), CATEGORY_ORDER)
    fig, axes = plt.subplots(1, len(expl), figsize=(4.2 * len(expl), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, expl):
        g = c[c.explainer == e].set_index("category").reindex(cats)
        x = np.arange(len(cats))
        ax.bar(x - 0.2, g[la], 0.4, label=la, color=COLORS.get(e))
        ax.bar(x + 0.2, g[lb], 0.4, label=lb, color=COLORS.get(e), alpha=0.4, hatch="//")
        for i, p in enumerate(g["mannwhitney_p"]):
            if p < 0.05:
                ax.text(i, max(g[la].iloc[i], g[lb].iloc[i]) + 0.02, "*", ha="center")
        ax.set_xticks(x, cats, rotation=20)
        ax.set_title(e)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(METRIC_LABELS[metric])
    axes[0].legend()
    fig.suptitle(f"Porównanie modeli: {la} vs {lb} (* p<0.05, Mann-Whitney)")
    save_fig(fig, plots_dir, f"model_comparison_{metric}")


def plot_convergence(df: pd.DataFrame, metric: str, plots_dir: str, label: str):
    """Running mean vs number of samples -> is N large enough?"""
    sub = main_subset(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    for e in ordered(sub.explainer_name.unique(), EXPLAINER_ORDER):
        ps = sub[sub.explainer_name == e].groupby("sample_id")[metric].mean().sort_index()
        run = ps.expanding().mean()
        ax.plot(np.arange(1, len(run) + 1), run.to_numpy(), color=COLORS.get(e), label=e)
    ax.set_xlabel("Liczba próbek")
    ax.set_ylabel(f"Skumulowana średnia {METRIC_LABELS[metric]}")
    ax.set_title(f"{label}: zbieżność estymatora względem N")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save_fig(fig, plots_dir, f"convergence_{metric}")


# ----------------------------------------------------------------------------------------
# 5. LaTeX
# ----------------------------------------------------------------------------------------
def latex_overall(ot: pd.DataFrame, label: str) -> str:
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{Stabilność metod XAI pod perturbacjami zachowującymi decyzję ({label}); średnia z 95\% CI (bootstrap po próbkach)}}",
        rf"\label{{tab:rq1_overall_{label.replace('.', '').replace('-', '').lower()}}}",
        r"\begin{tabular}{lccccc}", r"\hline",
        r"\textbf{Metoda} & \textbf{Spearman $\rho$} & \textbf{Kendall $\tau$} & \textbf{Jaccard@3} & \textbf{Noise floor $\rho$} & \textbf{Czas [s]} \\",
        r"\hline",
    ]
    for _, r in ot.iterrows():
        nf = "--" if np.isnan(r["noise_floor_spearman"]) else f"{r['noise_floor_spearman']:.3f}"
        lines.append(
            f"{r['explainer']} & {r['spearman_mean']:.3f} [{r['spearman_ci_lo']:.3f}, {r['spearman_ci_hi']:.3f}] & "
            f"{r['kendall_mean']:.3f} & {r['jaccard_k3_mean']:.3f} & {nf} & {r['time_per_expl_s']:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_category(ct: pd.DataFrame, metric: str, label: str) -> str:
    cols = list(ct.columns)
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{{METRIC_LABELS[metric].replace('ρ', '$\\rho$').replace('τ', '$\\tau$')} wg kategorii perturbacji ({label})}}",
        r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\hline",
        r"\textbf{Metoda} & " + " & ".join(rf"\textbf{{{c}}}" for c in cols) + r" \\", r"\hline",
    ]
    for e, r in ct.iterrows():
        best = r.max()
        cells = [(rf"\textbf{{{v:.3f}}}" if v == best else f"{v:.3f}") for v in r]
        lines.append(f"{e} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# 6. Report
# ----------------------------------------------------------------------------------------
def write_report(path: str, label: str, qc: Dict, fr: pd.DataFrame, sessions: pd.DataFrame, rec: Dict,
                 ot: pd.DataFrame, ct: pd.DataFrame, mt: Dict, pw: pd.DataFrame, cat_t: pd.DataFrame,
                 nf: pd.DataFrame, cov: pd.DataFrame, cmp: Optional[pd.DataFrame], cmp_label: Optional[str],
                 metric: str):
    L = [f"# Analiza RQ1 – {label}", ""]
    L += ["## 1. Kontrola jakości danych", ""]
    L.append(f"- Wiersze: **{qc['n_rows']}**, próbki: **{qc['n_samples']}** (id {qc['sample_id_range'][0]}–{qc['sample_id_range'][1]}), "
             f"brakujące id: {qc['missing_ids'] if qc['missing_ids'] else 'brak'}")
    L.append(f"- Metody: {', '.join(qc['explainers'])}; urządzenie: {', '.join(map(str, qc['devices']))}")
    L.append(f"- NaN w metrykach: {sum(qc['nan_metrics'].values())}, wartości poza [-1,1]: {sum(qc['out_of_range'].values())}")
    L.append(f"- Pary tożsame (perturbacja nie zmieniła tekstu): {qc['identity_rows']} {qc['identity_by_pert']} – "
             f"**wykluczone** z analizy głównej; perturbacje w 100% tożsame: {qc['fully_identity_perts']}")
    if qc["identity_det_min_spearman"]:
        L.append(f"- Test determinizmu (metody deterministyczne na tożsamym tekście, min ρ): "
                 + ", ".join(f"{k}={v:.4f}" for k, v in qc["identity_det_min_spearman"].items()))
    L += ["", "### Zachowanie klasyfikatora", ""]
    L.append(f"- Accuracy: **{qc['accuracy']:.3f}**, balanced accuracy: **{qc['balanced_accuracy']:.3f}** "
             f"(recall neg={qc['recall_neg']:.3f}, recall pos={qc['recall_pos']:.3f})")
    L.append(f"- Odsetek predykcji 'positive': **{qc['share_pred_positive']:.3f}** vs prawdziwy odsetek: {qc['share_true_positive']:.3f}; "
             f"średnie p(klasa przewidziana) = {qc['mean_p_orig']:.3f}")
    if qc["balanced_accuracy"] < 0.7 or abs(qc["share_pred_positive"] - qc["share_true_positive"]) > 0.2:
        L += ["", "> [!WARNING]",
              "> Klasyfikator jest silnie obciążony (prawie zawsze ta sama klasa). Wyjaśnienia opisują wtedy głównie "
              "*bias promptu/tokenu etykiety*, a nie rozumienie sentymentu. Filtr 'decision-preserved' jest dla "
              "większości próbek trywialnie spełniony. Wyniki należy raportować z tym zastrzeżeniem lub powtórzyć "
              "z kalibracją (np. contextual calibration, Zhao et al. 2021), wersją Instruct albo promptem few-shot."]
    L += ["", "### Flip rate (odsetek zmienionych decyzji)", "", df_to_md(fr), ""]

    L += ["## 2. Rekonstrukcja czasu obliczeń", ""]
    if not sessions.empty:
        L.append("Sesje dopasowano do zakresów próbek na podstawie liczników forward passes "
                 "(dopasowanie dokładne, gdy `match_error_fwd` = 0).")
        L += ["", df_to_md(sessions, ".1f"), ""]
        L.append(f"- Próbek z ≥1 odwróconą decyzją: **{rec['n_samples_with_flip']}** – przy każdym wznowieniu są one "
                 f"ponownie przetwarzane (predykcja + wyjaśnienie oryginału), bo pary odfiltrowane nie trafiają do "
                 f"`completed_units`. Zmarnowany czas w zalogowanych sesjach ≈ **{fmt_s(rec['revisit_waste_s'])}**.")
        L.append(f"- Współczynnik czas ścienny / suma czasów wyjaśnień: **{rec['wall_over_explain_ratio']:.3f}**")
        if rec.get("unlogged_n"):
            L.append(f"- Próbki bez logu sprzętowego: **{rec['unlogged_n']}** (id {rec['unlogged_range'][0]}–{rec['unlogged_range'][1]}), "
                     f"suma czasów wyjaśnień {fmt_s(rec['unlogged_explain_time_s'])} → szacowany czas ścienny "
                     f"**{fmt_s(rec['unlogged_est_wall_s'])}**")
        L.append(f"- Zalogowany czas: {fmt_s(rec['logged_wall_s'])}; szacowany łączny 'czysty' czas dla wszystkich "
                 f"{qc['n_samples']} próbek (bez narzutu wznowień): **{fmt_s(rec['est_total_wall_clean_s'])}**")
    else:
        L.append(f"Brak logów sprzętowych; suma czasów wyjaśnień: {fmt_s(rec['total_clean_time_s'])}")
    L.append("")

    L += ["## 3. Stabilność – wyniki główne", "",
          "Agregacja: najpierw średnia po perturbacjach w obrębie próbki, potem średnia po próbkach "
          "(95% CI: bootstrap po próbkach). Tylko pary z zachowaną decyzją, bez par tożsamych i kontroli.", ""]
    cols = ["explainer", "n_samples", "spearman_mean", "spearman_ci_lo", "spearman_ci_hi", "kendall_mean",
            "jaccard_k3_mean", "jaccard_k5_mean", "share_pairs_rho_lt_0.5", "noise_floor_spearman", "time_per_expl_s"]
    L += [df_to_md(ot[cols]), ""]
    L += [f"### {METRIC_LABELS[metric]} wg kategorii", "", df_to_md(ct.reset_index()), ""]
    if mt.get("friedman_p") is not None:
        L.append(f"**Test Friedmana (różnice między metodami, n={mt['n_samples']}):** χ²={mt['friedman_chi2']:.1f}, "
                 f"p={mt['friedman_p']:.2e}, Kendall W={mt['kendall_W']:.3f}. Średnie rangi (1=najstabilniejsza): "
                 + ", ".join(f"{k}={v:.2f}" for k, v in mt["mean_ranks"].items()))
        L += ["", "Porównania parami (Wilcoxon, korekta Holma):", "", df_to_md(pw, ".4f"), ""]
    if not cat_t.empty:
        L += ["**Wpływ kategorii perturbacji w obrębie metody (Friedman):**", "", df_to_md(cat_t, ".4f"), ""]
    if not nf.empty:
        L += ["### Noise floor (LIME/SHAP: ten sam tekst, inne ziarno)", "", df_to_md(nf), "",
              "`share_of_drop_explained_by_noise` = (1−noise floor)/(1−stabilność pod perturbacją): jaka część "
              "spadku stabilności wynika z samej losowości metody.", ""]
    L += ["### Zmienne towarzyszące", "", df_to_md(cov, ".4f"), ""]
    if cmp is not None and not cmp.empty:
        L += [f"## 4. Porównanie z {cmp_label}", "", df_to_md(cmp, ".4f"), "",
              "> [!NOTE]", "> Porównanie między modelami jest obciążone różnymi hiperparametrami (np. SHAP max_evals, "
              "IG steps), różną liczbą próbek i różną dokładnością modeli – traktować jako orientacyjne.", ""]

    # Automatic conclusions
    L += ["## 5. Wnioski (wygenerowane automatycznie – zweryfikuj)", ""]
    best = ot.sort_values("spearman_mean", ascending=False)
    L.append(f"1. Ranking stabilności (Spearman): " + " > ".join(
        f"{r.explainer} ({r.spearman_mean:.2f})" for r in best.itertuples()))
    hard = ct.mean(axis=0).sort_values()
    L.append(f"2. Najbardziej destabilizujące kategorie (średnio): " + ", ".join(f"{k} ({v:.2f})" for k, v in hard.items()))
    if not nf.empty:
        for r in nf.itertuples():
            L.append(f"3. {r.explainer}: noise floor ρ={r.noise_floor:.2f} vs pod perturbacją {r.under_perturbation:.2f} → "
                     f"~{100 * min(1, r.share_of_drop_explained_by_noise):.0f}% spadku to szum własny metody.")
    L.append(f"4. Klasyfikator: balanced acc. {qc['balanced_accuracy']:.2f}, predykcje 'positive' {100 * qc['share_pred_positive']:.0f}% – "
             "kluczowe zastrzeżenie interpretacyjne.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Analiza wyników RQ1 (stabilność).")
    ap.add_argument("--input-dir", default="results/rq1_stability_Qwen2.5-7B")
    ap.add_argument("--label", default=None, help="Nazwa modelu na wykresach (domyślnie z nazwy katalogu)")
    ap.add_argument("--compare-dir", default="results/rq1_stability", help="Drugi zbiór wyników do porównania ('' = brak)")
    ap.add_argument("--compare-label", default="Phi-2")
    ap.add_argument("--metric", default="spearman", choices=METRICS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    in_dir = args.input_dir if os.path.isabs(args.input_dir) else os.path.join(PROJECT_ROOT, args.input_dir)
    label = args.label or os.path.basename(in_dir.rstrip("/\\")).replace("rq1_stability_", "")
    out_dir = args.output_dir or os.path.join(in_dir, "analysis")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    m = args.metric

    print(f"[RQ1-analysis] Wczytywanie {in_dir}")
    df = load_raw(os.path.join(in_dir, "rq1_stability_raw.csv"))

    qc = quality_control(df)
    fr = decision_flip_rates(df)
    sessions, cost, rec = reconstruct_sessions(df, in_dir)
    rlog = build_reconstructed_log(in_dir, sessions, rec, df, cost)

    ot = overall_table(df, args.n_boot)
    ct = category_table(df, m)
    pt = perturbation_table(df, m)
    mt, pw = method_tests(df, m)
    cat_t = category_tests(df, m)
    nf = noise_floor_table(df, m, args.n_boot)
    cov = covariate_table(df, m)

    cmp = None
    if args.compare_dir:
        cdir = args.compare_dir if os.path.isabs(args.compare_dir) else os.path.join(PROJECT_ROOT, args.compare_dir)
        cpath = os.path.join(cdir, "rq1_stability_raw.csv")
        if os.path.exists(cpath) and os.path.abspath(cdir) != os.path.abspath(in_dir):
            dfc = load_raw(cpath)
            cmp = compare_models(df, dfc, label, args.compare_label, m)
            plot_model_comparison(cmp, label, args.compare_label, m, plots_dir)

    # Save tables
    fr.to_csv(os.path.join(out_dir, "table_flip_rates.csv"), index=False)
    ot.to_csv(os.path.join(out_dir, "table_overall.csv"), index=False)
    ct.to_csv(os.path.join(out_dir, f"table_category_{m}.csv"))
    pt.to_csv(os.path.join(out_dir, f"table_perturbation_{m}.csv"))
    pw.to_csv(os.path.join(out_dir, f"table_pairwise_{m}.csv"), index=False)
    nf.to_csv(os.path.join(out_dir, f"table_noise_floor_{m}.csv"), index=False)
    cov.to_csv(os.path.join(out_dir, f"table_covariates_{m}.csv"), index=False)
    cost.to_csv(os.path.join(out_dir, "per_sample_cost.csv"))
    if not sessions.empty:
        sessions.to_csv(os.path.join(out_dir, "session_reconstruction.csv"), index=False)
    if not rlog.empty:
        rlog.to_csv(os.path.join(out_dir, "hardware_session_log_reconstructed.csv"), index=False)
    if cmp is not None:
        cmp.to_csv(os.path.join(out_dir, f"table_model_comparison_{m}.csv"), index=False)
    with open(os.path.join(out_dir, "table_overall.tex"), "w", encoding="utf-8") as f:
        f.write(latex_overall(ot, label))
    with open(os.path.join(out_dir, f"table_category_{m}.tex"), "w", encoding="utf-8") as f:
        f.write(latex_category(ct, m, label))

    # Plots
    plot_heatmap(pt, m, plots_dir, label)
    plot_box_by_category(df, m, plots_dir, label)
    plot_jaccard_curves(df, plots_dir, label)
    plot_noise_floor(nf, m, plots_dir, label)
    plot_flip_rates(fr, plots_dir, label)
    plot_cost_vs_stability(ot, plots_dir, label)
    plot_covariates(df, m, plots_dir, label)
    plot_convergence(df, m, plots_dir, label)

    write_report(os.path.join(out_dir, "rq1_report.md"), label, qc, fr, sessions, rec, ot, ct, mt, pw, cat_t,
                 nf, cov, cmp, args.compare_label, m)
    print(f"[RQ1-analysis] Gotowe -> {out_dir}")


if __name__ == "__main__":
    main()
