"""
Phase 3d: Direct confound test via trial-level correlation + residualization.

Computes LZ complexity, permutation entropy, and variance/amplitude together for
every trial (so they're properly paired trial-by-trial, not just compared as separate
subject-level summaries), then:

  1. Checks the direct trial-level correlation between LZ complexity and variance.
     A strong correlation (e.g. |r| > 0.5-0.6) would suggest LZ is largely tracking
     amplitude. A weak/moderate correlation suggests they're capturing different things.

  2. Residualizes LZ complexity against variance (removes the part of LZ explainable
     by variance) and re-tests Seen vs Unseen on the residuals. If the Seen/Unseen
     difference survives on residualized LZ, that's strong evidence the effect isn't
     just an amplitude artifact.

This reuses the same epoch files as before - no new preprocessing needed.
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon, pearsonr
from antropy import perm_entropy, lziv_complexity

DATA_EPOCH = Path("../data/epochdata")
TMIN, TMAX = 0.6, 1.0
SUBJECT_IDS = ["10", "12", "13", "14", "15", "16", "19", "20", "21", "22", "23"]


def compute_all_measures_for_subject(subject_id):
    fif_path = DATA_EPOCH / f"sub-{subject_id}-epo.fif"
    if not fif_path.exists():
        return None

    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
    code_to_name = {v: k for k, v in epochs.event_id.items()}
    epochs_cropped = epochs.copy().crop(tmin=TMIN, tmax=TMAX)
    data = epochs_cropped.get_data(copy=True)

    rows = []
    for i, code in enumerate(epochs.events[:, 2]):
        name = code_to_name[code]
        if not name.startswith("Present"):
            continue
        awareness = "Seen" if "Seen" in name else "Unseen"

        pe_per_channel, lz_per_channel, var_per_channel = [], [], []
        for ch_idx in range(data.shape[1]):
            sig = data[i, ch_idx, :]
            pe_per_channel.append(perm_entropy(sig, order=3, delay=1, normalize=True))
            binarized = (sig > np.median(sig)).astype(int)
            lz_per_channel.append(lziv_complexity(binarized, normalize=True))
            var_per_channel.append(np.var(sig))

        rows.append({
            "subject": subject_id,
            "trial_index": i,
            "awareness": awareness,
            "perm_entropy": np.mean(pe_per_channel),
            "lz_complexity": np.mean(lz_per_channel),
            "variance": np.mean(var_per_channel),
        })

    return pd.DataFrame(rows)


def main():
    all_dfs = []
    for subject_id in SUBJECT_IDS:
        print(f"[processing] subject {subject_id}...")
        df = compute_all_measures_for_subject(subject_id)
        if df is not None and len(df) > 0:
            all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    out_csv = DATA_EPOCH / "trial_level_all_measures.csv"
    df.to_csv(out_csv, index=False)
    print(f"[save] {len(df)} trials saved to {out_csv}\n")

    # 1. direct trial-level correlation: LZ complexity vs variance 
    print("=== Trial-level correlation: LZ complexity vs Variance ===")
    r, p = pearsonr(df["lz_complexity"], df["variance"])
    print(f"  Pearson r = {r:.3f}, p = {p:.2e}  (n={len(df)} trials, pooled across subjects)")
    if abs(r) > 0.5:
        print("  -> STRONG correlation: LZ complexity may be substantially tracking amplitude.")
    elif abs(r) > 0.25:
        print("  -> MODERATE correlation: some shared variance, but LZ likely captures more than amplitude alone.")
    else:
        print("  -> WEAK correlation: LZ complexity is largely independent of amplitude in this data.")

    # 2. residualize LZ against variance, per subject (to respect within-subject structure) 
    print("\n=== Residualized LZ complexity (variance effect removed) -- Seen vs Unseen ===")
    df["lz_residual"] = np.nan
    for subject_id in df["subject"].unique():
        mask = df["subject"] == subject_id
        sub_df = df[mask]
        coeffs = np.polyfit(sub_df["variance"], sub_df["lz_complexity"], 1)
        predicted = np.polyval(coeffs, sub_df["variance"])
        df.loc[mask, "lz_residual"] = sub_df["lz_complexity"].values - predicted

    subj_resid_means = df.groupby(["subject", "awareness"])["lz_residual"].mean().reset_index()
    pivot = subj_resid_means.pivot(index="subject", columns="awareness", values="lz_residual").dropna()
    seen_resid = pivot["Seen"].values
    unseen_resid = pivot["Unseen"].values
    diffs = seen_resid - unseen_resid

    stat, p_resid = wilcoxon(seen_resid, unseen_resid)
    print(f"  n subjects: {len(pivot)}")
    print(f"  Mean residual (Seen): {seen_resid.mean():.5f}")
    print(f"  Mean residual (Unseen): {unseen_resid.mean():.5f}")
    print(f"  Subjects with negative diff (Unseen > Seen): {(diffs < 0).sum()}/{len(diffs)}")
    print(f"  Wilcoxon W = {stat}, p = {p_resid:.5f}")
    if p_resid < 0.05:
        print("  -> Seen/Unseen difference SURVIVES after removing the variance-explainable")
        print("     component of LZ complexity. This is fairly strong evidence the effect")
        print("     is not simply an amplitude artifact.")
    else:
        print("  -> Seen/Unseen difference does NOT survive after removing variance-explainable")
        print("     component. This suggests the original LZ effect may be substantially")
        print("     driven by the amplitude difference, and should be reported as such.")


if __name__ == "__main__":
    main()