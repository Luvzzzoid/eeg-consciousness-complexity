"""
Phase 3b: Formal statistics on entropy/complexity results.

Reads entropy_complexity_trial_level.csv (already computed by compute_entropy.py - 
no re-computation needed, this just runs stats on saved trial-level values).

Aggregates to subject-level means (Seen vs Unseen), then runs paired Wilcoxon
signed-rank tests - same statistical approach the original paper used for its own
awareness comparisons (see stats.py in the authors' GitHub code) - with FDR
correction across the two measures tested (permutation entropy, LZ complexity).
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import fdrcorrection
from pathlib import Path

DATA_EPOCH = Path("../data/epochdata")


def main():
    df = pd.read_csv(DATA_EPOCH / "entropy_complexity_trial_level.csv")
    print(f"[load] {len(df)} trials loaded across {df['subject'].nunique()} subjects")

    # aggregate to subject-level means, Seen vs Unseen, for each measure
    subj_means = df.groupby(["subject", "awareness"])[["perm_entropy_mean", "lz_complexity_mean"]].mean().reset_index()

    results = {}
    p_values = []
    measure_names = []

    for measure in ["perm_entropy_mean", "lz_complexity_mean"]:
        pivot = subj_means.pivot(index="subject", columns="awareness", values=measure)
        pivot = pivot.dropna()  # only subjects with both Seen and Unseen trials

        seen_vals = pivot["Seen"].values
        unseen_vals = pivot["Unseen"].values
        n_subjects = len(pivot)

        stat, p = wilcoxon(seen_vals, unseen_vals)
        diffs = seen_vals - unseen_vals

        results[measure] = {
            "n_subjects": n_subjects,
            "mean_seen": seen_vals.mean(),
            "mean_unseen": unseen_vals.mean(),
            "mean_diff": diffs.mean(),
            "sd_diff": diffs.std(),
            "n_negative": int((diffs < 0).sum()),
            "n_positive": int((diffs > 0).sum()),
            "W": stat,
            "p_uncorrected": p,
        }
        p_values.append(p)
        measure_names.append(measure)

    # FDR correction across the two measures tested (standard practice whenever
    # testing more than one outcome measure on the same comparison, to control
    # false discovery rate rather than treating each test as independent/isolated)
    reject, p_corrected = fdrcorrection(p_values, alpha=0.05)

    print("\n=== RESULTS: Seen vs Unseen (Present trials, experimental block, 100-600ms window) ===\n")
    for i, measure in enumerate(measure_names):
        r = results[measure]
        label = "Permutation Entropy" if "perm_entropy" in measure else "Lempel-Ziv Complexity"
        print(f"--- {label} ---")
        print(f"  n subjects: {r['n_subjects']}")
        print(f"  Mean (Seen): {r['mean_seen']:.4f}")
        print(f"  Mean (Unseen): {r['mean_unseen']:.4f}")
        print(f"  Mean difference (Seen - Unseen): {r['mean_diff']:+.4f} (SD={r['sd_diff']:.4f})")
        print(f"  Subjects with negative diff (Unseen > Seen): {r['n_negative']}/{r['n_subjects']}")
        print(f"  Subjects with positive diff (Seen > Unseen): {r['n_positive']}/{r['n_subjects']}")
        print(f"  Wilcoxon W = {r['W']}, p (uncorrected) = {r['p_uncorrected']:.5f}")
        print(f"  p (FDR-corrected) = {p_corrected[i]:.5f}  {'*** SIGNIFICANT ***' if reject[i] else '(not significant)'}")
        print()

    # save results table
    results_df = pd.DataFrame(results).T
    results_df["p_fdr_corrected"] = p_corrected
    results_df["significant_fdr"] = reject
    out_path = DATA_EPOCH / "entropy_complexity_stats_results.csv"
    results_df.to_csv(out_path)
    print(f"[save] full results table saved to {out_path}")


if __name__ == "__main__":
    main()