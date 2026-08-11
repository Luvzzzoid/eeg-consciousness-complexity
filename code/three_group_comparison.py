"""
Phase 3e: Three-group validation test.

Tests whether Lempel-Ziv complexity follows a graded pattern across three conditions
that differ in "strength" of conscious detection:

  1. Localizer Present+Seen   - obvious, suprathreshold, confidently detected stimulus
  2. Experimental Present+Seen - near-threshold, but still consciously reported
  3. Experimental Present+Unseen - near-threshold, NOT consciously reported

If the interpretation in the main analysis is correct (conscious detection is
associated with a more stereotyped, lower-complexity neural response), a graded
prediction follows: Localizer Seen should show the LOWEST complexity (strongest,
most reliable detection event), Experimental Unseen the HIGHEST (no detection event),
with Experimental Seen in between.

NOTE on what this does NOT test: this does not compare Localizer Seen vs Localizer
Unseen directly, because localizer "miss" trials are too rare (often 0-3 per subject)
to compute a reliable per-subject average. Comparing Present+Seen vs Absent trials
would confound awareness with stimulus presence, so that comparison is avoided here.

Uses the same epoch .fif files already produced (both experimental and localizer
files from run_preprocessing.py) - no new preprocessing needed.
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon, friedmanchisquare
from antropy import perm_entropy, lziv_complexity

DATA_EPOCH = Path("../data/epochdata")
TMIN, TMAX = 0.3, 0.6  # the window where the primary effect was found
SUBJECT_IDS = ["10", "12", "13", "14", "15", "16", "19", "20", "21", "22", "23"]
MIN_TRIALS = 10  # minimum trials required for a condition to be included for a subject


def compute_measures(epochs_path, block_label, subject_id):
    if not epochs_path.exists():
        return None
    epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
    code_to_name = {v: k for k, v in epochs.event_id.items()}
    epochs_cropped = epochs.copy().crop(tmin=TMIN, tmax=TMAX)
    data = epochs_cropped.get_data(copy=True)

    rows = []
    for i, code in enumerate(epochs.events[:, 2]):
        name = code_to_name[code]
        if not name.startswith("Present"):
            continue
        awareness = "Seen" if "Seen" in name else "Unseen"

        pe_per_channel, lz_per_channel = [], []
        for ch_idx in range(data.shape[1]):
            sig = data[i, ch_idx, :]
            pe_per_channel.append(perm_entropy(sig, order=3, delay=1, normalize=True))
            binarized = (sig > np.median(sig)).astype(int)
            lz_per_channel.append(lziv_complexity(binarized, normalize=True))

        rows.append({
            "subject": subject_id,
            "block": block_label,
            "awareness": awareness,
            "perm_entropy": np.mean(pe_per_channel),
            "lz_complexity": np.mean(lz_per_channel),
        })
    return pd.DataFrame(rows)


def main():
    all_dfs = []
    for subject_id in SUBJECT_IDS:
        print(f"[processing] subject {subject_id}...")
        exp_df = compute_measures(DATA_EPOCH / f"sub-{subject_id}-epo.fif", "experimental", subject_id)
        loc_df = compute_measures(DATA_EPOCH / f"sub-{subject_id}-epo_localizer.fif", "localizer", subject_id)
        for d in [exp_df, loc_df]:
            if d is not None and len(d) > 0:
                all_dfs.append(d)

    df = pd.concat(all_dfs, ignore_index=True)
    out_csv = DATA_EPOCH / "three_group_trial_level.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[save] {len(df)} trials saved to {out_csv}")

    # define the three groups
    df["group"] = None
    df.loc[(df["block"] == "localizer") & (df["awareness"] == "Seen"), "group"] = "Localizer_Seen"
    df.loc[(df["block"] == "experimental") & (df["awareness"] == "Seen"), "group"] = "Experimental_Seen"
    df.loc[(df["block"] == "experimental") & (df["awareness"] == "Unseen"), "group"] = "Experimental_Unseen"
    df_groups = df.dropna(subset=["group"])

    group_order = ["Localizer_Seen", "Experimental_Seen", "Experimental_Unseen"]

    # check trial counts per subject per group, drop subjects missing enough trials in any group
    counts = df_groups.groupby(["subject", "group"]).size().unstack(fill_value=0)
    print("\n=== Trial counts per subject per group ===")
    print(counts[group_order])
    valid_subjects = counts[(counts[group_order] >= MIN_TRIALS).all(axis=1)].index.tolist()
    print(f"\n[info] subjects with >= {MIN_TRIALS} trials in all 3 groups: {len(valid_subjects)} of {len(SUBJECT_IDS)}")
    if len(valid_subjects) < len(SUBJECT_IDS):
        dropped = set(SUBJECT_IDS) - set(valid_subjects)
        print(f"[info] dropped subjects (insufficient trials in >=1 group): {sorted(dropped)}")

    df_valid = df_groups[df_groups["subject"].isin(valid_subjects)]

    # subject-level means per group
    subj_means = df_valid.groupby(["subject", "group"])["lz_complexity"].mean().unstack()[group_order]
    print("\n=== Subject-level mean LZ complexity by group ===")
    print(subj_means.round(4))

    print(f"\n=== Overall means (n={len(valid_subjects)} subjects) ===")
    for g in group_order:
        print(f"  {g}: mean = {subj_means[g].mean():.4f}")

    # Friedman test: are the three groups different at all? (nonparametric repeated-measures ANOVA)
    stat, p = friedmanchisquare(subj_means["Localizer_Seen"], subj_means["Experimental_Seen"], subj_means["Experimental_Unseen"])
    print(f"\n=== Friedman test (all 3 groups differ?) ===")
    print(f"  chi2 = {stat:.3f}, p = {p:.5f}")

    # pairwise Wilcoxon tests for the predicted graded pattern
    print("\n=== Pairwise comparisons (predicted: Localizer_Seen < Experimental_Seen < Experimental_Unseen) ===")
    pairs = [
        ("Localizer_Seen", "Experimental_Seen"),
        ("Experimental_Seen", "Experimental_Unseen"),
        ("Localizer_Seen", "Experimental_Unseen"),
    ]
    for a, b in pairs:
        stat, p = wilcoxon(subj_means[a], subj_means[b])
        diff = (subj_means[a] - subj_means[b]).mean()
        direction = "as predicted (lower)" if diff < 0 else "OPPOSITE of predicted"
        print(f"  {a} vs {b}: mean diff = {diff:+.4f} ({direction}), W={stat}, p={p:.5f}")

    print("\nINTERPRETATION: if the graded prediction holds, Localizer_Seen should have the")
    print("lowest mean LZ complexity, Experimental_Unseen the highest, with Experimental_Seen")
    print("in between -- and all three pairwise comparisons should be significant in that direction.")


if __name__ == "__main__":
    main()