"""
Phase 3f: Multi-window three-group comparison.

Extends the three-group comparison (Localizer Seen / Experimental Seen / Experimental
Unseen) across multiple time windows, to test WHEN each condition's complexity
signature emerges and not just whether an effect exists in one preselected window.

Motivation: the single-window three-group test found Experimental_Seen had the LOWEST
complexity (not Localizer_Seen, as a naive "more obvious = more structured" prediction
would suggest) in the 300-600ms window. One possible explanation: obvious (localizer)
stimuli may be resolved FASTER, producing their own complexity dip in an EARLIER window,
while near-threshold (experimental) stimuli take longer to resolve, producing a dip in
a LATER window. This script tests that directly by sliding the analysis window across
the epoch for all three groups.

Each subject's epoch data is loaded and cropped once per window (not re-read from disk
per window), keeping this efficient despite testing multiple windows.
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon, friedmanchisquare
from antropy import perm_entropy, lziv_complexity

DATA_EPOCH = Path("../data/epochdata")
SUBJECT_IDS = ["10", "12", "13", "14", "15", "16", "19", "20", "21", "22", "23"]
MIN_TRIALS = 10

# windows to test, in seconds (post-stimulus onset). Includes the original 0.3-0.6
# window plus finer early/late slices to map out timing.
WINDOWS = [
    (0.0, 0.2),
    (0.1, 0.3),
    (0.2, 0.4),
    (0.3, 0.5),
    (0.3, 0.6),  # original window, kept for direct comparison
    (0.5, 0.7),
    (0.6, 0.8),
    (0.8, 1.0),
]


def load_epochs(subject_id, block_label):
    suffix = "-epo_localizer.fif" if block_label == "localizer" else "-epo.fif"
    path = DATA_EPOCH / f"sub-{subject_id}{suffix}"
    if not path.exists():
        return None
    return mne.read_epochs(path, preload=True, verbose=False)


def compute_measures_for_window(epochs, tmin, tmax, subject_id, block_label):
    code_to_name = {v: k for k, v in epochs.event_id.items()}
    epochs_cropped = epochs.copy().crop(tmin=tmin, tmax=tmax)
    data = epochs_cropped.get_data(copy=True)

    rows = []
    for i, code in enumerate(epochs.events[:, 2]):
        name = code_to_name[code]
        if not name.startswith("Present"):
            continue
        awareness = "Seen" if "Seen" in name else "Unseen"

        lz_per_channel = []
        for ch_idx in range(data.shape[1]):
            sig = data[i, ch_idx, :]
            binarized = (sig > np.median(sig)).astype(int)
            lz_per_channel.append(lziv_complexity(binarized, normalize=True))

        rows.append({
            "subject": subject_id, "block": block_label, "awareness": awareness,
            "lz_complexity": np.mean(lz_per_channel),
        })
    return pd.DataFrame(rows)


def main():
    # load each subject's epochs ONCE (both block types), reuse across all windows
    subject_epochs = {}
    for subject_id in SUBJECT_IDS:
        print(f"[loading] subject {subject_id}...")
        exp_epochs = load_epochs(subject_id, "experimental")
        loc_epochs = load_epochs(subject_id, "localizer")
        subject_epochs[subject_id] = (exp_epochs, loc_epochs)

    window_results = []

    for tmin, tmax in WINDOWS:
        print(f"\n[window] {tmin}-{tmax}s ...")
        all_dfs = []
        for subject_id in SUBJECT_IDS:
            exp_epochs, loc_epochs = subject_epochs[subject_id]
            if exp_epochs is not None:
                all_dfs.append(compute_measures_for_window(exp_epochs, tmin, tmax, subject_id, "experimental"))
            if loc_epochs is not None:
                all_dfs.append(compute_measures_for_window(loc_epochs, tmin, tmax, subject_id, "localizer"))

        df = pd.concat(all_dfs, ignore_index=True)
        df["group"] = None
        df.loc[(df["block"] == "localizer") & (df["awareness"] == "Seen"), "group"] = "Localizer_Seen"
        df.loc[(df["block"] == "experimental") & (df["awareness"] == "Seen"), "group"] = "Experimental_Seen"
        df.loc[(df["block"] == "experimental") & (df["awareness"] == "Unseen"), "group"] = "Experimental_Unseen"
        df_groups = df.dropna(subset=["group"])

        group_order = ["Localizer_Seen", "Experimental_Seen", "Experimental_Unseen"]
        counts = df_groups.groupby(["subject", "group"]).size().unstack(fill_value=0)
        valid_subjects = counts[(counts.reindex(columns=group_order, fill_value=0) >= MIN_TRIALS).all(axis=1)].index.tolist()
        df_valid = df_groups[df_groups["subject"].isin(valid_subjects)]

        if len(valid_subjects) < 3:
            print(f"  [skip] too few valid subjects ({len(valid_subjects)}) for this window")
            continue

        subj_means = df_valid.groupby(["subject", "group"])["lz_complexity"].mean().unstack()[group_order]

        friedman_stat, friedman_p = friedmanchisquare(
            subj_means["Localizer_Seen"], subj_means["Experimental_Seen"], subj_means["Experimental_Unseen"])

        row = {
            "window": f"{tmin}-{tmax}",
            "n_subjects": len(valid_subjects),
            "mean_LocSeen": subj_means["Localizer_Seen"].mean(),
            "mean_ExpSeen": subj_means["Experimental_Seen"].mean(),
            "mean_ExpUnseen": subj_means["Experimental_Unseen"].mean(),
            "friedman_p": friedman_p,
        }

        for a, b, label in [("Localizer_Seen", "Experimental_Seen", "LocSeen_vs_ExpSeen"),
                             ("Experimental_Seen", "Experimental_Unseen", "ExpSeen_vs_ExpUnseen"),
                             ("Localizer_Seen", "Experimental_Unseen", "LocSeen_vs_ExpUnseen")]:
            stat, p = wilcoxon(subj_means[a], subj_means[b])
            row[f"p_{label}"] = p
            row[f"diff_{label}"] = (subj_means[a] - subj_means[b]).mean()

        window_results.append(row)
        print(f"  LocSeen={row['mean_LocSeen']:.4f}  ExpSeen={row['mean_ExpSeen']:.4f}  "
              f"ExpUnseen={row['mean_ExpUnseen']:.4f}  Friedman p={friedman_p:.4f}")

    results_df = pd.DataFrame(window_results)
    out_csv = DATA_EPOCH / "multi_window_three_group_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n[save] window-by-window results saved to {out_csv}")

    print("\n=== SUMMARY: mean LZ complexity by group across windows ===")
    print(results_df[["window", "n_subjects", "mean_LocSeen", "mean_ExpSeen", "mean_ExpUnseen", "friedman_p"]].to_string(index=False))

    print("\n=== Which group has the LOWEST complexity in each window? ===")
    for _, r in results_df.iterrows():
        means = {"Localizer_Seen": r["mean_LocSeen"], "Experimental_Seen": r["mean_ExpSeen"], "Experimental_Unseen": r["mean_ExpUnseen"]}
        lowest = min(means, key=means.get)
        sig = "significant overall" if r["friedman_p"] < 0.05 else "not significant overall"
        print(f"  {r['window']}: lowest = {lowest} ({sig}, Friedman p={r['friedman_p']:.4f})")


if __name__ == "__main__":
    main()