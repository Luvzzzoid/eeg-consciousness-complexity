"""
Phase 3c: Amplitude confound check.

LZ complexity and permutation entropy can be sensitive to signal amplitude/variance. 
If Seen trials simply have a bigger, more stereotyped evoked response (which is 
expected since something happens neurally when a stimulus is consciously detected), 
that alone could lower LZ complexity mechanically without reflecting a deeper 
complexity difference.

This script checks: does trial-level signal variance differ between Seen and Unseen
in the same 100-600ms window used for the entropy/complexity analysis? If yes, the
LZ complexity result needs to be interpreted alongside this, not read in isolation.

Uses the same epoch .fif files as compute_entropy.py - fast, no reprocessing.
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path

DATA_EPOCH = Path("../data/epochdata")
TMIN, TMAX = 0.1, 0.6  # must match compute_entropy.py's window
SUBJECT_IDS = ["10", "12", "13", "14", "15", "16", "19", "20", "21", "22", "23"]


def compute_amplitude_for_subject(subject_id):
    fif_path = DATA_EPOCH / f"sub-{subject_id}-epo.fif"
    if not fif_path.exists():
        return None

    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
    code_to_name = {v: k for k, v in epochs.event_id.items()}
    epochs_cropped = epochs.copy().crop(tmin=TMIN, tmax=TMAX)
    data = epochs_cropped.get_data(copy=True)  # (n_epochs, n_channels, n_times)

    rows = []
    for i, code in enumerate(epochs.events[:, 2]):
        name = code_to_name[code]
        if not name.startswith("Present"):
            continue
        awareness = "Seen" if "Seen" in name else "Unseen"

        # per-channel variance and peak-to-peak amplitude, averaged across channels
        var_per_channel = np.var(data[i], axis=1)
        ptp_per_channel = np.ptp(data[i], axis=1)

        rows.append({
            "subject": subject_id,
            "awareness": awareness,
            "variance_mean": np.mean(var_per_channel),
            "peak_to_peak_mean": np.mean(ptp_per_channel),
        })

    return pd.DataFrame(rows)


def main():
    all_dfs = []
    for subject_id in SUBJECT_IDS:
        df = compute_amplitude_for_subject(subject_id)
        if df is not None and len(df) > 0:
            all_dfs.append(df)

    all_df = pd.concat(all_dfs, ignore_index=True)
    out_csv = DATA_EPOCH / "amplitude_confound_check.csv"
    all_df.to_csv(out_csv, index=False)

    print("=== Amplitude/Variance check: Seen vs Unseen (same window as entropy analysis) ===\n")
    subj_means = all_df.groupby(["subject", "awareness"])[["variance_mean", "peak_to_peak_mean"]].mean().reset_index()

    from scipy.stats import wilcoxon

    for measure in ["variance_mean", "peak_to_peak_mean"]:
        pivot = subj_means.pivot(index="subject", columns="awareness", values=measure).dropna()
        seen_vals = pivot["Seen"].values
        unseen_vals = pivot["Unseen"].values
        diffs = seen_vals - unseen_vals
        stat, p = wilcoxon(seen_vals, unseen_vals)

        label = "Variance" if measure == "variance_mean" else "Peak-to-peak amplitude"
        print(f"--- {label} ---")
        print(f"  Mean (Seen): {seen_vals.mean():.6g}")
        print(f"  Mean (Unseen): {unseen_vals.mean():.6g}")
        print(f"  Subjects with Seen > Unseen: {(diffs > 0).sum()}/{len(diffs)}")
        print(f"  Wilcoxon p = {p:.5f}")
        print()

    print("INTERPRETATION GUIDE:")
    print("  If amplitude/variance is significantly HIGHER in Seen trials, that's a plausible")
    print("  partial explanation for lower LZ complexity in Seen trials (bigger, more")
    print("  stereotyped signal -> more compressible -> lower LZ), and should be reported")
    print("  as a caveat/alternative explanation alongside the complexity result.")
    print("  If amplitude does NOT differ significantly, the LZ complexity effect is less")
    print("  likely to be a simple amplitude artifact, strengthening the finding.")


if __name__ == "__main__":
    main()