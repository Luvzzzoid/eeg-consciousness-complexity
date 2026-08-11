"""
Phase 3a: Entropy / complexity analysis.

Computes permutation entropy and Lempel-Ziv complexity per trial, per channel,
for experimental-block Present trials, split by Seen vs Unseen, using a
literature-justified post-stimulus window (100-600ms).

Reads the epoch .fif files already produced by run_preprocessing.py - 
no raw EEG reprocessing needed. 
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
from antropy import perm_entropy, lziv_complexity

DATA_EPOCH = Path("../data/epochdata")

# Time window: 100-600ms post-stimulus. Chosen before looking at any entropy results:
# avoids the earliest (<100ms) pure feedforward visual response, which is less likely
# to differ by awareness, and stays well before response-related motor preparation later
# in the epoch. Covers the window where the original paper found sustained awareness-
# related decoding.
TMIN, TMAX = 0.1, 0.6

# Subjects to include. Excludes subject 18 (abnormally high false-alarm rate, ~31% vs
# ~0-13% for all others) and known-bad subjects per the original authors (11, 24, 26, 38).
SUBJECT_IDS = ["10", "12", "13", "14", "15", "16", "19", "20", "21", "22", "23"]


def compute_entropy_for_subject(subject_id):
    fif_path = DATA_EPOCH / f"sub-{subject_id}-epo.fif"
    if not fif_path.exists():
        print(f"[skip] {fif_path} not found")
        return None

    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
    code_to_name = {v: k for k, v in epochs.event_id.items()}

    # crop to justified time window
    epochs_cropped = epochs.copy().crop(tmin=TMIN, tmax=TMAX)
    data = epochs_cropped.get_data(copy=True)  # shape: (n_epochs, n_channels, n_times)
    ch_names = epochs_cropped.info["ch_names"]

    rows = []
    for i, code in enumerate(epochs.events[:, 2]):
        name = code_to_name[code]
        if not name.startswith("Present"):
            continue  # only Present trials - Seen vs Unseen comparison requires stimulus present
        awareness = "Seen" if "Seen" in name else "Unseen"

        # compute entropy/complexity per channel, then average across channels for a
        # single trial-level summary value (simplest starting approach - per-channel
        # and per-region breakdowns can come later once this baseline works)
        pe_per_channel = []
        lz_per_channel = []
        for ch_idx in range(data.shape[1]):
            trial_signal = data[i, ch_idx, :]
            pe = perm_entropy(trial_signal, order=3, delay=1, normalize=True)
            binarized = (trial_signal > np.median(trial_signal)).astype(int)
            lz = lziv_complexity(binarized, normalize=True)
            pe_per_channel.append(pe)
            lz_per_channel.append(lz)

        rows.append({
            "subject": subject_id,
            "trial_index": i,
            "event_name": name,
            "awareness": awareness,
            "perm_entropy_mean": np.mean(pe_per_channel),
            "lz_complexity_mean": np.mean(lz_per_channel),
        })

    return pd.DataFrame(rows)


def main():
    all_dfs = []
    for subject_id in SUBJECT_IDS:
        print(f"[processing] subject {subject_id}...")
        df = compute_entropy_for_subject(subject_id)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            print(f"  -> {len(df)} Present trials processed")

    all_df = pd.concat(all_dfs, ignore_index=True)
    out_csv = DATA_EPOCH / "entropy_complexity_trial_level.csv"
    all_df.to_csv(out_csv, index=False)
    print(f"\n[save] {len(all_df)} total trials saved to {out_csv}")

    # summary: Seen vs Unseen, per subject and overall 
    print("\n=== Permutation Entropy: Seen vs Unseen (per subject) ===")
    for subject_id in SUBJECT_IDS:
        sub_df = all_df[all_df["subject"] == subject_id]
        if len(sub_df) == 0:
            continue
        seen_pe = sub_df[sub_df["awareness"] == "Seen"]["perm_entropy_mean"]
        unseen_pe = sub_df[sub_df["awareness"] == "Unseen"]["perm_entropy_mean"]
        if len(seen_pe) == 0 or len(unseen_pe) == 0:
            continue
        print(f"  subject {subject_id}: Seen mean={seen_pe.mean():.4f} (n={len(seen_pe)}), "
              f"Unseen mean={unseen_pe.mean():.4f} (n={len(unseen_pe)}), "
              f"diff={seen_pe.mean()-unseen_pe.mean():+.4f}")

    print("\n=== Lempel-Ziv Complexity: Seen vs Unseen (per subject) ===")
    for subject_id in SUBJECT_IDS:
        sub_df = all_df[all_df["subject"] == subject_id]
        if len(sub_df) == 0:
            continue
        seen_lz = sub_df[sub_df["awareness"] == "Seen"]["lz_complexity_mean"]
        unseen_lz = sub_df[sub_df["awareness"] == "Unseen"]["lz_complexity_mean"]
        if len(seen_lz) == 0 or len(unseen_lz) == 0:
            continue
        print(f"  subject {subject_id}: Seen mean={seen_lz.mean():.4f} (n={len(seen_lz)}), "
              f"Unseen mean={unseen_lz.mean():.4f} (n={len(unseen_lz)}), "
              f"diff={seen_lz.mean()-unseen_lz.mean():+.4f}")


if __name__ == "__main__":
    main()