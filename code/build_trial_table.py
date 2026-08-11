"""
Phase 2: Confirm trial table reproduces study's published results

Reads the saved epoch .fif files for a subject and builds a trial-level table +
summary statistics, for comparison against the published paper's reported numbers.

Run this after Phase 1 (run_preprocessing.py) has produced sub-XX-epo.fif and
sub-XX-epo_localizer.fif for a subject.
"""

import mne
import pandas as pd
from pathlib import Path

DATA_EPOCH = Path("../data/epochdata")


def summarize_epochs(subject_id, block_label, fif_path):
    if not fif_path.exists():
        print(f"[skip] {fif_path} not found")
        return None

    epochs = mne.read_epochs(fif_path, preload=False, verbose=False)

    rows = []
    # epochs.events[:,2] holds the recoded event code; epochs.event_id maps name->code
    code_to_name = {v: k for k, v in epochs.event_id.items()}

    for code in epochs.events[:, 2]:
        name = code_to_name[code]
        # parse the human-readable label back into presence/awareness/tilt fields
        present = "Present" if name.startswith("Present") else "Absent"
        seen = "Seen" if "Seen" in name else "Unseen"
        tilt = "Left" if "Left" in name else ("Right" if "Right" in name else None)

        rows.append({
            "subject": subject_id,
            "block": block_label,
            "event_name": name,
            "target_presence": present,
            "awareness": seen,
            "tilt": tilt,
        })

    df = pd.DataFrame(rows)
    return df


def main():
    subject_ids = ["10", "12", "13", "14", "15", "16", "18", "19", "20", "21", "22", "23"]  # update after each batch

    all_dfs = []
    for subject_id in subject_ids:
        exp_df = summarize_epochs(subject_id, "experimental", DATA_EPOCH / f"sub-{subject_id}-epo.fif")
        loc_df = summarize_epochs(subject_id, "localizer", DATA_EPOCH / f"sub-{subject_id}-epo_localizer.fif")
        for d in [exp_df, loc_df]:
            if d is not None:
                all_dfs.append(d)

    all_df = pd.concat(all_dfs, ignore_index=True)

    out_csv = DATA_EPOCH / "all_subjects_trial_table.csv"
    all_df.to_csv(out_csv, index=False)
    print(f"[save] trial table saved to {out_csv}")
    print(f"[save] {len(all_df)} total trials across {len(subject_ids)} subjects")

    # summary stats to compare against the paper, per subject 
    print("\n=== SUMMARY STATS PER SUBJECT (compare against paper's reported values) ===")
    for subject_id in subject_ids:
        subj_df = all_df[all_df["subject"] == subject_id]
        if len(subj_df) == 0:
            print(f"\n--- subject {subject_id}: NO DATA FOUND ---")
            continue
        print(f"\n--- subject {subject_id} ---")
        for block_label in ["localizer", "experimental"]:
            block_df = subj_df[subj_df["block"] == block_label]
            if len(block_df) == 0:
                continue
            for presence in ["Present", "Absent"]:
                sub = block_df[block_df["target_presence"] == presence]
                if len(sub) == 0:
                    continue
                n_seen = (sub["awareness"] == "Seen").sum()
                n_unseen = (sub["awareness"] == "Unseen").sum()
                pct_seen = n_seen / len(sub) * 100
                print(f"  {block_label} / {presence}: n={len(sub)} total "
                      f"(Seen={n_seen}, Unseen={n_unseen}), % Seen = {pct_seen:.1f}%")

    print("\n--- Expected from paper (Rodriguez-San Esteban et al.), sample averages ---")
    print("  Localizer, Present:     ~97.4% seen")
    print("  Localizer, Absent:      ~1.9% seen (false alarm rate)")
    print("  Experimental, Present:  ~49% seen (individual subjects vary around this)")
    print("  Experimental, Absent:   ~7.2% seen (false alarm rate)")


if __name__ == "__main__":
    main()