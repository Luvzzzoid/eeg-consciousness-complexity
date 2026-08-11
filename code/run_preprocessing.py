"""
Phase 1: Reproduce event decoding

Adapted from Rodriguez-San Esteban et al. preprocessing script.
Changes from original:
  - filenames matched to actual OpenNeuro BIDS naming (sub-XX_task-1_eeg.*)
  - montage read from per-subject CapTrak electrodes.tsv instead of easycap-M10.txt
  - fixed subject_id bug in the final loop (was calling preprocessing(id, block))
  - scoped to run one subject first for validation before scaling to all 33
  - added print statements + a montage sanity-check plot to validate it works
"""

import mne
from mne.preprocessing import EOGRegression
import numpy as np
import pandas as pd
from pathlib import Path

import pyprep
from pyprep.find_noisy_channels import NoisyChannels

# set up directories (relative to where you run this script from)
RAW_DATA = Path("../data/raw_data")
DATA_EPOCH = Path("../data/epochdata")
DATA_EPOCH.mkdir(parents=True, exist_ok=True)


def build_montage(subject_id):
    """Build a montage from the subject's own CapTrak electrode positions.
    Channels with n/a (non-numeric) x/y/z — typically the online reference electrode,
    here labeled FCz in the raw file — are skipped rather than passed as NaN, since MNE/pyprep
    cannot handle NaN positions. That channel gets renamed FCz->Iz downstream (per the
    original authors' convention) and is treated as a bad channel for interpolation."""
    elec_file = RAW_DATA / f"S{subject_id}" / f"sub-{subject_id}_space-CapTrak_electrodes.tsv"
    elec_df = pd.read_csv(elec_file, sep="\t")
    print(f"[montage] loaded {len(elec_df)} electrode rows for sub-{subject_id}")

    # coerce to numeric; any 'n/a' string becomes NaN, which can then filter out cleanly
    for col in ["x", "y", "z"]:
        elec_df[col] = pd.to_numeric(elec_df[col], errors="coerce")

    valid_rows = elec_df.dropna(subset=["x", "y", "z"])
    dropped = elec_df[~elec_df["name"].isin(valid_rows["name"])]["name"].tolist()
    if dropped:
        print(f"[montage] channels with no numeric position in electrodes.tsv, excluded from montage: {dropped}")

    ch_pos = {
        row["name"]: np.array([row["x"], row["y"], row["z"]])
        for _, row in valid_rows.iterrows()
    }
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    return montage


def preprocessing(subject_id, block_type):
    subj_dir = RAW_DATA / f"S{subject_id}"
    vhdr_path = subj_dir / f"sub-{subject_id}_task-1_eeg.vhdr"

    print(f"\n=== Subject {subject_id}, block_type={block_type} ===")
    print(f"[load] reading {vhdr_path}")

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, eog={"EOG1", "EOG2"})
    print(f"[load] channels: {raw.info['nchan']}, sfreq: {raw.info['sfreq']} Hz, "
          f"duration: {raw.times[-1]:.1f} sec")

    montage = build_montage(subject_id)

    # Note: original authors' script renamed FCz->Iz to fix a mislabeled channel during
    # their recording. OpenNeuro data doesn't have an FCz channel, so this doesn't
    # apply here - channels are already correctly labeled. Leaving this check in case
    # a future subject's file does have the issue.
    if "FCz" in raw.info["ch_names"]:
        print("[montage] FCz found — renaming to Iz per original authors' correction note")
        mne.rename_channels(raw.info, {"FCz": "Iz"})

    raw.set_montage(montage, on_missing="warn")

    # RANSAC (and other spatial methods downstream) cannot handle NaN channel positions.
    # If any channel has no valid position in the montage (this happened with 'Iz' for
    # this dataset), mark it as bad up front so it's excluded from RANSAC/interpolation
    # math, rather than crashing. It can still be interpolated back in afterward using
    # the other channels' positions, same as any other bad channel.
    montage_ch_names = set(montage.ch_names)
    no_pos_channels = [ch for ch, ch_type in zip(raw.info["ch_names"], raw.get_channel_types())
                        if ch_type == "eeg" and ch not in montage_ch_names]
    if no_pos_channels:
        print(f"[montage] channels with no position in montage (will be excluded from RANSAC, "
              f"marked bad for later interpolation): {no_pos_channels}")
        raw.info["bads"] = list(set(raw.info["bads"] + no_pos_channels))

    # quick sanity check: save a plot of sensor positions to visually confirm
    # this looks like a head
    fig = raw.plot_sensors(show=False)
    fig.savefig(f"../data/epochdata/sub-{subject_id}_montage_check.png")
    print(f"[montage] sanity check plot saved to data/epochdata/sub-{subject_id}_montage_check.png")

    # -- reduce EOG artifacts through regression --
    # set average reference on the object that'll keep being used, before filtering/fitting,
    # so both the fit step and the apply step see a consistently-referenced raw
    raw.set_eeg_reference("average")
    raw_eog = raw.copy()
    raw_eog.filter(0.3, None, picks="all")
    weights = EOGRegression().fit(raw_eog)
    raw = weights.apply(raw, copy=True)
    print("[preprocess] EOG regression applied")

    # -- notch filter for power line noise --
    # n_jobs limited (not -1) to avoid spawning too many parallel workers at once, which
    # multiplies peak memory use on machines with limited RAM 
    freqs = (50, 100, 150)
    raw_notch = raw.notch_filter(freqs=freqs, n_jobs=2, method="spectrum_fit", p_value=0.05, verbose=None)
    print("[preprocess] notch filter applied")
    del raw_eog  # no longer needed, free memory before the heavy RANSAC step

    # -- automatic bad channel detection --
    # RANSAC on the full 1000Hz, ~107min recording is extremely memory-heavy and can freeze
    # a laptop (happened to me twice). Bad-channel detection doesn't need full resolution or 
    # the full recording length, so you can run it on a downsampled, cropped copy just for 
    # this step. The actual epoching later still uses the full-resolution raw_interp (below), 
    # just with the same bad-channel list applied.
    print("[preprocess] running RANSAC bad channel detection on a lightweight copy (downsampled + cropped)...")
    raw_for_ransac = raw_notch.copy()

    # drop any channel with no valid montage position entirely - RANSAC's correlation math
    # cannot tolerate NaN positions even if the channel is just marked 'bad'
    if raw_for_ransac.info["bads"]:
        print(f"[preprocess] dropping channels with no position before RANSAC: {raw_for_ransac.info['bads']}")
        raw_for_ransac.drop_channels(raw_for_ransac.info["bads"])

    raw_for_ransac.resample(100)
    max_crop_sec = min(90, raw_for_ransac.times[-1])  # further reduced to 1.5 min
    raw_for_ransac.crop(tmin=0, tmax=max_crop_sec)
    print(f"[preprocess] RANSAC working on {max_crop_sec:.0f}s at {raw_for_ransac.info['sfreq']}Hz "
          f"({raw_for_ransac.info['nchan']} channels)")

    try:
        import psutil
        print(f"[memory] available before RANSAC: {psutil.virtual_memory().available / 1e9:.2f} GB")
    except ImportError:
        pass

    nd = NoisyChannels(raw_for_ransac)
    nd.find_bad_by_ransac(channel_wise=True)
    raw_notch.info["bads"] = nd.bad_by_ransac
    del raw_for_ransac
    print(f"[preprocess] bad channels found: {nd.bad_by_ransac}")

    # -- interpolate bad channels --
    # Important (root cause found via local testing, not guessing): raw.get_montage().ch_names
    # includes channels that were NEVER given a position (like 'Iz') - MNE keeps them in the
    # digitization list with position = [nan, nan, nan] rather than omitting them. Checking
    # "channel name not in montage.ch_names" is therefore always False for them, since they
    # are present in that list, just with NaN coordinates. The correct check is whether the
    # position values themselves are NaN.
    pos_dict = raw_notch.get_montage().get_positions()["ch_pos"]
    eeg_ch_names = [ch for ch, ch_type in zip(raw_notch.info["ch_names"], raw_notch.get_channel_types())
                     if ch_type == "eeg"]
    no_position_eeg_channels = [ch for ch in eeg_ch_names
                                 if ch in pos_dict and np.any(np.isnan(pos_dict[ch]))]

    if no_position_eeg_channels:
        print(f"[preprocess] dropping EEG channel(s) with no real position (cannot be used as "
              f"interpolation reference or interpolated themselves): {no_position_eeg_channels}")
        raw_notch.drop_channels(no_position_eeg_channels)

    print(f"[preprocess] interpolating {len(raw_notch.info['bads'])} bad channel(s): {raw_notch.info['bads']}")
    raw_interp = raw_notch.interpolate_bads(reset_bads=True)
    del raw_notch, nd

    # re-reference to average 
    raw_avg_ref = raw_interp.set_eeg_reference(ref_channels="average")
    del raw_interp
    raw = raw_avg_ref
    print("[preprocess] re-referenced to average")

    # extract events from annotations 
    events = mne.events_from_annotations(raw, event_id="auto", regexp="Stimulus/")
    print(f"[events] found {len(events[0])} stimulus events")

    def recode_gabor_events(input_events, block_type):
        """
        Recodes raw trigger stream into trial-level events anchored to Gabor onset (code=10).
        good_trial_types: 55=Present/Seen, 56=Absent/Unseen, 57=Absent/Seen, 59=Present/Unseen
        gabor_tilt: 74/75 = tilt orientation (only present on Present trials)
        block_type: 70=localizer, 71=experimental
        """
        good_trial_types = [55, 56, 57, 59]
        gabor_tilt = [74, 75]
        event_dict = {
            "Present/Seen/Left": 5574,
            "Absent/Unseen": 56,
            "Absent/Seen": 57,
            "Present/Unseen/Left": 5974,
            "Present/Seen/Right": 5575,
            "Present/Unseen/Right": 5975,
        }
        output_events = []
        trial_types_array = np.squeeze(input_events[:, 2])
        gabor_event_indexs = np.squeeze(np.argwhere(trial_types_array == 10))
        if gabor_event_indexs.ndim == 0:
            gabor_event_indexs = np.array([gabor_event_indexs])

        for i in range(len(gabor_event_indexs)):
            e = gabor_event_indexs[i]
            next_e = gabor_event_indexs[i + 1] if (i + 1) < len(gabor_event_indexs) else None
            time_stamp = input_events[e, 0]
            trials_events = trial_types_array[e + 1:next_e]
            bt = np.isin(trials_events, [block_type])
            if np.any(bt):
                mask1 = np.isin(trials_events, good_trial_types)
                mask2 = np.isin(trials_events, gabor_tilt)
                if np.any(mask1):
                    trial_type = trials_events[mask1][0]
                    if np.any(mask2):
                        tilt_type = trials_events[mask2][0]
                        event_type = 100 * trial_type + tilt_type
                    else:
                        event_type = trial_type
                    output_events.append((time_stamp, 0, event_type))

        return np.array(output_events), event_dict

    gabor_events, events_dict = recode_gabor_events(events[0], block_type)
    print(f"[recode] {len(gabor_events)} trials recoded for block_type={block_type}")

    if len(gabor_events) == 0:
        print(f"[WARNING] zero trials recoded for block_type={block_type} — something is wrong, stopping here.")
        return

    # -- build epochs --
    epochs = mne.Epochs(raw, events=gabor_events, event_id=events_dict,
                         tmin=-2.0, tmax=2.0, baseline=None, preload=True, on_missing="warn")
    epochs.apply_baseline(baseline=(-0.2, 0))
    epochs.resample(sfreq=256)
    print(f"[epochs] final epoch count: {len(epochs)}")
    print(epochs.event_id)

    # -- save --
    if block_type == 70:
        out_path = DATA_EPOCH / f"sub-{subject_id}-epo_localizer.fif"
    else:
        out_path = DATA_EPOCH / f"sub-{subject_id}-epo.fif"
    epochs.save(out_path, overwrite=True)
    print(f"[save] saved to {out_path}")


if __name__ == "__main__":
    # Validate on a small set of subjects (excluding 11, 24, 26, 38 - these were
    # excluded by the original authors themselves). Running sequentially, one at a time, 
    # is intentional - keeps peak memory use the same as the already-successful sub-10 
    # runs rather than compounding it.
    test_subjects = ["21", "22", "23"] # Edit subject list 
    block_types = [70, 71]  # 70 = localizer, 71 = experimental

    for subject_id in test_subjects:
        for block in block_types:
            preprocessing(subject_id, block)

    print("\n=== DONE. Check data/epochdata/ for output files and the montage_check.png plot. ===")