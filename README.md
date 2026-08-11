# EEG Analysis of Conscious vs. Unconscious Visual Perception

Exploratory analysis of entropy and complexity in single-trial EEG during a near-threshold visual detection task, using a public dataset. Investigates whether consciously reported ("Seen") stimuli differ from unreported ("Unseen") stimuli of identical physical intensity in signal entropy and Lempel-Ziv complexity. Further analysis of whether that signature generalizes to obvious, suprathreshold (localizer) stimuli.

**Full write-up:** see `doc/EEG_Neuro_Report_1.docx` in this repository for complete methods, results, figures, and discussion.

## Summary of findings

- Lempel-Ziv complexity is significantly lower for Seen than Unseen trials in a 300-600ms post-stimulus window (Wilcoxon signed-rank, n=11 subjects, W=0, p=0.00098, FDR-corrected p=0.00195), with all 11 subjects showing the same direction of effect.
- This effect survives statistical control for trial-level signal amplitude.
- Permutation entropy shows no consistent difference between conditions in the same window.
- Extending the comparison to obvious, suprathreshold (localizer) stimuli shows a temporal pattern rather than a simple monotonic relationship: localizer stimuli show their own low-complexity signature earlier (0-400ms), while near-threshold reported stimuli show a distinct low-complexity signature later (300-600ms). Neither group differs significantly after 600ms.
- These results should be interpreted narrowly: this is a within-subject, trial-level comparison of a single sensory event reaching conscious report or not, and is distinct from the global consciousness-state comparisons (e.g. anesthesia, sleep) more commonly studied with these same complexity measures. See the full write-up for discussion of this distinction.

## Dataset

This project uses OpenNeuro dataset **ds005273**, "Neural representation of consciously seen and unseen information," from:

> Rodriguez-San Esteban, P. et al. Neural representation of consciously seen and unseen information. *Scientific Reports* (2025).

Dataset: https://openneuro.org/datasets/ds005273/versions/1.0.0

Raw and processed EEG data are **not included in this repository**. Raw data must be downloaded directly from OpenNeuro (see "Reproducing this analysis" below).

The dataset's public metadata did not document its raw EEG trigger codes. These were reconstructed using logic adapted from the original authors' own published preprocessing code from their GitHub repository (https://github.com/rodriguez-p/EEGConsciouslySeenUnseen) and cross-referenced against the raw trigger stream to confirm correct decoding before any trial-level analysis was performed.

## Repository contents

```
code/
  run_preprocessing.py         preprocessing pipeline: raw EEG -> cleaned, epoched .fif 
                                files

  build_trial_table.py         builds trial-level behavioral table from epochs; validates
                                against paper's published behavioral statistics

  compute_entropy.py           computes permutation entropy and LZ complexity per trial

  run_stats.py                  Wilcoxon signed-rank + FDR-corrected statistics on results

  check_confound_direct.py     tests whether the LZ complexity effect is explained by
                                trial-level signal amplitude (correlation + 
                                residualization)

  three_group_comparison.py    compares Localizer Seen vs. Experimental Seen vs.
                                Experimental Unseen in a single time window

  multi_window_comparison.py   repeats the three-group comparison across 8 time windows
                                to map when each group's complexity signature emerges

doc/
  EEG_Neuro_Report_1.docx  full write-up: background, methods, results, discussion
```

Raw data, intermediate epoch files (`.fif`), and figure images (`.png`) are excluded via `.gitignore`. Small derived result tables (trial-level and summary CSVs) are committed alongside the code that produced them.

## Reproducing this analysis

1. Download subject data from OpenNeuro (ds005273): for each subject, the BrainVision EEG triplet (`.vhdr`, `.eeg`, `.vmrk`), the `events.tsv` file, and the `*_space-CapTrak_electrodes.tsv` file.
2. Place each subject's files in `data/raw_data/S<id>/`.
3. Install dependencies: `pip install mne pyprep pandas antropy statsmodels scipy matplotlib`
4. Run the pipeline in order:
   ```
   python code/run_preprocessing.py       # edit the subject list at the bottom of the                           
                                            file first
   python code/build_trial_table.py       # validates against published statistical                           
                                            results
   python code/compute_entropy.py
   python code/run_stats.py
   python code/check_confound_direct.py
   python code/three_group_comparison.py
   python code/multi_window_comparison.py
   ```

Subjects 11, 24, 26, and 38 are excluded, consistent with the original authors' own exclusions. Subject 18 is additionally excluded here due to an atypically high false-alarm rate (see write-up, Section 3.4).

## Credits and licensing

- Event-code decoding logic in `run_preprocessing.py` is adapted from the original authors' publicly released preprocessing code accompanying the publication above.
- All other code (trial-table construction, entropy/complexity computation, statistical testing, confound analysis, multi-group and multi-window comparisons) was independently written for this project and is not part of the original authors' published analyses.
- The source dataset remains subject to its original OpenNeuro license terms; see the dataset section for details.

## Status

This is an exploratory, single-analysis project intended as a methods validation and initial probe into entropy/complexity signatures of conscious perception, not a peer-reviewed or definitive result. See the "Limitations" and "Future Directions" sections of the full write-up for planned extensions, including channel-resolved analysis, additional complexity measures, a larger subject sample (which may also help resolve whether narrower time windows can be analyzed stably, see Section 5.3 of the write-up), and a proposed follow-up using a surprise/prediction-error paradigm to further test the interpretation offered here.