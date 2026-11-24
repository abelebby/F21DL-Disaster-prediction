# High-flow labels and historical flood evidence

This is a usable **high-flow classification dataset**, not verified daily flood ground truth. No reliable flood/no-flood threshold can currently be calibrated from the available evidence. Dataset discharge provenance, units and station correspondence remain unverified.

## Main files

- `train_high_flow.csv`: 10,890 frozen training rows with added labels.
- `test_high_flow.csv`: 2,430 frozen test rows with added labels.
- `station_thresholds.csv`: station-specific 90th, 95th and 99th percentiles from frozen training rows (2023-01-08 through 2025-12-31) only.
- `fold_2024_train.csv`, `fold_2024_validation.csv` and the analogous 2025 files: safe development splits; each fold's thresholds use only its earlier training years.
- `label_counts.csv`: positive counts by station and split.
- `historical_threshold_candidates.csv`: three candidate matches to Gautam & Dulal (2013), Table 5. NOT approved for labelling.
- `historical_event_windows.csv`: three station references from the SAME September 2024 event, not three independent flood events. A report-window match does not establish flooding on every day.
- `event_window_data_check.csv`: our existing measurements in those windows, for investigation only.
- `metadata.json`: method, input hashes, provenance and restrictions.

## Label meaning

`high_flow_p95 = 1` means recorded discharge is strictly greater than that station's training 95th percentile. Zero means below/equal to this statistical cutoff, NOT confirmed no flood. This is a predefined coursework proxy, not a threshold optimized or validated against observed flooding. p90 and p99 are sensitivity alternatives with different target definitions, not competing flood detectors with shared ground truth. Ties can make the training positive proportion differ from exactly 5%.

`confirmed_flood` is blank throughout. Never fill these blanks with zero. The event report does not provide a sufficiently complete daily positive/negative inventory to train a verified flood classifier or choose a best flood threshold.

## Use with models

Use ONLY the original predictor lists in `data/frozen/feature_config.json`; use `high_flow_p95` as the new target. Exclude `river_discharge_m3s`, `y`, all `q*` thresholds, all `high_flow_*` labels, `confirmed_flood` and evidence fields from predictor inputs. These contain the answer or label-definition information.

For tuning, use the fold-specific exports, not the full-training labels: full-training thresholds include later validation years. Choose model settings on validation. Test data has already been viewed in earlier regression work, so this extension is not a completely untouched confirmatory experiment. Do not pick p90/p95/p99 based on test performance. Report positive counts, precision, recall and F1, and account for station splits with no positives. No classification models have been trained by this script.

A regression output can also be compared with the same station threshold, but scoring it against these labels measures high-flow agreement, not confirmed flood detection. Current inputs include same-day weather; this remains same-day estimation/detection, not next-day forecasting.

## Historical source findings

Gautam, D.K. and Dulal, K. (2013), *Determination of Threshold Runoff for Flood Warning in Nepalese Rivers*, DOI 10.5595/idrim.2013.0061, Table 5 (PDF page 9):
https://www.idrimjournal.com/article/11709-determination-of-threshold-runoff-for-flood-early-warning-in-nepalese-rivers.pdf

Chatara warning/danger: 5,600/8,150 m3/s; Chisapani: 8,200/10,000; Kusum: 1,500/2,000. The paper recommends reassessment. We do not transfer Narayanghat thresholds to Devghat. Matching names alone does not establish matching gauges or measurement definitions. Current dataset maxima at Chatara and Chisapani are only 23.27 and 16.99 m3/s respectively. This large scale mismatch needs investigation, not an arbitrary conversion. Daily values also need not represent instantaneous event peaks.

DHM (8 October 2024), *Situational Report on Extreme Precipitation and Flooding Event of 27–29 September 2024*, Section 6 and Table 4 (PDF page 6):
https://dhm.gov.np/uploads/dhm/downloads/Situational_Report_on_Extreme_Precipitation_and_Flooding_Event_of_27-29_September_2024.pdf

The report supplies period peak gauge heights for Chatara, Devghat and Khokana. We retain the report's overall event window without inventing daily flood labels. Gauge heights are metres, not discharge in m3/s. This is a small manually curated evidence sample, not an exhaustive flood history.

## Reproduce

From the repository root: `python src/build_threshold_dataset.py` (requires pandas and numpy). It overwrites only generated files in this directory and leaves frozen data unchanged. Do not change the frozen data to force agreement with external thresholds.
