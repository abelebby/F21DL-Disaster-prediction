# Experimental Setup — Baseline Models and Clustering

Linear regression, decision-tree regression and random-forest regression were compared using the two frozen feature sets, with and without discharge history. These models provided a linear reference, nonlinear decision rules and an ensemble of bootstrap-trained trees, respectively. All predicted log(1 + discharge) using the shared expanding-window validation folds. Linear-regression inputs were standardised within each training fold.

Tree depths of 5, 10 and unrestricted were tested with five samples per leaf. Forests used 100 trees, minimum leaf sizes of one or five, and all features at each split. Settings were selected by mean validation log-RMSE, then models were refitted on 2023–2025 and tested on January–August 2026. Log-RMSE measured error on the fitted scale; original-scale MAE measured average absolute error, while RMSE emphasised larger mistakes. Predictions were constrained to non-negative values and compared with persistence and station–month climatology.

K-means grouped training days using nine standardised weather and rainfall-history variables, excluding discharge and station indicators. We tested two to six clusters with ten initialisations per fit, selecting the highest silhouette score on a reproducible 2,000-row sample. Stochastic procedures used seed 42.

# Results — Baseline Models and Clustering

Table 1 shows that linear regression with discharge history achieved the lowest validation and test log-RMSE. Its test MAE was 2.128, compared with 2.159 for persistence, an improvement of approximately 1.5%. Without discharge history, random forest performed best among the trained models. However, climatology achieved lower log-RMSE than this forest in both validation and testing, despite higher test MAE and RMSE.

**Table 1. Regression results. Validation log-RMSE averages the two folds; test results cover January–August 2026. MAE and RMSE use the dataset’s stated m³/s units. Lower values indicate better performance.**

| Model | Discharge history | Validation log-RMSE | Test log-RMSE | Test MAE | Test RMSE |
|---|---|---:|---:|---:|---:|
| Linear regression | Yes | 0.143 | 0.128 | 2.128 | 14.169 |
| Decision tree | Yes | 0.169 | 0.156 | 2.712 | 16.421 |
| Random forest | Yes | 0.146 | 0.133 | 2.204 | 15.618 |
| Linear regression | No | 0.610 | 0.433 | 9.615 | 49.077 |
| Decision tree | No | 0.577 | 0.403 | 6.417 | 35.200 |
| Random forest | No | 0.492 | 0.336 | 4.366 | 19.843 |
| Persistence | Previous day | 0.153 | 0.135 | 2.159 | 14.914 |
| Climatology | Monthly averages | 0.402 | 0.323 | 5.702 | 27.515 |

Two clusters produced the highest silhouette score (0.302), compared with 0.247–0.282 for larger cluster counts. Table 2 shows contrasting wetter and drier profiles. These groups describe weather conditions, not confirmed flood categories.

**Table 2. Selected cluster characteristics: training station-day counts and mean weather values.**

| Profile | Count | Daily precipitation (mm) | Seven-day precipitation (mm) | Humidity (%) |
|---|---:|---:|---:|---:|
| Wetter | 3,554 | 11.66 | 79.15 | 84.40 |
| Drier | 7,336 | 0.68 | 5.93 | 63.65 |

# Discussion — Interpretation and Limitations

Greater complexity did not consistently improve discharge estimation. Linear regression’s advantage with discharge history, and its small gain over persistence, suggest that recent flow dominates predictive performance. Decision trees nevertheless offer interpretable rules for examining combinations of weather and previous flow associated with different discharge estimates, supporting exploratory analysis of river behaviour.

The clusters reveal contrasting weather profiles, although their separation is limited and stability across years remains untested. Correlated rainfall totals may also disproportionately influence grouping. Feature selection and model tuning used the same validation years, so validation scores are development estimates. Unverified discharge provenance and units, together with possible station-specific and peak-flow errors, limit real-world conclusions. Same-day weather inputs mean these models estimate current discharge rather than forecast future floods. Furthermore, the neural-network experiments use different targets and splits, preventing direct performance ranking. Future comparisons should align prediction horizons, targets and evaluation periods, supported by verified measurements.

# Contribution Appendix

Abel implemented and evaluated the regression baselines and K-means clustering, exported predictions and results, and prepared their interpretation. Generative AI assisted code development and report drafting.
