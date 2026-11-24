"""Create high-flow proxy labels, NOT confirmed flood labels.
AI was used in implementation as requested by abel. But the Frozen inputs are never modified.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/derived/thresholds'
PAPER = 'https://www.idrimjournal.com/article/11709-determination-of-threshold-runoff-for-flood-early-warning-in-nepalese-rivers.pdf'
REPORT = 'https://dhm.gov.np/uploads/dhm/downloads/Situational_Report_on_Extreme_Precipitation_and_Flooding_Event_of_27-29_September_2024.pdf'


def thresholds(reference):
    return reference.groupby('location').river_discharge_m3s.quantile([.90, .95, .99]).unstack().rename(columns={.90:'q90', .95:'q95', .99:'q99'})


def label(frame, limits):
    result = frame.merge(limits, on='location', how='left', validate='many_to_one')
    if result[['q90', 'q95', 'q99']].isna().any().any():
        raise ValueError('Station has no training threshold')
    for p in (90, 95, 99):
        result[f'high_flow_p{p}'] = (result.river_discharge_m3s > result[f'q{p}']).astype('int8')
    result['confirmed_flood'] = pd.Series(pd.NA, index=result.index, dtype='Int8')
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {s: ROOT / f'data/frozen/{s}.csv' for s in ('train','test')}
    inputs = {s:pd.read_csv(p) for s,p in paths.items()}
    train, test = inputs['train'], inputs['test']
    for df in inputs.values():
        assert not df.duplicated(['date','location']).any()
        assert np.isfinite(df.river_discharge_m3s).all()
        assert (df.river_discharge_m3s >= 0).all()
    assert train.date.max() < test.date.min()
    limits = thresholds(train)
    limits.to_csv(OUT / 'station_thresholds.csv')
    summaries = []
    for split, frame in inputs.items():
        out = label(frame, limits)
        out.to_csv(OUT / f'{split}_high_flow.csv', index=False)
        counts = out.groupby('location').agg(rows=('date','size'), positives_p90=('high_flow_p90','sum'), positives_p95=('high_flow_p95','sum'), positives_p99=('high_flow_p99','sum'))
        counts['split'] = split
        summaries.append(counts.reset_index())
    pd.concat(summaries).to_csv(OUT / 'label_counts.csv', index=False)
    # Fold-specific limits prevent validation years influencing their own label definition.
    for cutoff, validation_year in [('2023-12-31',2024), ('2024-12-31',2025)]:
        ref = train[train.date <= cutoff]
        val = train[train.date.str.startswith(str(validation_year))]
        fold_limits = thresholds(ref)
        fold_limits.to_csv(OUT / f'thresholds_validate_{validation_year}.csv')
        for part, frame in [('train',ref),('validation',val)]:
            label(frame, fold_limits).to_csv(OUT / f'fold_{validation_year}_{part}.csv',index=False)
    historical = pd.DataFrame([
        ('Chatara','Koshi',5600,8150), ('Chisapani','Karnali',8200,10000), ('Kusum','West Rapti',1500,2000)
    ], columns=['location','river','historical_warning_m3s','historical_danger_m3s'])
    historical['source_year'] = 2013
    historical['source_url'] = PAPER
    historical['source_table'] = 'Table 5, PDF page 9'
    historical['approved_for_labelling'] = False
    historical['reason'] = 'Historical thresholds; station identity, units, time aggregation and applicability in 2023-2026 not verified'
    all_rows = pd.concat(inputs.values())
    maxima = all_rows.groupby('location').river_discharge_m3s.max().rename('dataset_max_m3s')
    historical = historical.merge(maxima,on='location')
    historical.to_csv(OUT / 'historical_threshold_candidates.csv',index=False)
    # The report gives period maxima: do NOT assert flooding on every date in the window.
    evidence = pd.DataFrame([
        ('Chatara','695',11.83), ('Devghat','450',13.62), ('Khokana','550.05',6.16)
    ],columns=['location','reported_station_id','reported_period_peak_gauge_m'])
    evidence['window_start'] = '2024-09-27'
    evidence['window_end'] = '2024-09-29'
    evidence['evidence_type'] = 'reported_flood_event_window_not_daily_label'
    evidence['source_url'] = REPORT
    evidence['source_section'] = 'Section 6 and Table 4, PDF page 6; overall report window used'
    evidence.to_csv(OUT / 'historical_event_windows.csv',index=False)
    event_rows=[]
    for e in evidence.to_dict('records'):
        rows=all_rows[(all_rows.location==e['location']) & all_rows.date.between(e['window_start'],e['window_end'])][['date','location','river_discharge_m3s']].copy()
        rows['evidence_type']=e['evidence_type']
        rows['source_url']=e['source_url']
        event_rows.append(rows)
    pd.concat(event_rows).to_csv(OUT / 'event_window_data_check.csv',index=False)
    config=json.loads((ROOT/'data/frozen/feature_config.json').read_text())
    metadata = {
        'label':'high_flow_p95', 'meaning':'Discharge strictly above station training-history 95th percentile; not a confirmed flood',
        'threshold_selection':'A priori coursework definition, not optimized against flood events or test scores',
        'quantile_method':'pandas linear interpolation', 'comparison':'>',
        'reference_start':train.date.min(), 'reference_end':train.date.max(),
        'test_used_to_fit_thresholds':False, 'confirmed_flood':'All missing: no defensible daily binary ground truth established',
        'feature_rule':'Use ONLY the original frozen feature lists; never include discharge, y, q90/q95/q99, labels or evidence columns as predictors',
        'input_sha256':{s:hashlib.sha256(p.read_bytes()).hexdigest() for s,p in paths.items()},
        'historical_source_accessed':'2026-11-16', 'pandas_version':pd.__version__
    }
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(pd.concat(summaries).to_string(index=False))
    print('Created:', OUT)

if __name__ == '__main__':
    main()
