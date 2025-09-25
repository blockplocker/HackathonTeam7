"""Predictive Analyst

Generates machine-level risk indicators and explanatory features from the
processed datasets produced by data_pipeline.py.

Inputs (expected in ../processed or ./processed):
  - machine_aggregates.parquet (claims + counts)
  - claims_all.parquet (for recency / fault time distributions)
Optional:
  - machine_text_documents.parquet (for severity keyword enrichment)

Outputs (written to processed/):
  - machine_risk_scores.parquet
  - machine_risk_scores.csv

Scoring Heuristics (initial version):
  risk_base = normalized(claim_frequency_z)
  + weight_open_ratio * open_ratio
  + weight_unique_faults * unique_faults_norm
  + weight_recent_spike * recent_spike_flag
  + weight_severity * severity_signal

This is intentionally transparent and explainable; can be replaced later
with a statistical / ML model (e.g., gradient boosting) once labeled
outcomes (failures, downtime) are available.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import math

BASE_DIR = Path(__file__).parent.parent  # go up from pyHackathon
PROCESSED_DIR_CANDIDATES = [
    BASE_DIR / "processed",
    Path(__file__).parent / "processed",
]

def _resolve_processed_dir() -> Path:
    for p in PROCESSED_DIR_CANDIDATES:
        if p.exists():
            return p
    # fallback create local
    fallback = Path(__file__).parent / "processed"
    fallback.mkdir(exist_ok=True)
    return fallback

PROCESSED_DIR = _resolve_processed_dir()

@dataclass
class RiskConfig:
    recent_days: int = 30
    spike_window_days: int = 90
    min_claims_for_spike: int = 3
    weight_open_ratio: float = 0.8
    weight_unique_faults: float = 0.5
    weight_recent_spike: float = 1.0
    weight_severity: float = 0.7
    severity_keywords: Tuple[str, ...] = (
        "leak", "overheat", "pressure", "derate", "injector", "nox", "temperature", "brake"
    )

CONFIG = RiskConfig()

# ---------------- Data Loading ---------------- #

def load_inputs() -> Dict[str, pd.DataFrame]:
    paths = {
        'aggregates': PROCESSED_DIR / 'machine_aggregates.parquet',
        'claims': PROCESSED_DIR / 'claims_all.parquet',
        'docs': PROCESSED_DIR / 'machine_text_documents.parquet',
    }
    data = {}
    for k, p in paths.items():
        if p.exists():
            try:
                data[k] = pd.read_parquet(p)
            except Exception:
                # fallback to CSV
                alt_csv = p.with_suffix('.csv')
                if alt_csv.exists():
                    data[k] = pd.read_csv(alt_csv)
                else:
                    data[k] = pd.DataFrame()
        else:
            data[k] = pd.DataFrame()
    return data

# ---------------- Feature Engineering ---------------- #

def add_recency_features(claims: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    if claims.empty or 'chassis_id' not in claims.columns:
        return pd.DataFrame()
    df = claims.copy()
    if 'requested_date' in df.columns:
        df['requested_date'] = pd.to_datetime(df['requested_date'], errors='coerce')
    else:
        return pd.DataFrame()
    cutoff_recent = df['requested_date'].max() - pd.Timedelta(days=cfg.recent_days)
    recent = df[df['requested_date'] >= cutoff_recent]
    recent_counts = recent.groupby('chassis_id').size().reset_index(name='recent_claims')
    total_counts = df.groupby('chassis_id').size().reset_index(name='total_claims_full')
    merged = total_counts.merge(recent_counts, on='chassis_id', how='left')
    merged['recent_claims'] = merged['recent_claims'].fillna(0)
    merged['recent_claim_ratio'] = merged['recent_claims'] / merged['total_claims_full'].replace(0, np.nan)
    return merged


def add_spike_flag(claims: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    if claims.empty or 'chassis_id' not in claims.columns:
        return pd.DataFrame()
    df = claims.copy()
    df['requested_date'] = pd.to_datetime(df['requested_date'], errors='coerce')
    # restrict to last spike_window_days
    cutoff = df['requested_date'].max() - pd.Timedelta(days=cfg.spike_window_days)
    window_df = df[df['requested_date'] >= cutoff]
    grp = window_df.groupby('chassis_id').size().reset_index(name='claims_window')
    # baseline: all-time average per day
    overall_days = max(1, (df['requested_date'].max() - df['requested_date'].min()).days)
    overall_counts = df.groupby('chassis_id').size().reset_index(name='total_claims')
    overall_counts['avg_per_day'] = overall_counts['total_claims'] / overall_days
    # window rate per day
    window_days = max(1, (window_df['requested_date'].max() - window_df['requested_date'].min()).days)
    grp['window_per_day'] = grp['claims_window'] / window_days
    merged = overall_counts.merge(grp, on='chassis_id', how='left')
    merged['claims_window'] = merged['claims_window'].fillna(0)
    merged['window_per_day'] = merged['window_per_day'].fillna(0)
    # spike if window rate > 2x historical AND minimum count threshold
    merged['recent_spike_flag'] = ((merged['window_per_day'] > 2 * merged['avg_per_day']) & (merged['claims_window'] >= cfg.min_claims_for_spike)).astype(int)
    return merged[['chassis_id', 'recent_spike_flag']]


def add_severity_signal(docs: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    if docs.empty or 'text' not in docs.columns:
        return pd.DataFrame()
    d = docs.copy()
    d['text_lower'] = d['text'].astype(str).str.lower()
    for kw in cfg.severity_keywords:
        d[f'kw_{kw}'] = d['text_lower'].str.contains(kw, na=False)
    agg_specs = {f'kw_{kw}': 'sum' for kw in cfg.severity_keywords}
    agg = d.groupby('chassis_id').agg(agg_specs).reset_index()
    # simple severity score = log1p(sum of keyword hits)
    kw_cols = [c for c in agg.columns if c.startswith('kw_')]
    agg['severity_raw_hits'] = agg[kw_cols].sum(axis=1)
    agg['severity_signal'] = np.log1p(agg['severity_raw_hits'])
    return agg[['chassis_id', 'severity_signal', 'severity_raw_hits']]


def assemble_features(data: Dict[str, pd.DataFrame], cfg: RiskConfig) -> pd.DataFrame:
    aggregates = data.get('aggregates', pd.DataFrame()).copy()
    claims = data.get('claims', pd.DataFrame()).copy()
    docs = data.get('docs', pd.DataFrame()).copy()
    if aggregates.empty:
        return pd.DataFrame()
    # rename for consistency
    if 'unique_fault_codes' not in aggregates.columns and 'unique_fault_codes' in aggregates.columns:
        pass
    feats = aggregates.copy()
    # recency
    rec = add_recency_features(claims, cfg)
    if not rec.empty:
        feats = feats.merge(rec, on='chassis_id', how='left')
    # spike
    spike = add_spike_flag(claims, cfg)
    if not spike.empty:
        feats = feats.merge(spike, on='chassis_id', how='left')
    # severity
    sev = add_severity_signal(docs, cfg)
    if not sev.empty:
        feats = feats.merge(sev, on='chassis_id', how='left')
    return feats

# ---------------- Scoring ---------------- #

def zscore(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    return (series - series.mean()) / (series.std(ddof=0) + 1e-9)


def score_features(df: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    # Establish base frequency metric
    if 'claims_total' not in out.columns and 'claims_total' in out.columns:
        pass
    # Some pipeline used 'claims_total'; verify
    freq_col = 'claims_total' if 'claims_total' in out.columns else ('claims_total_full' if 'claims_total_full' in out.columns else None)
    if freq_col:
        out['claim_freq_z'] = zscore(out[freq_col].fillna(0))
    else:
        out['claim_freq_z'] = 0
    # Unique fault normalization
    if 'unique_fault_codes' in out.columns:
        out['unique_faults_norm'] = out['unique_fault_codes'] / (out['unique_fault_codes'].max() or 1)
    else:
        out['unique_faults_norm'] = 0
    out['open_ratio'] = 0
    open_cols = [c for c in out.columns if c.startswith('state_') and 'closed' not in c]
    closed_cols = [c for c in out.columns if c.startswith('state_') and 'closed' in c]
    if open_cols:
        out['open_sum'] = out[open_cols].sum(axis=1)
        out['closed_sum'] = out[closed_cols].sum(axis=1) if closed_cols else 0
        denom = out['open_sum'] + out['closed_sum']
        out['open_ratio'] = out['open_sum'] / denom.replace(0, np.nan)
    out['recent_spike_flag'] = out.get('recent_spike_flag', 0).fillna(0)
    out['severity_signal'] = out.get('severity_signal', 0).fillna(0)
    # Risk score
    out['risk_score_raw'] = (
        out['claim_freq_z'].clip(lower=0) +
        cfg.weight_open_ratio * out['open_ratio'].fillna(0) +
        cfg.weight_unique_faults * out['unique_faults_norm'].fillna(0) +
        cfg.weight_recent_spike * out['recent_spike_flag'] +
        cfg.weight_severity * out['severity_signal']
    )
    # Normalize 0-100
    max_raw = out['risk_score_raw'].max() or 1
    out['risk_score'] = (out['risk_score_raw'] / max_raw * 100).round(2)
    # Explanation column
    def explain(row):
        reasons = []
        if row['claim_freq_z'] > 1: reasons.append('High claim frequency')
        if row['open_ratio'] > 0.3: reasons.append('Many open/non-closed cases')
        if row['recent_spike_flag'] == 1: reasons.append('Recent spike')
        if row['severity_signal'] > 0: reasons.append('Severity keywords present')
        if row['unique_faults_norm'] > 0.5: reasons.append('Broad fault diversity')
        return '; '.join(reasons) if reasons else 'Stable'
    out['risk_explanation'] = out.apply(explain, axis=1)
    return out

# ---------------- Public API ---------------- #

def generate_machine_risk(config: RiskConfig = CONFIG) -> pd.DataFrame:
    data = load_inputs()
    feats = assemble_features(data, config)
    scored = score_features(feats, config)
    return scored

def save_outputs(df: pd.DataFrame):
    if df.empty:
        print("No data to save.")
        return
    df_out = df.sort_values('risk_score', ascending=False)
    df_out.to_parquet(PROCESSED_DIR / 'machine_risk_scores.parquet', index=False)
    df_out.to_csv(PROCESSED_DIR / 'machine_risk_scores.csv', index=False)
    print(f"Saved risk scores to {PROCESSED_DIR}")

# ---------------- CLI Entry ---------------- #

if __name__ == '__main__':
    print(f"Using processed dir: {PROCESSED_DIR}")
    result = generate_machine_risk()
    save_outputs(result)
    # Preview
    if not result.empty:
        print(result[['chassis_id','risk_score','risk_explanation']].head(15).to_string(index=False))
