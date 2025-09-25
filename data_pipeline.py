"""Data integration and feature engineering pipeline for machine datasets.

Datasets expected in Data/ directory:
- ActiveCare Current Claims.csv
- ActiveCare Historical.csv
- Matris Log Data.csv
- TechSupport Data.csv

Primary key: chassis_id (some files use this explicitly). If a row lacks
chassis_id it will not contribute to machine-level aggregates, but its
text can still be embedded separately (tagged with a surrogate id).

Outputs:
- processed/claims_all.parquet (row-level cleaned claims)
- processed/machine_aggregates.parquet (aggregated numeric + categorical summaries)
- processed/machine_text_documents.parquet (documents for embedding / RAG)
- Optional: processed/integrated.duckdb (if duckdb installed)

Run: python data_pipeline.py
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple

DATA_DIR = Path(__file__).parent / "Hackathon_team7" / "Data" if (Path(__file__).parent / "Hackathon_team7" / "Data").exists() else Path(__file__).parent / "Data"
OUTPUT_DIR = Path(__file__).parent / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

FILES = {
    "claims_current": "ActiveCare Current Claims.csv",
    "claims_historical": "ActiveCare Historical.csv",
    "matris_logs": "Matris Log Data.csv",
    "tech_support": "TechSupport Data.csv",
}

SENTINEL_DATES = {"1900-01-01", "1899-12-30"}

# ---------------------- Loading Utilities ---------------------- #

def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"WARNING: Missing file {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    return df


def load_raw() -> Dict[str, pd.DataFrame]:
    raw = {}
    for key, fname in FILES.items():
        df = _read_csv_safe(DATA_DIR / fname)
        raw[key] = df
        print(f"Loaded {key:<16} rows={len(df):>6} cols={df.shape[1] if not df.empty else 0}")
    return raw

# ---------------------- Cleaning Functions ---------------------- #

def _clean_dates(df: pd.DataFrame, candidate_cols: List[str]) -> pd.DataFrame:
    for c in candidate_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            # Nullify sentinel baseline dates
            df.loc[df[c].dt.strftime("%Y-%m-%d").isin(SENTINEL_DATES), c] = pd.NaT
    return df


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})
    return df

# ---------------------- Claims Processing ---------------------- #

def prepare_claims(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _strip_strings(df)
    df = _clean_dates(df, ["requested_date"])
    # Standardize column names to lower snake
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "chassis_id" not in df.columns and "chassisid" in df.columns:
        df = df.rename(columns={"chassisid": "chassis_id"})
    # Machine hours to numeric
    if "machine_hours" in df.columns:
        df["machine_hours"] = pd.to_numeric(df["machine_hours"], errors="coerce")
    return df


def aggregate_claims(claims_all: pd.DataFrame) -> pd.DataFrame:
    if claims_all.empty or "chassis_id" not in claims_all.columns:
        return pd.DataFrame()
    grp = claims_all.groupby("chassis_id")
    agg = grp.agg(
        claims_total=("chassis_id", "count"),
        first_claim_date=("requested_date", "min"),
        last_claim_date=("requested_date", "max"),
        unique_fault_codes=("fault_code", lambda x: len(set([v for v in x if pd.notna(v)]))),
        avg_machine_hours=("machine_hours", "mean"),
        max_machine_hours=("machine_hours", "max"),
    ).reset_index()

    # Open / closed distribution if state present
    if "state" in claims_all.columns:
        state_counts = claims_all.pivot_table(index="chassis_id", columns="state", values="ac_key" if "ac_key" in claims_all.columns else "fault_code", aggfunc="count", fill_value=0)
        state_counts.columns = [f"state_{str(c).lower().replace(' ', '_')}" for c in state_counts.columns]
        agg = agg.merge(state_counts.reset_index(), on="chassis_id", how="left")
    return agg

# ---------------------- Tech Support Processing ---------------------- #

def prepare_tech_support(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _strip_strings(df)
    df = _clean_dates(df, ["requested_date"])
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Parse fault_codes JSON-like field
    if "fault_codes" in df.columns:
        def parse_codes(val):
            try:
                return json.loads(val) if isinstance(val, str) and val.startswith("[") else []
            except Exception:
                return []
        df["fault_codes_list"] = df["fault_codes"].apply(parse_codes)
        df["fault_codes_count"] = df["fault_codes_list"].apply(len)
    # Derive surrogate chassis id if missing
    if "chassis_id" not in df.columns:
        # create stable surrogate by hashing combination of model + first 8 chars complaint + date
        def surrogate(row):
            raw = f"{row.get('machine_salesmodel','')}|{row.get('requested_date','')}|{row.get('case_ts_number','')}"
            return "TS_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        df["surrogate_chassis_id"] = df.apply(surrogate, axis=1)
    return df

# ---------------------- Matris Logs Processing ---------------------- #

def prepare_matris(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _strip_strings(df)
    # Attempt typical date columns
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()][:3]
    df = _clean_dates(df, date_cols)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "chassis_id" not in df.columns:
        # Try alternative naming
        for alt in ["chassisid", "machine_id", "machineid", "unit_number"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "chassis_id"})
                break
    return df

# ---------------------- Text Document Generation ---------------------- #

def build_text_documents(claims: pd.DataFrame, tech: pd.DataFrame, matris: pd.DataFrame) -> pd.DataFrame:
    docs: List[Dict[str, str]] = []
    def add_doc(chassis_id: str, source: str, when: datetime | None, text: str):
        if not text or str(text).strip() == "":
            return
        docs.append({
            "chassis_id": chassis_id,
            "source": source,
            "timestamp": when.isoformat() if isinstance(when, pd.Timestamp) else (when if isinstance(when, str) else None),
            "text": str(text)[:4000]  # cap length
        })

    if not claims.empty:
        for _, r in claims.iterrows():
            add_doc(r.get("chassis_id"), "claim", r.get("requested_date"), f"Fault {r.get('fault_code')}: {r.get('description')}")
    if not tech.empty:
        chassis_col = "chassis_id" if "chassis_id" in tech.columns else "surrogate_chassis_id"
        for _, r in tech.iterrows():
            snippet = r.get("complaint") or r.get("description") or r.get("cause")
            add_doc(r.get(chassis_col), "tech_support", r.get("requested_date"), snippet)
    if not matris.empty and "chassis_id" in matris.columns:
        text_candidates = [c for c in matris.columns if any(tok in c for tok in ["desc", "message", "fault", "event"])]
        for _, r in matris.iterrows():
            text_bits = [str(r.get(c)) for c in text_candidates if pd.notna(r.get(c))][:3]
            if text_bits:
                add_doc(r.get("chassis_id"), "matris_log", None, " | ".join(text_bits))

    return pd.DataFrame(docs)

# ---------------------- Integration ---------------------- #

def integrate(raw: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    claims_cur = prepare_claims(raw.get("claims_current", pd.DataFrame()))
    claims_hist = prepare_claims(raw.get("claims_historical", pd.DataFrame()))
    claims_all = pd.concat([df for df in [claims_cur, claims_hist] if not df.empty], ignore_index=True)

    tech = prepare_tech_support(raw.get("tech_support", pd.DataFrame()))
    matris = prepare_matris(raw.get("matris_logs", pd.DataFrame()))

    claims_agg = aggregate_claims(claims_all)

    # Merge other aggregated signals if they have chassis_id
    if not matris.empty and "chassis_id" in matris.columns:
        matris_counts = matris.groupby("chassis_id").size().reset_index(name="matris_events")
        claims_agg = claims_agg.merge(matris_counts, on="chassis_id", how="left")
    if not tech.empty and "chassis_id" in tech.columns:
        tech_counts = tech.groupby("chassis_id").size().reset_index(name="tech_tickets")
        claims_agg = claims_agg.merge(tech_counts, on="chassis_id", how="left")

    text_docs = build_text_documents(claims_all, tech, matris)
    return claims_all, claims_agg, text_docs, tech

# ---------------------- Persistence ---------------------- #

def persist(claims_all: pd.DataFrame, claims_agg: pd.DataFrame, text_docs: pd.DataFrame, tech: pd.DataFrame):
    claims_all.to_parquet(OUTPUT_DIR / "claims_all.parquet", index=False)
    claims_agg.to_parquet(OUTPUT_DIR / "machine_aggregates.parquet", index=False)
    text_docs.to_parquet(OUTPUT_DIR / "machine_text_documents.parquet", index=False)
    tech.to_parquet(OUTPUT_DIR / "tech_support_clean.parquet", index=False)

    # Also CSV for quick inspection
    claims_agg.to_csv(OUTPUT_DIR / "machine_aggregates.csv", index=False)
    text_docs.to_csv(OUTPUT_DIR / "machine_text_documents.csv", index=False)

    try:
        import duckdb  # type: ignore
        con = duckdb.connect(str(OUTPUT_DIR / "integrated.duckdb"))
        con.register("claims_all_df", claims_all)
        con.register("claims_agg_df", claims_agg)
        con.register("text_docs_df", text_docs)
        con.execute("CREATE OR REPLACE TABLE claims_all AS SELECT * FROM claims_all_df")
        con.execute("CREATE OR REPLACE TABLE machine_aggregates AS SELECT * FROM claims_agg_df")
        con.execute("CREATE OR REPLACE TABLE machine_text_documents AS SELECT * FROM text_docs_df")
        con.close()
        print("DuckDB file created: processed/integrated.duckdb")
    except ImportError:
        print("duckdb not installed; skipping database export (pip install duckdb).")

# ---------------------- Entry Point ---------------------- #

def run_pipeline():
    print(f"Data dir resolved to: {DATA_DIR}")
    raw = load_raw()
    claims_all, claims_agg, text_docs, tech = integrate(raw)
    print(f"Claims rows: {len(claims_all)} | Machines aggregated: {len(claims_agg)} | Text docs: {len(text_docs)}")
    persist(claims_all, claims_agg, text_docs, tech)
    # Basic correlation preview
    if not claims_agg.empty:
        num = claims_agg.select_dtypes(include=[np.number])
        if not num.empty:
            corr = num.corr().round(3)
            corr.to_csv(OUTPUT_DIR / "machine_correlations.csv")
            print("Correlation matrix saved -> machine_correlations.csv")
    print("Pipeline complete.")

if __name__ == "__main__":
    run_pipeline()
