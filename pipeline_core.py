"""
pipeline_core.py
────────────────
All pipeline logic extracted from AI_AND_DS_PROJECT_FINAL_ISH.ipynb.
Matplotlib is set to a non-interactive backend so it works headlessly.
"""

import logging
import os
import random
import subprocess
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")          # Must be set BEFORE importing pyplot
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import requests

from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
TIER1_THRESHOLD = 0.7
TIER2_THRESHOLD = 0.5

COMPOSITE_WEIGHTS = {
    "druggability_proba":  0.40,
    "conservation_score":  0.25,
    "has_structure_bonus": 0.15,
    "length_druggability": 0.10,
    "has_function_bonus":  0.10,
}

MC_N_SAMPLES    = 1_000
MC_DIRICHLET_A  = 10.0

FEATURE_COLS = [
    "length", "molecular_weight_kDa", "pI", "hydropathy_index",
    "aromaticity", "instability_index", "conservation_score",
    "is_membrane", "is_cytoplasmic", "has_gene_name", "length_druggability",
]
BOOL_COLS = ["is_membrane", "is_cytoplasmic", "has_gene_name"]

random.seed(42)
np.random.seed(42)


# ── 1-3. Data Collection, Processing & Cleaning ──────────────────────────────

def fetch_listeria_proteome() -> list:
    params = {
        "query":  "organism_id:169963 AND reviewed:true",
        "format": "json",
        "size":   500,
        "fields": (
            "accession,id,gene_names,protein_name,length,sequence,"
            "cc_function,cc_subcellular_location,xref_pdb,xref_drugbank"
        ),
    }
    log.info("Fetching L. monocytogenes proteome from UniProt...")
    r = requests.get("https://rest.uniprot.org/uniprotkb/search", params=params, timeout=60)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        raise RuntimeError("UniProt returned 0 results. Check query or connectivity.")
    log.info("Retrieved %d entries.", len(results))
    return results


def process_uniprot_data(raw_entries: list) -> pd.DataFrame:
    records, skipped = [], 0
    for e in raw_entries:
        accession = e.get("primaryAccession", "").strip()
        sequence  = e.get("sequence", {}).get("value", "").strip()
        if not accession or not sequence:
            skipped += 1
            continue

        rec = {
            "accession":            accession,
            "entry_name":           e.get("uniProtkbId", ""),
            "protein_name":         "",
            "gene_name":            "",
            "length":               e.get("sequence", {}).get("length", 0),
            "sequence":             sequence,
            "function":             "",
            "subcellular_location": "",
            "pdb_structures":       "",
            "drugbank_targets":     "",
        }
        desc = e.get("proteinDescription", {})
        if "recommendedName" in desc:
            rec["protein_name"] = desc["recommendedName"].get("fullName", {}).get("value", "")
        genes = e.get("genes", [])
        if genes:
            rec["gene_name"] = genes[0].get("geneName", {}).get("value", "")

        for c in e.get("comments", []):
            ct = c.get("commentType", "")
            if ct == "FUNCTION":
                texts = c.get("texts", [])
                if texts:
                    rec["function"] = texts[0].get("value", "")
            elif ct == "SUBCELLULAR LOCATION":
                locs = c.get("subcellularLocations", [])
                if locs:
                    rec["subcellular_location"] = locs[0].get("location", {}).get("value", "")

        pdb, db = [], []
        for x in e.get("uniProtKBCrossReferences", []):
            if x.get("database") == "PDB":
                pdb.append(x.get("id", ""))
            elif x.get("database") == "DrugBank":
                db.append(x.get("id", ""))
        rec["pdb_structures"]   = "; ".join(pdb)
        rec["drugbank_targets"] = "; ".join(db)
        records.append(rec)

    if skipped:
        log.warning("Skipped %d entries missing accession or sequence.", skipped)
    df = pd.DataFrame(records)
    log.info("Processed %d proteins.", len(df))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["sequence"].notna() & (df["sequence"] != "")].copy()
    df = df[(df["length"] >= 50) & (df["length"] <= 2000)].copy()
    for col in ["function", "subcellular_location", "gene_name", "protein_name"]:
        df[col] = df[col].fillna("")
    log.info("After cleaning: %d proteins.", len(df))
    return df


def assign_length_druggability(df: pd.DataFrame) -> pd.DataFrame:
    bins      = [0, 150, 200, 450, 700, float("inf")]
    labels    = ["Very Small", "Small", "Optimal", "Large", "Very Large"]
    score_map = {"Very Small": 0.3, "Small": 0.6, "Optimal": 1.0, "Large": 0.8, "Very Large": 0.4}
    df = df.copy()
    df["length_category"]     = pd.cut(df["length"], bins=bins, labels=labels)
    df["length_druggability"] = df["length_category"].map(score_map)
    return df


def add_feature_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    flag_defs = {
        "has_function":   ("function",             lambda s: s.str.len() > 0),
        "has_structure":  ("pdb_structures",        lambda s: s.str.len() > 0),
        "has_drugbank":   ("drugbank_targets",      lambda s: s.str.len() > 0),
        "has_gene_name":  ("gene_name",             lambda s: s.str.len() > 0),
        "is_membrane":    ("subcellular_location",  lambda s: s.str.contains("membrane",  case=False, na=False)),
        "is_cytoplasmic": ("subcellular_location",  lambda s: s.str.contains("cytoplasm", case=False, na=False)),
    }
    for flag, (col, fn) in flag_defs.items():
        df[flag] = fn(df[col])
    for col in ["has_function", "has_structure", "has_drugbank", "has_gene_name", "is_membrane", "is_cytoplasmic"]:
        log.info("  %s: %d", col, df[col].sum())
    return df


# ── 4-7. Physicochemical Features & Conservation ─────────────────────────────

def calculate_molecular_weight(sequence: str) -> float:
    aa_masses = {
        "A": 71.04, "R": 156.10, "N": 114.04, "D": 115.03, "C": 103.01,
        "E": 129.04, "Q": 128.06, "G":  57.02, "H": 137.06, "I": 113.08,
        "L": 113.08, "K": 128.09, "M": 131.04, "F": 147.07, "P":  97.05,
        "S":  87.03, "T": 101.05, "W": 186.08, "Y": 163.06, "V":  99.07,
    }
    mw = sum(aa_masses.get(aa, 111.1) for aa in sequence.upper()) + 18.01
    return round(mw / 1000, 2)


def calculate_pI(sequence: str) -> float:
    pKa = {"D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1, "H": 6.0,
           "K": 10.5, "R": 12.5, "N_term": 8.0, "C_term": 3.1}
    counts = {aa: sequence.upper().count(aa) for aa in pKa if len(aa) == 1}

    def charge_at_pH(pH):
        q = 1 / (1 + 10 ** (pH - pKa["N_term"])) - 1 / (1 + 10 ** (pKa["C_term"] - pH))
        for aa in ("H", "K", "R"):
            q += counts[aa] / (1 + 10 ** (pH - pKa[aa]))
        for aa in ("D", "E", "C", "Y"):
            q -= counts[aa] / (1 + 10 ** (pKa[aa] - pH))
        return q

    lo, hi = 0.0, 14.0
    mid = 7.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if charge_at_pH(mid) > 0:
            lo = mid
        else:
            hi = mid
    return round(mid, 2)


def calculate_hydropathy(sequence: str) -> float:
    kd = {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
        "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
        "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
        "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
    }
    scores = [kd.get(aa, 0) for aa in sequence.upper()]
    return round(float(np.mean(scores)), 3) if scores else 0.0


def calculate_aromaticity(sequence: str) -> float:
    return round(sum(sequence.upper().count(aa) for aa in "FWY") / len(sequence), 4) if sequence else 0.0


def calculate_instability_index(sequence: str) -> float:
    if len(sequence) < 2:
        return 0.0
    try:
        return round(ProteinAnalysis(sequence.upper()).instability_index(), 2)
    except Exception:
        return 0.0


def add_physicochemical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["molecular_weight_kDa"] = df["sequence"].apply(calculate_molecular_weight)
    df["pI"]                   = df["sequence"].apply(calculate_pI)
    df["hydropathy_index"]     = df["sequence"].apply(calculate_hydropathy)
    df["aromaticity"]          = df["sequence"].apply(calculate_aromaticity)
    df["instability_index"]    = df["sequence"].apply(calculate_instability_index)
    df["is_stable"]            = df["instability_index"] < 40
    log.info("Physicochemical features added. Stable proteins: %d", df["is_stable"].sum())
    return df


def compute_clustal_conservation(df: pd.DataFrame) -> dict:
    records = [
        SeqRecord(Seq(r["sequence"]), id=r["accession"], description=r["gene_name"])
        for _, r in df.iterrows() if r["sequence"]
    ]
    log.info("Running ClustalOmega on %d sequences...", len(records))

    with tempfile.NamedTemporaryFile(dir=".", suffix=".fasta", delete=False, mode="w") as fh:
        SeqIO.write(records, fh, "fasta")
        infile = fh.name
    
    # Use basenames for WSL to resolve them in the current directory
    infile_base = os.path.basename(infile)
    outfile_base = infile_base.replace(".fasta", "_aln.fasta")
    outfile = infile.replace(".fasta", "_aln.fasta")

    def _fallback(reason: str) -> dict:
        log.warning("%s — returning 0.5 for all conservation scores.", reason)
        for f in (infile, outfile):
            if os.path.exists(f):
                os.unlink(f)
        return {acc: 0.5 for acc in df["accession"]}

    import sys
    cmd = ["wsl.exe", "clustalo"] if sys.platform == "win32" else ["clustalo"]
    cmd.extend(["-i", infile_base, "-o", outfile_base, "--force", "--outfmt=fasta", "--threads=4"])

    try:
        subprocess.run(
            cmd,
            check=True, capture_output=True,
        )
    except FileNotFoundError:
        return _fallback("clustalo (or wsl) not found on PATH")
    except subprocess.CalledProcessError as e:
        return _fallback(f"ClustalOmega error: {e.stderr.decode().strip()}")

    if not os.path.exists(outfile) or os.path.getsize(outfile) == 0:
        return _fallback("ClustalOmega produced no output file")

    try:
        alignment = AlignIO.read(outfile, "fasta")
    except Exception as e:
        return _fallback(f"Could not parse alignment: {e}")

    aln_len = alignment.get_alignment_length()
    if aln_len == 0:
        return _fallback("Alignment has zero length")

    # Precompute conservation score for each column ONCE (O(L*N) instead of O(L*N^2))
    col_scores_cache = {}
    for col in range(aln_len):
        col_no_gap = [c for c in alignment[:, col] if c != "-"]
        if not col_no_gap:
            col_scores_cache[col] = 0.0
            continue
        most_common = max(set(col_no_gap), key=col_no_gap.count)
        col_scores_cache[col] = col_no_gap.count(most_common) / len(col_no_gap)

    scores = {}
    for rec in alignment:
        seq_col_scores = [col_scores_cache[col] for col in range(aln_len) if rec.seq[col] != "-"]
        scores[rec.id] = round(sum(seq_col_scores) / len(seq_col_scores), 4) if seq_col_scores else 0.5

    os.unlink(infile)
    os.unlink(outfile)
    median_score = float(np.median(list(scores.values())))
    log.info("Conservation computed. Median: %.3f", median_score)
    return {acc: scores.get(acc, median_score) for acc in df["accession"]}


CONSERVATION_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conservation_cache.json")


def add_conservation_scores(df: pd.DataFrame) -> pd.DataFrame:
    import json as _json

    # ── Try loading pre-computed scores from cache (instant!) ────────────
    if os.path.exists(CONSERVATION_CACHE):
        with open(CONSERVATION_CACHE, "r") as fh:
            cached: dict = _json.load(fh)
        matched = sum(1 for acc in df["accession"] if acc in cached)
        log.info(
            "Loaded %d conservation scores from cache (%d/%d matched).",
            len(cached), matched, len(df),
        )
        df = df.copy()
        df["conservation_score"] = df["accession"].map(cached)
        df["conservation_score"].fillna(0.5, inplace=True)
        return df

    # ── No cache → fall back to live ClustalOmega alignment ─────────────
    log.info("No conservation_cache.json found — running live ClustalOmega alignment...")
    scores = compute_clustal_conservation(df)
    df = df.copy()
    df["conservation_score"] = df["accession"].map(scores)
    df["conservation_score"].fillna(df["conservation_score"].median(), inplace=True)
    return df


# ── 8-9. Model ───────────────────────────────────────────────────────────────

def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    for col in BOOL_COLS:
        X[col] = X[col].astype(int)
    X["length_druggability"] = X["length_druggability"].astype(float)
    if X.isnull().any().any():
        X = X.fillna(X.median())
    if np.isinf(X.values).any():
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    return X


def build_and_evaluate_model(df: pd.DataFrame):
    df = df.copy()
    df["druggable_label"] = df["has_drugbank"].astype(int)
    log.info(
        "Known positives: %d  |  Unlabelled: %d",
        df["druggable_label"].sum(),
        (df["druggable_label"] == 0).sum(),
    )

    X, y = _prepare_X(df), df["druggable_label"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_split=5, class_weight="balanced",
            random_state=42,
        )),
    ])
    log.info("Using balanced Random Forest (class_weight='balanced', n_estimators=200).")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)

    log.info("5-fold CV ROC-AUC Mean: %.3f ± %.3f", np.nanmean(cv_scores), np.nanstd(cv_scores))

    model.fit(X, y)
    df["druggability_proba"] = model.predict_proba(X)[:, 1]

    rf_fitted = model.named_steps["rf"]
    importances = pd.Series(
        rf_fitted.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)

    return df, model, cv_scores, importances, y_pred_cv, y


# ── 10-14. Scoring, Sensitivity & Ranking ────────────────────────────────────

def compute_target_score(row, weights=None) -> float:
    w = weights or COMPOSITE_WEIGHTS
    return round(
        w["druggability_proba"]  * row["druggability_proba"]
        + w["conservation_score"]  * row["conservation_score"]
        + w["has_structure_bonus"] * float(row["has_structure"])
        + w["length_druggability"] * float(row["length_druggability"])
        + w["has_function_bonus"]  * float(row["has_function"]),
        4,
    )


def assign_tier(score: float) -> str:
    if score >= TIER1_THRESHOLD:
        return "Tier 1 — High Priority"
    if score >= TIER2_THRESHOLD:
        return "Tier 2 — Medium Priority"
    return "Tier 3 — Low Priority"


def rank_targets(df: pd.DataFrame):
    df = df.copy()
    df["composite_target_score"] = df.apply(compute_target_score, axis=1)
    df["priority_tier"]          = df["composite_target_score"].apply(assign_tier)
    df["_loc_bonus"] = (df["is_membrane"].astype(int) + df["is_cytoplasmic"].astype(int)) * 0.01
    df = df.sort_values(
        ["composite_target_score", "conservation_score", "_loc_bonus"],
        ascending=False,
    ).drop(columns="_loc_bonus")
    tier_counts = df["priority_tier"].value_counts()
    return df, tier_counts, df.head(20)


def run_monte_carlo_sensitivity(df: pd.DataFrame, n_samples=MC_N_SAMPLES, alpha=MC_DIRICHLET_A) -> pd.DataFrame:
    weight_keys = list(COMPOSITE_WEIGHTS.keys())
    draws       = np.random.default_rng(42).dirichlet(np.full(len(weight_keys), alpha), size=n_samples)
    tier1_counts = np.zeros(len(df))
    score_matrix = np.zeros((n_samples, len(df)))

    for i, raw_w in enumerate(draws):
        scores = df.apply(
            lambda r: compute_target_score(r, weights=dict(zip(weight_keys, raw_w))),
            axis=1,
        ).values
        score_matrix[i]  = scores
        tier1_counts     += (scores >= TIER1_THRESHOLD).astype(int)

    df = df.copy()
    df["mc_tier1_freq"]  = tier1_counts / n_samples
    df["mc_score_mean"]  = score_matrix.mean(axis=0)
    df["mc_score_std"]   = score_matrix.std(axis=0)
    df["mc_rank_stable"] = df["mc_tier1_freq"] >= 0.80
    df["mc_score_cv"]    = np.where(
        df["mc_score_mean"] > 0,
        df["mc_score_std"] / df["mc_score_mean"],
        0.0,
    )
    return df


# ── Streaming pipeline (WebSocket-aware) ─────────────────────────────────────

class _QueueLogHandler(logging.Handler):
    """Forwards every log record as a plain string into a multiprocessing.Queue."""

    def __init__(self, queue):
        super().__init__()
        self._q = queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._q.put_nowait(self.format(record))
        except Exception:
            pass  # never crash the pipeline because of a log failure


def run_pipeline_streaming(queue, output_csv: str = "listeria_final_results.csv") -> dict:
    """
    Identical to run_full_pipeline but every log.info / log.warning line is
    also pushed into *queue* so the WebSocket endpoint can relay it live.

    Designed to run inside a ProcessPoolExecutor worker; all imports are
    already resolved at module level.
    """
    # Wire the queue handler into the root logger for this process
    handler = _QueueLogHandler(queue)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(handler)

    try:
        # ── Step 1 ──────────────────────────────────────────────────────────
        queue.put_nowait("[1/8] Fetching L. monocytogenes proteome from UniProt...")
        raw_data       = fetch_listeria_proteome()

        queue.put_nowait("[2/8] Processing UniProt entries...")
        proteins_df    = process_uniprot_data(raw_data)

        queue.put_nowait("[3/8] Cleaning & filtering sequences (50–2000 aa)...")
        proteins_clean = clean_data(proteins_df)
        proteins_clean = assign_length_druggability(proteins_clean)
        proteins_clean = add_feature_flags(proteins_clean)

        # ── Step 2 ──────────────────────────────────────────────────────────
        queue.put_nowait("[4/8] Computing physicochemical features (MW, pI, hydropathy, aromaticity, instability)...")
        proteins_clean = add_physicochemical_features(proteins_clean)

        queue.put_nowait("[5/8] Running ClustalOmega multiple-sequence alignment for conservation scores...")
        proteins_clean = add_conservation_scores(proteins_clean)

        # ── Step 3 ──────────────────────────────────────────────────────────
        queue.put_nowait("[6/8] Training PU-learning / Random Forest druggability model (5-fold CV)...")
        proteins_clean, model, cv_scores, importances, y_pred_cv, y = build_and_evaluate_model(proteins_clean)
        queue.put_nowait(
            f"      CV ROC-AUC: {np.nanmean(cv_scores):.4f} ± {np.nanstd(cv_scores):.4f}  "
            f"| Folds: {', '.join(f'{s:.3f}' if not np.isnan(s) else 'N/A' for s in cv_scores)}"
        )

        # ── Step 4 ──────────────────────────────────────────────────────────
        queue.put_nowait("[7/8] Running Monte-Carlo sensitivity analysis (1 000 weight draws)...")
        proteins_clean["composite_target_score"] = proteins_clean.apply(compute_target_score, axis=1)
        proteins_clean["priority_tier"]          = proteins_clean["composite_target_score"].apply(assign_tier)
        proteins_clean = run_monte_carlo_sensitivity(proteins_clean)

        # ── Step 5 ──────────────────────────────────────────────────────────
        queue.put_nowait("[8/8] Ranking targets and writing results CSV...")
        proteins_final, tier_counts, top20 = rank_targets(proteins_clean)
        proteins_final.to_csv(output_csv, index=False)

        # Build the result payload
        top20_records = top20[[
            "accession", "gene_name", "protein_name", "length", "conservation_score",
            "druggability_proba", "composite_target_score", "priority_tier",
            "has_structure", "subcellular_location", "mc_rank_stable",
        ]].rename(columns={
            "composite_target_score": "composite",
            "druggability_proba":     "druggability",
            "conservation_score":     "conservation",
        }).to_dict(orient="records")

        result = {
            "total_proteins": len(proteins_final),
            "cv_auc_mean":    round(float(np.nanmean(cv_scores)), 4),
            "cv_auc_std":     round(float(np.nanstd(cv_scores)), 4),
            "cv_fold_scores": [round(float(s), 4) if not np.isnan(s) else None for s in cv_scores],
            "tier_counts":    tier_counts.to_dict(),
            "top20":          top20_records,
            "output_csv":     output_csv,
        }
        queue.put_nowait("__DONE__")      # sentinel consumed by the WS endpoint
        return result

    except Exception as exc:
        queue.put_nowait(f"[ERROR] {exc}")
        queue.put_nowait("__ERROR__")
        raise
    finally:
        logging.getLogger().removeHandler(handler)


# ── Legacy (non-streaming) entry-point kept for REST /run-pipeline ────────────

def run_full_pipeline(output_csv: str = "listeria_final_results.csv") -> dict:
    """Blocking pipeline used by the REST background-task endpoint."""
    import multiprocessing
    q = multiprocessing.Queue()
    result = run_pipeline_streaming(q, output_csv=output_csv)
    return result

