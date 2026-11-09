from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from urllib.parse import quote

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
try:
    import requests
except Exception:
    requests = None

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from core.publication_utils import (
    DEFAULT_RANDOM_STATE,
    build_output_prefix,
    configure_publication_plots,
    is_component_feature,
    is_true_shap_source,
    resolve_publication_font,
    robust_minmax,
    set_global_seed,
    stage_task_inputs,
    style_publication_axis,
    task_output_dir,
    task_run_context,
)

configure_publication_plots()

STRING_API = "https://string-db.org/api/tsv"
NODE_COLUMNS = [
    "Protein",
    "GeneSymbol",
    "DisplayLabel",
    "UniProtID",
    "EntrezID",
    "GOAnnotations",
    "STRINGID",
    "STRINGPreferredName",
    "AnnotationSource",
    "Community",
    "IsCommunityHub",
    "IsSeedBiomarker",
    "IsExpandedInteractor",
    "ConnectedSeedProtein",
    "STRINGConfidence",
    "CorrelationCoefficient",
    "CorrelationMethod",
    "ImportanceValue",
    "ImportanceType",
    "SHAP",
    "ConsensusScore",
    "StabilityScore",
    "SupportFrequency",
    "ConfidenceScore",
    "HubStability",
    "Degree",
    "Betweenness",
    "Closeness",
    "HubRank",
]
EDGE_COLUMNS = [
    "Source",
    "Target",
    "EdgeType",
    "STRINGScore",
    "CoexpressionScore",
    "Confidence",
    "IsExpandedEdge",
    "SourceSTRINGID",
    "TargetSTRINGID",
    "ConnectedSeedProtein",
    "STRINGConfidence",
    "CorrelationCoefficient",
    "CorrelationMethod",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python PPI network analysis using provenance-aware protein importance, consensus, stability, and STRING.")
    parser.add_argument("--input", default=None, help="Task input directory containing SHAP and biomarker reliability outputs.")
    parser.add_argument("--outdir", default=".", help="Base output directory.")
    parser.add_argument("--feature_selection", choices=["yes", "no"], default="no", help="Accepted for CLI consistency.")
    parser.add_argument("--jobs", type=int, default=1, help="Accepted for CLI consistency.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("-s", "--score_thresholdHere", type=int, default=400)
    parser.add_argument("-c", "--combined_score_thresholdHere", type=int, default=800)
    parser.add_argument("-a", "--SHAPthresh", "--top_n", dest="SHAPthresh", type=int, default=100)
    parser.add_argument("-n", "--patternChosen", type=str, default="shap_values.csv")
    parser.add_argument("-v", "--converProId", type=str, default="TRUE")
    parser.add_argument("-x", "--proteinExpFile", "--protein_expression", dest="proteinExpFile", type=str, default="Case1.csv")
    parser.add_argument("-m", "--CorMethod", "--correlation_method", dest="CorMethod", type=str, default="spearman")
    parser.add_argument("-d", "--CorThreshold", "--correlation_threshold", dest="CorThreshold", type=float, default=0.8)
    parser.add_argument("-p", "--prefix", type=str, default="")
    parser.add_argument("--species", type=int, default=9606)
    parser.add_argument("--background-species", type=int, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--expanded-limit", type=int, default=5)
    return parser.parse_args()


def _read_shap(path: str) -> pd.DataFrame:
    """Read a protein-importance artifact while preserving attribution provenance."""
    df = pd.read_csv(path)
    protein_col = "Protein" if "Protein" in df.columns else "Feature" if "Feature" in df.columns else None
    if protein_col is None:
        return pd.DataFrame(columns=["Protein", "ImportanceValue", "ImportanceType", "SHAP"])
    if "ImportanceValue" in df.columns:
        values = pd.to_numeric(df["ImportanceValue"], errors="coerce").fillna(0.0)
    elif "MeanImportance" in df.columns:
        values = pd.to_numeric(df["MeanImportance"], errors="coerce").fillna(0.0)
    elif "MeanAbsSHAP" in df.columns:
        values = pd.to_numeric(df["MeanAbsSHAP"], errors="coerce").fillna(0.0)
    elif "MeanSHAP" in df.columns:
        values = pd.to_numeric(df["MeanSHAP"], errors="coerce").fillna(0.0)
    elif "Global Mean" in df.columns:
        values = pd.to_numeric(df["Global Mean"], errors="coerce").fillna(0.0)
    elif "Mean SHAP Value" in df.columns:
        values = pd.to_numeric(df["Mean SHAP Value"], errors="coerce").fillna(0.0)
    else:
        numeric = df.drop(columns=[protein_col], errors="ignore").select_dtypes(include="number")
        values = numeric.abs().mean(axis=1) if not numeric.empty else pd.Series(np.zeros(len(df)))
    source_col = next(
        (column for column in ("ImportanceType", "Importance_Source", "ImportanceSource") if column in df.columns),
        None,
    )
    source = (
        df[source_col].fillna("ModelSpecificImportance").astype(str)
        if source_col
        else pd.Series(["UnverifiedLegacyImportance"] * len(df), index=df.index)
    )
    if "IsTrueSHAP" in df.columns:
        is_shap = df["IsTrueSHAP"].fillna(False).astype(bool)
    else:
        is_shap = source.map(is_true_shap_source)
    out = pd.DataFrame({
        "Protein": df[protein_col].astype(str),
        "ImportanceValue": values,
        "ImportanceType": source,
        "SHAP": values.where(is_shap),
    })
    out = out[~out["Protein"].map(is_component_feature)]
    return out.sort_values("ImportanceValue", ascending=False)


def load_importance(prefix: str, pattern: str, top_n: int) -> pd.DataFrame:
    paths = sorted(glob.glob(f"{prefix}_*{pattern}")) if prefix else sorted(glob.glob(f"*{pattern}"))
    paths = [p for p in paths if "_fold" not in Path(p).name]
    frames = []
    for path in paths:
        df = _read_shap(path)
        if df.empty:
            continue
        model = Path(path).name.replace(f"{Path(prefix).name}_", "").replace("_shap_values.csv", "") if prefix else Path(path).name.replace("_shap_values.csv", "")
        df["Model"] = model
        df["NormalizedImportance"] = robust_minmax(df["ImportanceValue"])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["Protein", "ImportanceValue", "ImportanceType", "SHAP", "Occurrence"])
    combined = pd.concat(frames, ignore_index=True)
    summary = combined.groupby("Protein").agg(
        ImportanceValue=("NormalizedImportance", "mean"),
        ImportanceType=("ImportanceType", lambda values: ";".join(sorted(set(map(str, values))))),
        SHAP=("SHAP", "mean"),
        Occurrence=("Model", "nunique"),
    ).reset_index()
    return summary.sort_values(["ImportanceValue", "Occurrence"], ascending=False).head(top_n)


def _score_column(df: pd.DataFrame, options: list[str], default: float = 0.0) -> pd.Series:
    for col in options:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def _first_present(df: pd.DataFrame, options: list[str], default: str = "") -> pd.Series:
    for col in options:
        if col in df.columns:
            return df[col].fillna(default).astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype=str)


def _clean_annotation_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na"} else text


def _display_annotation_label(gene_symbol, protein) -> str:
    return _clean_annotation_text(gene_symbol) or str(protein)


def _empty_annotations(proteins: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "Protein": proteins,
        "GeneSymbol": [""] * len(proteins),
        "UniProtID": [""] * len(proteins),
        "EntrezID": [""] * len(proteins),
        "GOAnnotations": [""] * len(proteins),
        "STRINGID": [""] * len(proteins),
        "STRINGPreferredName": [""] * len(proteins),
        "AnnotationSource": ["input"] * len(proteins),
    })


def _merge_annotation(base: pd.DataFrame, update: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if update.empty:
        return base
    merged = base.merge(update, on="Protein", how="left", suffixes=("", "_new"))
    for col in ["GeneSymbol", "UniProtID", "EntrezID", "GOAnnotations", "STRINGID", "STRINGPreferredName"]:
        new = f"{col}_new"
        if new in merged.columns:
            current = merged[col].fillna("").astype(str)
            incoming = merged[new].fillna("").astype(str)
            merged[col] = current.where(current.str.len() > 0, incoming)
            merged = merged.drop(columns=[new])
    populated = update.set_index("Protein").reindex(merged["Protein"]) if "Protein" in update.columns else pd.DataFrame(index=merged.index)
    has_update = populated.fillna("").astype(str).apply(lambda row: any(v for v in row), axis=1) if not populated.empty else pd.Series(False, index=merged.index)
    merged.loc[has_update.to_numpy(), "AnnotationSource"] = merged.loc[has_update.to_numpy(), "AnnotationSource"].astype(str) + f";{source_name}"
    return merged


def _string_aliases(proteins: list[str], species: int, offline: bool) -> pd.DataFrame:
    if offline or requests is None or not proteins:
        return pd.DataFrame()
    ordered = list(dict.fromkeys(map(str, proteins)))
    rows = []
    for start in range(0, len(ordered), 100):
        batch = ordered[start : start + 100]
        identifiers = "%0d".join(quote(protein) for protein in batch)
        url = f"{STRING_API}/get_string_ids?identifiers={identifiers}&species={species}&limit=1"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except Exception:
            continue
        lines = response.text.strip().splitlines()
        if len(lines) < 2:
            continue
        header = lines[0].split("\t")
        for line in lines[1:]:
            rec = dict(zip(header, line.split("\t")))
            try:
                query = batch[int(rec.get("queryIndex", ""))]
            except (ValueError, IndexError):
                query = rec.get("queryItem")
            if not query:
                continue
            rows.append({
                "Protein": str(query),
                "GeneSymbol": rec.get("preferredName", ""),
                "STRINGID": rec.get("stringId", ""),
                "STRINGPreferredName": rec.get("preferredName", ""),
            })
    return pd.DataFrame(rows).drop_duplicates("Protein") if rows else pd.DataFrame()


def _mygene_annotations(proteins: list[str], species: int, offline: bool) -> pd.DataFrame:
    if offline or not proteins:
        return pd.DataFrame()
    try:
        import mygene

        mg = mygene.MyGeneInfo()
        records = mg.querymany(proteins, scopes="symbol,alias,uniprot", fields="symbol,entrezgene,uniprot,go", species=species, as_dataframe=False, verbose=False)
    except Exception:
        return pd.DataFrame()
    rows = []
    for rec in records:
        protein = rec.get("query")
        if not protein or rec.get("notfound"):
            continue
        uniprot = rec.get("uniprot", "")
        if isinstance(uniprot, dict):
            uniprot = uniprot.get("Swiss-Prot") or uniprot.get("TrEMBL") or ""
            if isinstance(uniprot, list):
                uniprot = uniprot[0] if uniprot else ""
        go = rec.get("go", "")
        go_terms = []
        if isinstance(go, dict):
            for value in go.values():
                values = value if isinstance(value, list) else [value]
                go_terms.extend(str(item.get("id", "")) for item in values if isinstance(item, dict) and item.get("id"))
        rows.append({
            "Protein": str(protein),
            "GeneSymbol": str(rec.get("symbol", "")),
            "UniProtID": str(uniprot),
            "EntrezID": str(rec.get("entrezgene", "")),
            "GOAnnotations": ";".join(sorted(set(go_terms))),
        })
    return pd.DataFrame(rows).drop_duplicates("Protein") if rows else pd.DataFrame()


def _uniprot_rest_annotations(proteins: list[str], species: int, offline: bool) -> pd.DataFrame:
    if offline or requests is None or not proteins:
        return pd.DataFrame()
    rows = []
    for protein in proteins:
        query = quote(f'({protein}) AND (organism_id:{species})')
        url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&fields=accession,gene_names,xref_geneid,go_id&format=tsv&size=1"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
        except Exception:
            continue
        lines = response.text.strip().splitlines()
        if len(lines) < 2:
            continue
        header = lines[0].split("\t")
        rec = dict(zip(header, lines[1].split("\t")))
        gene_names = rec.get("Gene Names", "").split()
        rows.append({
            "Protein": str(protein),
            "GeneSymbol": gene_names[0] if gene_names else "",
            "UniProtID": rec.get("Entry", ""),
            "EntrezID": rec.get("GeneID", ""),
            "GOAnnotations": rec.get("Gene Ontology IDs", ""),
        })
    return pd.DataFrame(rows).drop_duplicates("Protein") if rows else pd.DataFrame()


def annotate_proteins(proteins: list[str], species: int, offline: bool) -> pd.DataFrame:
    unique = sorted({str(p) for p in proteins if str(p)})
    annotations = _empty_annotations(unique)
    for source_name, loader in [
        ("STRING aliases", _string_aliases),
        ("mygene", _mygene_annotations),
        ("UniProt REST", _uniprot_rest_annotations),
    ]:
        annotations = _merge_annotation(annotations, loader(unique, species, offline), source_name)
    annotations["AnnotationSource"] = annotations["AnnotationSource"].str.replace("input;", "", regex=False)
    return annotations


_PUBLICATION_GENE_SYMBOL_CACHE: dict[tuple[int, bool, str], str] = {}


def publication_gene_symbol_map(
    proteins,
    *,
    species: int = 9606,
    offline: bool = False,
    annotations: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Return publication display names through the canonical PPI annotation path."""
    ordered = list(dict.fromkeys(str(protein) for protein in proteins if str(protein)))
    mapping: dict[str, str] = {}
    if annotations is not None and not annotations.empty and "Protein" in annotations.columns:
        annotation_table = annotations.copy()
        if "GeneSymbol" not in annotation_table.columns:
            annotation_table["GeneSymbol"] = ""
        for _, row in annotation_table.drop_duplicates("Protein").iterrows():
            protein = str(row["Protein"])
            mapping[protein] = _display_annotation_label(row.get("GeneSymbol", ""), protein)

    unresolved = []
    for protein in ordered:
        key = (int(species), bool(offline), protein)
        if protein in mapping:
            _PUBLICATION_GENE_SYMBOL_CACHE[key] = mapping[protein]
        elif key in _PUBLICATION_GENE_SYMBOL_CACHE:
            mapping[protein] = _PUBLICATION_GENE_SYMBOL_CACHE[key]
        else:
            unresolved.append(protein)
    if unresolved:
        resolved = annotate_proteins(unresolved, int(species), bool(offline))
        for _, row in resolved.iterrows():
            protein = str(row["Protein"])
            label = _display_annotation_label(row.get("GeneSymbol", ""), protein)
            mapping[protein] = label
            _PUBLICATION_GENE_SYMBOL_CACHE[(int(species), bool(offline), protein)] = label
    return {protein: mapping.get(protein, protein) for protein in ordered}


def load_annotation_scores(prefix: str) -> pd.DataFrame:
    frames = []
    for path in [f"{prefix}_consensus_biomarkers.csv", f"{prefix}_biomarker_frequency.csv", f"{prefix}_recommended_biomarkers.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "Protein" not in df.columns and "Feature" in df.columns:
                df = df.rename(columns={"Feature": "Protein"})
            if "Protein" in df.columns:
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["Protein", "ConsensusScore", "StabilityScore", "SupportFrequency", "ConfidenceScore", "GeneSymbol", "UniProtID", "EntrezID", "GOAnnotations", "STRINGID", "STRINGPreferredName", "AnnotationSource"])
    proteins = sorted(set().union(*[set(df["Protein"].astype(str)) for df in frames]))
    out = pd.DataFrame({"Protein": proteins})
    for df in frames:
        tmp = pd.DataFrame({
            "Protein": df["Protein"].astype(str),
            "ConsensusScore": _score_column(df, ["ConsensusScore", "Consensus_Score"]),
            "StabilityScore": _score_column(df, ["StabilityScore", "BiomarkerStabilityScore", "Biomarker_Stability_Score"]),
            "SupportFrequency": _score_column(df, ["SupportFrequency", "SelectionFrequency", "Selection_Frequency"]),
            "ConfidenceScore": _score_column(df, ["ConfidenceScore"]),
            "GeneSymbol": _first_present(df, ["GeneSymbol", "Gene Symbol", "Symbol"]),
            "UniProtID": _first_present(df, ["UniProtID", "UniProt", "UniProtKB"]),
            "EntrezID": _first_present(df, ["EntrezID", "Entrez", "GeneID"]),
            "GOAnnotations": _first_present(df, ["GOAnnotations", "GO Annotation", "GO"]),
            "STRINGID": _first_present(df, ["STRINGID", "STRING_ID"]),
            "STRINGPreferredName": _first_present(df, ["STRINGPreferredName", "STRING_PreferredName"]),
            "AnnotationSource": _first_present(df, ["AnnotationSource"], "workflow"),
        })
        out = out.merge(tmp, on="Protein", how="left", suffixes=("", "_new"))
        for col in ["ConsensusScore", "StabilityScore", "SupportFrequency", "ConfidenceScore"]:
            new = f"{col}_new"
            if new in out.columns:
                out[col] = out[col].fillna(out[new]) if col in out.columns else out[new]
                out = out.drop(columns=[new])
        for col in ["GeneSymbol", "UniProtID", "EntrezID", "GOAnnotations", "STRINGID", "STRINGPreferredName", "AnnotationSource"]:
            new = f"{col}_new"
            if new in out.columns:
                out[col] = out[col].where(out[col].astype(str).str.len() > 0, out[new]) if col in out.columns else out[new]
                out = out.drop(columns=[new])
    for col in ["ConsensusScore", "StabilityScore", "SupportFrequency", "ConfidenceScore"]:
        out[col] = pd.to_numeric(out.get(col, 0.0), errors="coerce").fillna(0.0)
    for col in ["GeneSymbol", "UniProtID", "EntrezID", "GOAnnotations", "STRINGID", "STRINGPreferredName", "AnnotationSource"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    return out


def _request_string(
    endpoint: str,
    proteins: list[str],
    required_score: int,
    species: int,
    limit: int | None,
    offline: bool,
    background_proteins: list[str] | None = None,
) -> pd.DataFrame:
    if offline or requests is None or not proteins:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    aliases = _string_aliases(proteins, species, offline)
    id_map = aliases.set_index("Protein")["STRINGID"].to_dict() if not aliases.empty and "STRINGID" in aliases.columns else {}
    mapping = aliases
    if background_proteins:
        background = _string_aliases(background_proteins, species, offline)
        if not background.empty:
            mapping = pd.concat([aliases, background], ignore_index=True).drop_duplicates("Protein")
    preferred_map = (
        mapping.drop_duplicates("STRINGID").set_index("STRINGID")["Protein"].to_dict()
        if not mapping.empty and "STRINGID" in mapping.columns
        else {}
    )
    identifiers_to_query = [id_map.get(str(p), str(p)) for p in proteins]
    identifiers = "%0d".join(quote(str(p)) for p in identifiers_to_query)
    url = f"{STRING_API}/{endpoint}?identifiers={identifiers}&species={species}&required_score={required_score}"
    if limit is not None:
        url += f"&limit={int(limit)}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except Exception:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    lines = response.text.strip().splitlines()
    if len(lines) < 2:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    header = lines[0].split("\t")
    rows = []
    seed_set = set(proteins)
    for line in lines[1:]:
        rec = dict(zip(header, line.split("\t")))
        a = rec.get("preferredName_A") or rec.get("stringId_A")
        b = rec.get("preferredName_B") or rec.get("stringId_B")
        if not a or not b or a == b:
            continue
        score = float(rec.get("score", rec.get("combined_score", 0.0)))
        source_id = rec.get("stringId_A", "")
        target_id = rec.get("stringId_B", "")
        source_name = preferred_map.get(source_id, a)
        target_name = preferred_map.get(target_id, b)
        rows.append({
            "Source": source_name,
            "Target": target_name,
            "EdgeType": "STRING",
            "STRINGScore": score,
            "CoexpressionScore": 0.0,
            "Confidence": score,
            "IsExpandedEdge": not (source_name in seed_set and target_name in seed_set),
            "SourceSTRINGID": source_id,
            "TargetSTRINGID": target_id,
        })
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=EDGE_COLUMNS)


def string_network(proteins: list[str], required_score: int, species: int, offline: bool = False) -> pd.DataFrame:
    return _request_string("network", proteins, required_score, species, None, offline)


def string_expanded_network(
    proteins: list[str],
    required_score: int,
    species: int,
    limit: int,
    offline: bool = False,
    background_proteins: list[str] | None = None,
) -> pd.DataFrame:
    expanded = _request_string(
        "interaction_partners",
        proteins,
        required_score,
        species,
        limit,
        offline,
        background_proteins,
    )
    if expanded.empty:
        return string_network(proteins, required_score, species, offline)
    return expanded


def _expression_matrix(protein_exp_file: str) -> pd.DataFrame:
    if not os.path.exists(protein_exp_file):
        return pd.DataFrame()
    df = pd.read_csv(protein_exp_file)
    feature_cols = [column for column in df.columns if column not in {"SampleID", "Label"}]
    return df[feature_cols].apply(pd.to_numeric, errors="coerce")


def coexpression_edges(protein_exp_file: str, proteins: list[str], method: str, threshold: float) -> pd.DataFrame:
    expression = _expression_matrix(protein_exp_file)
    if expression.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    selected = [p for p in proteins if p in expression.columns]
    if len(selected) < 2:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    corr = expression[selected].corr(method=method.lower())
    rows = []
    for i, a in enumerate(selected):
        for b in selected[i + 1:]:
            value = corr.loc[a, b]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append({"Source": a, "Target": b, "EdgeType": "coexpression", "STRINGScore": 0.0, "CoexpressionScore": float(abs(value)), "Confidence": float(abs(value)), "IsExpandedEdge": False})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=EDGE_COLUMNS)


def filter_expanded_interactions(
    string_edges: pd.DataFrame,
    seeds: list[str],
    protein_exp_file: str,
    method: str,
    correlation_threshold: float,
    string_threshold: int = 800,
) -> pd.DataFrame:
    if string_edges.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    expression = _expression_matrix(protein_exp_file)
    if expression.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    measured_lookup = {str(column).casefold(): str(column) for column in expression.columns}
    seed_lookup = {str(seed).casefold(): str(seed) for seed in seeds}
    required_confidence = float(string_threshold) / 1000.0 if float(string_threshold) > 1.0 else float(string_threshold)
    rows = []
    for _, row in string_edges.iterrows():
        source = str(row.get("Source", ""))
        target = str(row.get("Target", ""))
        source_seed = seed_lookup.get(source.casefold())
        target_seed = seed_lookup.get(target.casefold())
        if bool(source_seed) == bool(target_seed):
            continue
        seed = source_seed or target_seed
        partner_raw = target if source_seed else source
        partner = measured_lookup.get(partner_raw.casefold())
        seed_column = measured_lookup.get(str(seed).casefold())
        if partner is None or seed_column is None or partner.casefold() in seed_lookup:
            continue

        string_confidence = float(row.get("STRINGScore", row.get("Confidence", 0.0)) or 0.0)
        normalized_confidence = string_confidence / 1000.0 if string_confidence > 1.0 else string_confidence
        if normalized_confidence < required_confidence:
            continue
        correlation = expression[seed_column].corr(expression[partner], method=method.lower())
        if pd.isna(correlation) or abs(float(correlation)) < float(correlation_threshold):
            continue

        record = {column: row.get(column, np.nan) for column in EDGE_COLUMNS}
        record.update(
            {
                "Source": str(seed),
                "Target": partner,
                "EdgeType": "STRING",
                "STRINGScore": string_confidence,
                "CoexpressionScore": abs(float(correlation)),
                "Confidence": string_confidence,
                "IsExpandedEdge": True,
                "ConnectedSeedProtein": str(seed),
                "STRINGConfidence": string_confidence,
                "CorrelationCoefficient": float(correlation),
                "CorrelationMethod": method.lower(),
            }
        )
        rows.append(record)
    if not rows:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    return pd.DataFrame(rows, columns=EDGE_COLUMNS).drop_duplicates(subset=["Source", "Target"])


def build_graph(
    edges: pd.DataFrame,
    seeds: list[str],
    importance: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    expanded_metadata: bool = False,
) -> nx.Graph:
    graph = nx.Graph()
    graph.graph["ExpandedNetwork"] = bool(expanded_metadata)
    if not importance.empty:
        value_column = "ImportanceValue" if "ImportanceValue" in importance.columns else "SHAP"
        importance_map = importance.set_index("Protein")[value_column].to_dict()
        importance_type_map = (
            importance.set_index("Protein")["ImportanceType"].to_dict()
            if "ImportanceType" in importance.columns
            else {protein: "UnverifiedLegacyImportance" for protein in importance["Protein"].astype(str)}
        )
    else:
        importance_map = {}
        importance_type_map = {}
    shap_map = importance.set_index("Protein")["SHAP"].to_dict() if not importance.empty and "SHAP" in importance.columns else {}
    score_map = scores.set_index("Protein").to_dict(orient="index") if not scores.empty else {}
    seed_set = set(seeds)
    proteins = set(seed_set)
    if not edges.empty:
        proteins |= set(edges["Source"].dropna().astype(str)) | set(edges["Target"].dropna().astype(str))
    expanded_attributes: dict[str, dict] = {}
    if expanded_metadata and not edges.empty:
        expanded_mask = edges["IsExpandedEdge"].astype(bool) if "IsExpandedEdge" in edges.columns else pd.Series(False, index=edges.index)
        for _, row in edges.loc[expanded_mask].iterrows():
            source = str(row.get("Source", ""))
            target = str(row.get("Target", ""))
            partner = target if source in seed_set else source if target in seed_set else ""
            if partner:
                expanded_attributes[partner] = {
                    "ConnectedSeedProtein": str(row.get("ConnectedSeedProtein", source if source in seed_set else target)),
                    "STRINGConfidence": float(row.get("STRINGConfidence", row.get("STRINGScore", np.nan))),
                    "CorrelationCoefficient": float(row.get("CorrelationCoefficient", np.nan)),
                    "CorrelationMethod": str(row.get("CorrelationMethod", "")),
                }
    for protein in sorted(proteins):
        attrs = score_map.get(protein, {})
        is_expanded = protein not in seed_set
        omit_biomarker_scores = expanded_metadata and is_expanded
        expansion = expanded_attributes.get(protein, {})
        graph.add_node(
            protein,
            IsSeedBiomarker=protein in seed_set,
            IsExpandedInteractor=is_expanded,
            ConnectedSeedProtein=str(expansion.get("ConnectedSeedProtein", "")),
            STRINGConfidence=float(expansion.get("STRINGConfidence", np.nan)),
            CorrelationCoefficient=float(expansion.get("CorrelationCoefficient", np.nan)),
            CorrelationMethod=str(expansion.get("CorrelationMethod", "")),
            ImportanceValue=np.nan if omit_biomarker_scores else float(importance_map.get(protein, 0.0)),
            ImportanceType="" if omit_biomarker_scores else str(importance_type_map.get(protein, "ModelSpecificImportance")),
            SHAP=np.nan if omit_biomarker_scores else float(shap_map.get(protein, np.nan)),
            ConsensusScore=np.nan if omit_biomarker_scores else float(attrs.get("ConsensusScore", 0.0)),
            StabilityScore=np.nan if omit_biomarker_scores else float(attrs.get("StabilityScore", 0.0)),
            SupportFrequency=np.nan if omit_biomarker_scores else float(attrs.get("SupportFrequency", 0.0)),
            ConfidenceScore=np.nan if omit_biomarker_scores else float(attrs.get("ConfidenceScore", 0.0)),
            GeneSymbol=_clean_annotation_text(attrs.get("GeneSymbol", "")),
            UniProtID=_clean_annotation_text(attrs.get("UniProtID", "")),
            EntrezID=_clean_annotation_text(attrs.get("EntrezID", "")),
            GOAnnotations=_clean_annotation_text(attrs.get("GOAnnotations", "")),
            STRINGID=_clean_annotation_text(attrs.get("STRINGID", "")),
            STRINGPreferredName=_clean_annotation_text(attrs.get("STRINGPreferredName", "")),
            AnnotationSource=_clean_annotation_text(attrs.get("AnnotationSource", "")),
        )
    for _, row in edges.iterrows():
        source = str(row.get("Source", ""))
        target = str(row.get("Target", ""))
        if not source or not target or source == target:
            continue
        graph.add_edge(source, target, **{col: row.get(col, 0.0) for col in EDGE_COLUMNS if col not in {"Source", "Target"}})
    return graph


def _community_map(graph: nx.Graph) -> dict[str, int]:
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_edges() == 0:
        return {node: idx + 1 for idx, node in enumerate(sorted(graph.nodes))}
    try:
        communities = nx.community.greedy_modularity_communities(graph)
    except Exception:
        communities = [set(component) for component in nx.connected_components(graph)]
    mapping: dict[str, int] = {}
    for idx, community in enumerate(sorted(communities, key=len, reverse=True), start=1):
        for node in community:
            mapping[str(node)] = idx
    return mapping


def _community_hubs(graph: nx.Graph, community: dict[str, int]) -> set[str]:
    hubs: set[str] = set()
    for community_id in sorted(set(community.values())):
        members = [node for node in graph.nodes if community.get(str(node)) == community_id]
        if not members:
            continue
        hub = sorted(
            members,
            key=lambda node: (
                graph.degree(node),
                graph.nodes[node].get("ConsensusScore", 0.0),
                graph.nodes[node].get("StabilityScore", 0.0),
                graph.nodes[node].get("ImportanceValue", 0.0),
                str(node),
            ),
            reverse=True,
        )[0]
        hubs.add(str(hub))
    return hubs


def node_table(graph: nx.Graph) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=NODE_COLUMNS)
    degree = dict(graph.degree())
    betweenness = nx.betweenness_centrality(graph, normalized=True) if graph.number_of_edges() else {n: 0.0 for n in graph.nodes}
    closeness = nx.closeness_centrality(graph) if graph.number_of_edges() else {n: 0.0 for n in graph.nodes}
    community = _community_map(graph)
    community_hubs = _community_hubs(graph, community)
    rows = []
    for node, attrs in graph.nodes(data=True):
        hub_stability = (
            0.45 * float(attrs.get("StabilityScore", 0.0))
            + 0.35 * float(attrs.get("ConsensusScore", 0.0))
            + 0.20 * (1.0 if attrs.get("IsSeedBiomarker", False) else 0.0)
        )
        rows.append({
            "Protein": node,
            "GeneSymbol": _clean_annotation_text(attrs.get("GeneSymbol", "")),
            "DisplayLabel": _display_annotation_label(attrs.get("GeneSymbol", ""), node),
            "UniProtID": _clean_annotation_text(attrs.get("UniProtID", "")),
            "EntrezID": _clean_annotation_text(attrs.get("EntrezID", "")),
            "GOAnnotations": _clean_annotation_text(attrs.get("GOAnnotations", "")),
            "STRINGID": _clean_annotation_text(attrs.get("STRINGID", "")),
            "STRINGPreferredName": _clean_annotation_text(attrs.get("STRINGPreferredName", "")),
            "AnnotationSource": _clean_annotation_text(attrs.get("AnnotationSource", "")),
            "Community": int(community.get(str(node), 0)),
            "IsCommunityHub": str(node) in community_hubs,
            "IsSeedBiomarker": bool(attrs.get("IsSeedBiomarker", False)),
            "IsExpandedInteractor": bool(attrs.get("IsExpandedInteractor", False)),
            "ConnectedSeedProtein": str(attrs.get("ConnectedSeedProtein", "")),
            "STRINGConfidence": float(attrs.get("STRINGConfidence", np.nan)),
            "CorrelationCoefficient": float(attrs.get("CorrelationCoefficient", np.nan)),
            "CorrelationMethod": str(attrs.get("CorrelationMethod", "")),
            "ImportanceValue": float(attrs.get("ImportanceValue", np.nan)),
            "ImportanceType": str(attrs.get("ImportanceType", "")),
            "SHAP": float(attrs.get("SHAP", np.nan)),
            "ConsensusScore": float(attrs.get("ConsensusScore", 0.0)),
            "StabilityScore": float(attrs.get("StabilityScore", 0.0)),
            "SupportFrequency": float(attrs.get("SupportFrequency", 0.0)),
            "ConfidenceScore": float(attrs.get("ConfidenceScore", 0.0)),
            "HubStability": float(hub_stability),
            "Degree": int(degree.get(node, 0)),
            "Betweenness": float(betweenness.get(node, 0.0)),
            "Closeness": float(closeness.get(node, 0.0)),
        })
    df = pd.DataFrame(rows)
    order = df.sort_values(["Degree", "Betweenness", "HubStability", "ConsensusScore"], ascending=False).index
    df["HubRank"] = 0
    df.loc[order, "HubRank"] = np.arange(1, len(order) + 1)
    return df.sort_values("HubRank")[NODE_COLUMNS]


def edge_table(graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for source, target, attrs in graph.edges(data=True):
        rows.append({
            "Source": source,
            "Target": target,
            "EdgeType": attrs.get("EdgeType", "STRING"),
            "STRINGScore": float(attrs.get("STRINGScore", 0.0)),
            "CoexpressionScore": float(attrs.get("CoexpressionScore", 0.0)),
            "Confidence": float(attrs.get("Confidence", 0.0)),
            "IsExpandedEdge": bool(attrs.get("IsExpandedEdge", False)),
            "SourceSTRINGID": attrs.get("SourceSTRINGID", ""),
            "TargetSTRINGID": attrs.get("TargetSTRINGID", ""),
            "ConnectedSeedProtein": attrs.get("ConnectedSeedProtein", ""),
            "STRINGConfidence": float(attrs.get("STRINGConfidence", np.nan)),
            "CorrelationCoefficient": float(attrs.get("CorrelationCoefficient", np.nan)),
            "CorrelationMethod": attrs.get("CorrelationMethod", ""),
        })
    return pd.DataFrame(rows, columns=EDGE_COLUMNS)


def _normalize_layout(pos: dict) -> dict[str, np.ndarray]:
    if not pos:
        return {}
    nodes = list(pos)
    coordinates = np.asarray([pos[node] for node in nodes], dtype=float)
    coordinates -= coordinates.mean(axis=0, keepdims=True)
    span = float(np.max(np.ptp(coordinates, axis=0)))
    if span > 0:
        coordinates /= span
    return {node: coordinates[index] for index, node in enumerate(nodes)}


def _open_connected_component_layout(
    graph: nx.Graph,
    pos: dict,
    minimum_aspect: float = 0.88,
) -> dict[str, np.ndarray]:
    """Open compressed connected components without changing graph topology."""
    opened = {
        node: np.asarray(coordinate, dtype=float).copy()
        for node, coordinate in _normalize_layout(pos).items()
    }
    for members in nx.connected_components(graph):
        if len(members) < 3:
            continue
        ordered = sorted(members, key=str)
        coordinates = np.asarray([opened[node] for node in ordered], dtype=float)
        center = coordinates.mean(axis=0)
        local = coordinates - center
        spans = np.ptp(local, axis=0)
        major = float(np.max(spans))
        minor_axis = int(np.argmin(spans))
        if major <= 1e-12 or spans[minor_axis] <= 1e-12:
            continue
        aspect = float(spans[minor_axis] / major)
        if aspect < minimum_aspect:
            local[:, minor_axis] *= float(minimum_aspect / aspect)
        for node, coordinate in zip(ordered, local + center):
            opened[node] = coordinate
    return _normalize_layout(opened)


def _expand_dense_core_regions(
    graph: nx.Graph,
    pos: dict,
    iterations: int = 560,
) -> dict[str, np.ndarray]:
    """Open only crowded neighborhoods in the largest connected component."""
    expanded = {
        node: np.asarray(coordinate, dtype=float).copy()
        for node, coordinate in pos.items()
    }
    components = list(nx.connected_components(graph))
    if not components:
        return expanded
    core = max(components, key=len)
    if len(core) < 3:
        return expanded
    ordered = sorted(core, key=str)
    coordinates = np.asarray([expanded[node] for node in ordered], dtype=float)
    anchors = coordinates.copy()
    center = coordinates.mean(axis=0)
    span = max(float(np.max(np.ptp(coordinates, axis=0))), 1e-9)
    distances = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :],
        axis=2,
    )
    distances[distances <= 1e-12] = np.inf
    nearest = np.min(distances, axis=1)
    target_distance = max(
        0.155 * span,
        float(np.quantile(nearest, 0.60)) * 1.70,
    )
    maximum_displacement = 0.36 * span

    for _ in range(iterations):
        movement = 0.0
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                delta = coordinates[right] - coordinates[left]
                distance = float(np.linalg.norm(delta))
                if distance >= target_distance:
                    continue
                if distance <= 1e-12:
                    angle = 2.0 * np.pi * (right + 1) / len(ordered)
                    direction = np.array([np.cos(angle), np.sin(angle)])
                else:
                    direction = delta / distance
                offset = 0.58 * (target_distance - distance) * direction
                coordinates[left] -= offset
                coordinates[right] += offset
                movement += float(np.linalg.norm(offset))
        coordinates += 0.014 * (anchors - coordinates)
        coordinates += center - coordinates.mean(axis=0)
        displacement = coordinates - anchors
        displacement_length = np.linalg.norm(displacement, axis=1)
        over_limit = displacement_length > maximum_displacement
        if over_limit.any():
            displacement[over_limit] *= (
                maximum_displacement / displacement_length[over_limit]
            )[:, None]
            coordinates[over_limit] = anchors[over_limit] + displacement[over_limit]
        if movement < 1e-5 * span:
            break

    for node, coordinate in zip(ordered, coordinates):
        expanded[node] = coordinate
    return expanded


def _arrange_disconnected_components(graph: nx.Graph, pos: dict) -> dict[str, np.ndarray]:
    """Give the largest connected component priority and pack isolates compactly."""
    components = sorted(
        (set(component) for component in nx.connected_components(graph)),
        key=lambda members: (-len(members), tuple(sorted(str(node) for node in members))),
    )
    if len(components) <= 1:
        return _normalize_layout(pos)

    connected = [members for members in components if len(members) > 1]
    isolates = [next(iter(members)) for members in components if len(members) == 1]
    if not isolates:
        return _normalize_layout(pos)

    arranged: dict[str, np.ndarray] = {}
    if connected:
        for component_index, members in enumerate(connected):
            ordered = sorted(members, key=str)
            local = np.asarray([pos[node] for node in ordered], dtype=float)
            local -= local.mean(axis=0, keepdims=True)
            if len(ordered) == 2:
                local = np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=float)
            else:
                span = float(np.max(np.ptp(local, axis=0)))
                if span > 1e-12:
                    local /= span
            if not np.isfinite(local).all() or float(np.max(np.ptp(local, axis=0))) <= 1e-12:
                angles = np.linspace(0.0, 2.0 * np.pi, len(ordered), endpoint=False)
                local = np.column_stack([np.cos(angles), np.sin(angles)])
            if component_index == 0:
                center = np.array([0.0, 0.36])
                if len(ordered) > 2:
                    component_span = np.maximum(np.ptp(local, axis=0), 1e-12)
                    scale = np.array([1.34, 1.34]) / component_span
                else:
                    scale = np.array([1.34, 1.34])
            else:
                side = -1.0 if component_index % 2 else 1.0
                row = (component_index - 1) // 2
                center = np.array([0.62 * side, 0.02 - 0.22 * row])
                component_scale = max(0.22, 0.36 / np.sqrt(max(1, len(connected) - 1)))
                scale = np.array([component_scale, component_scale * 0.78])
            for node, coordinate in zip(ordered, local):
                arranged[node] = center + coordinate * scale

    ordered_isolates = sorted(isolates, key=str)
    if connected:
        columns = min(8, max(4, int(np.ceil(np.sqrt(len(ordered_isolates) * 2.4)))))
        rows = int(np.ceil(len(ordered_isolates) / columns))
        x_positions = np.linspace(-0.72, 0.72, columns)
        y_positions = np.linspace(-0.58, -0.94, max(1, rows))
    else:
        columns = max(1, int(np.ceil(np.sqrt(len(ordered_isolates)))))
        rows = int(np.ceil(len(ordered_isolates) / columns))
        x_positions = np.linspace(-0.80, 0.80, columns)
        y_positions = np.linspace(0.70, -0.70, max(1, rows))
    for index, node in enumerate(ordered_isolates):
        row, column = divmod(index, columns)
        arranged[node] = np.array([x_positions[column], y_positions[row]])

    return _normalize_layout(arranged)


def _minimum_node_distance(node_count: int) -> float:
    return float(np.clip(0.36 / np.sqrt(max(1, int(node_count))), 0.035, 0.11))


def _repel_node_positions(
    pos: dict,
    minimum_distance: float | None = None,
    iterations: int = 1000,
) -> dict[str, np.ndarray]:
    normalized = _normalize_layout(pos)
    nodes = sorted(normalized, key=str)
    coordinates = np.asarray([normalized[node] for node in nodes], dtype=float)
    minimum_distance = (
        _minimum_node_distance(len(nodes))
        if minimum_distance is None
        else float(minimum_distance)
    )
    for _ in range(iterations):
        moved = False
        for left in range(len(nodes)):
            for right in range(left + 1, len(nodes)):
                delta = coordinates[right] - coordinates[left]
                distance = float(np.linalg.norm(delta))
                if distance >= minimum_distance:
                    continue
                if distance <= 1e-12:
                    angle = 2.0 * np.pi * (right + 1) / max(2, len(nodes))
                    direction = np.array([np.cos(angle), np.sin(angle)])
                else:
                    direction = delta / distance
                offset = 0.52 * (minimum_distance - distance) * direction
                coordinates[left] -= offset
                coordinates[right] += offset
                moved = True
        coordinates -= coordinates.mean(axis=0, keepdims=True)
        span = float(np.max(np.ptp(coordinates, axis=0)))
        if span > 0:
            coordinates /= span
        if not moved:
            break
    return {node: coordinates[index] for index, node in enumerate(nodes)}


def _nearest_node_distances(pos: dict, nodes: list[str]) -> dict[str, float]:
    """Return deterministic nearest-neighbour spacing for local label bounds."""
    coordinates = np.asarray([pos[node] for node in nodes], dtype=float)
    if len(nodes) <= 1:
        return {node: 1.0 for node in nodes}
    distances = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :],
        axis=2,
    )
    distances[distances <= 1e-12] = np.inf
    return {
        node: float(np.min(distances[index]))
        for index, node in enumerate(nodes)
    }


def _label_distance_bounds(
    pos: dict,
    nodes: list[str],
    degrees: dict[str, int] | None = None,
) -> dict[str, tuple[float, float]]:
    """Keep label offsets proportional to each node's local free space."""
    degrees = degrees or {}
    coordinates = np.asarray([pos[node] for node in nodes], dtype=float)
    span = max(float(np.max(np.ptp(coordinates, axis=0))), 1.0)
    nearest = _nearest_node_distances(pos, nodes)
    bounds: dict[str, tuple[float, float]] = {}
    for node in nodes:
        if degrees.get(node, 1) == 0:
            bounds[node] = (0.028 * span, 0.052 * span)
            continue
        local = nearest[node]
        minimum = max(0.032 * span, min(0.052 * span, 0.32 * local))
        maximum = min(0.082 * span, max(0.058 * span, 0.56 * local))
        bounds[node] = (minimum, max(minimum + 0.010 * span, maximum))
    return bounds


def _leader_line_required(
    node_position: np.ndarray,
    label_position: np.ndarray,
    span: float,
) -> bool:
    """Use a leader whenever a label is visibly separated from its node."""
    return float(np.linalg.norm(label_position - node_position)) > 0.024 * span


def _network_layout(graph: nx.Graph, community: dict[str, int] | None = None) -> dict[str, np.ndarray]:
    if graph.number_of_nodes() == 0:
        return {}
    community = community or {}

    def finalize(pos):
        arranged = _open_connected_component_layout(graph, pos)
        arranged = _arrange_disconnected_components(graph, arranged)
        arranged = _expand_dense_core_regions(graph, arranged)
        arranged = _repel_node_positions(arranged)
        coordinates = np.asarray([arranged[node] for node in graph.nodes], dtype=float)
        if coordinates.shape != (graph.number_of_nodes(), 2) or not np.isfinite(coordinates).all():
            raise ValueError("PPI layout returned invalid node coordinates.")
        rounded = np.round(coordinates, decimals=12)
        if len(np.unique(rounded, axis=0)) != graph.number_of_nodes():
            raise ValueError("PPI layout returned duplicate node coordinates.")
        return arranged

    for layout in (
        getattr(getattr(nx, "nx_agraph", None), "graphviz_layout", None),
        getattr(getattr(nx, "nx_pydot", None), "graphviz_layout", None),
    ):
        if layout is None:
            continue
        try:
            return finalize(layout(graph, prog="sfdp", args=f"-Gstart={DEFAULT_RANDOM_STATE} -Goverlap=prism -Gsep=+18"))
        except Exception:
            pass
    try:
        from fa2 import ForceAtlas2

        forceatlas2 = ForceAtlas2(
            outboundAttractionDistribution=False,
            linLogMode=False,
            adjustSizes=True,
            edgeWeightInfluence=1.0,
            jitterTolerance=1.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=1.0,
            verbose=False,
        )
        return finalize(forceatlas2.forceatlas2_networkx_layout(graph, pos=nx.spring_layout(graph, seed=DEFAULT_RANDOM_STATE), iterations=1000))
    except Exception:
        pass
    try:
        layout = getattr(nx, "forceatlas2_layout")
        return finalize(layout(graph, seed=DEFAULT_RANDOM_STATE, max_iter=1000))
    except Exception:
        pass
    try:
        return finalize(nx.kamada_kawai_layout(graph))
    except Exception:
        return finalize(nx.spring_layout(graph, seed=DEFAULT_RANDOM_STATE, k=1.1, iterations=500))


def _label_nodes(graph: nx.Graph) -> list[str]:
    return sorted(graph.nodes, key=str)


def _display_node_label(graph: nx.Graph, node: str) -> str:
    return _display_annotation_label(graph.nodes[node].get("GeneSymbol", ""), node)


def _repelled_label_positions(
    pos: dict,
    nodes: list[str],
    labels: dict[str, str] | None = None,
    degrees: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    coordinates = np.asarray([pos[node] for node in nodes], dtype=float)
    degrees = degrees or {}
    connected_coordinates = np.asarray(
        [pos[node] for node in nodes if degrees.get(node, 1) > 0],
        dtype=float,
    )
    center = (
        connected_coordinates.mean(axis=0)
        if connected_coordinates.size
        else coordinates.mean(axis=0)
    )
    span = max(float(np.max(np.ptp(coordinates, axis=0))), 1.0)
    bounds = _label_distance_bounds(pos, nodes, degrees)
    nearest = _nearest_node_distances(pos, nodes)
    label_positions: dict[str, np.ndarray] = {}
    anchors: dict[str, np.ndarray] = {}
    for index, node in enumerate(nodes):
        node_position = np.asarray(pos[node], dtype=float)
        if degrees.get(node, 1) == 0:
            direction = np.array([-1.0 if node_position[0] <= center[0] else 1.0, 0.0])
        else:
            direction = node_position - center
            length = float(np.linalg.norm(direction))
            if length < 1e-9:
                angle = 2.0 * np.pi * index / max(1, len(nodes))
                direction = np.array([np.cos(angle), np.sin(angle)])
            else:
                direction /= length
        minimum, maximum = bounds[node]
        offset = min(maximum, max(minimum, 0.46 * nearest[node]))
        anchors[node] = node_position + direction * offset
        label_positions[node] = anchors[node].copy()

    labels = labels or {node: str(node) for node in nodes}
    widths = {node: (0.030 + 0.013 * len(labels[node])) * span for node in nodes}
    height = 0.046 * span
    lower = coordinates.min(axis=0) - 0.12 * span
    upper = coordinates.max(axis=0) + 0.12 * span
    for _ in range(700):
        movement = 0.0
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1 :]:
                delta = label_positions[right] - label_positions[left]
                overlap_x = (widths[left] + widths[right]) / 2.0 - abs(delta[0])
                overlap_y = height - abs(delta[1])
                if overlap_x <= 0 or overlap_y <= 0:
                    continue
                if overlap_x < overlap_y:
                    sign = 1.0 if delta[0] >= 0 else -1.0
                    shift = np.array([sign * (overlap_x / 2.0 + 0.003 * span), 0.0])
                else:
                    sign = 1.0 if delta[1] >= 0 else -1.0
                    shift = np.array([0.0, sign * (overlap_y / 2.0 + 0.003 * span)])
                label_positions[left] -= shift
                label_positions[right] += shift
                movement += float(np.linalg.norm(shift))
        for node in nodes:
            for coordinate in coordinates:
                delta = label_positions[node] - coordinate
                distance = float(np.linalg.norm(delta))
                minimum = max(0.036 * span, min(0.055 * span, 0.38 * nearest[node]))
                if distance < minimum:
                    if distance < 1e-9:
                        delta = np.array([0.0, 1.0])
                        distance = 1.0
                    shift = delta / distance * (minimum - distance)
                    label_positions[node] += shift
                    movement += float(np.linalg.norm(shift))
            label_positions[node] += 0.24 * (anchors[node] - label_positions[node])
            label_positions[node] = np.clip(label_positions[node], lower, upper)
        label_positions = _constrain_label_positions(
            pos,
            nodes,
            label_positions,
            degrees,
        )
        if movement < 1e-5 * span:
            break
    return label_positions


def _constrain_label_positions(
    pos: dict,
    nodes: list[str],
    label_positions: dict[str, np.ndarray],
    degrees: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """Keep publication labels local to their nodes after collision adjustment."""
    degrees = degrees or {}
    node_coordinates = np.asarray([pos[node] for node in nodes], dtype=float)
    center = node_coordinates.mean(axis=0)
    bounds = _label_distance_bounds(pos, nodes, degrees)
    constrained: dict[str, np.ndarray] = {}
    for index, node in enumerate(nodes):
        node_position = np.asarray(pos[node], dtype=float)
        label_position = np.asarray(label_positions[node], dtype=float)
        delta = label_position - node_position
        distance = float(np.linalg.norm(delta))
        minimum, maximum = bounds[node]
        if distance <= 1e-12:
            if degrees.get(node, 1) == 0:
                direction = np.array([-1.0 if node_position[0] <= center[0] else 1.0, 0.0])
            else:
                direction = node_position - center
                length = float(np.linalg.norm(direction))
                if length <= 1e-12:
                    angle = 2.0 * np.pi * index / max(1, len(nodes))
                    direction = np.array([np.cos(angle), np.sin(angle)])
                else:
                    direction /= length
        else:
            direction = delta / distance
        bounded_distance = float(np.clip(distance, minimum, maximum))
        constrained[node] = node_position + direction * bounded_distance
    return constrained


def _confidence01(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size and np.nanmax(arr) > 1.0:
        arr = arr / 1000.0
    return np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0)


def _publication_colormap(name: str):
    try:
        return matplotlib.colormaps.get_cmap(name)
    except AttributeError:
        legacy_get_cmap = getattr(cm, "get_cmap")
        return legacy_get_cmap(name)


def _community_colors(graph: nx.Graph) -> dict[int, tuple[float, float, float, float]]:
    communities = sorted({int(graph.nodes[n].get("Community", 0) or 0) for n in graph.nodes})
    n_colors = max(1, len(communities))
    positions = np.array([0.5]) if n_colors == 1 else np.linspace(0.0, 1.0, n_colors, endpoint=False)
    cmap = _publication_colormap("tab20")
    return {community: cmap(float(position)) for community, position in zip(communities, positions)}


def _edge_render_style(attrs: dict) -> dict[str, str | float]:
    """Return the single rendering and legend style for one PPI edge."""
    confidence = float(_confidence01([attrs.get("Confidence", 0.0)])[0])
    edge_type = str(attrs.get("EdgeType", "STRING")).lower()
    is_expanded = bool(attrs.get("IsExpandedEdge", False))
    if edge_type == "coexpression":
        color, linestyle, label = "#B45309", "--", "Co-expression interaction"
    elif is_expanded:
        color, linestyle, label = "#6B7280", ":", "Expanded STRING interaction"
    else:
        color, linestyle, label = "#6B7280", "-", "STRING interaction"
    return {
        "color": color,
        "linestyle": linestyle,
        "label": label,
        "width": 0.30 + 3.00 * confidence,
        "alpha": 0.16 + 0.56 * confidence,
    }


def _rendered_edges(graph: nx.Graph) -> list[tuple[str, str, dict]]:
    """Return the exact candidate edges passed to the network edge renderer."""
    return [(source, target, attrs) for source, target, attrs in graph.edges(data=True)]


def _visualization_graph(
    graph: nx.Graph,
    rendered_edges: list[tuple[str, str, dict]] | None = None,
) -> nx.Graph:
    """Return a rendering-only copy with exactly the endpoints of rendered edges."""
    edges = _rendered_edges(graph) if rendered_edges is None else rendered_edges
    return graph.edge_subgraph([(source, target) for source, target, _ in edges]).copy()


def draw_network(graph: nx.Graph, title: str, out: str) -> bool:
    full_node_count = graph.number_of_nodes()
    rendered_edges = _rendered_edges(graph)
    display_graph = _visualization_graph(graph, rendered_edges)
    node_count = display_graph.number_of_nodes()
    figure_width = min(17.5, max(12.5, 12.0 + 0.060 * node_count))
    figure_height = min(13.5, max(9.5, 9.0 + 0.050 * node_count))
    fig = plt.figure(figsize=(figure_width, figure_height))
    ax = fig.add_axes([0.035, 0.155, 0.93, 0.735])
    label_font = resolve_publication_font()
    if full_node_count == 0:
        plt.close(fig)
        output = Path(out)
        for path in (output, output.with_suffix(".pdf"), output.with_suffix(".tiff")):
            if path.exists():
                path.unlink()
        logging.warning("PPI network figure omitted because no valid network nodes were available: %s", out)
        return False
    if node_count == 0:
        ax.axis("off")
        fig.suptitle(title, y=0.982, fontsize=17, fontweight="bold")
        ax.text(
            0.5,
            0.55,
            "No qualifying PPI interactions were identified.",
            ha="center",
            va="center",
            fontsize=13,
            color="#374151",
            transform=ax.transAxes,
        )
        fig.text(
            0.5,
            0.12,
            "Biomarkers without qualifying interactions remain part of the biomarker recommendation set.",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#4B5563",
        )
        fig.savefig(out, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        return True
    graph = display_graph
    node_list = list(graph.nodes)
    edge_count = len(rendered_edges)
    community = _community_map(graph)
    pos = _network_layout(graph, community)
    largest_component = max(
        (set(component) for component in nx.connected_components(graph)),
        key=len,
        default=set(),
    )
    hubs = _community_hubs(graph, community)
    for node in node_list:
        graph.nodes[node]["Community"] = community.get(str(node), 0)
        graph.nodes[node]["IsCommunityHub"] = str(node) in hubs
    expanded_network = bool(graph.graph.get("ExpandedNetwork", False))
    expanded_mask = np.array([bool(graph.nodes[n].get("IsExpandedInteractor", False)) for n in node_list])
    importance_vals = np.nan_to_num(np.array([graph.nodes[n].get("ImportanceValue", 0.0) for n in node_list], dtype=float), nan=0.0)
    confidence_vals = _confidence01(np.nan_to_num([graph.nodes[n].get("ConfidenceScore", 0.0) for n in node_list], nan=0.0))
    if expanded_network:
        color_values = importance_vals
        color_label = "Model attribution / importance"
    else:
        color_values = importance_vals if np.nanmax(np.abs(importance_vals)) > 0 else confidence_vals
        color_label = "Model attribution / importance" if np.nanmax(np.abs(importance_vals)) > 0 else "Confidence score"
    size_basis = importance_vals
    stability = np.nan_to_num(np.array([graph.nodes[n].get("StabilityScore", 0.0) for n in node_list], dtype=float), nan=0.0)
    density_scale = min(1.0, (45.0 / max(45, len(node_list))) ** 0.45)
    sizes = density_scale * (230 + 820 * robust_minmax(pd.Series(size_basis)).to_numpy())
    borders = 0.9 + 3.6 * robust_minmax(pd.Series(stability)).to_numpy()
    alphas = 0.60 + 0.36 * robust_minmax(pd.Series(confidence_vals)).to_numpy()
    if expanded_network:
        sizes[expanded_mask] = density_scale * 300.0
        borders[expanded_mask] = 1.0
        alphas[expanded_mask] = 0.78
    vmin = float(np.nanmin(color_values))
    vmax = float(np.nanmax(color_values))
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = _publication_colormap("viridis")

    ellipse_extents: list[np.ndarray] = []
    if len(largest_component) > 1:
        from matplotlib.patches import Ellipse

        component_coordinates = np.asarray(
            [pos[node] for node in sorted(largest_component, key=str)],
            dtype=float,
        )
        component_center = component_coordinates.mean(axis=0)
        component_span = np.ptp(component_coordinates, axis=0)
        ellipse_width = max(0.70, float(component_span[0]) + 0.34)
        ellipse_height = max(0.48, float(component_span[1]) + 0.28)
        ellipse_extents.extend(
            [
                component_center - np.array([ellipse_width / 2.0, ellipse_height / 2.0]),
                component_center + np.array([ellipse_width / 2.0, ellipse_height / 2.0]),
            ]
        )
        ax.add_patch(
            Ellipse(
                component_center,
                width=ellipse_width,
                height=ellipse_height,
                facecolor="#EFF6FF",
                edgecolor="#BFDBFE",
                linewidth=0.8,
                alpha=0.62,
                zorder=0,
            )
        )

    rendered_edge_styles: dict[str, dict[str, str | float]] = {}
    for source, target, attrs in rendered_edges:
        edge_style = _edge_render_style(attrs)
        rendered_edge_styles.setdefault(str(edge_style["label"]), edge_style)
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(source, target)],
            ax=ax,
            width=float(edge_style["width"]),
            edge_color=str(edge_style["color"]),
            alpha=float(edge_style["alpha"]),
            style=str(edge_style["linestyle"]),
            arrows=False,
        )

    shape_groups = {
        "Seed biomarker": ("o", [n for n in node_list if graph.nodes[n].get("IsSeedBiomarker")]),
        "Expanded interactor": ("D", [n for n in node_list if graph.nodes[n].get("IsExpandedInteractor")]),
        "Network protein": (
            "^",
            [n for n in node_list if not graph.nodes[n].get("IsSeedBiomarker") and not graph.nodes[n].get("IsExpandedInteractor")],
        ),
    }
    node_index = {node: idx for idx, node in enumerate(node_list)}
    node_artist = None
    rendered_node_groups: list[tuple[str, str, tuple[float, ...]]] = []
    for group_name, (shape, nodes_for_shape) in shape_groups.items():
        if not nodes_for_shape:
            continue
        idx = [node_index[n] for n in nodes_for_shape]
        if expanded_network and group_name == "Expanded interactor":
            rgba = np.tile(np.array([0.56, 0.59, 0.63, 0.78]), (len(idx), 1))
        else:
            rgba = cmap(norm(color_values[idx]))
            rgba[:, 3] = alphas[idx]
        artist = nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes_for_shape,
            ax=ax,
            node_shape=shape,
            node_color=rgba,
            node_size=sizes[idx],
            linewidths=borders[idx],
            edgecolors="#111827",
        )
        rendered_node_groups.append((group_name, shape, tuple(rgba[0])))
        node_artist = artist if node_artist is None else node_artist

    label_nodes = _label_nodes(graph)
    display_labels = {node: _display_node_label(graph, node) for node in label_nodes}
    node_degrees = {node: int(graph.degree(node)) for node in label_nodes}
    coordinate_array = np.asarray([pos[node] for node in label_nodes], dtype=float)
    label_positions = _repelled_label_positions(
        pos,
        label_nodes,
        display_labels,
        node_degrees,
    )
    label_size = 10.0 if len(label_nodes) <= 24 else 8.5 if len(label_nodes) <= 60 else 7.5
    texts = []
    for node in label_nodes:
        label_x, label_y = label_positions[node]
        texts.append(ax.text(
            label_x,
            label_y,
            display_labels[node],
            fontsize=label_size,
            family=label_font,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.16",
                "fc": "#FFFFFF",
                "ec": "#D1D5DB",
                "lw": 0.45,
                "alpha": 0.94,
            },
        ))
    label_positions = _constrain_label_positions(
        pos,
        label_nodes,
        label_positions,
        node_degrees,
    )
    for node, text in zip(label_nodes, texts):
        text.set_position(label_positions[node])
    label_span = max(float(np.max(np.ptp(coordinate_array, axis=0))), 1.0)
    for node in label_nodes:
        node_xy = np.asarray(pos[node], dtype=float)
        label_xy = label_positions[node]
        if not _leader_line_required(node_xy, label_xy, label_span):
            continue
        ax.plot(
            [node_xy[0], label_xy[0]],
            [node_xy[1], label_xy[1]],
            color="#9CA3AF",
            linewidth=0.42,
            alpha=0.82,
            solid_capstyle="round",
            zorder=1.8,
        )
    if node_artist is not None:
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(color_values)
        colorbar_axis = fig.add_axes([0.77, 0.925, 0.18, 0.014])
        cbar = fig.colorbar(mappable, cax=colorbar_axis, orientation="horizontal")
        cbar.ax.tick_params(labelsize=7, length=2, pad=1)
        cbar.set_label(color_label, fontsize=8, labelpad=2)

    node_legend_labels = {
        "Seed biomarker": "Recommended biomarker (circle; importance-colored)",
        "Expanded interactor": "Expanded interactor (diamond; neutral gray)",
        "Network protein": "Network protein (triangle; importance-colored)",
    }
    legend_items = [
        Line2D(
            [0],
            [0],
            marker=shape,
            color="none",
            label=node_legend_labels[group_name],
            markerfacecolor=facecolor,
            markeredgecolor="#111827",
            markersize=8,
        )
        for group_name, shape, facecolor in rendered_node_groups
    ]
    legend_items.extend(
        Line2D(
            [0],
            [0],
            color=str(edge_style["color"]),
            lw=float(edge_style["width"]),
            alpha=float(edge_style["alpha"]),
            linestyle=str(edge_style["linestyle"]),
            label=str(edge_style["label"]),
        )
        for edge_style in rendered_edge_styles.values()
    )
    fig.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.064),
        frameon=False,
        fontsize=8.2,
        ncol=max(1, min(4, len(legend_items))),
        handlelength=2.8,
        columnspacing=1.5,
    )
    ax.axis("off")
    coordinate_groups = [
        np.asarray(list(pos.values()), dtype=float),
        np.asarray(list(label_positions.values()), dtype=float),
    ]
    if ellipse_extents:
        coordinate_groups.append(np.asarray(ellipse_extents, dtype=float))
    coordinates = np.vstack(coordinate_groups)
    x_min, y_min = coordinates.min(axis=0)
    x_max, y_max = coordinates.max(axis=0)
    x_pad = max(0.04, (x_max - x_min) * 0.045)
    y_pad = max(0.04, (y_max - y_min) * 0.045)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    fig.suptitle(title, y=0.982, fontsize=17, fontweight="bold")
    fig.text(
        0.04,
        0.942,
        (
            f"{node_count} proteins with qualifying interactions  |  {edge_count} observed interactions  |  "
            f"largest connected component: {len(largest_component)}"
        ),
        ha="left",
        va="center",
        fontsize=8.5,
        color="#4B5563",
    )
    node_encoding_note = (
        "Recommended biomarker fill encodes model attribution or importance; expanded interactors use neutral gray. "
        "Border width encodes stability; thin leader lines associate displaced labels with nodes."
        if expanded_network
        else
        "Node fill encodes model attribution or importance; border width encodes stability. "
        "Thin leader lines associate displaced labels with nodes."
    )
    fig.text(
        0.5,
        0.036,
        node_encoding_note,
        ha="center",
        va="center",
        fontsize=7.8,
        color="#4B5563",
    )
    fig.text(
        0.5,
        0.014,
        "Biomarkers without qualifying interactions are omitted from this network visualization but remain recommended. "
        "Nodes without Gene Symbols retain their original protein accession.",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#6B7280",
    )
    fig.savefig(out, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return True


def plot_hub_proteins(nodes: pd.DataFrame, title: str, out: str) -> bool:
    """Visualize the existing production hub ranking without recomputing centrality."""
    required = {"HubRank", "Degree"}
    if nodes.empty or not required.issubset(nodes.columns):
        return False
    view = nodes.copy()
    view["HubRank"] = pd.to_numeric(view["HubRank"], errors="coerce")
    view["Degree"] = pd.to_numeric(view["Degree"], errors="coerce")
    connected = view[view["Degree"] > 0]
    if not connected.empty:
        view = connected
    view = view.dropna(subset=["HubRank", "Degree"]).sort_values("HubRank").head(15)
    if view.empty:
        return False
    labels = (
        view["DisplayLabel"].fillna("").astype(str)
        if "DisplayLabel" in view.columns
        else view["GeneSymbol"].fillna("").astype(str)
        if "GeneSymbol" in view.columns
        else view["Protein"].astype(str)
    )
    if "Protein" in view.columns:
        labels = [
            _display_annotation_label(label, protein)
            for label, protein in zip(labels, view["Protein"].astype(str))
        ]
    display = view.assign(_DisplayLabel=labels).iloc[::-1]
    configure_publication_plots()
    fig, ax = plt.subplots(figsize=(8.3, max(4.8, 0.33 * len(display) + 1.8)))
    y = np.arange(len(display))
    degree = display["Degree"].to_numpy(dtype=float)
    ax.hlines(y, 0.0, degree, color="#9CA3AF", linewidth=1.8, zorder=1)
    ax.scatter(
        degree,
        y,
        s=70,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.7,
        zorder=2,
    )
    ax.set_yticks(y, display["_DisplayLabel"].astype(str))
    ax.set_xlabel("Network degree (number of observed interactions)")
    ax.set_ylabel("Protein")
    ax.set_title(f"{title}\nNetwork Degree")
    max_degree = float(np.nanmax(degree)) if degree.size else 0.0
    offset = max(max_degree * 0.025, 0.12)
    for position, value in zip(y, degree):
        ax.text(value + offset, position, f"{value:.0f}", va="center", fontsize=8)
    ax.set_xlim(0, max_degree + max(offset * 5.0, 1.0))
    style_publication_axis(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def _write_network(prefix: str, tag: str, graph: nx.Graph, title: str, mirror_png: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = node_table(graph)
    edges = edge_table(graph)
    nodes.to_csv(f"{prefix}_{tag}_nodes.csv", index=False)
    edges.to_csv(f"{prefix}_{tag}_edges.csv", index=False)
    hubs_path = f"{prefix}_hub_proteins.csv"
    nodes.head(50).to_csv(hubs_path, index=False)
    figure_name = f"{prefix}_{tag}.png" if tag.endswith("network") else f"{prefix}_{tag}_network.png"
    if tag == "ppi_network":
        figure_name = f"{prefix}_ppi_network.png"
    if tag == "ppi_expanded_network":
        figure_name = f"{prefix}_ppi_expanded_network.png"
    draw_network(graph, title, figure_name)
    plot_hub_proteins(
        nodes,
        f"{title}: Hub Proteins",
        str(Path(figure_name).with_name(Path(figure_name).stem + "_hub_proteins.png")),
    )
    if mirror_png:
        draw_network(graph, title, mirror_png)
    return nodes, edges


def _network_statistics(name: str, graph: nx.Graph) -> list[dict[str, str | int | float]]:
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    components = nx.number_connected_components(graph) if nodes else 0
    largest_component = max((len(c) for c in nx.connected_components(graph)), default=0) if nodes else 0
    average_degree = float(np.mean([degree for _, degree in graph.degree()])) if nodes else 0.0
    density = float(nx.density(graph)) if nodes > 1 else 0.0
    clustering = float(nx.average_clustering(graph)) if nodes > 1 else 0.0
    return [
        {"Section": name, "Metric": "Number of Nodes", "Value": nodes},
        {"Section": name, "Metric": "Number of Edges", "Value": edges},
        {"Section": name, "Metric": "Connected Components", "Value": components},
        {"Section": name, "Metric": "Average Degree", "Value": average_degree},
        {"Section": name, "Metric": "Graph Density", "Value": density},
        {"Section": name, "Metric": "Clustering Coefficient", "Value": clustering},
        {"Section": name, "Metric": "Largest Connected Component", "Value": largest_component},
    ]


def _community_summary(name: str, graph: nx.Graph) -> list[dict[str, str | int | float]]:
    mapping = _community_map(graph)
    if not mapping:
        return []
    rows = []
    community_series = pd.Series(mapping)
    for community_id, proteins in community_series.groupby(community_series):
        members = sorted(proteins.index.astype(str).tolist())
        rows.append({
            "Section": f"{name} Communities",
            "Metric": f"Community {community_id}",
            "Value": ";".join(members[:25]),
        })
    return rows


def write_community_summary(prefix: str, nodes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for network, table in nodes.items():
        if table.empty or "Community" not in table.columns:
            continue
        for community_id, group in table.groupby("Community", sort=True):
            ranked = group.sort_values(["Degree", "Betweenness", "HubStability"], ascending=False)
            rows.append({
                "Network": network,
                "CommunityID": int(community_id),
                "NumberOfNodes": int(len(group)),
                "TopHubProtein": ranked.iloc[0]["Protein"] if not ranked.empty else "",
                "MeanConfidenceScore": float(pd.to_numeric(group["ConfidenceScore"], errors="coerce").fillna(0.0).mean()),
                "MeanImportance": float(pd.to_numeric(group["ImportanceValue"], errors="coerce").fillna(0.0).mean()),
                "MeanStability": float(pd.to_numeric(group["StabilityScore"], errors="coerce").fillna(0.0).mean()),
            })
    out = pd.DataFrame(rows, columns=["Network", "CommunityID", "NumberOfNodes", "TopHubProtein", "MeanConfidenceScore", "MeanImportance", "MeanStability"])
    out.to_csv(f"{prefix}_community_summary.csv", index=False)
    out.to_csv("community_summary.csv", index=False)
    return out


def _supporting_model_count(prefix: str) -> dict[str, int]:
    path = f"{prefix}_consensus_biomarkers.csv"
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "Protein" not in df.columns:
        return {}
    if "SupportingModels" not in df.columns:
        return {str(p): 1 for p in df["Protein"].astype(str)}
    return {
        str(row["Protein"]): len([m for m in str(row.get("SupportingModels", "")).split(";") if m])
        for _, row in df.iterrows()
    }


def _outer_fold_counts(prefix: str) -> dict[str, int]:
    path = f"{prefix}_biomarker_frequency.csv"
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "Protein" not in df.columns:
        return {}
    count_col = "Frequency" if "Frequency" in df.columns else "AppearsAcrossOuterFolds" if "AppearsAcrossOuterFolds" in df.columns else None
    if count_col is None:
        return {}
    return dict(zip(df["Protein"].astype(str), pd.to_numeric(df[count_col], errors="coerce").fillna(0).astype(int)))


def write_hub_stability(prefix: str, nodes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    supporting_models = _supporting_model_count(prefix)
    outer_folds = _outer_fold_counts(prefix)
    rows = []
    proteins = sorted(set().union(*[set(table["Protein"].astype(str)) for table in nodes.values() if not table.empty]))
    for protein in proteins:
        records = []
        for network, table in nodes.items():
            if table.empty:
                continue
            subset = table[table["Protein"].astype(str) == protein]
            if subset.empty:
                continue
            row = subset.iloc[0]
            records.append({
                "Degree": float(row.get("Degree", 0.0)),
                "Betweenness": float(row.get("Betweenness", 0.0)),
                "Closeness": float(row.get("Closeness", 0.0)),
                "ConsensusScore": float(row.get("ConsensusScore", 0.0)),
                "StabilityScore": float(row.get("StabilityScore", 0.0)),
            })
        if not records:
            continue
        rec = pd.DataFrame(records)
        rows.append({
            "Protein": protein,
            "AppearsAcrossConsensusModels": int(supporting_models.get(protein, 0)),
            "AppearsAcrossOuterFolds": int(outer_folds.get(protein, 0)),
            "DegreeMean": float(rec["Degree"].mean()),
            "DegreeSD": float(rec["Degree"].std(ddof=0)),
            "BetweennessMean": float(rec["Betweenness"].mean()),
            "BetweennessSD": float(rec["Betweenness"].std(ddof=0)),
            "ClosenessMean": float(rec["Closeness"].mean()),
            "ClosenessSD": float(rec["Closeness"].std(ddof=0)),
            "ConsensusScore": float(rec["ConsensusScore"].max()),
            "StabilityScore": float(rec["StabilityScore"].max()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["HubStabilityScore"] = (
            0.30 * robust_minmax(out["DegreeMean"])
            + 0.20 * robust_minmax(out["BetweennessMean"])
            + 0.20 * robust_minmax(out["AppearsAcrossConsensusModels"])
            + 0.20 * robust_minmax(out["AppearsAcrossOuterFolds"])
            + 0.10 * robust_minmax(out["StabilityScore"])
        )
        out = out.sort_values("HubStabilityScore", ascending=False)
    else:
        out = pd.DataFrame(columns=[
            "Protein", "AppearsAcrossConsensusModels", "AppearsAcrossOuterFolds", "DegreeMean", "DegreeSD",
            "BetweennessMean", "BetweennessSD", "ClosenessMean", "ClosenessSD", "ConsensusScore",
            "StabilityScore", "HubStabilityScore"
        ])
    out.to_csv(f"{prefix}_hub_stability.csv", index=False)
    out.to_csv("hub_stability.csv", index=False)
    return out


def _top_table_rows(prefix: str, filename: str, label: str, score_columns: list[str]) -> list[dict[str, str | int | float]]:
    path = f"{prefix}_{filename}"
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    protein_col = "Protein" if "Protein" in df.columns else "Feature" if "Feature" in df.columns else None
    if protein_col is None:
        return []
    score_col = next((c for c in score_columns if c in df.columns), None)
    if score_col:
        df = df.sort_values(score_col, ascending=False)
    rows = []
    for _, row in df.head(20).iterrows():
        value = row.get(score_col, "") if score_col else ""
        rows.append({"Section": label, "Metric": str(row.get(protein_col, "")), "Value": value})
    return rows


def write_ppi_report(
    prefix: str,
    graphs: dict[str, nx.Graph],
    nodes: dict[str, pd.DataFrame],
    edges: dict[str, pd.DataFrame],
    community_summary: pd.DataFrame,
    hub_stability: pd.DataFrame,
    species: int,
    annotation_source: str,
) -> None:
    rows: list[dict[str, str | int | float]] = []
    for name, graph in graphs.items():
        rows.extend(_network_statistics(name, graph))
        rows.extend(_community_summary(name, graph))
        edge_df = edges.get(name, pd.DataFrame())
        string_score = pd.to_numeric(edge_df["STRINGScore"], errors="coerce").fillna(0.0) if not edge_df.empty and "STRINGScore" in edge_df.columns else pd.Series([0.0])
        coexpression = pd.to_numeric(edge_df["CoexpressionScore"], errors="coerce").fillna(0.0) if not edge_df.empty and "CoexpressionScore" in edge_df.columns else pd.Series([0.0])
        rows.extend([
            {"Section": name, "Metric": "Average STRING Confidence", "Value": float(string_score.mean())},
            {"Section": name, "Metric": "Average Co-expression", "Value": float(coexpression.mean())},
        ])
    rows.extend([
        {"Section": "Run Metadata", "Metric": "Species", "Value": species},
        {"Section": "Run Metadata", "Metric": "Annotation Source", "Value": annotation_source},
    ])
    rows.extend(_top_table_rows(prefix, "recommended_biomarkers.csv", "Top Recommended Biomarkers", ["RecommendedBiomarkerScore", "ConfidenceScore"]))
    rows.extend(_top_table_rows(prefix, "consensus_biomarkers.csv", "Top Consensus Biomarkers", ["ConsensusScore"]))
    for name, table in nodes.items():
        for _, row in table.head(20).iterrows():
            rows.append({
                "Section": f"{name} Top Hub Proteins",
                "Metric": str(row.get("Protein", "")),
                "Value": f"Rank={row.get('HubRank', '')}; Degree={row.get('Degree', '')}; HubStability={row.get('HubStability', '')}",
            })
    if not community_summary.empty:
        for _, row in community_summary.sort_values(["Network", "NumberOfNodes"], ascending=[True, False]).head(30).iterrows():
            rows.append({
                "Section": "Top Communities",
                "Metric": f"{row['Network']} community {row['CommunityID']}",
                "Value": f"Nodes={row['NumberOfNodes']}; TopHub={row['TopHubProtein']}; MeanImportance={row['MeanImportance']:.4g}",
            })
    report = pd.DataFrame(rows, columns=["Section", "Metric", "Value"])
    report.to_csv(f"{prefix}_ppi_report.csv", index=False)
    report.to_csv("ppi_report.csv", index=False)
    try:
        with pd.ExcelWriter(f"{prefix}_ppi_report.xlsx") as writer:
            report.to_excel(writer, sheet_name="PPI Report", index=False)
            community_summary.to_excel(writer, sheet_name="Communities", index=False)
            hub_stability.to_excel(writer, sheet_name="Hub Stability", index=False)
            for name, table in nodes.items():
                table.to_excel(writer, sheet_name=name[:31], index=False)
        with pd.ExcelWriter("ppi_report.xlsx") as writer:
            report.to_excel(writer, sheet_name="PPI Report", index=False)
            community_summary.to_excel(writer, sheet_name="Communities", index=False)
            hub_stability.to_excel(writer, sheet_name="Hub Stability", index=False)
    except Exception:
        try:
            from engines.reliability.biomarker_reliability import write_simple_xlsx

            write_simple_xlsx(f"{prefix}_ppi_report.xlsx", {
                "PPI Report": report,
                "Communities": community_summary,
                "Hub Stability": hub_stability,
            })
            write_simple_xlsx("ppi_report.xlsx", {
                "PPI Report": report,
                "Communities": community_summary,
                "Hub Stability": hub_stability,
            })
        except Exception:
            pass
    with open(f"{prefix}_ppi_report.md", "w", encoding="utf-8") as handle:
        handle.write("# PPI Network Report\n\n")
        for section, group in report.groupby("Section", sort=False):
            handle.write(f"## {section}\n\n")
            for _, row in group.iterrows():
                handle.write(f"- {row['Metric']}: {row['Value']}\n")
            handle.write("\n")
    with open("ppi_report.md", "w", encoding="utf-8") as handle:
        handle.write("# PPI Network Report\n\n")
        for section, group in report.groupby("Section", sort=False):
            handle.write(f"## {section}\n\n")
            for _, row in group.iterrows():
                handle.write(f"- {row['Metric']}: {row['Value']}\n")
            handle.write("\n")


def run_network(args: argparse.Namespace) -> None:
    prefix = args.prefix.strip("_")
    prefix_out = prefix if prefix else "biomarkerml"
    importance = load_importance(prefix, args.patternChosen, args.SHAPthresh)
    seeds = importance["Protein"].astype(str).tolist()
    standard_edges = string_network(seeds, args.combined_score_thresholdHere, args.species, args.offline)
    if not standard_edges.empty:
        seed_set = set(seeds)
        standard_edges = standard_edges[
            standard_edges["Source"].astype(str).isin(seed_set)
            & standard_edges["Target"].astype(str).isin(seed_set)
        ].copy()
    coexpr = coexpression_edges(args.proteinExpFile, seeds, args.CorMethod, args.CorThreshold)
    if standard_edges.empty and not coexpr.empty:
        standard_edges = coexpr
    measured_proteins = list(map(str, _expression_matrix(args.proteinExpFile).columns))
    expanded_candidates = string_expanded_network(
        seeds,
        args.combined_score_thresholdHere,
        args.species,
        args.expanded_limit,
        args.offline,
        measured_proteins,
    )
    validated_expansion = filter_expanded_interactions(
        expanded_candidates,
        seeds,
        args.proteinExpFile,
        args.CorMethod,
        args.CorThreshold,
        args.combined_score_thresholdHere,
    )
    expanded_frames = [frame for frame in (standard_edges, validated_expansion) if not frame.empty]
    expanded_edges = (
        pd.concat(expanded_frames, ignore_index=True).drop_duplicates(subset=["Source", "Target", "EdgeType"])
        if expanded_frames
        else pd.DataFrame(columns=EDGE_COLUMNS)
    )
    observed = set(seeds)
    for edge_df in [standard_edges, expanded_edges]:
        if not edge_df.empty:
            observed |= set(edge_df["Source"].dropna().astype(str)) | set(edge_df["Target"].dropna().astype(str))
    scores = load_annotation_scores(prefix_out)
    annotation_species = getattr(args, "background_species", None) or args.species
    annotations = annotate_proteins(sorted(observed), annotation_species, args.offline)
    scores = pd.DataFrame({"Protein": sorted(observed)}).merge(scores, on="Protein", how="left").merge(annotations, on="Protein", how="left", suffixes=("", "_annotation"))
    for col in ["ConsensusScore", "StabilityScore", "SupportFrequency", "ConfidenceScore"]:
        current = scores[col] if col in scores.columns else pd.Series([0.0] * len(scores), index=scores.index)
        scores[col] = pd.to_numeric(current, errors="coerce").fillna(0.0)
    for col in ["GeneSymbol", "UniProtID", "EntrezID", "GOAnnotations", "STRINGID", "STRINGPreferredName", "AnnotationSource"]:
        annotated = f"{col}_annotation"
        if annotated in scores.columns:
            current = scores[col] if col in scores.columns else pd.Series([""] * len(scores), index=scores.index)
            current = current.fillna("").astype(str)
            incoming = scores[annotated].fillna("").astype(str)
            scores[col] = current.where(current.str.len() > 0, incoming)
            scores = scores.drop(columns=[annotated])
        if col not in scores.columns:
            scores[col] = ""
        scores[col] = scores[col].fillna("").astype(str)
    annotation_source = ";".join(sorted(set(";".join(scores["AnnotationSource"].astype(str)).split(";")) - {""}))

    standard_graph = build_graph(standard_edges, seeds, importance, scores)
    expanded_graph = build_graph(expanded_edges, seeds, importance, scores, expanded_metadata=True)

    standard_nodes, standard_edge_table = _write_network(
        prefix_out,
        "ppi_network",
        standard_graph,
        "Recommended Biomarkers PPI",
        mirror_png="Network_1.png",
    )
    expanded_nodes, expanded_edge_table = _write_network(prefix_out, "ppi_expanded_network", expanded_graph, "Expanded PPI", mirror_png="Network_2.png")
    node_tables = {"Recommended Biomarkers PPI": standard_nodes, "Expanded PPI": expanded_nodes}
    edge_tables = {"Recommended Biomarkers PPI": standard_edge_table, "Expanded PPI": expanded_edge_table}
    community_summary = write_community_summary(prefix_out, node_tables)
    hub_stability = write_hub_stability(prefix_out, node_tables)
    write_ppi_report(
        prefix_out,
        {"Recommended Biomarkers PPI": standard_graph, "Expanded PPI": expanded_graph},
        node_tables,
        edge_tables,
        community_summary,
        hub_stability,
        args.species,
        annotation_source or "offline",
    )

    try:
        from engines.reliability.biomarker_reliability import write_publication_summary
        write_publication_summary(prefix_out)
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    args.prefix = build_output_prefix(args.input, args.prefix)
    outdir = task_output_dir(args.outdir, "ppi")
    stage_task_inputs(args.input, outdir, args.prefix)
    if args.proteinExpFile == "Case1.csv" and args.input and Path(args.input).is_file():
        args.proteinExpFile = str(Path(args.input).resolve())
    with task_run_context(args.outdir, "ppi"):
        run_network(args)


if __name__ == "__main__":
    main()
