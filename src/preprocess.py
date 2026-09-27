"""Build a drug-to-pathway coverage matrix from DGIdb and Reactome inputs.

The generated matrix is a research abstraction for the RL simulator. It represents
normalized pathway coverage derived from mapped drug-gene interactions; it is not a
measured pharmacological effect matrix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import mygene
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EffectMatrix:
    drug_names: List[str]
    pathway_ids: List[str]
    pathway_names: List[str]
    effects: np.ndarray


def _normalize_symbol(symbol: str) -> str:
    if symbol is None:
        return ""
    return re.sub(r"\s+", "", str(symbol)).upper()


def load_dgidb_interactions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    columns = {column.lower(): column for column in frame.columns}

    drug_col = None
    gene_col = None

    for candidate in [
        "drug_name",
        "drug",
        "drugclaim",
        "drug_claim_name",
    ]:
        for normalized, original in columns.items():
            if candidate in normalized:
                drug_col = original
                break
        if drug_col:
            break

    for candidate in [
        "gene_name",
        "gene",
        "geneclaim",
        "gene_claim_name",
        "gene_symbol",
        "genesymbol",
    ]:
        for normalized, original in columns.items():
            if candidate in normalized:
                gene_col = original
                break
        if gene_col:
            break

    if drug_col is None or gene_col is None:
        raise ValueError(
            "Could not detect drug/gene columns in DGIdb file. "
            f"Columns={list(frame.columns)}"
        )

    output = frame[[drug_col, gene_col]].rename(
        columns={
            drug_col: "drug",
            gene_col: "gene_symbol",
        }
    )
    output["drug"] = output["drug"].astype(str).str.strip()
    output["gene_symbol"] = (
        output["gene_symbol"].astype(str).map(_normalize_symbol)
    )
    output = output[
        (output["drug"] != "")
        & (output["gene_symbol"] != "")
    ].drop_duplicates()

    if output.empty:
        raise ValueError("DGIdb input contained no usable drug-gene rows.")
    return output


def load_reactome_ensembl2reactome(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if frame.shape[1] < 6:
        raise ValueError(
            "Ensembl2Reactome.txt format unexpected (expected >=6 columns)"
        )

    frame = frame.iloc[:, :6].copy()
    frame.columns = [
        "ensembl",
        "pathway_id",
        "reactome_url",
        "pathway_name",
        "evidence",
        "species",
    ]

    for column in [
        "ensembl",
        "pathway_id",
        "pathway_name",
        "species",
    ]:
        frame[column] = frame[column].astype(str).str.strip()

    frame = frame[
        frame["species"].str.lower().isin(
            [
                "homo sapiens",
                "homo\u00a0sapiens",
                "homo sapiens (human)",
            ]
        )
    ]
    frame = frame[
        (frame["ensembl"] != "")
        & (frame["pathway_id"] != "")
    ]
    frame = frame.drop_duplicates(
        subset=["ensembl", "pathway_id"]
    )

    if frame.empty:
        raise ValueError(
            "Reactome input contained no usable human pathway rows."
        )

    return frame[["ensembl", "pathway_id", "pathway_name"]]


def map_symbols_to_ensembl(
    symbols: Iterable[str],
    cache_path: Path,
    species: str = "human",
) -> Dict[str, str]:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache: Dict[str, str] = {}
    if cache_path.exists():
        cache_frame = pd.read_csv(
            cache_path,
            sep="\t",
            dtype=str,
        )
        required = {"symbol", "ensembl"}
        if not required.issubset(cache_frame.columns):
            raise ValueError(
                "Symbol cache must contain columns 'symbol' and 'ensembl'."
            )
        cache = dict(
            zip(
                cache_frame["symbol"],
                cache_frame["ensembl"],
                strict=True,
            )
        )

    requested = sorted(
        {
            symbol
            for symbol in symbols
            if symbol
        }
    )
    missing = [
        symbol
        for symbol in requested
        if symbol not in cache
    ]

    if missing:
        gene_client = mygene.MyGeneInfo()
        batch_size = 1000

        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            result = gene_client.querymany(
                batch,
                scopes="symbol",
                fields="ensembl.gene",
                species=species,
                as_dataframe=True,
                returnall=False,
                verbose=False,
            )
            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame(result)

            if "ensembl.gene" in result.columns:
                mapped = result["ensembl.gene"].copy()
                mapped = mapped.apply(
                    lambda value: (
                        value[0]
                        if isinstance(value, list) and value
                        else value
                    )
                )
            elif "ensembl" in result.columns:

                def extract_gene(value: object) -> object:
                    if isinstance(value, dict):
                        return value.get("gene")
                    if (
                        isinstance(value, list)
                        and value
                        and isinstance(value[0], dict)
                    ):
                        return value[0].get("gene")
                    return None

                mapped = result["ensembl"].apply(extract_gene)
            else:
                mapped = pd.Series(
                    index=result.index,
                    data=None,
                    dtype=object,
                )

            for symbol, ensembl in mapped.items():
                if not isinstance(symbol, str):
                    continue
                if ensembl is None or str(ensembl) == "nan":
                    cache[symbol] = ""
                else:
                    cache[symbol] = str(ensembl).split(".")[0]

        cache_frame = pd.DataFrame({"symbol": sorted(cache)})
        cache_frame["ensembl"] = cache_frame["symbol"].map(cache)
        cache_frame.to_csv(cache_path, sep="\t", index=False)

    return cache


def build_effect_matrix(
    dgidb_df: pd.DataFrame,
    reactome_df: pd.DataFrame,
    *,
    top_drugs: int = 60,
    top_pathways: int = 40,
    symbol_cache_path: Path = Path(
        "data/processed/symbol_to_ensembl.tsv"
    ),
) -> EffectMatrix:
    if top_drugs <= 0:
        raise ValueError("top_drugs must be greater than 0")
    if top_pathways <= 0:
        raise ValueError("top_pathways must be greater than 0")

    symbol_to_ensembl = map_symbols_to_ensembl(
        dgidb_df["gene_symbol"].unique(),
        cache_path=symbol_cache_path,
    )

    interactions = dgidb_df.copy()
    interactions["ensembl"] = (
        interactions["gene_symbol"]
        .map(symbol_to_ensembl)
        .fillna("")
    )
    interactions = interactions[
        interactions["ensembl"] != ""
    ].drop_duplicates(subset=["drug", "ensembl"])

    joined = interactions.merge(
        reactome_df,
        on="ensembl",
        how="inner",
    )
    if joined.empty:
        raise ValueError(
            "After symbol-to-Ensembl mapping and Reactome joining, "
            "no rows remained. Check the raw files and symbol cache."
        )

    drug_counts = (
        joined.groupby("drug")["pathway_id"]
        .nunique()
        .sort_values(ascending=False, kind="stable")
    )
    drugs = drug_counts.head(top_drugs).index.tolist()
    joined = joined[joined["drug"].isin(drugs)]

    pathway_counts = (
        joined.groupby("pathway_id")["drug"]
        .nunique()
        .sort_values(ascending=False, kind="stable")
    )
    pathway_ids = pathway_counts.head(top_pathways).index.tolist()
    joined = joined[joined["pathway_id"].isin(pathway_ids)]

    pathway_name_lookup = (
        joined.drop_duplicates(subset=["pathway_id"])
        .set_index("pathway_id")["pathway_name"]
        .to_dict()
    )
    pathway_names = [
        pathway_name_lookup.get(pathway_id, pathway_id)
        for pathway_id in pathway_ids
    ]

    pivot = (
        joined.groupby(["drug", "pathway_id"])["ensembl"]
        .nunique()
        .reset_index()
        .pivot(
            index="drug",
            columns="pathway_id",
            values="ensembl",
        )
        .fillna(0.0)
    )
    pivot = pivot.reindex(
        index=drugs,
        columns=pathway_ids,
    ).fillna(0.0)

    matrix = pivot.to_numpy(dtype=np.float32)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    effects = (matrix / row_sums).astype(np.float32)

    return EffectMatrix(
        drug_names=drugs,
        pathway_ids=pathway_ids,
        pathway_names=pathway_names,
        effects=effects,
    )


def save_effects(
    effect_matrix: EffectMatrix,
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        effects=effect_matrix.effects.astype(np.float32),
        drug_names=np.asarray(effect_matrix.drug_names, dtype=str),
        pathway_ids=np.asarray(effect_matrix.pathway_ids, dtype=str),
        pathway_names=np.asarray(effect_matrix.pathway_names, dtype=str),
    )
    return path


def load_effects(path: Path) -> EffectMatrix:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        effects = archive["effects"].astype(np.float32)
        drug_names = list(archive["drug_names"].astype(str))
        pathway_ids = list(archive["pathway_ids"].astype(str))
        pathway_names = list(archive["pathway_names"].astype(str))

    if effects.shape != (
        len(drug_names),
        len(pathway_names),
    ):
        raise ValueError(
            "Stored effect matrix shape does not match "
            "drug/pathway metadata lengths."
        )

    return EffectMatrix(
        drug_names=drug_names,
        pathway_ids=pathway_ids,
        pathway_names=pathway_names,
        effects=effects,
    )
