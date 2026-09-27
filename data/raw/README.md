# External data and provenance

The maintained repository does not vendor large downloaded copies of DGIdb, HGNC, or Reactome datasets.

## Core training inputs

Place the following files in `data/raw/`:

```text
dgidb_interactions.tsv
Ensembl2Reactome.txt
```

The preprocessing code maps DGIdb gene symbols to Ensembl IDs with `mygene` and writes a local mapping cache under `data/processed/`.

## PSC-oriented optional inputs

The PSC helper can additionally use:

```text
PSC_risk_genes.csv
hgnc_complete_set.tsv
```

The small project-specific `PSC_risk_genes.csv` may be kept in the repository. Large third-party downloads should be obtained from their authoritative source rather than committed.

## Reproducibility metadata

For a publication-scale run, record:

- source/database name;
- release/version when available;
- retrieval date;
- exact filename;
- SHA-256 checksum;
- filtering or preprocessing performed before use.

External biological databases change over time. A filename such as `interactions.tsv` is not sufficient provenance by itself.

## Processed data

Files under `data/processed/` are generated and ignored by Git.

A typical output is:

```text
drug_pathway_effects_N60_P40.npz
```

It contains:

- normalized pathway-coverage matrix;
- drug names;
- Reactome pathway IDs;
- pathway names.

The maintained format stores normal string arrays and is loaded with `allow_pickle=False`.
