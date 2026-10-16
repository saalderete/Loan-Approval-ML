# Data

Raw and processed HMDA files are **not committed**. Recreate them with:

```bash
loan-approval fetch     # downloads data/raw/hmda_2025_TX_actions_1_2_3.csv and records its SHA-256
loan-approval prepare   # writes data/processed/hmda_2025_TX_model.parquet + .quality.json
```

- `data_manifest.yaml` records the source URL, query, retrieval time, size, row count and checksum.
- `synthetic/` holds generated HMDA-format files (`loan-approval synthetic --rows 5000`) for offline
  development. Synthetic data says nothing about real lending.

See [`reports/dataset_card.md`](../reports/dataset_card.md) for provenance, filters and known biases.
