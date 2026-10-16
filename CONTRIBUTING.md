# Contributing

1. Create a virtual environment and install `pip install -r requirements-dev.txt && pip install -e .`.
2. Make your change with tests. Keep the rules in `docs/MODEL_LIMITATIONS.md` intact: no protected,
   post-decision or proxy columns as model inputs, and no hard-coded metrics in the UI.
3. Run `ruff check src app tests`, `ruff format src app tests` and `pytest -m "not e2e"`.
4. If you change the training protocol (`configs/training.yaml`), retrain, review the promotion gates
   and model card, and promote the new version explicitly. Never edit committed artifacts by hand;
   the registry checksums will reject them.
