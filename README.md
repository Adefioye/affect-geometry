# Affect Geometry

A reproducible pipeline for extracting every-layer Qwen residual-stream activations from
single-label GoEmotions examples, constructing representational dissimilarity matrices (RDMs),
and comparing MDS and Isomap affect manifolds.

See [`doc/implementation.md`](doc/implementation.md) for installation, commands, outputs, and
the mathematical conventions used by the implementation.

```bash
python -m pip install -e ".[dev]"
affect-geometry run --config configs/default.json
```

All generated artifacts are written below `outputs/<run_id>/`.

