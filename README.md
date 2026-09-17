# IGRCA

Minimal NITS(1)/NITS(3) generation, IGRCA training and GC/RCA evaluation.
Includes two trained checkpoints; synthetic data are generated on demand.

```bash
python generate.py
python run.py evaluate --check-reference
python run.py train --output retrained
python run.py evaluate --checkpoints retrained --output retrained_eval
```

Use the versions in `requirements.txt` (recorded environment: Linux, Python
3.11.11, PyTorch 2.0.0+cu117). No dependency installation is automatic.
The default device is `cuda:0`; use `--device cpu` if needed. Set
`CUDA_VISIBLE_DEVICES=1` to restrict a multi-GPU machine to physical GPU 1.

Configuration: `config.json`. Data seed: **2026082255**; training seed:
**2026182255**, selected post hoc for the highest mean IGRCA Avg*@500 over
the two anchor settings among ten seeds. This is a selected-seed demonstration,
not a multi-seed average. Training uses 8/2 healthy train/validation sequences
of length 2,000, 100 test sequences, and 140/860 maximum Stage 1/2 epochs.
It replays the archived supplementary settings, not every original paper table.

The archived model/metric calculations are preserved, including pooled unlabeled
test-score POT calibration. Exact replay is checked against the bundled weights;
retraining may vary across software/hardware. Upstream copyright is retained in
`LICENSE`.
