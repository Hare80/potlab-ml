# potlab-ml — Training Guide

Metrics, visualization, and the test checklist. Interfaces: `training/trainer.py`,
`training/metrics.py`, `training/callbacks.py` ([DESIGN.md](../DESIGN.md#7-trainer-and-callbacks)).

## Metrics to record

### Per epoch (main table, `runs/<name>/metrics.csv`)

| column | meaning |
|---|---|
| `epoch` | epoch index |
| `train_loss` | per-molecule MSE, summed over the epoch then divided by the total molecule count |
| `val_mae` | validation MAE **in display units** (meV for QM9 energy targets) |
| `val_rmse` | optional |
| `lr` | learning-rate snapshot at epoch end |
| `epoch_time` | wall time of the epoch |
| `grad_norm` | optional — global gradient norm after the last backward (exploding/vanishing diagnostics) |

### Per step (secondary table, `runs/<name>/lr_steps.csv`)

`step, lr` — the cosine schedule changes the learning rate **every step**
(`T_max = num_epochs × steps_per_epoch`), so an epoch-end snapshot cannot show its shape.
Log every step (or every K steps).

### Sum-then-divide rule

Batches differ in size (the last one is usually smaller), so both the training loss and the
validation MAE follow one pattern: accumulate `reduction='sum'` values, divide **once** by
the total molecule count. Never average per-batch means — that weights small batches
incorrectly. In code, name the variables explicitly (`loss_sum`, `loss_mean`) so the two
different divisions (per batch for `backward()`, once per epoch for display) stay obvious.

## Runs layout

```
runs/<run_name>/
├── metrics.csv         # source of truth for all plotting
├── lr_steps.csv
├── config.yaml         # config snapshot (reproducibility)
├── plots/latest.png    # overwritten every N epochs
└── checkpoints/        # model + standardizer + optimizer + scheduler + config + epoch
```

CSV is the deliberate core choice: figures are always re-plottable, pandas-friendly, and
independent of the training process. TensorBoard events are a *view*, not the record.

## Visualization

### TensorBoard (live monitoring)

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(runs/<name>)          # same dir as the CSV
writer.add_scalar("train/loss", loss, epoch)
writer.add_scalar("val/mae_meV", val_mae, epoch)
writer.add_scalar("train/lr", lr, epoch)
writer.add_graph(model, example_inputs)      # the PaiNN graph, for free
```

- Launch: `tensorboard --logdir runs` → `http://localhost:6006`. If the command is missing
  on Windows: `python -m tensorboard.main --logdir runs`.
- Use the **smoothing slider** for noisy loss curves; put multiple runs side by side when
  comparing hyperparameters.

### matplotlib panel (report-ready)

`PlotCallback` redraws a 2×2 panel to `plots/latest.png` every N epochs:

```
┌──────────────────────┬──────────────────────┐
│ train loss (log y)   │ val MAE (meV, linear)│
├──────────────────────┼──────────────────────┤
│ lr vs epoch          │ grad norm (log y)    │
└──────────────────────┴──────────────────────┘
```

Details:

- **Log-scale the loss** — it spans orders of magnitude; linear scale hides the early drop.
- Mark the **best epoch** with a vertical line on the val-MAE panel — deciding the early
  stopping patience is a matter of looking at this curve.
- The lr panel is a wiring self-check: run a 2-epoch smoke test first and confirm the cosine
  curve has the expected shape (a wrong `T_max` is a classic silent bug).
- Live viewing with zero dependencies: open `plots/latest.png` in VS Code — the preview
  auto-refreshes when the file is overwritten.

## Checkpointing and resume

A checkpoint contains: model `state_dict`, standardizer state, optimizer + scheduler state,
config snapshot, `epoch`, `best_val_mae`. `--resume runs/<name>` restores all of it and
continues from the same epoch — a kill mid-training must lose nothing.

## Test checklist (M4)

The permanent exam every model and dataset must pass before being merged:

1. **Rotation invariance of the energy** — rotate all positions by a random rotation matrix;
   `energy` must be unchanged to `1e-6` (invariance is why vector layers have no bias).
2. **Equivariance of forces** — forces computed on the rotated geometry must equal the
   original forces rotated by the same matrix.
3. **Gradient check** — `energy_and_forces` vs finite differences on positions:
   relative error < `1e-4`. Confirms the autograd path end to end.
4. **Standardizer roundtrip** — `inverse(transform(y, z, batch), z, batch)` restores `y`
   to `1e-6`, including atom references.
5. **TorchScript parity** — `torch.jit.script(core)` output matches eager mode to `1e-6`;
   this test runs from M2 onward, not just at export time.
6. **Data contract** — batch keys and shapes match [DESIGN.md](../DESIGN.md#3-data-contract-the-batch);
   split sizes sum to the dataset size (guards the silent-ignored-third-value bug).
7. **Split discipline** — for trajectory datasets, train/val/test must not share simulation
   runs (see [docs/data.md](data.md)).

## Interpreting results (QM9 baseline)

Target: test MAE ≈ **5.4 meV** on QM9 U0 (reference: PaiNN paper ≈ 5.85 meV; "chemical
accuracy" 1 kcal/mol ≈ 43 meV is only the literature benchmark line — whether a given error
is *useful* depends on the application: kinetics needs ~0.1 kcal/mol, screening tolerates
more; and the model error sits on top of the reference method's own error vs experiment).
