# deep-b-spline-approximation

Research repository for B-spline curve approximation with neural networks, originally based on a Master's thesis and then extended with a more reproducible experimental pipeline.

This README is intentionally long: besides technical usage instructions, it also stores the experiment log, failed attempts, and working scientific hypotheses, so it can later be turned into a PDF or a thesis/report section.

## Current status in one sentence

The best neural variant so far is not a PPN that predicts the parametrization from scratch, but a **monotone residual correction on top of centripetal parametrization**.

Main benchmark: non-uniformly sampled B-spline curves, 100 points, 7 internal knots in the data generator, and the same 7 fixed internal knots in the LSQ fitting stage.

| Method | Fit MSE | Fit DHD | Note |
|---|---:|---:|---|
| uniform | 0.0316 | 0.4550 | classical baseline |
| chord_length | 0.0194 | 0.3991 | classical baseline |
| centripetal | 0.0153 | 0.3473 | strongest classical baseline |
| PPN MLP bezier-only | 0.0327 | 0.4814 | poor OOD generalization |
| PPN MLP B-spline + fixed-loss7 | 0.0381 | 0.5008 | architecture/loss mismatch not fully solved |
| PPN MLP + t supervision | 0.0328 | 0.4659 | improves, but still worse than centripetal |
| convex mixture baseline | 0.0194 | 0.3991 | collapses to chord_length |
| **centripetal_residual** | **0.0125** | **0.3226** | first neural variant that beats centripetal |

Preliminary interpretation: the problem seems highly sensitive to **synthetic generator bias**, and the standard PPN tends to learn an almost uniform parametrization, or at least one that is not more useful than geometric baselines. Constraining the model to start from a strong baseline and learn only a small monotone correction seems to be a better direction.

## Repository contents

### Legacy code / original pipeline

| Path | Role |
|---|---|
| `deep_b_spline_approximation/deep_b_spline_approximation.py` | legacy PPN + KPN + fitting inference orchestrator |
| `deep_b_spline_approximation/BSpline.py` | LSQ layer and B-spline utilities |
| `deep_b_spline_approximation/ppn.py` | PPN MLP/CNN networks + new experimental variants |
| `deep_b_spline_approximation/kpn.py` | legacy KPN |
| `deep_b_spline_approximation/preprocessing.py` | segmentation, normalization, reparametrization pipeline |
| `deep_b_spline_approximation/directedHausdorff.py` | DHD/MSE metrics used in benchmarks |
| `deep_b_spline_approximation/training.py` | legacy training script, not reproducible out of the box |
| `deep_b_spline_approximation/resumetraining.py` | legacy resume-training script |
| `deep_b_spline_approximation/evalmodel.py` | legacy evaluation script |
| `deep_b_spline_approximation/loadfromtxt.py` | legacy text dataset loader |

### Modules added for reproducible experiments

| Path | Role |
|---|---|
| `deep_b_spline_approximation/parametrization.py` | uniform / chord-length / centripetal baselines |
| `deep_b_spline_approximation/synthetic_dataset.py` | controllable synthetic data generator |
| `deep_b_spline_approximation/reproducibility.py` | global Python/NumPy/PyTorch seeding |
| `deep_b_spline_approximation/parametrization_features.py` | input features built from classical parametrizations |
| `scripts/train_ppn_mlp.py` | training CLI for PPN variants |
| `scripts/evaluate_ppn_mlp.py` | checkpoint vs baseline evaluation CLI |
| `scripts/benchmark_ppn_mlp.py` | quick benchmark over ID/OOD scenarios |
| `scripts/benchmark_fixed_knots_parametrization.py` | fair comparison with shared fitting knots |
| `scripts/inspect_parametrization_quality.py` | direct inspection of predicted `t` |

## Recommended setup

### Requirements

- Windows + PowerShell
- Python 3.11 recommended
- PyTorch, NumPy, SciPy, Matplotlib, PrettyTable

Important note: `torch` installation did not work here on Python 3.13, so a Python 3.11 environment named `.venv311` was created instead.

### Create the virtual environment

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TEMP = (Resolve-Path .tmp).Path
$env:TMP = (Resolve-Path .tmp).Path
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch numpy scipy matplotlib prettytable
```

### If you already have the environment created in this working copy

```powershell
.\.venv311\Scripts\Activate.ps1
```

## How to use the repo

### 1) Train a standard PPN MLP

```powershell
.\.venv311\Scripts\python.exe scripts\train_ppn_mlp.py `
  --output-dir outputs\ppn_mlp_mid `
  --n-train 2000 `
  --n-val 500 `
  --hidden-size 1000 `
  --batch-size 50 `
  --epochs 10 `
  --seed 52 `
  --device cpu
```

### 2) Train on random B-splines + LSQ loss with 7 fixed internal knots

```powershell
.\.venv311\Scripts\python.exe scripts\train_ppn_mlp.py `
  --output-dir outputs\ppn_mlp_bspline_fixedloss7 `
  --n-train 2000 `
  --n-val 500 `
  --min-internal-knots 1 `
  --max-internal-knots 8 `
  --fit-internal-knots 7 `
  --hidden-size 1000 `
  --batch-size 50 `
  --epochs 10 `
  --seed 52 `
  --device cpu
```

### 3) Train the current best variant: centripetal residual

```powershell
.\.venv311\Scripts\python.exe scripts\train_ppn_mlp.py `
  --output-dir outputs\ppn_centripetal_residual_fixedloss7_nonuniform `
  --model-kind centripetal_residual `
  --n-train 2000 `
  --n-val 500 `
  --min-internal-knots 1 `
  --max-internal-knots 8 `
  --fit-internal-knots 7 `
  --non-uniform `
  --normalization none `
  --hidden-size 256 `
  --batch-size 50 `
  --epochs 10 `
  --seed 52 `
  --device cpu
```

### 4) Evaluate a checkpoint

```powershell
.\.venv311\Scripts\python.exe scripts\evaluate_ppn_mlp.py `
  --checkpoint outputs\ppn_mlp_mid\ppn_mlp.pt `
  --output-json outputs\ppn_mlp_mid\eval_metrics.json `
  --n-eval 200 `
  --n-points 100 `
  --batch-size 50 `
  --seed 123 `
  --device cpu
```

### 5) Run a fair benchmark with fixed fitting knots

```powershell
.\.venv311\Scripts\python.exe scripts\benchmark_fixed_knots_parametrization.py `
  --checkpoint outputs\ppn_centripetal_residual_fixedloss7_nonuniform\ppn_mlp.pt `
  --output-json outputs\ppn_centripetal_residual_fixedloss7_nonuniform\bench_7k_nonuniform.json `
  --n-eval 200 `
  --n-points 100 `
  --seed 123 `
  --n-internal-knots 7 `
  --data-internal-knots 7 `
  --non-uniform `
  --device cpu
```

### 6) Inspect the predicted parametrization `t`

```powershell
.\.venv311\Scripts\python.exe scripts\inspect_parametrization_quality.py `
  --checkpoint outputs\ppn_centripetal_residual_fixedloss7_nonuniform\ppn_mlp.pt `
  --output-json outputs\ppn_centripetal_residual_fixedloss7_nonuniform\t_quality_7k_nonuniform.json `
  --n-eval 200 `
  --n-points 100 `
  --data-internal-knots 7 `
  --non-uniform `
  --seed 123 `
  --device cpu
```

## Model variants and useful flags

### `--model-kind`

| Value | Description |
|---|---|
| `ppn_mlp` | standard PPN MLP that predicts `t` from scratch |
| `baseline_mixture` | global convex mixture of uniform/chord/centripetal |
| `centripetal_residual` | small monotone correction on top of `t_centripetal` |

### Other relevant flags

| Flag | Effect |
|---|---|
| `--fit-internal-knots K` | use `K` fixed internal knots in the LSQ training layer |
| `--min-internal-knots`, `--max-internal-knots` | control generated-curve complexity |
| `--non-uniform` | non-uniform curve sampling |
| `--point-noise-std` | Gaussian noise on sampled points |
| `--normalization batchnorm|layernorm|none` | MLP backbone normalization |
| `--param-loss-weight` | adds MSE supervision on `t_true` |
| `--param-inputs uniform,chord_length,centripetal` | concatenates baseline `t` values as extra input channels |
| `--centripetal-margin-weight` | penalizes the model if its fit is worse than centripetal |

## Reference methodology from the thesis

The original thesis describes a pipeline like this:

```text
Dense curve
-> segment by complexity/curvature
-> resample each segment to N+1 points
-> normalize coordinates to [0,1]
-> PPN predicts the segment parametrization
-> KPN proposes one new internal knot in the segment with maximum error
-> LSQ B-spline fit
-> iterative refinement
```

The part we focused on in this phase is mainly **PPN + parametrization**, because the first benchmarks showed that the legacy PPN's out-of-distribution behavior is the most scientifically suspicious point.

## Thesis reading and initial theory-vs-code mismatches

These points were identified by comparing the thesis PDF with the legacy implementation.

| Topic | Thesis | Legacy code | Implication |
|---|---|---|---|
| PPN training data | 240k synthetic Bezier curves, 100 points | legacy scripts point to local hardcoded files | dataset is not directly reproducible |
| Baseline parametrizations | centripetal/chord/uniform used for comparison | `parametrization` module was missing | legacy evaluation was incomplete |
| KPN MLP | FC + BatchNorm + Softplus | ReLU + Dropout, BN commented out | architectural mismatch |
| KPN knot clipping | threshold like `[eps, 1-eps]` | clipping to `[t[:,1], t[:,-2]]` | stricter, data-dependent constraint |
| Geometric distance | distance to orthogonal projection on the curve | discrete `directed_hausdorff` over sampled points | metric differs from the theoretical definition |
| Randomness | experiments should be repeatable | partial seeding, random sampling in several places | weak reproducibility |
| PPN BatchNorm | inference batch dependence not discussed | `track_running_stats=False` + duplicated single-sample hack | inference is batch-dependent |
| Packaging | installable Python package | minimal `setup.py`, no modern config | structure can be improved |
| Checkpoints | pretrained models distributed externally | duplicated `.pt` files in two folders | repository is heavy and redundant |

## Experimental research log

This section intentionally keeps both positive and negative results. The numbers below are those observed in the benchmarks run in this working copy, so they are useful as a historical reference and as a basis for deciding what to reproduce more rigorously later.

### P0 smoke test: minimal end-to-end pipeline

Training:

```powershell
.\.venv311\Scripts\python.exe scripts\train_ppn_mlp.py --output-dir outputs\smoke_ppn --n-train 20 --n-val 10 --hidden-size 32 --batch-size 5 --epochs 1 --seed 52 --device cpu
```

Result:

| Run | train_mse | val_mse |
|---|---:|---:|
| smoke_ppn | 0.00249101 | 0.00259682 |

Evaluation:

```powershell
.\.venv311\Scripts\python.exe scripts\evaluate_ppn_mlp.py --checkpoint outputs\smoke_ppn\ppn_mlp.pt --output-json outputs\smoke_ppn\eval_metrics.json --n-eval 10 --n-points 100 --batch-size 5 --seed 53 --device cpu
```

| Method | MSE | DHD |
|---|---:|---:|
| ppn_mlp | 0.0026427 | 0.11563 |
| uniform | ~0 | ~0 |
| chord_length | 0.0068979 | 0.17473 |
| centripetal | 0.0023941 | 0.11756 |

Interpretation: on the equispaced Bezier generator, uniform parametrization is almost perfect. This was the first strong hint that a PPN trained only on this generator may mostly learn **synthetic sampling bias**.

### P1 baseline PPN MLP trained on equispaced Bezier curves

Training:

```powershell
.\.venv311\Scripts\python.exe scripts\train_ppn_mlp.py --output-dir outputs\ppn_mlp_mid --n-train 2000 --n-val 500 --hidden-size 1000 --batch-size 50 --epochs 10 --seed 52 --device cpu
```

Training curves:

| Initial epoch | Final epoch |
|---:|---:|
| train_mse 8.5691e-4 | train_mse 5.5090e-5 |
| val_mse 3.7898e-4 | val_mse 3.9210e-5 |

Benchmark with `scripts\benchmark_ppn_mlp.py`:

| Scenario | PPN DHD | uniform DHD | chord DHD | centripetal DHD | Reading |
|---|---:|---:|---:|---:|---|
| id_bezier_uniform | 0.0246 | ~0 | 0.1719 | 0.1027 | PPN learns the generator, but uniform remains perfect |
| ood_bezier_non_uniform | 0.1589 | 0.1558 | 0.1845 | 0.1261 | PPN is roughly uniform, centripetal is better |
| ood_bezier_noise | 0.1378 | 0.1373 | 0.1706 | 0.1480 | no clear PPN advantage |
| ood_bspline_internal_knots | 1.3182 | 1.3120 | 0.8140 | 0.9466 | PPN fails badly on B-splines with internal knots |

Interpretation: this is the first clear evidence that a standard PPN MLP trained only on Bezier curves **does not generalize well** to B-splines with internal knots and tends to behave close to uniform parametrization.

### PPN training on B-splines with random internal knots, but still with a Bezier-style LSQ loss

Training:

```powershell
.\.venv311\Scripts\python.exe scripts\train_ppn_mlp.py --output-dir outputs\ppn_mlp_bspline_knots --n-train 2000 --n-val 500 --min-internal-knots 1 --max-internal-knots 8 --hidden-size 1000 --batch-size 50 --epochs 10 --seed 52 --device cpu
```

Qualitative results:

| Observation | Outcome |
|---|---|
| `ood_bspline_internal_knots` improves compared with the Bezier-only model | DHD 1.318 -> 0.940 |
| performance on Bezier cases gets much worse | `id_bezier_uniform` DHD 0.025 -> 0.263 |
| validation loss remains high | around 0.1337 |

Interpretation: changing the training distribution truly changes what the network learns, so the **generator bias** concern is real. However, this training setup is still misaligned, because the curves contain internal knots while the PPN LSQ layer still uses a knot-free Bezier-like fitting structure.

### Fair benchmark with fixed knots

To isolate **parametrization quality** from knot-placement effects, `scripts\benchmark_fixed_knots_parametrization.py` was introduced. In this benchmark, all parametrizations are evaluated with **the same fixed internal fitting knots**.

Main scenario: non-uniform B-spline data, 7 internal knots in the generator, 7 internal knots in the fit.

| Method | Fit MSE | Fit DHD |
|---|---:|---:|
| PPN bezier_only | 0.0326911 | 0.481433 |
| PPN bspline_data_bezier_loss | 0.0535118 | 0.722687 |
| uniform | 0.0316231 | 0.454996 |
| chord_length | 0.0194353 | 0.399095 |
| centripetal | 0.0153280 | 0.347287 |

Interpretation: even after removing the "which internal knots should I choose?" factor, the standard PPN does not beat geometric baselines. Therefore the bottleneck is not only KPN or knot placement: there is a genuine problem in the learned parametrization.

### Direct diagnostics of the predicted `t`

Script:

```powershell
.\.venv311\Scripts\python.exe scripts\inspect_parametrization_quality.py --checkpoint outputs\ppn_mlp_mid\ppn_mlp.pt --output-json outputs\ppn_mlp_mid\t_quality.json --n-eval 200 --n-points 100 --data-internal-knots 7 --non-uniform --seed 123 --device cpu
```

Results on non-uniformly sampled B-splines:

| Method | MAE(t,true_t) | corr(t,true_t) | monotonicity violations |
|---|---:|---:|---:|
| PPN bezier_only | 0.0312 | 0.9961 | 0 |
| PPN bspline_data_bezier_loss | 0.0718 | 0.9810 | 0 |
| uniform | 0.0308 | 0.9961 | 0 |
| chord_length | 0.0650 | 0.9886 | 0 |
| centripetal | 0.0368 | 0.9961 | 0 |

Important interpretation: the problem is **not** that the PPN outputs a non-monotone parametrization. The problem is that the learned `t` is not geometrically better than the baselines, and often ends up very close to uniform.

### LSQ loss with fixed internal knots during training

Change: `PointParametrizationNetwork` can now use fixed `internal_knots` inside the LSQ layer. Training:

```powershell
.\.venv311\Scripts\python.exe scripts\train_ppn_mlp.py --output-dir outputs\ppn_mlp_bspline_fixedloss7 --n-train 2000 --n-val 500 --min-internal-knots 1 --max-internal-knots 8 --fit-internal-knots 7 --hidden-size 1000 --batch-size 50 --epochs 10 --seed 52 --device cpu
```

Results on the fair 7k non-uniform benchmark:

| Method | Fit DHD | MAE(t,true_t) |
|---|---:|---:|
| bspline_data_bezier_loss | 0.7227 | 0.0718 |
| bspline_data_fixedloss7 | 0.5008 | 0.0531 |
| centripetal | 0.3473 | 0.0368 |

Interpretation: using fixed internal knots in the loss fixes a large part of the "B-spline data + Bezier loss" failure, but it is **still not enough** to beat geometric baselines. So the loss mismatch was real, but it was not the only issue.

### BatchNorm / no-normalization ablation

Training no-norm:

```powershell
.\.venv311\Scripts\python.exe scripts\train_ppn_mlp.py --output-dir outputs\ppn_mlp_bspline_fixedloss7_nonorm --n-train 2000 --n-val 500 --min-internal-knots 1 --max-internal-knots 8 --fit-internal-knots 7 --normalization none --hidden-size 1000 --batch-size 50 --epochs 10 --seed 52 --device cpu
```

Comparison:

| Variant | Fit DHD | MAE(t,true_t) |
|---|---:|---:|
| BatchNorm ON | 0.5008 | 0.0531 |
| No norm | 0.5034 | 0.0332 |
| centripetal | 0.3473 | 0.0368 |

Interpretation: removing BatchNorm makes `t` closer to the true sampling parameters, but does not actually improve LSQ fitting. This suggests that imitating the generator's `t_true` is not the same as finding the parametrization that minimizes B-spline approximation error.

### Hybrid loss with `t_true` supervision

Training:

```powershell
.\.venv311\Scripts\python.exe scripts\train_ppn_mlp.py --output-dir outputs\ppn_mlp_bspline_fixedloss7_nonorm_tsup --n-train 2000 --n-val 500 --min-internal-knots 1 --max-internal-knots 8 --fit-internal-knots 7 --non-uniform --normalization none --param-loss-weight 1.0 --hidden-size 1000 --batch-size 50 --epochs 10 --seed 52 --device cpu
```

Results:

| Variant | Fit DHD | Fit MSE | MAE(t,true_t) |
|---|---:|---:|---:|
| no_norm_no_tsup | 0.5034 | 0.0342 | 0.0332 |
| no_norm_tsup | 0.4659 | 0.0328 | 0.0299 |
| centripetal | 0.3473 | 0.0153 | 0.0368 |

Interpretation: direct supervision on `t_true` helps, but the model is still worse than centripetal. Therefore, "learning the generator sampling" is not enough.

### Margin loss against centripetal

Idea tested: add a penalty term if the model's LSQ error is worse than the one obtained by centripetal on the same knots.

Training:

```powershell
.\.venv311\Scripts\python.exe scripts\train_ppn_mlp.py --output-dir outputs\ppn_mlp_bspline_fixedloss7_nonorm_centrmargin --n-train 2000 --n-val 500 --min-internal-knots 1 --max-internal-knots 8 --fit-internal-knots 7 --non-uniform --normalization none --centripetal-margin-weight 1.0 --centripetal-margin 0.0 --hidden-size 1000 --batch-size 50 --epochs 10 --seed 52 --device cpu
```

Results:

| Variant | Fit DHD | MAE(t,true_t) | Note |
|---|---:|---:|---|
| centrmargin | 0.4788 | 0.0335 | worse than the `t_true` supervision loss |
| no_norm_tsup | 0.4659 | 0.0299 | better than the centripetal-margin loss |
| centripetal | 0.3473 | 0.0368 | still clearly better |

Conclusion: this margin loss, in the form tested here, **was not effective**.

### Enriched input with concatenated classical parametrizations

Idea: feed the network not only `(x, y)`, but also `t_uniform`, `t_chord_length`, and `t_centripetal`, hoping that it learns when each baseline is more appropriate.

Training:

```powershell
.\.venv311\Scripts\python.exe scripts\train_ppn_mlp.py --output-dir outputs\ppn_mlp_bspline_fixedloss7_nonorm_tsup_paraminputs --n-train 2000 --n-val 500 --min-internal-knots 1 --max-internal-knots 8 --fit-internal-knots 7 --non-uniform --normalization none --param-inputs uniform,chord_length,centripetal --param-loss-weight 1.0 --hidden-size 1000 --batch-size 50 --epochs 10 --seed 52 --device cpu
```

Results:

| Variant | Fit DHD | MAE(t,true_t) |
|---|---:|---:|
| no_norm_tsup | 0.4659 | 0.0299 |
| no_norm_tsup + param_inputs | 0.4818 | 0.0315 |
| centripetal | 0.3473 | 0.0368 |

Conclusion: naive concatenation of baseline parametrizations as extra features **did not help**.

### Global convex mixture of baselines

Idea: instead of passing baseline parametrizations as unconstrained features, predict three global softmax weights and take a convex combination of uniform/chord/centripetal.

Training:

```powershell
.\.venv311\Scripts\python.exe scripts\train_ppn_mlp.py --output-dir outputs\ppn_baseline_mixture_fixedloss7_nonuniform --model-kind baseline_mixture --n-train 2000 --n-val 500 --min-internal-knots 1 --max-internal-knots 8 --fit-internal-knots 7 --non-uniform --normalization none --hidden-size 256 --batch-size 50 --epochs 10 --seed 52 --device cpu
```

Results:

| Method | Fit DHD | MAE(t,true_t) |
|---|---:|---:|
| baseline_mixture | 0.3991 | 0.0650 |
| chord_length | 0.3991 | 0.0650 |
| centripetal | 0.3473 | 0.0368 |

Learned average weights:

```text
[uniform, chord_length, centripetal] ~= [0.0, 1.0, 0.0]
```

Observed issue: the model **fully collapses to chord_length** instead of learning a truly adaptive gating mechanism. During training, many messages like `Calcolo pseudoinversa, matrice non invertibile...` also appeared, together with a huge `train_mse` spike, suggesting that this global mixture does not really stabilize the numerical problem.

Conclusion: this variant **does not seem to be the right direction** either, at least not in this global form without finer local constraints.

### Best variant so far: residual correction on top of centripetal

Idea: instead of predicting `t` from scratch, start from `t_centripetal` and learn only a small monotone correction on `Delta t`, of the form:

```text
Delta t_pred = Delta t_centripetal * (1 + 0.25 * tanh(r))
t_pred = normalize(cumsum(Delta t_pred))
```

This constrains the network to remain in the space of reasonable parametrizations and prevents it from destroying a strong geometric baseline.

Training:

```powershell
.\.venv311\Scripts\python.exe scripts\train_ppn_mlp.py --output-dir outputs\ppn_centripetal_residual_fixedloss7_nonuniform --model-kind centripetal_residual --n-train 2000 --n-val 500 --min-internal-knots 1 --max-internal-knots 8 --fit-internal-knots 7 --non-uniform --normalization none --hidden-size 256 --batch-size 50 --epochs 10 --seed 52 --device cpu
```

Training dynamics:

| Initial epoch | Final epoch |
|---:|---:|
| train_mse 0.00920946 | train_mse 0.00862584 |
| val_mse 0.00805496 | val_mse 0.00798611 |

Benchmark fair 7k non-uniform:

| Method | Fit MSE | Fit DHD | MAE(t,true_t) | corr(t,true_t) |
|---|---:|---:|---:|---:|
| uniform | 0.0316231 | 0.454996 | 0.0308 | 0.9961 |
| chord_length | 0.0194353 | 0.399095 | 0.0650 | 0.9886 |
| centripetal | 0.0153280 | 0.347287 | 0.0368 | 0.9961 |
| **centripetal_residual** | **0.0125049** | **0.322628** | **0.0319** | **0.9965** |

Monotonicity checks:

| Method | mean min Delta t | mean max Delta t | monotonicity violations |
|---|---:|---:|---:|
| centripetal | 0.0008322 | 0.0335260 | 0 |
| centripetal_residual | 0.0008488 | 0.0305513 | 0 |

Conclusion: this was the **first neural variant that beats centripetal** on the fair fixed-knot benchmark, without losing monotonicity.

### OOD stress test for the residual variant

Same checkpoint `outputs\ppn_centripetal_residual_fixedloss7_nonuniform\ppn_mlp.pt`, always fitting with 7 fixed internal knots, but changing the data generator.

| Scenario | residual DHD | centripetal DHD | chord DHD | uniform DHD | Reading |
|---|---:|---:|---:|---:|---|
| 7k_nonuniform | 0.3226 | 0.3473 | 0.3991 | 0.4550 | residual is better |
| 7k_nonuniform_noise005 | 0.3529 | 0.3703 | 0.3852 | 0.4760 | residual is still better under noise |
| 3k_nonuniform_fit7 | 0.1527 | 0.1510 | 0.1866 | 0.2124 | almost tied with centripetal |
| 11k_nonuniform_fit7 | 0.6678 | 0.6930 | 0.7113 | 0.9009 | residual is better on curves more complex than the fit |

| Scenario | residual MSE | centripetal MSE |
|---|---:|---:|
| 7k_nonuniform | 0.012505 | 0.015328 |
| 7k_nonuniform_noise005 | 0.016313 | 0.018772 |
| 3k_nonuniform_fit7 | 0.002569 | 0.002701 |
| 11k_nonuniform_fit7 | 0.053948 | 0.063732 |

Interpretation: the residual-over-centripetal advantage **seems to hold up fairly well** and does not look like trivial overfitting to a single scenario, although on the 3k-data / 7k-fit case it is basically tied with centripetal.

## Failures and negative results worth keeping

This table is here on purpose, so we do not waste time repeating experiments that were already tried.

| Attempt | Outcome | Why it matters |
|---|---|---|
| PPN MLP trained only on equispaced Bezier curves | good in-distribution, poor on B-splines with internal knots | strong evidence of generator bias |
| PPN trained on random B-splines but LSQ loss without internal knots | slightly better OOD, but worse on Bezier and still poor | mismatch between data and loss |
| Fixed-loss with 7 internal knots | better than Bezier-loss, but still worse than centripetal | the issue is not just the loss |
| BatchNorm OFF | improves `t_mae` but not the fit | being close to generator `t_true` does not imply better LSQ fit |
| `t_true` supervision | improves, but still below centripetal | imitating sampling is not enough |
| Margin loss against centripetal | no improvement over `t_true` supervision | objective not effective enough in this form |
| Concatenated `t_uniform,t_chord,t_centripetal` inputs | no improvement | naive feature engineering is not enough |
| Global convex mixture of baselines | collapses to chord_length | local/constrained correction seems better than free global gating |

## Provisional scientific hypotheses

- A standard PPN, if left free to predict `t` from scratch, tends to learn a parametrization close to synthetic generator bias rather than robust geometric structure.
- The quality of `t` with respect to `t_true` is not a perfect proxy for LSQ B-spline fit quality. A parametrization can be more "true" to the sampling process and still be worse for approximation.
- Constraining the model to start from a strong geometric baseline and learn only a small monotone residual seems to greatly reduce pathological solutions.
- The most credible neural advantage so far does not seem to be "replace centripetal", but **refine centripetal in a controlled way**.

## Operational issues and troubleshooting notes

| Problem | Symptom | Workaround / note |
|---|---|---|
| Missing `.git/index` after clone | all files appear as `U` / untracked | `git reset --hard HEAD` if there are no local edits to keep |
| Stale `.git/index.lock` | `fatal: Unable to create ... .git/index.lock: File exists.` | close VS Code / git processes, then `Remove-Item .git\index.lock` |
| `torch` not installable in `.venv` with Python 3.13 | `No matching distribution found for torch` | use Python 3.11 and `.venv311` |
| `venv` / `ensurepip` PermissionError in `%TEMP%` | environment creation fails | force `TEMP` and `TMP` to `.tmp` inside the repo |
| `git status --short` fails on this machine | `sh.exe ... couldn't create signal pipe, Win32 error 5` | likely a local Git/MSYS issue, not caused by this code |
| many `Calcolo pseudoinversa...` messages | numerical instability in the LSQ layer | worth investigating, especially for some parametrizations |
| BatchNorm with `track_running_stats=False` | inference depends on batch composition | one reason why `--normalization none` was tested |

## Suggested roadmap

### P0 - reproducibility and clean baselines

- Keep `train_ppn_mlp.py`, `benchmark_fixed_knots_parametrization.py`, and `inspect_parametrization_quality.py` stable.
- Always save metrics and checkpoint config as JSON/CSV.
- Explicitly document for each run: data generator, true knots, fit knots, sampling strategy, noise, seed.

### P1 - stronger scientific validation

- Extend the OOD stress tests for `centripetal_residual` over more seeds and more true-knot / fit-knot combinations.
- Analyze where the residual correction modifies centripetal: high-curvature segments, nearly linear regions, areas with uneven sampling density, etc.
- Compare not only DHD/MSE, but also the distribution of `Delta t_pred / Delta t_centripetal`.
- Check whether the gain still holds with a metric closer to the orthogonal-projection distance described in the thesis.

### P2 - targeted refactor, without replacing the geometric core

- Better separate `geometry/`, `models/`, `data/`, and `scripts/`.
- Introduce a minimal YAML or TOML config system to avoid overly long CLI commands.
- Reduce duplication of heavy checkpoint files in the repo.
- Add minimal tests for `t` monotonicity, tensor shapes, and LSQ layer consistency.

## Honest note on the legacy code

The original package works as a research baseline, but contains several fragile points typical of a thesis prototype:

- legacy scripts with local absolute paths;
- very large modules with mixed responsibilities;
- dependency on large checkpoints stored inside the repo;
- nontrivial differences between the thesis description and the actual implementation;
- delicate numerical choices in the LSQ layer.

For this reason, the strategy here was not "clean everything immediately", but **first build a minimal, reproducible, comparable experimental pipeline**, and only then think about refactoring.

## TL;DR for tomorrow

If you want to restart from the most promising scientific point, look at these files and outputs first:

| Item | Path |
|---|---|
| training CLI | `scripts/train_ppn_mlp.py` |
| benchmark fair | `scripts/benchmark_fixed_knots_parametrization.py` |
| `t` diagnostics | `scripts/inspect_parametrization_quality.py` |
| residual model | `deep_b_spline_approximation/ppn.py` |
| checkpoint residual | `outputs/ppn_centripetal_residual_fixedloss7_nonuniform/ppn_mlp.pt` |

Command to rerun in order to verify the current best result:

```powershell
.\.venv311\Scripts\python.exe scripts\benchmark_fixed_knots_parametrization.py `
  --checkpoint outputs\ppn_centripetal_residual_fixedloss7_nonuniform\ppn_mlp.pt `
  --output-json outputs\ppn_centripetal_residual_fixedloss7_nonuniform\bench_7k_nonuniform.json `
  --n-eval 200 `
  --n-points 100 `
  --seed 123 `
  --n-internal-knots 7 `
  --data-internal-knots 7 `
  --non-uniform `
  --device cpu
```

Expected result, if everything is consistent:

```text
centripetal_residual: DHD ~ 0.3226, MSE ~ 0.0125
centripetal baseline: DHD ~ 0.3473, MSE ~ 0.0153
```

## Note on legacy pretrained models

To download the legacy checkpoints distributed by the original package:

```powershell
python -m deep_b_spline_approximation download-models
```

That is useful for legacy inference, but the experiments in this README mostly use checkpoints generated under `outputs/` by the new scripts.
