# Market Regime Detection CNN — Design

Classifies the current stock market regime (Bull / Bear / Sideways) from
daily OHLCV data using a small CNN quantized for the Coral Edge TPU.

## Architecture overview

```
 yfinance          regime/data.py          regime/train.py        Edge TPU
┌────────┐   OHLCV  ┌──────────────┐  (30,5,1)  ┌────────────┐   .tflite
│ Yahoo  │ ───────> │  5 features  │ ────────>  │  CNN ~5.9k │ ─────────>
│Finance │          │  30-day win  │            │  params    │  quantized
└────────┘          │  regime lbl  │            │  int8 PTQ  │   uint8
                    └──────────────┘            └────────────┘
                           │                                      │
                           │  3 months           regime/infer.py  │
                           │  lookback          ┌──────────────┐  │
                           └──────────────────> │  load model  │ <┘
                                                │  quantize in │
                                                │  invoke TPU  │
                                                │  dequant out │
                                                └──────┬───────┘
                                                       │
                                                Bear / Sideways / Bull
```

Three modules in the `regime/` package share a common data module:

| Module | Role | Runtime deps |
|--------|------|--------------|
| `regime/data.py` | Feature engineering, labeling, windowing | `numpy`, `yfinance` |
| `regime/train.py` | Model definition, training, TFLite export | `tensorflow` (optional extra) |
| `regime/infer.py` | Edge TPU / CPU inference on live data | `tflite-runtime`, `yfinance` |

## Data pipeline

### Source

`yfinance` downloads 10 years of daily OHLCV for a given ticker (default SPY).
The pipeline handles MultiIndex column flattening that yfinance sometimes
returns for single-ticker downloads.

### Features

Five dimensionless ratios computed per trading day.  All are scale-invariant
across tickers and price levels.

| # | Feature | Formula | Intuition |
|---|---------|---------|-----------|
| 1 | Log return | `ln(close[t] / close[t-1])` | Daily price momentum |
| 2 | Normalized range | `(high - low) / close` | Intraday volatility |
| 3 | Volume ratio | `volume / SMA(volume, 20)` | Relative activity |
| 4 | Price vs SMA | `(close - SMA20) / SMA20` | Trend position |
| 5 | Rolling volatility | `std(log_ret, 20) * sqrt(252)` | Annualized vol |

The first 20 rows are NaN (insufficient SMA/std history) and are dropped
during windowing.  Rolling helpers use cumulative-sum for mean (O(n)) and
a simple loop for std (O(n*w)) — adequate for ~2500 rows.

### Regime labeling

Forward-looking 20-day returns are computed at each point:

```
fwd_return[t] = close[t+20] / close[t] - 1
```

Percentile thresholds on the full valid range split returns into three
regimes:

| Regime | Code | Condition | Expected share |
|--------|------|-----------|----------------|
| Bear | 0 | fwd_return < 25th percentile | 25% |
| Sideways | 1 | 25th–75th percentile | 50% |
| Bull | 2 | fwd_return > 75th percentile | 25% |

The last 20 days are unlabelable (no forward data) and are excluded.

### Windowing and normalization

Each sample is a sliding window of 30 consecutive days, shaped `(30, 5, 1)`
for Conv2D input.  The label corresponds to the regime of the window's last
day.

Feature normalization uses per-feature mean/std computed from the training
set only.  These statistics are saved to `models/regime_feature_stats.json`
and loaded at inference time to ensure identical preprocessing.

### Train/test split

Chronological 80/20 split — no shuffling, no look-ahead bias.  With 10
years of SPY data this yields ~1950 training and ~490 test samples.

## CNN model

All operations are Edge TPU-mappable: Conv2D, ReLU (via ReLU6), average
pooling, Dense, Softmax.

```
Layer                          Output shape       Params
─────────────────────────────────────────────────────────
Input                          (batch, 30, 5, 1)       0
Conv2D 16, (5,1), same         (batch, 30, 5, 16)     96
ReLU6                          (batch, 30, 5, 16)      0
Conv2D 32, (5,1), same         (batch, 30, 5, 32)  2,592
ReLU6                          (batch, 30, 5, 32)      0
Conv2D 32, (3,1), same         (batch, 30, 5, 32)  3,104
ReLU6                          (batch, 30, 5, 32)      0
AveragePooling2D (30,5)        (batch, 1, 1, 32)       0
Reshape                        (batch, 32)              0
Dense 3 + Softmax              (batch, 3)              99
─────────────────────────────────────────────────────────
Total                                              ~5,891
```

### Design choices

**Conv2D (k,1) kernels** — Convolve along the time axis only.  TFLite's
Edge TPU delegate does not support Conv1D, so the 5 features are treated
as a spatial height dimension and the kernel width is fixed at 1.

**ReLU6** — Clips activations to [0, 6], giving the int8 quantizer a
bounded range to work with.  Standard ReLU would let outliers dominate the
quantization scale.

**AveragePooling2D + Reshape** — Replaces GlobalAveragePooling2D, which
compiles to a MEAN op that may not map to the Edge TPU.  Explicit pool size
`(30, 5)` followed by reshape produces the same result with guaranteed Edge
TPU compatibility.

**Class weights {Bear: 2.0, Sideways: 1.0, Bull: 2.0}** — The 25/50/25
label split means the Sideways class dominates.  Doubling the loss
contribution for Bear and Bull prevents the model from collapsing to
always-Sideways predictions.

### Quantization

Post-training integer quantization using 200 representative samples from
the training set.  Both input and output tensors use `uint8` dtype.  Target
op set: `TFLITE_BUILTINS_INT8`.

The quantized model is saved as `models/regime_model_quant.tflite` (~8 KB).

## Inference

`regime/infer.py` follows the same Edge TPU delegate pattern as `examples/classify.py`:

1. Load `regime_feature_stats.json` (normalization parameters from training)
2. Load TFLite model with Edge TPU delegate (or plain CPU interpreter with `--cpu`)
3. Fetch 3 months of recent OHLCV via yfinance
4. Compute the 5 features, take the last 30-day window, normalize
5. Quantize float32 input to uint8 using the model's input scale/zero_point
6. Invoke the interpreter
7. Dequantize uint8 output to float32 probabilities
8. Argmax for regime label, print probabilities

### Fallback behavior

If the Edge TPU-compiled model is not found but the uncompiled quantized
model exists, the script falls back to CPU inference with a warning.  The
`--cpu` flag forces CPU mode explicitly.

## File inventory

```
regime/
  __init__.py                      Package init, re-exports key constants
  data.py                          Data pipeline (shared)
  train.py                         Training + TFLite export
  infer.py                         Edge TPU / CPU inference
examples/
  classify.py                      Edge TPU image classification demo
models/
  regime_model_quant.tflite        Quantized int8 model (generated by training)
  regime_model_quant_edgetpu.tflite  Edge TPU-compiled model (generated by edgetpu_compiler)
  regime_feature_stats.json        Feature normalization stats (generated by training)
docs/
  DESIGN.md                        This file
  SETUP.md                         Hardware/driver setup guide
  EDGETPU_COMPILER_AARCH64.md      QEMU-based compilation notes
  EDGETPU_COMPILATION.md           TFLite export and compilation guide
  compile_edgetpu.ipynb            Colab compilation notebook
```

## Progress

| Step | Status | Notes |
|------|--------|-------|
| Data pipeline (`regime/data.py`) | Done | 1956 train / 490 test samples, 26.7/48.1/25.3 label split |
| CNN + training script (`regime/train.py`) | Done | ~5.9k params, int8 PTQ export, feature stats saved |
| Inference script (`regime/infer.py`) | Done | Edge TPU + CPU modes, auto-fallback |
| Dependency config (`pyproject.toml`) | Done | `yfinance` added, `[train]` extra for TensorFlow |
| Train model | Done | Pi 5 aarch64, TF 2.16.2, 17 epochs (early stop), val acc 53.9% |
| TFLite int8 export | Done | 12.4 KB, uint8 I/O, via SavedModel (Keras 3 workaround) |
| CPU inference test (`--cpu`) | Done | SPY → Sideways (Bear 35.9%, Sideways 42.6%, Bull 21.5%) |
| Edge TPU compilation | Done | Via QEMU user-mode (`qemu-user-static`) on Pi 5, all 8 ops mapped |
| Edge TPU inference test | Done | 0.4 ms inference, Edge TPU delegate |

## Platform requirements

| Step | Pi 5 (aarch64) | x86_64 | Notes |
|------|:-:|:-:|-------|
| Training | Yes | Yes | TensorFlow has aarch64 wheels since 2.10 |
| TFLite export | Yes | Yes | Part of the training script |
| Edge TPU compilation | No | **Required** | Google only ships x86_64 binaries |
| CPU inference | Yes | Yes | `tflite-runtime` only |
| Edge TPU inference | **Required** | No | Needs the Coral hardware |

Training on the Pi 5 is practical for this model: ~5.9k parameters and
~2000 samples means CPU training is fast even without a GPU.  The main cost
is TensorFlow's install footprint (~600 MB of wheels).

The Edge TPU compiler (`edgetpu_compiler`) is the only step that cannot run
on aarch64.  Use an x86_64 machine or Google Colab for this single step,
then copy the compiled `_edgetpu.tflite` back to the Pi.

## Next steps

### 1. Train the model

On the Pi 5 (or any machine with Python 3.11+):

```sh
uv pip install -e ".[train]"
uv run python regime/train.py -t SPY
```

This produces `models/regime_model_quant.tflite` and
`models/regime_feature_stats.json`.

Expect validation accuracy above the 33.3% random baseline.  With
EarlyStopping (patience 10), training typically converges within 30–50
epochs.

### 2. Verify CPU inference

On the Pi (or any machine):

```sh
uv run python regime/infer.py -t SPY --cpu
```

This confirms the data pipeline, feature normalization, and TFLite
model work end-to-end without the Edge TPU.

### 3. Compile for Edge TPU (x86_64 only)

On an x86_64 machine (or Google Colab) with the Edge TPU compiler
installed.  There is no aarch64 build of `edgetpu_compiler`.

```sh
edgetpu_compiler -s models/regime_model_quant.tflite -o models/
```

Check the compiler log — all ops should map to the Edge TPU.  If
`AveragePooling2D` maps as `MEAN` and falls back to CPU, the model
architecture already uses the explicit pool+reshape workaround, so this
should not happen.

Copy the resulting `regime_model_quant_edgetpu.tflite` to the Pi's
`models/` directory.

### 4. Run on Edge TPU

On the Pi 5 with the compiled model in place:

```sh
uv run python regime/infer.py -t SPY
```

Expected inference latency is under 2 ms for this model size (~5.9k params,
~8 KB quantized).

### 5. Future improvements

- **Multi-ticker training** — Train on a basket (SPY, QQQ, IWM) to improve
  generalization across market segments.
- **Feature expansion** — Add RSI, MACD, or Bollinger Band width as
  additional channels.  Model input shape becomes `(30, N, 1)`.
- **Online retraining** — Periodically retrain on recent data to adapt to
  regime shift dynamics.
- **Ensemble** — Run multiple tickers through the TPU and aggregate regime
  signals for a market-wide view.
- **Alerts** — Trigger notifications on regime transitions (e.g., Sideways
  to Bear).
