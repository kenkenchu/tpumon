# Coral Edge TPU Developer's Guide

A practical guide for developing, deploying, and debugging ML models on the
Coral Edge TPU (PCIe M.2) connected to a Raspberry Pi 5.

This guide assumes the hardware and driver setup from [SETUP.md](SETUP.md) is
already complete, and focuses on what you need to know to build and ship models
on the TPU.

---

## Table of contents

1. [Platform overview](#1-platform-overview)
2. [Project layout](#2-project-layout)
3. [Environment setup](#3-environment-setup)
4. [Model design constraints](#4-model-design-constraints)
5. [Training a model](#5-training-a-model)
6. [TFLite export and quantization](#6-tflite-export-and-quantization)
7. [Edge TPU compilation](#7-edge-tpu-compilation)
8. [Running inference](#8-running-inference)
9. [Writing your own inference script](#9-writing-your-own-inference-script)
10. [Performance characteristics](#10-performance-characteristics)
11. [Troubleshooting](#11-troubleshooting)
12. [Reference: hardware and driver internals](#12-reference-hardware-and-driver-internals)
13. [Monitoring with tpumon](#13-monitoring-with-tpumon)

---

## 1. Platform overview

| Component | Details |
|-----------|---------|
| Hardware | Coral Edge TPU (PCIe M.2 module), Global Unichip Corp. |
| Host | Raspberry Pi 5, aarch64, kernel 6.12 |
| PCIe topology | TPU on bus `0001:06`, behind ASMedia 1184 switch alongside 3 NVMe drives |
| Device node | `/dev/apex_0` (group `apex`) |
| Driver | `gasket-dkms` 1.0-18 + 4 custom patches for kernel 6.12 |
| Runtime | `libedgetpu1-std` 16.0 |
| Interrupt mode | hrtimer polling at 250 us (MSI vectors exhausted; INTx broken on Pi 5) |
| Python runtime | `tflite-runtime` for inference, `tensorflow` for training |

The TPU accelerates int8-quantized TFLite models. Inference runs on the TPU's
systolic array; unsupported ops fall back to the CPU. The goal is always 100%
TPU-mapped ops for maximum performance.

## 2. Project layout

```
edge_tpu/
  regime/                        # Market regime detection package
    __init__.py                  #   Re-exports: REGIME_LABELS, WINDOW_SIZE, NUM_FEATURES
    data.py                      #   Data pipeline: fetch, features, labels, windowing
    train.py                     #   CNN definition, training, TFLite export
    infer.py                     #   Edge TPU / CPU inference
  examples/
    classify.py                  # Image classification demo (MobileNet V2 bird)
  models/
    regime_model_quant.tflite    # Quantized int8 model (from train.py)
    regime_model_quant_edgetpu.tflite  # Edge TPU-compiled model
    regime_feature_stats.json    # Feature normalization params (from train.py)
    mobilenet_v2_*.tflite        # Pre-built classification model
    inat_bird_labels.txt         # Label file for classification demo
    parrot.jpg                   # Test image
  patches/                       # Kernel driver patches (4 sequential)
  docs/                          # Documentation
  pyproject.toml                 # Dependencies and project config
```

## 3. Environment setup

### Python dependencies

The project uses `uv` for dependency management. Core inference dependencies
and the optional training extra are declared in `pyproject.toml`:

```toml
# Core (always needed)
dependencies = [
    "tflite-runtime",
    "numpy<2",
    "Pillow",
    "yfinance",
]

# Training only
[project.optional-dependencies]
train = ["tensorflow>=2.13,<2.17"]
```

Install for inference only:

```sh
uv sync
```

Install with training support:

```sh
uv pip install -e ".[train]"
```

### Verify the TPU is accessible

```sh
ls -la /dev/apex_0
# Expected: crw-rw---- 1 root apex 120, 0 ... /dev/apex_0

dmesg | grep -E 'apex|wire|poll'
# Expected: "INTx mode: polling wire interrupts"
```

Quick smoke test:

```sh
uv run python examples/classify.py
# Expected: ~14 ms inference, top result "Ara macao (Scarlet Macaw)"
```

## 4. Model design constraints

The Edge TPU has a fixed set of supported operations. A model must satisfy all
of the following to compile for the TPU with zero CPU fallback.

### Supported ops (common subset)

| Op | Notes |
|----|-------|
| `Conv2D` | Fully supported. Use `(k, 1)` kernels for time-series (no `Conv1D` support). Equal x/y dilation required. |
| `DepthwiseConv2D` | Supported. |
| `TransposeConv` | Supported (compiler v13+). |
| `Dense` (FullyConnected) | Supported. |
| `AveragePooling2D` | Supported with explicit pool size. No fused activation. |
| `MaxPooling2D` | Supported. No fused activation. |
| `Mean` | Supported, but see notes on `GlobalAveragePooling2D` below. |
| `Pad` | Supported. |
| `ReLU`, `ReLU6` | Supported. Prefer `ReLU6` for better quantization (bounded range). |
| `Tanh`, `Logistic` | Supported. |
| `Softmax` | Supported. 1-D input, max 16,000 elements. |
| `L2Normalization` | Supported. |
| `Reshape` | Supported (no computation, just metadata). |
| `Concatenation` | Supported (max 2 inputs if one is constant). |
| `Add`, `Sub`, `Mul` | Supported (element-wise). |
| `Maximum`, `Minimum` | Supported (element-wise). |
| `ReduceMax`, `ReduceMin` | Supported (compiler v14+). |
| `LSTM` | Supported (compiler v14+, unidirectional only). |

This is a common subset. The full list of supported ops with version-specific
constraints is in the
[official Edge TPU model requirements](https://coral.ai/docs/edgetpu/models-intro/#supported-operations).

### Ops to avoid

| Op | Problem | Workaround |
|----|---------|------------|
| `Conv1D` | Not supported by the TPU. | Use `Conv2D` with `(k, 1)` kernels. Treat feature count as spatial height. |
| `GlobalAveragePooling2D` | Keras may emit this as a `MEAN` op with dimensions that cause CPU fallback in practice, even though `Mean` is nominally supported. | Use `AveragePooling2D(pool_size=(H, W))` + `Reshape` for reliable TPU mapping. |
| `BatchNormalization` | Fused during export, but verify in compiler log. | Usually OK if followed by activation. |
| `GRU` | Not supported. | Use CNN, LSTM (v14+), or fully-connected alternatives. |

### Tensor dimensionality

If a tensor has more than 3 dimensions, only the 3 innermost dimensions may
have a size greater than 1. In practice this means batch size must be 1 (which
you already need for static shapes). Unusual tensor layouts with multiple outer
dimensions > 1 will fail compilation.

### Quantization requirements

- **Full integer quantization** -- all weights and activations must be int8.
- **Input/output dtype**: `uint8` (or `int8`).
- Partial quantization (float input/output with int8 internals) will cause
  CPU fallback for the non-quantized segments.
- Use **post-training quantization** with a representative dataset of
  100-200 samples from the training set.

### Static tensor shapes

Every tensor must have a fully static `shape_signature`. The Edge TPU compiler
rejects models with `-1` (dynamic) in any dimension, even if the runtime
`shape` is static. This is the most common compilation failure.

**Cause**: `TFLiteConverter.from_saved_model()` preserves the SavedModel's
dynamic batch dimension (`None` -> `-1`).

**Fix**: Use `TFLiteConverter.from_concrete_functions()` with a fixed
`tf.TensorSpec`:

```python
input_shape = [1] + list(model.input_shape[1:])

@tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
def inference(x):
    return model(x, training=False)

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [inference.get_concrete_function()], model
)
```

### Compilation is single-partition

The Edge TPU compiler maps a contiguous sequence of supported ops to the TPU.
As soon as it encounters an unsupported op, **that op and everything after it
runs on the CPU**, even if later ops are individually supported. This means op
placement order matters: a single unsupported op in the middle of your graph
forces the entire tail to CPU.

Design models so that all TPU-compatible ops form one unbroken block from the
input onwards. If you must include an unsupported op (e.g., custom
post-processing), place it at the very end.

### Activation function choice

Prefer `ReLU6` (`max_value=6.0`) over standard `ReLU`. The bounded output
range `[0, 6]` gives the int8 quantizer a tight scale to work with. Unbounded
`ReLU` lets outlier activations dominate the quantization range, reducing
effective precision for typical values.

## 5. Training a model

Training runs on the Pi 5 or any machine with Python 3.11+ and TensorFlow.
For small models (< 10k parameters, < 5000 samples), CPU training on the Pi
is practical.

### Train the regime detection model

```sh
uv run python regime/train.py -t SPY
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `-t`, `--ticker` | `SPY` | Yahoo Finance ticker symbol |
| `-e`, `--epochs` | `100` | Max epochs (early stopping at patience 10) |
| `-b`, `--batch-size` | `64` | Training batch size |

The training script:

1. Fetches 10 years of daily OHLCV data via `yfinance`
2. Computes 5 dimensionless features per day (log return, normalized range,
   volume ratio, price vs SMA, rolling volatility)
3. Labels each day's regime (Bear/Sideways/Bull) using forward 20-day returns
   with 25th/75th percentile thresholds
4. Builds sliding 30-day windows, splits 80/20 chronologically
5. Trains a 3-layer CNN (~5.9k params) with class weights `{Bear: 2, Sideways: 1, Bull: 2}`
6. Exports a quantized int8 TFLite model and feature normalization stats

**Outputs** (in `models/`):

- `regime_model_quant.tflite` -- int8 quantized model (~8-12 KB)
- `regime_feature_stats.json` -- per-feature mean/std from training set

### Model architecture

```
Input          (1, 30, 5, 1)       # 30 days x 5 features x 1 channel
Conv2D 16      (5,1) same + ReLU6  # 96 params
Conv2D 32      (5,1) same + ReLU6  # 2,592 params
Conv2D 32      (3,1) same + ReLU6  # 3,104 params
AvgPool2D      (30,5)              # Replaces GlobalAveragePooling2D
Reshape        (32,)
Dense 3        + Softmax           # 99 params
                                   # Total: ~5,891 params
```

Every layer uses an Edge TPU-compatible op. The design avoids `Conv1D`,
`GlobalAveragePooling2D`, and unbounded `ReLU` -- see section 4 for rationale.

## 6. TFLite export and quantization

The export is handled by `regime/train.py` automatically after training. If
you need to export manually or adapt it for a new model, here is the pattern.

### Quantization approaches

There are two ways to produce a fully int8-quantized model:

- **Post-training quantization** (used by `regime/train.py`): train in float32,
  then calibrate with a representative dataset during TFLite conversion. Simple
  and works well for most models.
- **Quantization-aware training (QAT)**: insert fake quantization nodes into
  the graph during training so the model learns to compensate for quantization
  error. Generally produces better accuracy, especially for models sensitive to
  precision loss. See the
  [TensorFlow QAT guide](https://www.tensorflow.org/model_optimization/guide/quantization/training).

The recipe below uses post-training quantization. To switch to QAT, apply
`tfmot.quantization.keras.quantize_model()` before training and use
`tf.lite.TFLiteConverter.from_keras_model()` for export.

### Full export recipe (post-training quantization)

```python
import numpy as np
import tensorflow as tf

# 1. Pin the input shape to batch=1 (avoids dynamic shape_signature)
input_shape = [1] + list(model.input_shape[1:])

@tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
def inference(x):
    return model(x, training=False)

# 2. Convert from concrete function (not from_saved_model)
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [inference.get_concrete_function()], model
)

# 3. Full int8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# 4. Representative dataset for calibration
def representative_dataset():
    indices = np.linspace(0, len(X_train) - 1, 200, dtype=int)
    for i in indices:
        yield [X_train[i:i+1]]

converter.representative_dataset = representative_dataset

# 5. Convert and save
tflite_model = converter.convert()
Path("models/my_model_quant.tflite").write_bytes(tflite_model)
```

### Verify before compilation

Check that `shape_signature` is fully static:

```python
from tflite_runtime.interpreter import Interpreter

interp = Interpreter(model_path="models/my_model_quant.tflite")
interp.allocate_tensors()
for d in interp.get_input_details():
    sig = d.get("shape_signature")
    assert all(dim > 0 for dim in sig), f"Dynamic dim in {sig}"
```

## 7. Edge TPU compilation

The Edge TPU compiler is an x86_64-only binary. It takes a quantized TFLite
model and generates TPU microcode, wrapping accelerated ops into a
`edgetpu-custom-op` TFLite operator. Three approaches work from the Pi.

### Option A: QEMU user-mode emulation (recommended)

Run the x86_64 compiler natively on aarch64 via QEMU:

```sh
# One-time setup
sudo apt install qemu-user-static binfmt-support

# Download the compiler (amd64 deb, x86_64 binary with bundled libc)
mkdir -p /tmp/edgetpu-compiler && cd /tmp/edgetpu-compiler
curl -qsSL https://packages.cloud.google.com/apt/dists/coral-edgetpu-stable/main/binary-amd64/Packages \
  | grep -A1 '^Package: edgetpu-compiler' | grep Filename
# Download the URL from the Filename field above:
curl -LO https://packages.cloud.google.com/apt/pool/coral-edgetpu-stable/<filename>
dpkg -x edgetpu-compiler_*.deb extracted/

# Compile
extracted/usr/bin/edgetpu_compiler \
  -s models/regime_model_quant.tflite \
  -o models/
```

binfmt_misc transparently routes the x86_64 ELF through `qemu-x86_64-static`.
Compilation of a ~12 KB model takes ~4 seconds under emulation.

### Option B: Google Colab

Use the notebook at `docs/compile_edgetpu.ipynb`:

1. Open in Colab
2. Run all cells
3. Upload `models/regime_model_quant.tflite` when prompted
4. Download the compiled `regime_model_quant_edgetpu.tflite`
5. Copy to `models/` on the Pi

### Option C: Any x86_64 machine

```sh
edgetpu_compiler -s regime_model_quant.tflite -o ./
scp regime_model_quant_edgetpu.tflite pi@<pi-host>:~/development/source/gaming/edge_tpu/models/
```

### Compiler flags

The examples above use `-s` (show ops) and `-o` (output dir). Other useful
flags:

| Flag | Default | Description |
|------|---------|-------------|
| `-s, --show_operations` | off | Print operation mapping (also written to `.log` file) |
| `-o, --out_dir` | `.` | Output directory for compiled model and log |
| `-m, --min_runtime_version` | latest | Target an older Edge TPU runtime version for compatibility |
| `-d, --search_delegate` | off | Recursive search through graph on rare compiler failures |
| `-k, --delegate_search_step` | 1 | Step size for `-d` search (higher = faster but coarser) |
| `-t, --timeout_sec` | 180 | Compiler timeout in seconds |
| `-n, --num_segments` | 1 | Split model into pipeline segments for multiple TPUs |

### Compiler / runtime version compatibility

Compiled models require a minimum Edge TPU runtime version. Models are
forward-compatible (work with newer runtimes). Use `-m` to target an older
runtime if needed.

| Compiler version | Minimum runtime |
|-----------------|----------------|
| 16.0 | 14 |
| 15.0 | 13 |
| 14.1 | 13 |
| 2.1 | 13 |
| 2.0 | 12-13 |
| 1.0 | 10 |

This platform runs compiler 16.0 and runtime 16.0 (`libedgetpu1-std`), so
compatibility is not an issue unless sharing models with older devices.

### Co-compilation (shared parameter caching)

When compiling multiple models together, they share a caching token so
parameter data can coexist in TPU SRAM without cache eviction between
model switches:

```sh
edgetpu_compiler -s model_A.tflite model_B.tflite -o models/
```

Models listed first get priority cache allocation. This is useful if you run
multiple models on the same TPU (e.g., different ticker models).

### Verify the compiler log

The compiler writes a `*_edgetpu.log` next to the output model. Check for
two things:

**1. CPU fallback ops** (target: 0):

```
Edge TPU Compiler version 16.0.384591198
Number of operations that will run on Edge TPU: 8
Number of operations that will run on CPU: 0          <-- target: 0
```

If any ops fall back to CPU, revisit your model architecture (section 4).
Remember the single-partition rule: one unsupported op forces everything
after it to CPU as well.

**2. On-chip memory usage**:

```
On-chip memory available for caching model parameters: 6.91MiB
On-chip memory used for caching model parameters: 4.21MiB
Off-chip memory used for streaming uncached model parameters: 0.00B
```

The Edge TPU has ~8 MB of on-chip SRAM for caching model parameters. If
off-chip usage is non-zero, some parameters must be streamed from host memory
each inference, which is slower. Small models (like the regime CNN at ~12 KB)
fit entirely on-chip.

## 8. Running inference

### Image classification demo

```sh
uv run python examples/classify.py
uv run python examples/classify.py -m models/my_model_edgetpu.tflite -l labels.txt -i photo.jpg
```

### Market regime inference

```sh
# Edge TPU (default -- uses regime_model_quant_edgetpu.tflite)
uv run python regime/infer.py -t SPY

# CPU only (uses regime_model_quant.tflite)
uv run python regime/infer.py -t SPY --cpu

# Custom model path
uv run python regime/infer.py -t SPY -m models/custom_edgetpu.tflite
```

**Automatic fallback**: If the Edge TPU model is not found but the CPU model
exists, `infer.py` falls back to CPU with a warning.

### Expected output

```
Ticker:    SPY
Date:      2026-01-30
Backend:   Edge TPU
Inference: 0.4 ms
Regime:    Sideways
      Bear: 35.9%
  Sideways: 42.6%
      Bull: 21.5%
```

## 9. Writing your own inference script

Here is the minimal pattern for Edge TPU inference with a uint8-quantized
model. Both `examples/classify.py` and `regime/infer.py` follow this pattern.

```python
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# 1. Load the Edge TPU delegate and model
delegate = load_delegate("libedgetpu.so.1")
interpreter = Interpreter(
    model_path="models/my_model_edgetpu.tflite",
    experimental_delegates=[delegate],
)
interpreter.allocate_tensors()

# 2. Get tensor metadata
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# 3. Prepare input -- must match model's expected shape and dtype
#    For uint8 models, quantize float input:
input_scale, input_zero_point = input_details["quantization"]
input_quantized = np.clip(
    np.round(input_float / input_scale + input_zero_point), 0, 255
).astype(np.uint8)

# 4. Run inference
interpreter.set_tensor(input_details["index"], input_quantized)
interpreter.invoke()

# 5. Read and dequantize output
raw_output = interpreter.get_tensor(output_details["index"]).squeeze()
output_scale, output_zero_point = output_details["quantization"]
output_float = (raw_output.astype(np.float32) - output_zero_point) * output_scale
```

### Key points

- **Always load `libedgetpu.so.1` as a delegate**. Without it, the model runs
  entirely on CPU (the `edgetpu-custom-op` will fail or fall back).
- **Quantize inputs yourself** if the model has uint8 input. Use the `scale`
  and `zero_point` from `input_details["quantization"]`.
- **Dequantize outputs** using the output tensor's quantization parameters.
- For **CPU-only** mode, omit the delegate: `Interpreter(model_path=...)`.
  Use the non-`_edgetpu.tflite` model.
- The **first inference** after loading triggers TPU initialization (firmware
  upload). Subsequent calls are faster.

## 10. Performance characteristics

The Edge TPU delivers 4 TOPS (trillion int8 operations per second) at 2W
power draw.

| Metric | Value | Notes |
|--------|-------|-------|
| Theoretical peak | 4 TOPS | 2 TOPS/W |
| Image classification (MobileNet V2) | ~14 ms | 224x224 input, ~3.4M params |
| Regime detection CNN | ~0.4 ms | 30x5 input, ~5.9k params |
| TPU initialization | ~100-500 ms | First inference only (firmware load) |
| hrtimer poll period | 250 us | Adds < 0.25 ms latency to each inference |
| Int8 model size (regime) | ~12 KB | ~5.9k params quantized to 8-bit |
| On-chip parameter SRAM | ~8 MB | Models fitting entirely on-chip run fastest |

Inference latency scales roughly with model parameter count and the number of
operations. The regime CNN is small enough that the poll timer latency
dominates. Larger models like MobileNet spend most time in the TPU systolic
array.

### Parameter caching

The TPU has ~8 MB of on-chip SRAM used as a compiler-allocated scratchpad for
model parameters (weights and biases). Parameters that fit on-chip are accessed
at full speed; parameters that don't fit must be streamed from host memory over
PCIe each inference. The compiler log shows the split (see section 7). For
models under ~6 MB of parameters (after int8 quantization), everything
typically fits on-chip.

## 11. Troubleshooting

### "Encountered unresolved custom op: edgetpu-custom-op"

The Edge TPU delegate is not loaded. Ensure you pass
`experimental_delegates=[load_delegate("libedgetpu.so.1")]` to the
Interpreter.

### "Could not load library libedgetpu.so.1"

`libedgetpu1-std` is not installed, or `/dev/apex_0` is not accessible.
Check:

```sh
dpkg -l libedgetpu1-std
ls -la /dev/apex_0
groups  # must include 'apex'
```

### "Internal error: Failed to apply delegate"

Usually means the model was not compiled for Edge TPU. Use the
`*_edgetpu.tflite` file, not the plain `*_quant.tflite`.

### Dynamic-sized tensors error during compilation

```
ERROR: Attempting to use a delegate that only supports static-sized tensors
with a graph that has dynamic-sized tensors.
```

The model's `shape_signature` contains `-1`. Re-export using
`from_concrete_functions` with fixed `tf.TensorSpec` (see section 6).

Do **not** try to patch the flatbuffer directly -- re-serialized files produce
the same error due to structural differences that the compiler's embedded
TFLite runtime rejects.

### Compiler reports CPU fallback ops

Check the `*_edgetpu.log`. Common causes:

- `GlobalAveragePooling2D` emitting a `MEAN` variant that falls back --
  replace with explicit `AveragePooling2D` + `Reshape`
- `Conv1D` ops -- use `Conv2D` with `(k, 1)` kernels
- Non-int8 tensors -- ensure full integer quantization
- Remember the single-partition rule: one unsupported op forces everything
  after it to CPU as well

### Detecting CPU fallback at runtime

The tflite_runtime has no API to report whether ops actually executed on the
TPU or CPU after inference. However, the Edge TPU compiler partitions the
model at compile time, and the delegate either loads the entire TPU partition
or fails entirely -- there is no partial silent fallback at runtime.

**Delegate load failure (explicit fallback)**

Both `tpumon.py` and `regime/infer.py` catch `ValueError`/`OSError` from
`load_delegate()` and fall back to CPU with a message. This covers the case
where the delegate fails to load at all (missing library, device not
accessible, wrong model file).

**Interrupt activity (strongest runtime signal)**

If inference is running on CPU, no TPU interrupts fire. Run `tpumon` alongside
your inference workload:

```sh
uv run tpumon
```

If the Activity gauge stays at 0% while inference is running, the TPU is not
being used. Any non-zero activity during inference confirms TPU execution.

**Latency threshold**

TPU inference for typical models runs at ~14 ms (MobileNet) or ~0.4 ms
(regime CNN). CPU fallback is 5-10x slower. Sustained latency well above the
expected TPU latency for your model is strong evidence of CPU execution.

**Op inspection (compile-time partition)**

The tflite_runtime has an experimental private API to inspect the model's
operator graph:

```python
ops = interpreter._get_ops_details()
for op in ops:
    print(op["op_name"])
    # "CUSTOM" = mapped to Edge TPU delegate
    # Standard names (CONV_2D, FULLY_CONNECTED, etc.) = CPU
```

This shows the compile-time partition, not runtime behavior. But if you
loaded an `_edgetpu.tflite` model without a delegate, the `CUSTOM` ops would
fail with an error rather than silently falling back.

### Inference hangs (no result returned)

The TPU interrupt path may not be working. Verify the hrtimer poller is
active:

```sh
dmesg | grep 'polling wire interrupts'
# Expected: "INTx mode: polling wire interrupts (pending=0x48778 mask=0x48780)"
```

If this message is absent, the driver patches may need reapplication (e.g.,
after a kernel or `gasket-dkms` package update). See [SETUP.md](SETUP.md)
section 4.

### Permission denied on /dev/apex_0

```sh
sudo usermod -aG apex $USER
# Log out and back in for group membership to take effect
```

Verify the udev rule exists at `/etc/udev/rules.d/65-apex.rules`:

```
SUBSYSTEM=="apex", MODE="0660", GROUP="apex"
```

## 12. Reference: hardware and driver internals

This section covers the platform-specific details of the driver patches that
make the Edge TPU work on the Pi 5. You typically don't need this for model
development, but it's essential for debugging interrupt or driver issues.

### Why custom patches are needed

The stock `gasket-dkms` 1.0-18 driver fails on this platform for two reasons:

1. **Kernel API changes** (6.1 -> 6.12): renamed symbols, changed function
   signatures.
2. **PCIe interrupt exhaustion**: the Pi 5's MSI controller has only 8 MSI
   vectors for the external PCIe bus, all consumed by the NVMe drives and
   ASMedia switch ports. The TPU gets zero MSI vectors, and legacy INTx
   delivery via GICv2 does not work for ongoing interrupts on this platform.

### Patch summary

| Patch | Purpose | Files |
|-------|---------|-------|
| `0001` | Fix kernel 6.12 API renames (`no_llseek`, `class_create`, `eventfd_signal`) | `gasket_core.c`, `gasket_interrupt.c` |
| `0002` | MSI-X -> MSI -> INTx fallback using `pci_alloc_irq_vectors()` | `gasket_interrupt.c` |
| `0003` | Unmask wire interrupts for INTx mode, add shared-IRQ handling | `gasket_core.h`, `gasket_interrupt.c`, `apex_driver.c` |
| `0004` | hrtimer polling at 250 us (replaces broken INTx delivery) | `gasket_interrupt.c` |

Patches must be applied sequentially. After a `gasket-dkms` package update,
reapply all four and rebuild:

```sh
cd /var/lib/dkms/gasket/1.0/source
sudo patch -p1 < patches/0001-gasket-fix-build-for-kernel-6.12.patch
sudo patch -p1 < patches/0002-gasket-pcie-interrupt-fallback-msix-msi-intx.patch
sudo patch -p1 < patches/0003-gasket-add-intx-wire-interrupt-support.patch
sudo patch -p1 < patches/0004-gasket-wire-int-hrtimer-polling.patch
sudo dkms build gasket/1.0 -k $(uname -r) --force
sudo dkms install gasket/1.0 -k $(uname -r) --force
```

### Interrupt path (how inference completions reach userspace)

```
TPU completes operation
  -> Sets bit in WIRE_INT_PENDING_BIT_ARRAY (BAR offset 0x48778)
  -> hrtimer fires every 250 us, reads pending register
  -> Dispatches per set bit via gasket_handle_interrupt()
  -> Signals eventfd to wake the userspace TFLite runtime
  -> Inference call returns
```

The wire interrupt mask register (`0x48780`) is left in its default state.
The hrtimer poller reads and W1C-clears pending bits directly, bypassing the
PCI interrupt delivery path entirely.

### PCIe topology and MSI vector allocation

```
[0001:00]---00.0-[01-06]--00.0-[02-06]--+-01.0-[03]--00.0  Samsung 980 (4 MSI-X vectors)
                                         +-03.0-[04]--00.0  WD_BLACK SN7100 (INTx fallback)
                                         +-05.0-[05]--00.0  Micron P310 (INTx fallback)
                                         \-07.0-[06]--00.0  Coral Edge TPU (hrtimer poll)
```

The 8 available MSI vectors are consumed by the Samsung 980 (4 MSI-X) and
the 4 ASMedia switch downstream ports (1 each). All other devices fall back
to INTx, which is non-functional for ongoing interrupts on this platform.
NVMe drives on INTx lines use kernel polling mode.

### Verifying driver state

```sh
# Check driver loaded with polling mode
dmesg | grep -E 'apex|wire|poll'

# Check device node permissions
ls -la /dev/apex_0

# Check interrupt assignment
grep apex /proc/interrupts

# Check PCI device status
lspci -vvs 0001:06:00.0
```

## 13. Monitoring with tpumon

`tpumon` is a live terminal dashboard for monitoring the Edge TPU, similar to
`btop` or `nvtop`.

### Quick start

```sh
uv run tpumon                  # hardware monitoring only
uv run tpumon --bench          # with inference benchmark
uv run tpumon -b -m models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite
```

### What it shows

- **TPU status**: device health, temperature, activity (interrupt-rate derived)
- **CPU/fan**: SoC temperature, fan speed
- **PCIe**: link speed/width, AER error counts
- **Temperature history**: 60-second sparkline for TPU and CPU
- **Benchmark** (optional): inference latency stats (min/avg/p50/p95/p99/max),
  throughput, latency sparkline
- **Interrupts**: IRQ rate, per-vector counts, IOMMU page mappings

### Keybindings

| Key | Action |
|-----|--------|
| q | Quit |
| b | Toggle benchmark |
| r | Reset benchmark stats |
| +/- | Adjust polling rate |

### Note on TPU utilization

The Coral Edge TPU does not expose utilization metrics like NVIDIA GPUs.
The "Activity" gauge derives an approximation from the interrupt vector
count rate -- idle TPU = zero interrupts/sec, active TPU = high rate.
