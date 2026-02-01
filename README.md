# Edge TPU

Coral Edge TPU demos and tools for Raspberry Pi 5. Includes image
classification, market regime detection, a TUI-based live monitoring dashboard, and
kernel driver patches for the Pi 5 PCIe stack.

## Hardware

- **Board**: Raspberry Pi 5 (aarch64, kernel 6.12)
- **TPU**: Coral Edge TPU (M.2 A+E) behind an ASMedia 1184 PCIe switch
- **Device node**: `/dev/apex_0`
- **Driver**: `gasket-dkms` 1.0-18 with 4 custom patches for kernel 6.12
- **Runtime**: `libedgetpu1-std` 16.0
- **Interrupt mode**: hrtimer polling at 250 µs (MSI vectors exhausted; INTx
  broken on Pi 5 GICv2)
- **Inference latency**: ~14 ms (MobileNet V2 image classification)

## Setup

See [docs/SETUP.md](docs/SETUP.md) for full installation instructions
covering boot configuration, driver patching, udev rules, and verification.

## Project layout

```
edge_tpu/
├── examples/
│   └── classify.py            Bird image classification demo
├── regime/
│   ├── data.py                Feature engineering & data pipeline
│   ├── train.py               CNN training & int8 quantization
│   └── infer.py               Regime inference (Edge TPU or CPU)
├── models/                    Pre-trained models & labels (git-ignored)
├── patches/                   gasket-dkms kernel driver patches
├── docs/                      Setup, design, and compilation guides
├── tests/
│   └── test_tpumon.py         Unit tests for tpumon (no hardware needed)
├── tpumon.py                  Live TPU monitoring dashboard
└── pyproject.toml
```

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
# Install dependencies
uv sync

# Run the bird classification demo
uv run python examples/classify.py

# Monitor the TPU
uv run tpumon
```

## Examples

### Image classification

Classifies an image using a quantized MobileNet V2 bird model on the Edge TPU.

```sh
uv run python examples/classify.py
```

```
Image:     parrot.jpg
Inference: 13.9 ms
Top-5 results:
  75.7%  Ara macao (Scarlet Macaw)
   7.1%  Platycercus elegans (Crimson Rosella)
   2.0%  Coracias caudatus (Lilac-breasted Roller)
   1.2%  Trichoglossus haematodus (Rainbow Lorikeet)
   1.2%  Alisterus scapularis (Australian King-Parrot)
```

Options: `--model`, `--labels`, `--image`, `--top-k`.

### Market regime detection

A small CNN (5,891 parameters) classifies daily market conditions as
Bear, Sideways, or Bull using 5 dimensionless features derived from OHLCV
data (log return, normalized range, volume ratio, price vs SMA, rolling
volatility).

**Train** (requires `tensorflow`):

```sh
uv sync --extra train
uv run python regime/train.py -t SPY
```

Produces an int8-quantized TFLite model compiled for the Edge TPU.

**Infer**:

```sh
# On Edge TPU
uv run python regime/infer.py -t SPY

# CPU fallback
uv run python regime/infer.py -t SPY --cpu
```

See [docs/DESIGN.md](docs/DESIGN.md) for architecture details.

## tpumon

Live terminal dashboard for Coral Edge TPU monitoring, built with
[Textual](https://textual.textualize.io/).

```sh
uv run tpumon                    # monitor
uv run tpumon --bench            # monitor + inference benchmarking
uv run tpumon --interval 0.5     # 500 ms polling
uv run tpumon --bench -m MODEL   # benchmark a specific model
```

Panels: TPU status, temperature history, PCIe link & driver info,
interrupt distribution, activity sparkline, and optional benchmark
statistics (min/avg/p50/p95/p99/max latency, throughput).

## Driver patches

Four sequential patches in `patches/` fix `gasket-dkms` 1.0-18 for
kernel 6.12 on the Pi 5:

| Patch | Purpose |
|-------|---------|
| 0001 | Fix kernel 6.12 API renames (`no_llseek`, `class_create`, `eventfd_signal`) |
| 0002 | MSI-X/MSI/INTx interrupt fallback via `pci_alloc_irq_vectors()` |
| 0003 | Wire interrupt unmasking for INTx mode |
| 0004 | hrtimer-based interrupt polling at 250 µs (workaround for broken INTx on Pi 5) |

See [docs/SETUP.md](docs/SETUP.md) for patching instructions and the
reference section there for detailed technical explanations.

## Documentation

- [docs/SETUP.md](docs/SETUP.md) -- Hardware setup & driver patching
- [docs/DESIGN.md](docs/DESIGN.md) -- Market regime CNN design
- [docs/EDGETPU_COMPILATION.md](docs/EDGETPU_COMPILATION.md) -- Edge TPU model compilation
- [docs/EDGETPU_COMPILER_AARCH64.md](docs/EDGETPU_COMPILER_AARCH64.md) -- Compiler setup on aarch64
- [docs/TPU_DEVELOPERS_GUIDE.md](docs/TPU_DEVELOPERS_GUIDE.md) -- Extended TPU reference

## Tests

```sh
uv run pytest tests/
```

No TPU hardware required -- tests exercise parsing and metrics logic only.
