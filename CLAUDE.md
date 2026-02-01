## Edge TPU capability

This system has a Coral Edge TPU (PCIe M.2 module) available for ML inference acceleration.

- **Device**: Global Unichip Corp. Coral Edge TPU at PCIe `0001:06:00.0`
- **Device node**: `/dev/apex_0` (group `apex`)
- **Platform**: Raspberry Pi 5 (aarch64, kernel 6.12), TPU sits behind an ASMedia 1184 PCIe switch
- **Driver**: `gasket-dkms` 1.0-18 with 4 custom patches (in `patches/`) for kernel 6.12 compatibility and hrtimer-based interrupt polling
- **Runtime**: `libedgetpu1-std` 16.0 (userspace library)
- **Interrupt mode**: hrtimer polling at 250 µs (MSI vectors exhausted by NVMe drives; INTx delivery broken on Pi 5 GICv2)
- **Inference latency**: ~14 ms (image classification)
- **Run inference**: `uv run python examples/classify.py` (see `examples/classify.py` for example usage)
- **Train regime model**: `uv run python regime/train.py -t SPY`
- **Regime inference**: `uv run python regime/infer.py -t SPY` (or `--cpu` for CPU-only)
- **Monitor TPU**: `uv run tpumon` (or `uv run tpumon --bench` for inference benchmarking)
