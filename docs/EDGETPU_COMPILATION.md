# Edge TPU Compilation on aarch64 (Pi 5)

Learnings from getting the x86_64-only Edge TPU compiler running on the Pi 5
and producing models that compile successfully.

## QEMU user-mode emulation setup

The Edge TPU compiler is a self-contained x86_64 binary with its own bundled
glibc and dynamic linker. QEMU user-mode emulation runs it transparently on
aarch64 via kernel binfmt_misc.

### One-time setup

```bash
sudo apt install qemu-user-static binfmt-support
```

### Getting the compiler

The `edgetpu-compiler` deb is amd64-only and lives in Google's Coral apt repo.
`apt download` won't find it on an aarch64 host. Download it directly:

```bash
# Find the current filename from the repo metadata
curl -s https://packages.cloud.google.com/apt/dists/coral-edgetpu-stable/main/binary-amd64/Packages \
  | grep -A1 '^Package: edgetpu-compiler' | grep Filename

# Download and extract (no need to install — the binary is self-contained)
mkdir -p /tmp/edgetpu-compiler && cd /tmp/edgetpu-compiler
curl -LO https://packages.cloud.google.com/apt/pool/coral-edgetpu-stable/<filename from above>
dpkg -x edgetpu-compiler_*.deb extracted/
```

The compiler binary ends up at:
```
extracted/usr/bin/edgetpu_compiler          # bash wrapper
extracted/usr/bin/edgetpu_compiler_bin/
  edgetpu_compiler                          # actual x86_64 ELF
  ld-linux-x86-64.so.2                     # bundled dynamic linker
  libc.so.6  libc++.so.1  ...             # bundled libraries
```

### Running the compiler

```bash
/tmp/edgetpu-compiler/extracted/usr/bin/edgetpu_compiler \
  -s models/regime_model_quant.tflite \
  -o models/
```

binfmt_misc detects the x86_64 ELF and routes it through `qemu-x86_64-static`
automatically. No explicit QEMU invocation needed.

### Performance

Compilation of a 12 KB model (~5.9k params) takes ~4 seconds under emulation.
This is a one-shot batch operation so emulation overhead is irrelevant.

## TFLite export: avoiding dynamic batch dimensions

### The problem

The Edge TPU compiler rejects models with dynamic tensor dimensions:

```
ERROR: Attempting to use a delegate that only supports static-sized tensors
with a graph that has dynamic-sized tensors.
```

TFLite models have two shape fields per tensor:
- `shape`: the concrete shape used at runtime (e.g., `[1, 30, 5, 1]`)
- `shape_signature`: the symbolic shape from the original graph (e.g., `[-1, 30, 5, 1]`)

The Edge TPU compiler checks `shape_signature`. A `-1` in any dimension causes
the error above, even though the runtime `shape` is fully static.

### What causes dynamic shape_signature

`TFLiteConverter.from_saved_model()` preserves the serving signature's dynamic
batch dimension. TF SavedModels use `None` (→ `-1`) for the batch dim by
default because they're designed for variable batch serving.

This conversion path produces the problematic `-1`:
```python
# BAD: produces shape_signature [-1, 30, 5, 1]
model.export(saved_model_dir)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
```

### The fix: from_concrete_functions with fixed input spec

Use `from_concrete_functions` with an explicit `tf.TensorSpec` that has
`batch_size=1`:

```python
# GOOD: produces shape_signature [1, 30, 5, 1]
input_shape = [1] + list(model.input_shape[1:])

@tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
def inference(x):
    return model(x, training=False)

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [inference.get_concrete_function()], model
)
```

This pins every tensor in the graph to a static shape. The Edge TPU compiler
then accepts the model without issues.

### Why not patch the flatbuffer?

Re-serializing the TFLite flatbuffer through the Python `flatbuffers` library
(even just to set `shape_signature` to `[1,...]` or remove it) produces a
structurally valid but byte-incompatible file. The compiler's embedded TFLite
runtime (older vintage) fails to load these re-serialized models with the same
"dynamic-sized tensors" error. Re-exporting from TF is the reliable path.

## Edge TPU compilation checklist

Before running the compiler on a new model:

1. **All ops must be TPU-mappable.** Check the [supported ops list](https://www.coral.ai/docs/edgetpu/models-intro/#supported-operations).
   Common pitfalls:
   - `GlobalAveragePooling2D` → may emit a `MEAN` op (not supported). Use
     `AveragePooling2D(pool_size=(H, W))` + `Reshape` instead.
   - `Conv1D` → not supported. Use `Conv2D` with `(k, 1)` kernels.
   - `ReLU` → works, but `ReLU6` gives better int8 quantization (bounded range).

2. **Full integer quantization.** Input and output must be `uint8` (or `int8`):
   ```python
   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
   converter.inference_input_type = tf.uint8
   converter.inference_output_type = tf.uint8
   ```

3. **Static shape_signature.** Use `from_concrete_functions` as described above.
   Verify before compiling:
   ```python
   from tflite_runtime.interpreter import Interpreter
   interp = Interpreter(model_path="model.tflite")
   interp.allocate_tensors()
   for d in interp.get_input_details():
       sig = d.get("shape_signature")
       assert all(dim > 0 for dim in sig), f"Dynamic dim in {sig}"
   ```

4. **Check the compiler log.** The output lists every op and whether it mapped
   to Edge TPU or fell back to CPU. Target: all ops mapped, 0 CPU fallback.
   The log file is written next to the output model as `*_edgetpu.log`.

## File locations

| File | Purpose |
|------|---------|
| `/tmp/edgetpu-compiler/extracted/usr/bin/edgetpu_compiler` | Compiler wrapper (persists until reboot) |
| `models/regime_model_quant.tflite` | Input: int8 quantized model |
| `models/regime_model_quant_edgetpu.tflite` | Output: Edge TPU-compiled model |
| `models/regime_model_quant_edgetpu.log` | Compilation log (op mapping details) |
