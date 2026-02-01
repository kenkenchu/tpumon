# Edge TPU Compiler on aarch64: Feasibility Assessment

Google does not ship an aarch64 build of `edgetpu_compiler` and has explicitly
ruled it out (GitHub issue google-coral/edgetpu#311, closed Jan 2023). The
compiler is a proprietary binary that generates undocumented Edge TPU microcode.
No open-source reimplementation exists, and the newer LiteRT/ai-edge-torch
stack does not include a replacement.

## What the compiler actually is

The `edgetpu-compiler` Debian package (version 16.0, amd64 only) installs a
3-line bash wrapper that invokes a fully self-contained x86_64 binary with its
own bundled dynamic linker and libraries:

```
/usr/bin/edgetpu_compiler               # bash wrapper
/usr/bin/edgetpu_compiler_bin/
  edgetpu_compiler                      # actual ELF binary
  ld-linux-x86-64.so.2                  # bundled glibc dynamic linker
  libc.so.6  libc++.so.1  libc++abi.so.1
  libdl.so.2  libm.so.6  libpthread.so.0
  libresolv.so.2  librt.so.1
```

The wrapper script:

```bash
#!/bin/bash
d="$(dirname "${BASH_SOURCE[0]}")"
${d}/edgetpu_compiler_bin/ld-linux-x86-64.so.2 \
  --library-path ${d}/edgetpu_compiler_bin \
  ${d}/edgetpu_compiler_bin/edgetpu_compiler "$@"
```

The compiler is not a thin wrapper around libedgetpu. It is a separate
compilation toolchain that partitions TFLite ops, generates Edge TPU microcode
for the systolic array, allocates on-chip SRAM caching, and wraps TPU-targeted
ops into a single `edgetpu-custom-op` TFLite custom operator.

## Viable approaches (ranked)

| Approach | Feasibility | Notes |
|----------|-------------|-------|
| QEMU user-mode (`qemu-user-static`) | High | Lightest-weight. Bundled libc avoids host/guest ABI issues. |
| Docker multiarch + QEMU | High | Well-documented images exist. Heavier than bare QEMU. |
| Google Colab | High | Free, zero local setup, manual upload/download. |
| x86_64 machine + scp | High | Simplest if hardware is available. |
| Box64 | Low | Bundled libc defeats Box64's native-wrapping optimization. |
| Legacy aarch64 binary (pre-2.1) | Very low | "Internal compiler error. Aborting!" on modern systems. |
| Open-source reimplementation | None | Edge TPU ISA is undocumented. |

## Recommended: QEMU user-mode emulation

The self-contained nature of the binary (bundled libc, ld-linux) is favorable
for QEMU user-mode emulation. The kernel's binfmt_misc transparently routes
x86_64 ELFs through QEMU.

### Setup

```bash
# Install QEMU x86_64 emulator and binfmt registration
sudo apt install qemu-user-static binfmt-support

# Download and extract the edgetpu-compiler Debian package
mkdir -p /tmp/edgetpu-compiler && cd /tmp/edgetpu-compiler
apt download edgetpu-compiler
dpkg -x edgetpu-compiler_*.deb extracted/

# Compile a model
extracted/usr/bin/edgetpu_compiler -s models/regime_model_quant.tflite -o models/
```

### Docker multiarch alternative

```bash
# Register x86_64 binfmt handler
docker run --rm --privileged tonistiigi/binfmt --install amd64

# Run the compiler in an x86_64 container
docker run --platform linux/amd64 --rm \
  -v $(pwd)/models:/home/edgetpu \
  tomassams/docker-edgetpu-compiler \
  edgetpu_compiler -s /home/edgetpu/regime_model_quant.tflite \
                   -o /home/edgetpu/
```

### Performance expectations

For a 12.4 KB model (~5.9k parameters), compilation should finish in seconds
even under full x86_64 emulation. This is a one-shot batch operation.

## What does not work

- **No Python-only alternative** — compilation requires generating hardware
  microcode for the undocumented Edge TPU ISA.
- **Ultralytics `export(format="edgetpu")`** — explicitly raises `SystemError`
  on ARM64 hosts.
- **LiteRT / ai-edge-torch** — newer Google AI Edge stack does not include an
  Edge TPU compiler replacement.
- **Coral project status** — effectively end-of-life. The google-coral/edgetpu
  README states: "The code that remains in this repo is legacy and might be
  removed in the future, and all the code is no longer maintained."

## References

- [google-coral/edgetpu#311 — Compiler for ARM](https://github.com/google-coral/edgetpu/issues/311)
- [Edge TPU Compiler docs](https://www.coral.ai/docs/edgetpu/compiler)
- [Compiler wrapper script source](https://github.com/google-coral/edgetpu/blob/master/compiler/x86_64/edgetpu_compiler)
- [tomassams/docker-edgetpu-compiler](https://github.com/tomassams/docker-edgetpu-compiler)
- [Colab compilation notebook](https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb)
- [multiarch/qemu-user-static](https://github.com/multiarch/qemu-user-static)
