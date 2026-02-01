# Coral Edge TPU Setup (Raspberry Pi 5)

Setup guide for the Coral Edge TPU (PCIe M.2 module) on a Raspberry Pi 5.
The TPU sits behind an ASMedia 1184 PCIe switch alongside NVMe drives.

## 1. Physical installation

Install the Coral M.2 Accelerator in a PCIe M.2 slot connected to the Pi 5.
If using an ASMedia 1184-based carrier board, the TPU will appear on a
downstream port (e.g. bus `0001:06`).

## 2. Boot configuration

The kernel must use 4KB pages.  The Pi 5's default kernel (`kernel_2712.img`)
uses 16KB pages, which is incompatible with the gasket driver.  Force the 4KB
page-size kernel by adding this to `/boot/firmware/config.txt`:

```
kernel=kernel8.img
```

Also enable PCIe and set Gen 3 speed in the same file:

```
dtparam=pciex1
dtparam=pciex1_gen=3
```

Reboot after making these changes.

## 3. Install packages

Add the Coral APT repository and install the driver + runtime:

```sh
# Add the Coral package signing key
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg

# Add the Coral apt source
echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

sudo apt update
sudo apt install gasket-dkms libedgetpu1-std
```

- `gasket-dkms` (1.0-18) — kernel driver (DKMS), creates `/dev/apex_0`
- `libedgetpu1-std` (16.0) — userspace runtime library

## 4. Patch the driver for kernel 6.12 + Pi 5 PCIe

The packaged `gasket-dkms` source requires several fixes for kernel 6.12 API
changes and Pi 5 PCIe interrupt limitations.  Four patches in `patches/` fix:
API renames, MSI-X/MSI/INTx fallback, wire interrupt support, and hrtimer
polling for platforms with broken INTx delivery.

Apply all four patches sequentially, then rebuild with DKMS:

```sh
cd /var/lib/dkms/gasket/1.0/source

sudo patch -p1 < /path/to/patches/0001-gasket-fix-build-for-kernel-6.12.patch
sudo patch -p1 < /path/to/patches/0002-gasket-pcie-interrupt-fallback-msix-msi-intx.patch
sudo patch -p1 < /path/to/patches/0003-gasket-add-intx-wire-interrupt-support.patch
sudo patch -p1 < /path/to/patches/0004-gasket-wire-int-hrtimer-polling.patch

sudo dkms build gasket/1.0 -k $(uname -r) --force
sudo dkms install gasket/1.0 -k $(uname -r) --force
```

To reapply after a `gasket-dkms` package update, run the same patch + build
commands above.

## 5. System configuration

### Create the `apex` group and add your user

```sh
sudo groupadd apex
sudo usermod -aG apex $USER
```

### Udev rule

Create `/etc/udev/rules.d/65-apex.rules` so `/dev/apex_0` is accessible to
the `apex` group:

```
SUBSYSTEM=="apex", MODE="0660", GROUP="apex"
```

### Module auto-loading

Create `/etc/modules-load.d/apex.conf` to load the driver at boot:

```
gasket
apex
```

To temporarily disable auto-loading for debugging:

```sh
sudo mv /etc/modules-load.d/apex.conf /etc/modules-load.d/apex.conf.disabled
# Optionally blacklist to prevent udev modalias auto-loading:
echo -e "blacklist gasket\nblacklist apex" | sudo tee /etc/modprobe.d/blacklist-apex.conf
```

## 6. Reboot and verify

Reboot, then check that the driver loaded and the device node exists:

```sh
dmesg | grep -E 'apex|wire|poll'
ls -la /dev/apex_0
```

### Expected dmesg output (hrtimer polling mode)

```
apex 0001:06:00.0: enabling device (0000 -> 0002)
apex 0001:06:00.0: Allocated 1 IRQ vectors for 13 interrupts
apex 0001:06:00.0: INTx mode: polling wire interrupts (pending=0x48778 mask=0x48780)
```

The "polling wire interrupts" message confirms the hrtimer poller is
active.  The driver reads `WIRE_INT_PENDING_BIT_ARRAY` every 250 µs
and dispatches completions via eventfds.

**Bad (driver cannot allocate interrupts — not seen since patch 0002):**
```
apex 0001:06:00.0: Couldn't initialize interrupts: -28
```

### Expected device node

```
crw-rw---- 1 root apex 120, 0 ... /dev/apex_0
```

### Classification demo

```sh
uv run python classify.py
```

Expected output:
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

---

## Reference: technical notes

Detailed explanations of each patch and the hardware constraints that
motivated them.  Read this section if you need to understand *why* the
patches exist or debug interrupt issues.

### Patch 0001 — Build fixes (API renames)

`patches/0001-gasket-fix-build-for-kernel-6.12.patch`

Fixes three kernel API changes between 6.1 and 6.12:
- `no_llseek` renamed to `noop_llseek`
- `class_create()` lost its module parameter
- `eventfd_signal()` lost its count parameter

### Patch 0002 — MSI-X/MSI/INTx interrupt fallback

`patches/0002-gasket-pcie-interrupt-fallback-msix-msi-intx.patch`

The original driver uses `pci_enable_msix_exact()` which fails with `-ENOSPC`
on the Pi 5's PCIe bus (behind an ASMedia switch).  The fix replaces it with
the modern `pci_alloc_irq_vectors()` API that falls back from MSI-X to MSI
to INTx:

In `gasket_interrupt.c`, function `gasket_interrupt_msix_init`:
- Replace the `pci_enable_msix_exact()` retry loop with a single
  `pci_alloc_irq_vectors(pci_dev, 1, n, PCI_IRQ_MSIX | PCI_IRQ_MSI | PCI_IRQ_INTX)` call
  (min=1 to accept fewer vectors than interrupts)
- After allocation, set each `msix_entries[i].vector` via `pci_irq_vector(pci_dev, i)`
- Change `request_irq` flags from `0` to `IRQF_SHARED`

In `gasket_interrupt_msix_cleanup`:
- Replace `pci_disable_msix()` with `pci_free_irq_vectors()`

In `gasket_msix_interrupt_handler`:
- Return `IRQ_NONE` (instead of `IRQ_HANDLED`) when the interrupt was not
  for this device, as required by the shared-IRQ contract

With INTx the driver gets 1 vector; all 13 interrupt entries share it via
the existing modulo mapping.

#### Why MSI vectors are exhausted

The Pi 5's MSI controller (`brcm,bcm2712-mip` at `msi-controller@1000131000`)
has only **8 MSI vectors** for the external PCIe bus (device tree
`msi-ranges` count).  The hardware supports 64, but only 8 map to contiguous
GIC SPIs (247–254).  On this system those 8 are fully consumed:

| Consumer | Vectors |
|----------|---------|
| ASMedia 1184 switch downstream ports (×4) | 4 |
| NVMe at bus 03 (MSI-X, 4 queues) | 4 |
| **Total** | **8 / 8** |

NVMe drives at buses 04 and 05 already fell back to legacy GICv2 interrupts.
The Edge TPU gets none, causing `-ENOSPC` (`-28`) even with the MSI-X→MSI
fallback.

### Patch 0003 — Wire interrupt unmasking for INTx mode

`patches/0003-gasket-add-intx-wire-interrupt-support.patch`

The Edge TPU has two independent interrupt paths in hardware:

- **MSI-X path**: INTVECCTL registers, MSIX table, MSIX PBA — the driver
  programs all of these during normal init
- **Wire/INTx path**: `WIRE_INT_PENDING_BIT_ARRAY` (0x48778),
  `WIRE_INT_MASK_ARRAY` (0x48780) — the original driver never touches these

When MSI-X is disabled at PCI level (INTx fallback), wire interrupts remain
masked in hardware.  The TPU never asserts INTx, so inference hangs.

Additionally, the interrupt handler had no hardware-level check for INTx
shared-IRQ identification — it would spuriously signal all 13 eventfds on
every interrupt from other devices sharing the same IRQ line.

The patch adds:

In `gasket_core.h`:
- `wire_int_pending_reg` and `wire_int_mask_reg` fields in
  `gasket_driver_desc` (BAR offsets, 0 = not supported)

In `gasket_interrupt.c`:
- Store gasket_dev back-pointer and wire register offsets in interrupt data
- Detect INTx mode after `pci_alloc_irq_vectors()` (`!msix_enabled &&
  !msi_enabled`) and write 0 to mask register to unmask wire interrupts
- In the interrupt handler: read `WIRE_INT_PENDING_BIT_ARRAY`, return
  `IRQ_NONE` if zero (not our interrupt), dispatch per set bit, W1C clear
- Re-mask wire interrupts (`~0ULL` to mask reg) on cleanup
- Re-detect and re-unmask on reinit

In `apex_driver.c`:
- Set `.wire_int_pending_reg` and `.wire_int_mask_reg` to the Apex register
  offsets (enum values already defined at 0x48778 and 0x48780)

Key assumptions (no datasheet available):
- `WIRE_INT_MASK_ARRAY`: bit=1 means masked, write 0 to unmask
- `WIRE_INT_PENDING_BIT_ARRAY`: bit=1 means pending, write-1-to-clear (W1C)
- Bits 0–12 correspond to the 13 interrupt sources

### Patch 0004 — hrtimer polling (broken INTx delivery on Pi 5)

`patches/0004-gasket-wire-int-hrtimer-polling.patch`

Testing revealed that the wire interrupt unmasking approach (patch 0003) is
**necessary but not sufficient** on the Pi 5.  The root cause is that PCIe
legacy INTx delivery via GICv2 SPIs does not work for ongoing interrupt
signalling on this platform.

**Evidence:**

The Edge TPU is assigned GICv2 SPI 254 (IRQ 41), shared with two NVMe
drives (`nvme1q0`, `nvme1q1`):

```
/proc/interrupts:
 41:  21778  0  0  0  GICv2 254 Level  nvme1q0, nvme1q1, apex
```

The interrupt count is frozen — no new interrupts arrive on this line, not
even from NVMe I/O.  The NVMe driver has silently fallen back to polling
mode.  The same is true for GICv2 SPI 252 (IRQ 39, `nvme2q0/q1`).

Direct register reads via `/dev/mem` confirmed the TPU sets pending bits
(`WIRE_INT_PENDING_BIT_ARRAY = 0x11`, bits 0 and 4 for instruction-queue
and scalar-core-host-0 completions), but the PCI status register never
shows `INTx+` — the interrupt signal never reaches the GIC regardless of
mask register state.

The 8 GICv2 SPIs (247–254) allocated to the MIP MSI controller work for
MSI/MSI-X message-signalled interrupts during device init, but the same
SPIs do not function as level-triggered legacy interrupt inputs.  Any
device falling back to INTx on this PCIe topology will have dead interrupts.

**Fix:**

Replace the wire interrupt unmasking with `hrtimer`-based polling in
`gasket_interrupt.c`:

- Add `struct hrtimer wire_int_timer` to `gasket_interrupt_data`
- Add `gasket_wire_int_poll_fn()` callback: reads `WIRE_INT_PENDING_BIT_ARRAY`,
  dispatches per set bit via `gasket_handle_interrupt()`, W1C clears pending
  bits, reschedules at 250 µs period
- In `gasket_wire_int_setup()`: clear stale pending bits, init and start the
  hrtimer (replaces the mask register write)
- In `gasket_interrupt_msix_cleanup()`: cancel the hrtimer (replaces the
  mask register re-write)
- The existing IRQ handler's wire-interrupt path is retained as a no-cost
  fallback in case the shared IRQ does fire

The 250 µs polling period adds negligible latency to inference calls that
take several milliseconds, while keeping CPU overhead low (~4000 MMIO
reads/sec, each a single 64-bit BAR read).

### Verified state (2026-01-31)

All patches verified — inference runs successfully with auto-loading enabled.

#### Environment

- Kernel: `6.12.62+rpt-rpi-v8` (aarch64)
- Packages: `gasket-dkms` 1.0-18, `libedgetpu1-std` 16.0
- DKMS builds installed for: `6.12.47+rpt-rpi-2712`, `6.12.47+rpt-rpi-v8`,
  `6.12.62+rpt-rpi-2712`, `6.12.62+rpt-rpi-v8`, `6.12.63+deb13-arm64`

#### PCIe topology

```
-[0001:00]---00.0-[01-06]----00.0-[02-06]--+-01.0-[03]----00.0  Samsung 980 NVMe
                                           +-03.0-[04]----00.0  WD_BLACK SN7100 NVMe
                                           +-05.0-[05]----00.0  Micron P310 NVMe
                                           \-07.0-[06]----00.0  Coral Edge TPU
```

All four devices sit behind an ASMedia 1184 PCIe switch.

#### Interrupt assignments

```
 38:    0  0  0  0  GICv2 251 Level  PCIe PME, aerdrv
 39:16231  0  0  0  GICv2 252 Level  nvme2q0, nvme2q1
 41:20999  0  0  0  GICv2 254 Level  nvme1q0, nvme1q1, apex
 45:   12  0  0  0  MIP-MSI-PCI-MSIX-0001:03:00.0 0 Edge  nvme0q0
 46: 4385  0  0  0  MIP-MSI-PCI-MSIX-0001:03:00.0 1 Edge  nvme0q1
 47:    0  0 4640 0  MIP-MSI-PCI-MSIX-0001:03:00.0 2 Edge  nvme0q2
 48:    0  0  0 3395 MIP-MSI-PCI-MSIX-0001:03:00.0 3 Edge  nvme0q3
```

- Samsung 980 (bus 03): 4 MSI-X vectors via MIP controller — uses all 8 available MSI vectors
- WD_BLACK SN7100 (bus 04): legacy GICv2 SPI 252 (IRQ 39), shared
- Micron P310 (bus 05): same IRQ 39 line (not shown, shares nvme2 queues)
- Coral Edge TPU (bus 06): legacy GICv2 SPI 254 (IRQ 41), shared with nvme1

The IRQ 39 and 41 counters are frozen — GICv2 SPIs do not deliver
ongoing level-triggered interrupts on this platform.  NVMe drives on
these lines use kernel polling mode.  The Edge TPU uses the hrtimer
poller (patch 0004).

#### PCI device details

```
0001:06:00.0 System peripheral: Global Unichip Corp. Coral Edge TPU
  Control: Mem+ BusMaster+  DisINTx-
  Interrupt: pin A routed to IRQ 41
  Region 0: Memory at 1800100000 (64-bit, prefetchable) [size=16K]
  Region 2: Memory at 1800000000 (64-bit, prefetchable) [size=1M]
  Kernel driver in use: apex
```

`DisINTx-` confirms INTx is not disabled at PCI command register level.
`INTx-` in the status register confirms the device is not currently
asserting INTx (expected — the hrtimer poller clears pending bits via
W1C before they propagate).

#### dmesg output

```
gasket: loading out-of-tree module taints kernel.
apex 0001:06:00.0: enabling device (0000 -> 0002)
apex 0001:06:00.0: Allocated 1 IRQ vectors for 13 interrupts
apex 0001:06:00.0: INTx mode: polling wire interrupts (pending=0x48778 mask=0x48780)
apex 0001:06:00.0: Apex performance not throttled due to temperature
```

Second init (triggered by first inference call):
```
apex 0001:06:00.0: Allocated 1 IRQ vectors for 13 interrupts
apex 0001:06:00.0: INTx mode: polling wire interrupts (pending=0x48778 mask=0x48780)
```

#### Patch verification

All 4 patches in `patches/` were verified by applying them sequentially
to a fresh extraction of the upstream `gasket-dkms_1.0-18_all.deb`
source and comparing the result against the live DKMS source at
`/var/lib/dkms/gasket/1.0/source/`.  All 4 modified files
(`gasket_core.c`, `gasket_core.h`, `gasket_interrupt.c`,
`apex_driver.c`) are byte-identical after patching.

The patches touch:

| Patch | Files modified |
|-------|----------------|
| 0001 | `gasket_core.c`, `gasket_interrupt.c` |
| 0002 | `gasket_interrupt.c` |
| 0003 | `gasket_core.h`, `gasket_interrupt.c`, `apex_driver.c` |
| 0004 | `gasket_interrupt.c` |

A backup of the original interrupt source is at
`/var/lib/dkms/gasket/1.0/source/gasket_interrupt.c.bak`.
