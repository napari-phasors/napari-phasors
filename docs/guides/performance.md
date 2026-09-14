# Performance Settings

Large fluorescence lifetime (FLIM) and hyperspectral datasets can contain millions of pixels across multiple channels, harmonics, and time points. **napari-phasors** is designed to process these large datasets smoothly by distributing computationally intensive tasks across your computer's CPU cores.

Out of the box, parallel processing and memory budget capping are turned off by default to run sequentially with minimal memory footprint. This guide explains how acceleration works, how you can fine-tune performance in the user interface, and best practices for managing large datasets.

---

## The Performance Controls

You can adjust how napari-phasors utilizes your system's resources in the **Performance** section of the **Plot Settings** tab:

**Plot Settings -> Performance**

```{note}
The Performance section includes controls that take effect immediately for all subsequent operations. Any operation already running finishes with the settings it started with.
```

| Setting | What it does | Recommended use |
|---|---|---|
| **Parallel images** | Processes multiple files, layers, or images concurrently across CPU cores. | Disabled by default. Turn ON for multi-layer filtering, stack loading, and batch processing. Turn OFF if importing a very large image stack on a memory-constrained machine. |
| **Parallel regions** | Accelerates a single large image by computing spatial bands concurrently. | Disabled by default. Turn ON to ensure faster median filtering and phasor transforms on high-resolution images. |
| **Memory budget** | Toggle to cap the share of currently free RAM (5–95%) that concurrent workers may use (defaults to **50%** when active). | Disabled by default. When enabled, **50%** works well for most setups. Lower to **20–30%** on shared computers or laptops; raise to **70–80%** on dedicated high-memory workstations. |
| **Phasor precision** | Selects storage precision for newly opened phasor layers: `As read` (`float64`) or `float32 (half memory)`. | Select **`float32`** when working with large time-lapses or multi-gigabyte datasets to cut memory usage in half. |

A live hint line underneath the controls displays your machine's detected CPU cores and the current memory allocation.

---

### Parallel Images vs. Parallel Regions

Understanding the difference between these two toggles helps you choose the best settings for your workflow:

* **Parallel images (fanning out over items):**
  When you import a 3D/time-lapse stack, filter multiple selected layers at once, or run batch analysis, each worker processes an entire image. Because each worker holds a full image in memory, total memory usage scales with the worker count. The **Memory budget** toggle and spinbox allow you to automatically cap how many images are processed concurrently so your machine does not run out of RAM.

* **Parallel regions (splitting single large images):**
  When you are working with a single high-resolution image (e.g., 1024×1024 or 2048×2048 pixels), this switch splits the image into horizontal regions computed across all CPU cores. Because the image is already in memory, this uses almost no extra RAM while dramatically speeding up process like the median filter.

---

### Managing Memory: Budget & Precision

#### Memory budget (% of free RAM)
Parallel processing speeds up computation, but loading multiple images simultaneously increases peak RAM usage. The **Memory budget** switch and spinbox set an upper boundary: when active, napari-phasors checks how much free memory your computer has right now, and sizes its concurrency pools to stay within that budget. When disabled (default), concurrency is not capped by memory budget.

* **When to enable it:** Turn the toggle ON if parallel images is enabled and you want to prevent large batch/stack operations from exhausting system RAM.
* **When to lower it:** If napari or other applications on your computer feel sluggish during large imports, reduce the budget to 25–30%.
* **When to raise it:** If you are working on a powerful workstation with ample free RAM and want maximum import and export speed, raise the budget to 70–80%.

#### Phasor precision (`float32` vs. `float64`)
A phasor image layer maintains several full-resolution arrays (intensity, mean, real phasor coordinate $G$, and imaginary phasor coordinate $S$, plus unfiltered backups). Storing these arrays in 64-bit double precision (`float64`) uses 8 bytes per pixel per array.

Switching **Phasor precision** to **`float32 (half memory)`** reduces storage to 4 bytes per pixel, cutting the memory footprint of opened layers in half:

* **Is scientific precision affected?** In experimental FLIM and hyperspectral imaging, measurement uncertainty is dominated by photon Poisson noise (typically $pprox 10^{-2}$ to $10^{-3}$). In contrast, `float32` provides approximately 7 significant decimal digits of precision (errors $< 10^{-7}$), which is orders of magnitude finer than any experimental detector noise.
* **When to use it:** Highly recommended when analyzing large 3D stacks, long time-lapse series, or many images simultaneously on a machine with 16 GB of RAM or less.
* **Scope:** Applies to images opened *after* changing the setting. Existing open layers retain the precision they were created with.

---

(identical-results)=
## Scientific Accuracy: Identical Results Guarantee

Turning parallelism on or off is strictly a **speed versus system resource** decision—it never alters your scientific data.

Every parallel workflow in napari-phasors produces results that are **bit-for-bit identical** to running sequentially on a single core:
* The same $G$ and $S$ coordinate values down to the last decimal bit.
* The exact same thresholded pixels and NaN mask locations.
* The exact same FRET trajectories, component percentages, and cluster statistics.

The only setting that introduces a difference in numbers is **Phasor precision (`float32`)**, which is why it is strictly opt-in.

---

## Where Acceleration is Applied

| Workflow / Tool | How it is accelerated | Benefit to the user |
|---|---|---|
| **Raw file & stack import** | Files in a stack are decoded across multiple CPU threads. | Significantly faster loading of multi-file time-lapses and z-stacks. |
| **Phasor transform** | High-resolution images are split into regions and computed in parallel. | Instant conversion from raw decay or spectra to phasor coordinates. |
| **Median filtering** | Filter passes are computed in parallel across image bands. | Faster median filtering. |
| **Multi-layer analysis** | Filtering, thresholding, FRET mapping, and component fitting process layers in parallel. | Batch operations across 5–10+ layers complete in the time of a single layer. |
| **Exporting results** | Multi-layer exports compress and save images across threads. | Faster saving of large analysis outputs and TIFF stacks. |
| **Batch analysis** | Pipeline steps fan out across available cores with safe memory limits. | Maximizes throughput when analyzing directories of acquisitions. |

### Smart Safeguards
* **Small images are not split:** For small fields of view (under ~1 megapixel), single-core computation finishes in milliseconds. napari-phasors automatically skips region splitting for small images to avoid thread management overhead.
* **No oversubscription:** Reading a stack of files in parallel does not launch nested threads for internal filters. The plugin automatically coordinates workloads so your CPU is never overwhelmed.

---

## Hardware Recommendations

Depending on your computer setup, here are our recommended settings:

### Standard Laptop (8–16 GB RAM, 4–8 Cores)
* **Parallel images:** ON (turn OFF only if opening very large 3D volumes).
* **Parallel regions:** ON.
* **Memory budget:** 30–40% (leaves headroom for your web browser and OS).
* **Phasor precision:** `float32 (half memory)` (greatly extends how many images you can keep open simultaneously).

### High-End Analysis Workstation (32–128+ GB RAM, 12–32 Cores)
* **Parallel images:** ON.
* **Parallel regions:** ON.
* **Memory budget:** 50–75%.
* **Phasor precision:** `As read` (`float64`) or `float32`.

### Shared Servers & HPC Compute Nodes
When running napari on a shared multi-user server or cluster node with allocated CPU limits, you can restrict napari-phasors to a specific number of threads using the `NAPARI_PHASORS_WORKERS` environment variable before launching:

```bash
NAPARI_PHASORS_WORKERS=4 napari
```

Setting `NAPARI_PHASORS_WORKERS=1` acts as a complete single-threaded mode for baseline performance testing or low-resource environments.
