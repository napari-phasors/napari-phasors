# Performance

napari-phasors spreads the expensive work over the machine's cores. Most of
the time it spends on a large image is inside phasorpy's Cython kernels or
inside NumPy, and both release the GIL while they work, so plain threads are
enough to use every core — without paying to copy multi-hundred-megabyte `G`
and `S` arrays across a process boundary, which would cost more than the
compute it saves.

This is on by default and needs no configuration. The **Performance** section
of the **Plot Settings** tab is there for the cases where the defaults are not
what you want.

For measured numbers on synthetic data — and a notebook you can re-run on your
own hardware — see {doc}`benchmarks`.

## The Performance section

**Plot Settings -> Performance**. All three settings are plugin-wide, and all
three apply from the next operation onwards: anything already running finishes
with the setting it started under.

| Setting | Description |
|---------|-------------|
| **Parallel processing** | Master switch for the thread pools. On, filtering, the phasor transform, component fits and the per-layer analyses are spread over the cores; off, every one of them runs sequentially on the calling thread. |
| **Memory budget** | The share of currently free RAM that concurrent work may occupy, 5–95 %. Defaults to 50 %. |
| **Phasor precision** | The precision newly opened layers store their phasor arrays at: `As read` (the default) or `float32 (half memory)`. |

A hint line underneath reports what the settings currently amount to — the
worker count in use, and how much memory that works out to given what is free
right now.

### Parallel processing

Turning it off is a **speed-versus-resources** choice, not an accuracy one.
The split points were chosen so that every piece of work is independent of the
others, and the results are identical either way — see
[the identical-results guarantee](#identical-results) below.

Turn it off when:

- the machine is shared with another heavy job, or you are on a cluster node
  with a core allocation;
- you are reproducing a timing and want a single-threaded reference;
- you are short on memory (a pool of `N` workers holds `N` items at once,
  where a sequential run holds one);
- you are debugging, and a sequential traceback is easier to read.

### Memory budget

Parallelism changes the memory profile, not just the timing. A serial file
read holds one decoded signal; a pool of `N` holds `N`. The budget is what
sizes every pool whose items are large enough to matter:

- how many files a 3D stack read decodes at once;
- how many layers an export writes at once;
- how many batch-analysis files are decoded ahead of being written out.

Lower it when reading a large stack runs the machine out of memory; raise it
on a machine with headroom and nothing else running. It is the single knob
that makes the whole plugin less memory-hungry, because every pool sized
against whole images reads it.

The default of 50 % deliberately leaves headroom for whatever the caller
accumulates *alongside* the workers — for a stack read, the stacked canvas
being built.

### Phasor precision

This is the one control here that changes the numbers rather than only the
schedule, which is why it is separate and opt-in.

A phasor layer holds six full-size arrays — the intensity twice, and `G`, `S`
and their unfiltered originals — so the storage precision is the largest
single lever on what an open image costs. Storing them as `float32` halves
both the resident layer and every transient copy a filter makes.

The cost is precision: phasor coordinates lie in `[-1, 1]`, where `float32`
carries about seven significant digits against `float64`'s sixteen. That is
far below photon noise in any real acquisition, but it is **not**
bit-identical, which is why it is off by default.

It applies only to images opened **after** it is set. Images already open keep
the precision they were read with; nothing is converted behind your back.

(identical-results)=
## Identical results

Every parallel path in the plugin produces **bit**-identical output to the
sequential one: the same values, the same dtypes, the same NaN positions. Not
"within tolerance" — identical.

Two kinds of parallelism are involved, and each earns that guarantee
differently:

- **Fanning out over items** — one worker per file, per layer or per tile. The
  items are independent, so there is nothing to reconcile.
- **Splitting one array into bands** — for the case that has only one item: a
  single very large image. A band is valid only for a kernel whose output at a
  pixel depends on a bounded neighbourhood. The phasor transform is point-wise,
  so its bands need no overlap at all. The median filter reaches `size // 2`
  pixels per pass, so each band is grown by `repeat * (size // 2)` rows on both
  sides and trimmed back afterwards.

The test suite asserts this directly across a matrix of shapes, harmonic
counts, dtypes and filter parameters, with the splitting threshold patched down
to a single pixel so even small test arrays exercise the split path.

**Phasor precision is the sole exception**, and is opt-in for that reason.

## Where parallelism is applied

| Operation | Unit of work |
|---|---|
| Reading a stack of raw files | one worker per file, pool sized against free memory |
| The phasor transform | bands of one image |
| Median filtering | bands of one image — the hottest interactive path, since it re-runs on every slider move |
| Filtering and thresholding several selected layers | one worker per layer |
| FRET efficiency maps, phasor mapping output maps, component fits, linear projections | one worker per selected layer |
| Exporting layers | one file per worker; the encoders release the GIL while compressing |
| Batch analysis | files decoded ahead of being written out |

Two limits keep this from backfiring:

- **Small arrays are never split.** Below roughly a megapixel the kernels
  finish in a few milliseconds and the thread hand-off — plus, for the median
  filter, the halo recomputation — costs more than it saves. The helpers
  decide this themselves, so no call site has to guard for it.
- **Pools never nest.** A reader that fans out over files calls helpers that
  themselves fan out over bands. Letting both levels spawn `N` threads would
  oversubscribe the machine badly, so a thread already running inside a pool
  runs any nested fan-out sequentially.

However many cores are reported, no more than 16 threads are used: beyond
roughly that point the array work is memory-bandwidth bound, and extra threads
only add contention and peak memory.

## Errors are collected, not fatal

Previously a single failing layer aborted a whole batch. Now each layer's
exception is collected and the failures are reported together, so one bad file
in a batch no longer costs you the rest of the run.

## Overriding the worker count

`NAPARI_PHASORS_WORKERS` sets the thread count for the whole plugin:

```bash
NAPARI_PHASORS_WORKERS=4 napari
```

Setting it to `1` disables concurrency entirely — the escape hatch when
debugging, or when you need a single-threaded reference without touching the
GUI.

Precedence, highest first:

1. the **Parallel processing** switch (off wins over everything);
2. `NAPARI_PHASORS_WORKERS`;
3. any worker count an individual call asks for.

## Scripting

```python
from napari_phasors._parallel import (
    parallel_filter_median,
    parallel_map,
    parallel_phasor_from_signal,
    set_memory_fraction,
    set_parallel_enabled,
)

# Drop-in replacements for the phasorpy calls.
mean, real, imag = parallel_phasor_from_signal(signal, axis=0)
mean, real, imag = parallel_filter_median(mean, real, imag, repeat=3, size=5)

# Fan out over independent items, preserving order.
results = parallel_map(process_one, items)

# The same settings the GUI controls.
set_parallel_enabled(False)
set_memory_fraction(0.25)
```

| Helper | Purpose |
|---|---|
| `parallel_map(func, items)` | Order-preserving map over a pool. `on_error="collect"` returns exceptions in place instead of aborting. |
| `parallel_compute_apply(...)` | Compute in the pool, apply results in order on the calling thread — the split that keeps Qt and napari objects off worker threads. |
| `parallel_phasor_from_signal(...)` | Band-parallel `phasor_from_signal`. |
| `parallel_filter_median(...)` | Band-parallel `phasor_filter_median`, with the halo that keeps it bit-identical. |
| `band_bounds(size)` / `parallel_bands(...)` | Split an array into contiguous row bands. |
| `default_workers()` / `worker_limit_from_env()` | The resolved worker count. |
| `workers_for_memory(...)` / `items_for_memory(...)` | Size a pool so that concurrent items stay inside the memory budget. |

Two rules to follow when using these directly, both of which the plugin holds
throughout:

1. **Qt values are read on the calling thread and passed in**, never read from
   inside a worker. Reading a spinbox from a worker thread is a crash waiting
   to happen.
2. **napari progress bars are Qt objects owned by the calling thread.** Workers
   never touch them; report progress as results are collected.
