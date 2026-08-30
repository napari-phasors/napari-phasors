# Large Images

A phasor image with tens of millions of pixels is slow in every direction at
once: filtering takes seconds and re-runs on every slider move, the 2-D
histogram behind the phasor plot has to bin every pixel, and each analysis tab
walks the whole array again. Meanwhile a zoomed-out view **cannot display more
than a couple of million pixels anyway** — most of that work is thrown away by
the screen.

The **Level of Detail** section of the **Plot Settings** tab trades resolution
for speed the way a map application does. A layer is shown *and analysed* at a
coarse bin factor while you look at the whole of it; zoom in and the visible
region alone is recomputed at a finer factor, down to full resolution.

The full-resolution arrays are never discarded, so any level — including full
resolution — can be restored at any time.

## Binning phasor data costs resolution, never correctness

Binning phasor data is not the same as downsampling an image. G and S are
binned **photon-weighted**:

$$G_\text{bin} = \frac{\sum_i \bar{I}_i\, G_i}{\sum_i \bar{I}_i},
\qquad
S_\text{bin} = \frac{\sum_i \bar{I}_i\, S_i}{\sum_i \bar{I}_i}$$

where $\bar{I}_i$ is pixel $i$'s mean intensity. The consequence is the
important part:

> **A binned level is identical to phasor-transforming the summed raw signal
> over each block.** It is not an approximation of the phasor data.

So binning is a legitimate analysis choice, not only a speed setting: binning
$N \times N$ buys $N^2$ photons per point, which is exactly what a dim dataset
needs. What you give up is spatial detail, and nothing else. A plain
unweighted mean of G would instead bias every bin towards its dimmest pixels.

Two supporting decisions worth knowing:

- **Intensity is the block mean, not the sum**, so a binned level keeps the
  same intensity scale as full resolution. Thresholds, contrast limits and
  colormaps carry across levels unchanged.
- **Blocks are origin-aligned and padded, never trimmed**, so bin `(i, j)`
  always covers full-resolution rows `[i*f, (i+1)*f)`. That exact
  correspondence is what lets a zoomed region be refined and pasted back into
  the coarse canvas without anything shifting.

Blocks in which no pixel is valid come back as `NaN` in all three arrays,
which is how the rest of the plugin already marks "no data".

Bin factors are **powers of two**, so levels stay nested: every level is an
exact re-bin of the one above it, and refining never has to go back to full
resolution. The largest factor is `32x32`.

## The Level of Detail controls

They live in **Plot Settings**, below the phasor-center controls — not inside
an analysis tab, because applying a level swaps the arrays that *every* tab
reads.

| Control | Description |
|---------|-------------|
| **Bin large images** | Turns binning on for the layers currently selected in the plotter. Off restores every managed layer to full resolution and releases the pyramid. |
| **Binning** | `Auto` picks the coarsest factor that still brings the image under the pixel budget (4 million pixels by default). A fixed factor — `2x2` through `32x32` — pins the level and stays there. |
| **Refine on zoom** | Recompute the visible region at finer detail as you zoom in, down to full resolution once the region is small enough. Available only under `Auto`. |
| **Full resolution** | Put every managed layer back at full resolution while leaving binning enabled. Available only under `Auto`. |
| Status label | `Full resolution`, `Binned 4x4`, or a list when the selected layers are on different levels. |

**Refine on zoom** and **Full resolution** are offered only under `Auto` on
purpose: both move the level out from under a fixed factor, so with `4x4`
selected a single zoom would silently undo the choice. With a fixed factor,
turn **Bin large images** off to get back to full resolution.

## What follows the level

Applying a level swaps `original_mean`, `G_original` and `S_original` in the
layer's metadata and re-runs the layer's stored filter and threshold on top of
them. Everything downstream reads those same entries, so it all follows the
level automatically:

- the phasor plot, its 2-D histogram and the phasor centers;
- the filter and threshold tab;
- component analysis, FRET, phasor mapping;
- the statistics table.

The layer's `scale` and `translate` are adjusted so it keeps covering the same
ground in world coordinates. **A level swap does not move the image under the
camera**, and napari's own zoom is left alone.

Selection masks are carried across levels by block *maximum*: a bin holding
any selected pixel stays selected. When zoomed out that errs towards showing
data rather than hiding it, which is the safer mistake for a mask marking a
region of interest.

## Refining on zoom

With **Refine on zoom** enabled, camera movements are debounced and the
visible region — plus a small margin — is recomputed at the finest factor that
keeps it under the pixel budget. Because levels are nested and blocks are
origin-aligned, the refined region drops into place in the coarse canvas
exactly where it belongs.

The cost of extra detail is therefore bounded by the size of the viewport, not
by the size of the image. Zooming back out returns to the coarse level from
the pyramid rather than recomputing it.

## Choosing a level deliberately

Two reasons to reach for a fixed factor rather than `Auto`:

- **Photon statistics.** A dim acquisition spread over many pixels gives a
  diffuse phasor cloud. `4x4` gives every point sixteen times the photons,
  tightening the cloud, at a quarter of the spatial resolution in each
  direction. Since the binned level *is* the phasor of the summed signal,
  this is the same thing you would have got by binning at acquisition.
- **Reproducibility.** A fixed factor is a stated analysis parameter;
  `Auto` depends on the image size and the pixel budget.

For a first look at a large image, or for interactive exploration, `Auto`
with **Refine on zoom** is the better setting: full detail wherever you are
actually looking, coarse everywhere else.

## Memory

The pyramid holds references to the layer's existing full-resolution arrays
rather than a second copy of them, and caches the three most recently used
whole-image levels. Full resolution is never cached, since it is the source
data.

A coarse level costs $1/f^2$ of the full-resolution footprint, so the whole
cache is small next to the data it describes.

The manager itself is created **lazily**: a session that never bins anything
never allocates one, and no camera connection exists until binning is asked
for.

## Scripting

```python
from napari_phasors._binning import (
    PhasorPyramid,
    bin_factor_for_shape,
    bin_phasor_arrays,
)

metadata = layer.metadata
mean, real, imag = (
    metadata["original_mean"],
    metadata["G_original"],
    metadata["S_original"],
)

# Bin once, directly.
mean_4, real_4, imag_4 = bin_phasor_arrays(mean, real, imag, 4)

# Or build a pyramid and ask it for levels and regions.
pyramid = PhasorPyramid(mean, real, imag)
factor = bin_factor_for_shape(mean.shape[-2:], budget=4_000_000)
level = pyramid.level(factor)
region = pyramid.region(2, 1000, 3000, 1000, 3000)
```

| Object | Purpose |
|--------|---------|
| `_binning.bin_phasor_arrays(mean, real, imag, factor)` | Photon-weighted binning of one set of arrays. |
| `_binning.bin_factor_for_shape(shape, budget)` | Coarsest power-of-two factor bringing *shape* under *budget*. |
| `_binning.binned_shape(shape, factor)` | Shape a level would have. |
| `_binning.PhasorPyramid` | Level and region cache over one layer's arrays. |
| `_lod.PhasorLod` | Per-layer level state: `apply(factor)`, `refine_to(region)`, `is_full_detail`. |
| `_lod.LodManager` | Attaches layers, follows the camera, refines the visible region. |
