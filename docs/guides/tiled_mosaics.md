# Tiled Mosaics

Large samples are often acquired as a mosaic: the stage steps across the
specimen and records a grid of overlapping tiles. napari-phasors can read such
an acquisition and give you back **one** intensity layer with **one** set of
phasor coordinates behind it, so every other part of the plugin — the phasor
plot, filtering, selections, component analysis, FRET, mapping, export — treats
the mosaic exactly like any single image.

The stitching happens **in phasor space**, not on the raw data. Each tile is
phasor-transformed on its own and only the resulting mean intensity, G and S
are blended together. This is what makes a mosaic that would never fit in
memory as a raw signal stack tractable: peak memory stays at roughly one
tile's signal, not the whole acquisition's.

## Opening a mosaic

Tile mode lives in the **Phasor Custom Import** widget, alongside the
single-file and 3D-stack import paths described in {doc}`open_files`:

**Plugins -> napari-phasors -> Phasor Custom Import**

1. Click **Open tiled mosaic**.
2. Choose where the tiles are:
   - **Select files...** — pick the tile files individually.
   - **Select folder...** — use every supported file the folder contains.
3. Describe the layout in the **Tile Layout** dialog that opens (see below)
   and click **OK**.
4. Set the import options as usual (harmonics, frames, channels, and any
   format-specific option), then click **Stitch Mosaic (N tiles)**.

A mosaic can also live inside a *single* file — a Zeiss CZI mosaic, or any
file with a mosaic, view, block or scene dimension. Select that one file and
the dialog offers its tile dimension under **Tiles inside each file**.

### Supported formats

Every raw FLIM and hyperspectral format the plugin reads can be tiled:
`.ome.tif`, `.tif`, `.tiff`, `.lsm`, `.ptu`, `.fbd`, `.sdt`, `.czi`, `.flif`,
`.bh`, `.b&h`, `.bhz`, `.bin`, `.r64`, `.ref`, `.ifli` and `.lif`.

All tiles of one mosaic must share a single extension — a mosaic mixing
formats is rejected rather than guessed at.

## The Tile Layout dialog

The dialog previews the arrangement **before** anything is read, because
reading a mosaic the wrong way round is both slow to discover and slow to
undo. The preview redraws as you change any control, and the status line
reports the resulting canvas size.

| Control | Description |
|---------|-------------|
| **Tiles inside each file** | Which dimension of a file holds its tiles — `Mosaic tiles` for a CZI, or `M`, `V`, `B`, `S`. Choose `One tile per file` when each selected file is a single tile. Only shown when the files actually carry such a dimension. |
| **Binning** | Sum square blocks of pixels while reading, from `None` up to `16 x 16`. See [Binning on read](#binning-on-read). Only shown when tile positions are known in advance. |
| **Layout from** | Where the arrangement comes from — see the table below. |
| **Tiles per row** | Manual layout only. Tiles in each row, comma separated (`5, 7, 9, 9, 7, 5`), or `3x9` as shorthand for three rows of nine. Rows may hold different numbers of tiles. |
| **Order** | `Raster` restarts every row on the same side; `Snake` alternates direction, as most stage controllers do. |
| **Start** | Which corner the first tile occupies: `Top-left`, `Top-right`, `Bottom-left`, `Bottom-right`. |
| **Short rows** | Where a row with fewer tiles sits relative to the longest one: `Center`, `Left` or `Right`. |
| **Pattern** | File-name layout only. A regular expression matched against each file name, defining `row` and `col` groups, or a single `index` group. Defaults to `(?P<row>\d+)\D+(?P<col>\d+)`. |
| **Overlap Y (%)** / **Overlap X (%)** | Nominal overlap between neighbouring tiles, 0–90 %. A starting value only — it can be re-tuned or measured after the tiles are read. |
| **Blending** | How intensity is combined where tiles overlap. See [Blend modes](#blend-modes). |

### Where the layout comes from

The **Layout from** options are listed in increasing order of trust, and the
most trustworthy one available is preselected.

| Source | Use when |
|--------|----------|
| **Tile positions recorded in the file** | The acquisition stored the position of each tile. Nothing has to be described by hand; the positions are clustered onto a grid. Offered only when such positions exist. |
| **Stage positions in files** | Each file records the stage coordinates at which it was acquired. Positions are clustered into rows and columns, so an irregular or drifting grid still resolves. OME-TIFF only — other formats rarely carry absolute stage coordinates. |
| **File names** | The tile indices are in the file names (`tile_r02_c05.ptu`). Give a regular expression under **Pattern**. |
| **Rows (manual)** | Nothing about the arrangement is recorded. Type the shape yourself under **Tiles per row**, with the traversal order and starting corner. |

If a layout resolves to a single tile — a `1x1` spec, a pattern that matched
nothing — the dialog says so rather than importing a mosaic of one.

## After stitching: the Mosaic stitching section

Once the layout is set, a **Mosaic stitching** section appears in the import
widget, above the transform button. Its controls stay live *after* the mosaic
has been stitched, and changing any of them re-blends the tiles **already held
in memory** — nothing is read from disk again, so trying a different overlap
is close to instant.

| Control | Description |
|---------|-------------|
| **Overlap Y** / **Overlap X** | Sliders, in tenths of a percent, from 0 to 90 %. The mosaic re-stitches when you release the slider (immediately for arrow keys). |
| **Blending** | `Feather`, `Average` or `Sum counts`. See below. |
| **Estimate overlap from data** | Cross-correlates neighbouring tiles and sets the sliders to the overlap that aligns them best. Enabled once the tiles have been read. |

The status line underneath reports what happened — the estimated overlap, the
stitched canvas size, or why the tiles could not be matched.

The stitched layer is updated **in place** when you re-stitch, so it keeps its
position in the layer list along with the contrast limits and colormap you
have set.

### Estimating the overlap

**Estimate overlap from data** matches each pair of neighbouring tiles by
normalised cross-correlation over a range of candidate overlaps, and takes
the median of the per-pair results for each axis. It reports Y and X
separately, and reports only the axis it could measure — a single row of
tiles gives an X overlap and nothing for Y.

If the tiles cannot be matched at all — too little overlap, no structure in
the overlapping strip, or a layout whose order is wrong — the status line
says so and the sliders are left alone. That is usually a sign the traversal
order or starting corner needs fixing, not the overlap.

## The blend is photon-weighted

This is the one piece of physics worth understanding. Where tiles overlap, G
and S are combined weighted by intensity:

$$G_\text{blend} = \frac{\sum_i \bar{I}_i\, G_i}{\sum_i \bar{I}_i},
\qquad
S_\text{blend} = \frac{\sum_i \bar{I}_i\, S_i}{\sum_i \bar{I}_i}$$

where $\bar{I}_i$ is tile $i$'s mean intensity at that pixel.

The phasor of a sum of signals is the photon-weighted average of the
individual phasors. Photon weighting is therefore not a smoothing choice —
it is the only combination that gives **the same answer as summing the raw
signals and phasor-transforming once**. A plain unweighted mean of G and S
would quietly bias every seam towards the dimmer tile.

Two consequences:

- A pixel contributes only where all of its mean, G and S are finite.
  `phasor_from_signal` returns NaN wherever a pixel collected no photons, and
  those pixels are excluded from the average rather than poisoning it.
- Pixels no tile covered come back as `0` intensity and `NaN` phasor
  coordinates, so they are simply absent from the phasor plot.

(blend-modes)=
### Blend modes

The blend mode changes **only the intensity normalisation**. G and S are
identical in all three modes.

| Mode | Intensity in an overlap | Use when |
|------|------------------------|----------|
| **Feather** | Weighted average with a linear cross-fade across the overlap, so a tile only ramps on the sides that actually have a neighbour. | The default. Gives a seamless, uniformly bright image. |
| **Average** | Weighted average with flat per-tile weights. | You want the plain mean without a cross-fade. |
| **Sum counts** | Photon counts are added. Overlaps are brighter, but carry the full statistics of every tile that contributed. | The intensity image is being used quantitatively, or the overlaps are where you need the best photon statistics. |

Because feathering is edge-aware, a tile at the border of the mosaic does not
fade out into nothing on its outer sides.

(binning-on-read)=
## Binning on read

A mosaic's canvas is much larger than any of its tiles, and a large one can
imply an array of many gigabytes. The **Binning** selector sums square blocks
of pixels while each tile is read, before the phasor transform.

Binning by a factor $f$ cuts the memory the mosaic needs by $f^2$. Because
binning **adds photons together**, the phasor of a binned pixel is the
photon-weighted phasor of the pixels it covers — the same rule the overlaps
use — so it costs spatial resolution and nothing else. Signal-to-noise per
pixel improves.

The dialog starts on a factor that keeps the stitched image manageable rather
than on a setting that would try to allocate more memory than the machine has,
and shows the estimated footprint next to the selector.

## What you get

Stitching produces one layer per channel, named after the folder the tiles
came from (or after the file, for a mosaic held in a single file):

```
<name> Mosaic Intensity Image: Channel <n>
```

Its metadata carries everything the rest of the plugin needs
(`original_mean`, `G`, `S`, `G_original`, `S_original`, `harmonics`,
`settings`), plus three entries specific to mosaics:

| Key | Contents |
|-----|----------|
| `tile_files` | The tiles that went into the mosaic, in placement order. |
| `tile_geometry` | The full layout — tile shape, placements, overlap and blend mode — as a dictionary. |
| `tile_coverage` | How many tiles contributed to each pixel, as a `uint16` image. Useful for checking that the layout is right: it should be `1` in tile interiors and `2` or `4` along seams and at corners. |

From this point on the mosaic is an ordinary phasor layer:
{doc}`calibration`, {doc}`filtering_thresholding`, {doc}`phasor_selection`,
{doc}`component_analysis`, {doc}`fret_analysis`, {doc}`phasor_mapping`,
{doc}`histogram_statistics` and {doc}`exporting` all work on it unchanged.

## Troubleshooting

Tile mode reports these as messages rather than tracebacks:

| Message | Cause |
|---------|-------|
| Nothing selected / picker cancelled | No tiles were chosen. |
| Mixed extensions | The selection spans more than one file format. |
| Unsupported extension | The format cannot be read as a tile. |
| A single untiled file | One file was selected that holds only one tile. Use **Select file(s) to be read** instead. |
| Layout resolves to one tile | The rows spec, or the file-name pattern, describes a mosaic of one. |
| Tile count disagrees with recorded positions | The number of files does not match the number of positions in them. |
| Tiles of differing shapes | Every tile in a mosaic must have the same height and width. |
| Non-2D tiles | A tile resolved to something that is not a single plane. |
| Mosaic produced nothing | The mosaic path returned no tiles at all. |

Mixed excitation frequencies across tiles **warn** rather than fail — the
mosaic is still stitched, since a frequency mismatch is often a metadata
problem rather than a real one, but calibration will not be meaningful across
the seam.

If the mosaic looks scrambled rather than merely misaligned, the layout is
wrong, not the overlap: check **Order** (raster vs snake), **Start** corner,
and — for the file-name source — that the pattern is picking up the row and
column you think it is.

## Scripting

The same machinery is importable, so a mosaic can be stitched without the
GUI.

```python
from glob import glob

from napari_phasors._reader import raw_file_tile_reader
from napari_phasors._stitching import layout_from_rows

tiles = sorted(glob("mosaic/tile_*.ptu"))

geometry = layout_from_rows(
    tiles,
    tiles_per_row=[3, 3, 3],
    overlap_y=0.1,
    overlap_x=0.1,
    blend_mode="feather",
)

layer_data = raw_file_tile_reader(
    tiles, geometry, reader_options={"frame": -1}, harmonics=[1, 2]
)
viewer.add_image(layer_data[0][0], **layer_data[0][1])
```

To try several overlaps without re-reading the tiles, read once and stitch
repeatedly:

```python
from napari_phasors._reader import read_tile_phasors

tile_set = read_tile_phasors(tiles, harmonics=[1, 2])
layers = tile_set.stitch(geometry.with_overlap(0.12, 0.12))
```

Other useful entry points:

| Function | Purpose |
|----------|---------|
| `_reader.probe_tile_axes(path)` | Which dimensions of a file could hold tiles. |
| `_reader.czi_mosaic_info(path)` | Tile count, tile shape and canvas shape of a CZI mosaic, read from the sub-block index without touching pixels. |
| `_stitching.layout_from_filenames(paths, pattern)` | Layout from indices in the file names. |
| `_stitching.layout_from_positions(paths, positions, tile_shape)` | Layout from recorded tile positions, kept exact through per-tile corrections. |
| `_stitching.layout_from_stage_positions(paths)` | Layout from stage coordinates stored in the files. |
| `_stitching.estimate_overlap(means, geometry)` | Measure the overlap from the data. |
| `_stitching.blend_phasor_tiles(tiles, geometry)` | Blend `(mean, G, S)` triples directly. |
