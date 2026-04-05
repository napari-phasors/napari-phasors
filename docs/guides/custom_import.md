# Phasor Custom Import Widget

The **Phasor Custom Import** widget (**PhasorTransformWidget**) provides
format-aware import options for FLIM and hyperspectral files.

Open it from:

**Plugins -> napari-phasors -> Phasor Transform**

## Default opening vs custom import

Standard opening methods in napari:

- drag-and-drop files into the viewer
- **File -> Open File(s)**

These methods open files with default reader parameters for the detected file
format.

By default, standard opening reads the first two harmonics (when available).

Use the **Phasor Custom Import** widget when you need to override those
defaults (for example, choose a specific channel/frame, set LIF image/dimension,
or build a custom 3D stack with z spacing and axis order).

### Default reader parameters by format

The defaults below come from the format mapping in
`src/napari_phasors/_reader.py`.

| Format | Default parameters used by standard opening |
|------|-------------|
| `.ptu` | `frame=-1`, `keepdims=False` |
| `.fbd` | `frame=-1`, `keepdims=False`, `channel=None` |
| `.sdt` | no extra defaults |
| `.lsm` | no extra defaults |
| `.tif`, `.tiff` | no extra defaults |
| `.czi` | no extra defaults |
| `.flif` | no extra defaults |
| `.bh`, `.b&h` | no extra defaults |
| `.bhz` | no extra defaults |
| `.lif` (raw) | `image=None`, `dim="λ"` |
| `.bin` | no extra defaults |
| `.json` (raw) | `channel=0`, `dtype=None` |
| `.ome.tif`, `.ome.tiff` | no extra defaults |
| `.r64`, `.ref` | no extra defaults |
| `.ifli` | `channel=0` |
| `.lif` (processed) | `image=None` |
| `.json` (processed) | `channel=0` |

Notes:

- `.lif` and `.json` are ambiguous extensions (raw or processed), so the
   reader tries raw first and then processed if needed.
- If you need behavior different from these defaults, use the custom import
   widget.
- This includes selecting harmonics other than the default first two (when
   available).

## What this widget does

- Detects file format and shows relevant import options
- Previews signal data before transformation
- Supports importing one file, multiple separate files, or a stacked 3D volume

## Example 1: Import one file or multiple separate files

Use this when you want each selected file to become its own layer.

1. Click **Select file(s) to be read**.
2. Choose one or more supported files.
3. Adjust per-format options (for example channel, frame, harmonics) in the
   widget panel.
4. Click **Phasor Transform**.

If you select multiple files, the widget groups files by extension and applies
the chosen settings per group. The files are imported as separate layers, not
as a stack.

## Example 2: Create a 3D stack from multiple files

Use this when each file should be treated as one slice of a 3D volume.

1. Click **Open 3D stack**.
2. Select files that all share the same extension.
3. In the reorder dialog:
   - reorder files if needed,
   - set **Z spacing (um)**,
   - optionally set axis order/labels.
4. Confirm and click **Phasor Transform**.

The widget stacks files along a new first axis and creates 3D output layers.

## Notes

- Multi-file stacking requires all selected files to have the same extension.
- For supported formats, see {doc}`flim_workflow` and
  {doc}`hyperspectral_workflow`.
