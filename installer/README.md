# Installer assets

This directory contains scripts and assets used by
[constructor](https://github.com/conda/constructor) to build
standalone installers for napari-phasors.

## Adding an installer icon

To set a custom icon for the installer, place an `icon.png` file in
this directory. The build workflow will detect it automatically and
pass it to constructor via the `icon_image` field.

### Requirements

- **Format:** PNG (constructor expects PNG for `icon_image`).
- **Recommended size:** 256 × 256 pixels or larger.
- If you only have a Windows `.ico` file, convert it to PNG first
  (e.g. with ImageMagick: `magick icon.ico icon.png`).
  You may keep both files in this directory; only `icon.png` is
  referenced by the build.

### Windows shortcuts

The `post_install.bat` script will also pick up `icon.ico` (if
present in the install prefix) to set the icon on desktop and
Start Menu shortcuts. To ship the `.ico`, add it here and copy it
in the post-install script or embed it via constructor's
`extra_files`.
