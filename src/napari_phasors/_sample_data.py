"""
This module provides functions to fetch sample data for demonstration purposes.

The functions are:
    - `convallaria_FLIM_sample_data`: Convallaria FLIM image and calibration
      image consisting of Rhodamine110 solution. Files are in FBD format.
    - `embryo_FLIM_sample_data`: FLIM Embryo image and calibration image
      consisting of Fluorescein solution. Both files are from the FLUTE
      dataset. Files are in TIFF format.
    - `paramecium_HSI_sample_data`: Paramecium Hyperspectral image in LSM
      format.
    - `fret_FLIM_sample_data`: FLIM-FRET phasor analysis training dataset of
      live HeLa cells (donor only, background autofluorescence and two FRET
      constructs). Files are calibrated OME-TIFFs hosted on Zenodo
      (https://doi.org/10.5281/zenodo.22261325).

"""

from __future__ import annotations

import pooch

from napari_phasors._reader import napari_get_reader
from napari_phasors._utils import show_activity_progress


def convallaria_FLIM_sample_data():
    """Fetch Convallaria image and Calibration"""
    downloader = pooch.create(
        path=pooch.os_cache("napari-phasors"),
        base_url="https://github.com/napari-phasors/napari-phasors-data/raw/main/sample_data/",
        registry={
            'Convallaria_$EI0S.fbd': (
                'sha256:'
                '3751891b02e3095fedd53a09688d8a22ff2a0083544dd5c0726b9267d11df1bc'
            ),
            'Calibration_Rhodamine110_$EI0S.fbd': (
                'sha256:'
                'd745cbcdd4a10dbaed83ee9f1b150f0c7ddd313031e18233293582cdf10e4691'
            ),
        },
    )

    pbr = show_activity_progress(
        desc="Downloading Convallaria sample data...", total=4
    )
    try:
        pbr.set_description("Downloading Convallaria image...")
        data = downloader.fetch("Convallaria_$EI0S.fbd", progressbar=True)
        pbr.update(1)
        pbr.set_description("Downloading calibration data...")
        calibration_data = downloader.fetch(
            "Calibration_Rhodamine110_$EI0S.fbd", progressbar=True
        )
        pbr.update(1)

        reader_options = {'channel': 0}
        reader = napari_get_reader(data, reader_options=reader_options)
        pbr.set_description("Reading Convallaria image...")
        result_data = reader(data)[0]
        pbr.update(1)
        pbr.set_description("Reading calibration data...")
        result_calibration = reader(calibration_data)[0]
        pbr.update(1)
    finally:
        pbr.close()
    return [result_data, result_calibration]


def embryo_FLIM_sample_data():
    """Fetch FLUTE's FLIM Embryo image and Calibration"""
    downloader = pooch.create(
        path=pooch.os_cache("napari-phasors"),
        base_url="https://github.com/napari-phasors/napari-phasors-data/raw/main/sample_data/",
        registry={
            'Embryo.tif': (
                'sha256:'
                'd1107de8d0f3da476e90bcb80ddf40231df343ed9f28340c873cf858ca869e20'
            ),
            'Fluorescein_Embryo.tif': (
                'sha256:'
                '53cb66439a6e921aef1aa7f57ef542260c51cdb8fe56a643f80ea88fe2230bc8'
            ),
        },
    )

    pbr = show_activity_progress(
        desc="Downloading Embryo sample data...", total=4
    )
    try:
        pbr.set_description("Downloading Embryo image...")
        data = downloader.fetch("Embryo.tif", progressbar=True)
        pbr.update(1)
        pbr.set_description("Downloading calibration data...")
        calibration_data = downloader.fetch(
            "Fluorescein_Embryo.tif", progressbar=True
        )
        pbr.update(1)
        reader = napari_get_reader(data)
        pbr.set_description("Reading Embryo image...")
        result_data = reader(data)[0]
        pbr.update(1)
        pbr.set_description("Reading calibration data...")
        result_calibration = reader(calibration_data)[0]
        pbr.update(1)
    finally:
        pbr.close()
    return [result_data, result_calibration]


def paramecium_HSI_sample_data():
    """Fetch Paramecium Hyperspectral image"""
    downloader = pooch.create(
        path=pooch.os_cache("napari-phasors"),
        base_url="https://github.com/phasorpy/phasorpy-data/raw/main/tests",
        registry={
            'paramecium.lsm': (
                'sha256:'
                'b3b3b80be244a41352c56390191a50e4010d52e5ca341dc51bd1d7c89f10cedf'
            ),
        },
    )

    pbr = show_activity_progress(
        desc="Downloading Paramecium sample data...", total=2
    )
    try:
        pbr.set_description("Downloading Paramecium image...")
        data = downloader.fetch("paramecium.lsm", progressbar=True)
        pbr.update(1)
        reader = napari_get_reader(data)
        pbr.set_description("Reading Paramecium image...")
        result = reader(data)
        pbr.update(1)
    finally:
        pbr.close()
    return result


def fret_FLIM_sample_data():
    """Fetch the FLIM-FRET training dataset."""
    filenames = [
        'Donor_Only.ome.tif',
        'Background_Autofluorescence.ome.tif',
        'FRET_Construct_1.ome.tif',
        'FRET_Construct_2.ome.tif',
    ]
    downloader = pooch.create(
        path=pooch.os_cache("napari-phasors"),
        base_url="https://zenodo.org/records/22261325/files/",
        registry={
            'Donor_Only.ome.tif': (
                'sha256:'
                'c7bdd5e13e1c80f76445bd13b2c94ed97c721506f9fbe77a45898c340aea8d80'
            ),
            'Background_Autofluorescence.ome.tif': (
                'sha256:'
                'ed79b791ac4dc931ec1629bb420e78fc473498cf78fb9fecdd023c1626106ada'
            ),
            'FRET_Construct_1.ome.tif': (
                'sha256:'
                '0dbf586a5763c5a44134b9a19451ba78d638402fae40f1aced435ab1a0a9d994'
            ),
            'FRET_Construct_2.ome.tif': (
                'sha256:'
                '6438ad4a2108c1fa1d573131c5a514f61122300b1f56fdd3700e4b9e8223c746'
            ),
        },
    )

    pbr = show_activity_progress(
        desc="Downloading FRET sample data...", total=2 * len(filenames)
    )
    try:
        paths = []
        for filename in filenames:
            pbr.set_description(f"Downloading {filename}...")
            paths.append(downloader.fetch(filename, progressbar=True))
            pbr.update(1)

        results = []
        for filename, path in zip(filenames, paths, strict=True):
            reader = napari_get_reader(path)
            pbr.set_description(f"Reading {filename}...")
            results.append(reader(path)[0])
            pbr.update(1)
    finally:
        pbr.close()
    return results
