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

"""

from __future__ import annotations

import pooch

from napari_phasors._reader import napari_get_reader


def convallaria_FLIM_sample_data():
    """Fetch Convallaria image and Calibration"""
    downloader = pooch.create(
        path=pooch.os_cache("napari-phasors"),
        base_url="https://zenodo.org/records/14026720/files/",
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

    data = downloader.fetch("Convallaria_$EI0S.fbd", progressbar=True)
    calibration_data = downloader.fetch(
        "Calibration_Rhodamine110_$EI0S.fbd", progressbar=True
    )

    reader_options = {'channel': 0}
    reader = napari_get_reader(data, reader_options=reader_options)
    return [reader(data)[0], reader(calibration_data)[0]]


def embryo_FLIM_sample_data():
    """Fetch FLUTE's FLIM Embryo image and Calibration"""
    downloader = pooch.create(
        path=pooch.os_cache("napari-phasors"),
        base_url="doi:10.5281/zenodo.8046636",
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

    data = downloader.fetch("Embryo.tif", progressbar=True)
    calibration_data = downloader.fetch(
        "Fluorescein_Embryo.tif", progressbar=True
    )
    reader = napari_get_reader(data)
    return [reader(data)[0], reader(calibration_data)[0]]


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

    data = downloader.fetch("paramecium.lsm", progressbar=True)
    reader = napari_get_reader(data)
    return reader(data)
