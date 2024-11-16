"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""

from __future__ import annotations

import pooch
from napari_phasors._reader import napari_get_reader
from napari.utils.notifications import show_info


def convallaria_FLIM_sample_data():
    """Fetch Convallaria image and Calibration"""
    downloader = pooch.create(
        path=pooch.os_cache("napari-phasors"),
        base_url="https://zenodo.org/records/14026720/files/",
        registry={
            "Convallaria_$EI0S.fbd": "3751891b02e3095fedd53a09688d8a22ff2a0083544dd5c0726b9267d11df1bc",
            "Calibration_Rhodamine110_$EI0S.fbd": "d745cbcdd4a10dbaed83ee9f1b150f0c7ddd313031e18233293582cdf10e4691",
        },
    )

    # Show a message when downloading starts
    show_info("Gathering sample files, please wait...")

    # Download the files with a progress bar
    data = downloader.fetch("Convallaria_$EI0S.fbd", progressbar=True)
    calibration_data = downloader.fetch("Calibration_Rhodamine110_$EI0S.fbd", progressbar=True)
    
    reader_options = {'channel': 0}
    reader = napari_get_reader(data, reader_options=reader_options)
    return [reader(data)[0], reader(calibration_data)[0]]
