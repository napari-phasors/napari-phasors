"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""

from __future__ import annotations

import pooch
from napari_phasors._reader import napari_get_reader


def convallaria_FLIM_sample_data():
    """Fetch Convallaria image and Calibration"""
    data = pooch.retrieve('https://zenodo.org/records/14026720/files/Convallaria_$EI0S.fbd?download=1', known_hash='3751891b02e3095fedd53a09688d8a22ff2a0083544dd5c0726b9267d11df1bc')
    calibration_data = pooch.retrieve('https://zenodo.org/records/14026720/files/Calibration_Rhodamine110_$EI0S.fbd?download=1', known_hash='d745cbcdd4a10dbaed83ee9f1b150f0c7ddd313031e18233293582cdf10e4691')
    reader_options = {'channel': 0}
    reader = napari_get_reader(data, reader_options=reader_options)
    return [reader(data)[0], reader(calibration_data)[0]]
