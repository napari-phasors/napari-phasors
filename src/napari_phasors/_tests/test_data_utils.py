"""Shared utilities for test data management."""

import pooch
from phasorpy.datasets import fetch

# Create a pooch downloader for test data
test_data_downloader = pooch.create(
    path=pooch.os_cache("napari-phasors-test-data"),
    base_url="https://github.com/napari-phasors/napari-phasors-data/raw/main/test_files/",
    registry={
        'test_file.ptu': (
            'sha256:'
            'a1a8b5d22fc5d88dae5f0e73ce597e3af3f64d15e1f9f6e9fcc44f6244f92614'
        ),
        'test_file$EI0S.fbd': (
            'sha256:'
            '3751891b02e3095fedd53a09688d8a22ff2a0083544dd5c0726b9267d11df1bc'
        ),
        'test_file.lsm': (
            'sha256:'
            '4e5d1ef62b82a5c7eb1460583b229a8ca47bbe3a55b15bfe410d81898297a425'
        ),
        'test_file.ome.tif': (
            'sha256:'
            '2db2bca351ff51d136a3db1613b7f826519a2f1752b566ce0060db0e7d6c9ef9'
        ),
    },
)


def get_test_file_path(filename):
    """Get path to test file, downloading if necessary."""
    if filename == "seminal_receptacle_FLIM_single_image.sdt":
        # Special case for SDT file from phasorpy
        return fetch(filename)
    else:
        # Use pooch for other test files
        return test_data_downloader.fetch(filename, progressbar=True)
