#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for handing a pass to PPS alongside the FDR."""

import datetime as dt

import numpy as np
import satpy
import xarray as xr

from pygac_fdr.pps import scene_for_pps, write_pps_file

SCANLINE_TIMES = np.array(["2009-07-01T12:16", "2009-07-01T12:27"], dtype="datetime64[ms]")


def _swath_array(name, values, dims=("y", "x"), **attrs):
    """An array along the scanlines, shaped as the avhrr_l1b_gaclac reader delivers it."""
    return xr.DataArray(np.array(values), dims=dims, coords={"acq_time": ("y", SCANLINE_TIMES)},
                        attrs={"name": name, **attrs})


def _channel(name, values, wavelength):
    """A channel carrying what the avhrr_l1b_gaclac reader gives every channel."""
    return _swath_array(name, values, wavelength=wavelength, platform_name="noaa19", sensor="avhrr-3",
                        start_time=dt.datetime(2009, 7, 1, 12, 16), end_time=dt.datetime(2009, 7, 1, 12, 27))


def _scene_pps_can_convert():
    """The smallest reader-shaped scene the PPS conversion accepts."""
    scene = satpy.Scene()
    scene["1"] = _channel("1", [[10.0, 11.0], [12.0, 13.0]], [0.58, 0.63, 0.68, "um"])
    scene["2"] = _channel("2", [[20.0, 21.0], [22.0, 23.0]], [0.725, 0.8625, 1.0, "um"])
    scene["4"] = _channel("4", [[280.0, 281.0], [282.0, 283.0]], [10.3, 10.8, 11.3, "um"])
    scene["latitude"] = _swath_array("latitude", [[60.0, 61.0], [62.0, 63.0]])
    scene["longitude"] = _swath_array("longitude", [[15.0, 16.0], [17.0, 18.0]])
    for name in ("solar_zenith_angle", "sensor_zenith_angle", "sun_sensor_azimuth_difference_angle"):
        scene[name] = _swath_array(name, [[10.0, 20.0], [30.0, 40.0]])
    scene["qual_flags"] = _swath_array("qual_flags", [[1, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0]],
                                       dims=("y", "num_flags"))
    return scene


def _scene_with_channel_4():
    """A scene holding one channel, shaped as the avhrr_l1b_gaclac reader delivers it."""
    scene = satpy.Scene()
    scene["4"] = _swath_array("4", [[280.0, 281.0], [282.0, 283.0]])
    return scene


def test_the_pps_copy_shares_the_pixels_but_owns_its_metadata():
    """PPS renames and re-tags every channel, and the FDR writer must not see any of that.

    The conversion to PPS works on the scene in place, so it needs a scene of
    its own. Copying the pixels as well would double the memory of a full
    pass for nothing, because neither writer changes them.
    """
    scene = _scene_with_channel_4()
    copy = scene_for_pps(scene)
    copy["4"].attrs["name"] = "image3"
    assert (copy["4"].data is scene["4"].data, scene["4"].attrs["name"]) == (True, "4")


def test_the_pps_copy_leaves_out_what_pps_never_reads():
    """Terrain-corrected geolocation, control points and uncertainties are FDR products, not PPS input.

    Handed to the PPS conversion, they would be written into the PPS file as they are,
    the per-pixel uncertainty cube included.
    """
    scene = _scene_with_channel_4()
    for name in ("tc_latitude", "gcp_x", "random_uncertainty"):
        scene[name] = xr.DataArray(np.zeros(2), dims=("y",), attrs={"name": name})
    assert sorted(dataset_id["name"] for dataset_id in scene_for_pps(scene).keys()) == ["4"]


def test_writing_for_pps_puts_one_pps_file_in_the_output_directory(tmp_path):
    """PPS finds a pass by its file name in the directory it watches, so that is what arrives there.

    The conversion itself belongs to level1c4pps; what pygac-fdr owes PPS is
    one file per pass, under the name PPS looks for.
    """
    write_pps_file(_scene_pps_can_convert(), tmp_path)
    assert [path.name for path in tmp_path.iterdir()] == [
        "S_NWC_avhrr_noaa19_00000_20090701T1216000Z_20090701T1227000Z.nc"]
