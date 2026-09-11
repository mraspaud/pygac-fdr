#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for handing a pass to PPS alongside the FDR."""

import numpy as np
import satpy
import xarray as xr

from pygac_fdr.pps import scene_for_pps


def _scene_with_channel_4():
    """A scene holding one channel, shaped as the avhrr_l1b_gaclac reader delivers it."""
    scene = satpy.Scene()
    scene["4"] = xr.DataArray(
        np.array([[280.0, 281.0], [282.0, 283.0]]),
        dims=("y", "x"),
        coords={"acq_time": ("y", np.array([0, 1], dtype="datetime64[s]"))},
        attrs={"name": "4"},
    )
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
