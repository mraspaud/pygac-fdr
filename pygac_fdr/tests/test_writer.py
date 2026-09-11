#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 pygac-fdr developers
#
# This file is part of pygac-fdr.
#
# pygac-fdr is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pygac-fdr is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pygac-fdr. If not, see <http://www.gnu.org/licenses/>.

import datetime as dt
import os
import unittest

import numpy as np
import pytest
import satpy
import xarray as xr
from pyresample.geometry import SwathDefinition
from satpy.tests.utils import make_dataid

from pygac_fdr.writer import DEFAULT_ENCODING, NetcdfWriter


# As pygac delivers them: scan line number, then fatal error, calibration, earth location,
# and blackbody contamination in channels 3, 4 and 5. Line 1 lacks calibration data;
# line 2 is fatal and has channel 5 contaminated.
SEVEN_COLUMN_QUALITY_FLAGS = [[1, 0, 1, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0, 1]]


class NetcdfWriterTest(unittest.TestCase):
    def test_default_encoding(self):
        bt_range = np.arange(170, 330, 1, dtype="f8")
        refl_range = np.arange(0, 1.5, 0.1, dtype="f8")
        test_data = {
            "reflectance_channel_1": refl_range,
            "reflectance_channel_2": refl_range,
            "brightness_temperature_channel_3": bt_range,
            "reflectance_channel_3a": refl_range,
            "brightness_temperature_channel_3b": bt_range,
            "brightness_temperature_channel_4": bt_range,
            "brightness_temperature_channel_5": bt_range,
        }
        for ch, data in test_data.items():
            enc = DEFAULT_ENCODING[ch]
            offset = enc.get("add_offset", 0.0)
            data_enc = ((data - offset) / enc["scale_factor"]).astype(enc["dtype"])
            data_dec = data_enc * enc["scale_factor"] + offset
            np.testing.assert_allclose(data_dec, data, rtol=0.1)


class TestNetcdfWriter:
    @pytest.fixture
    def scene_lonlats(self):
        lons = [[5.0, 6.0], [7.0, 8.0]]
        lats = [[1.0, 2.0], [3.0, 4.0]]
        return lons, lats

    @pytest.fixture(params=[True, False])
    def with_orbital_parameters(self, request):
        return request.param

    @pytest.fixture
    def scene_dataset_attrs(self, scene_lonlats, with_orbital_parameters):
        lons, lats = scene_lonlats
        attrs = {
            "platform_name": "noaa15",
            "sensor": "avhrr-1",
            "sun_earth_distance_correction_factor": 0.9,
            "calib_coeffs_version": "patmos-x 2012",
            "long_name": "my_long_name",
            "units": "my_units",
            "standard_name": "my_standard_name",
            "area": SwathDefinition(lons, lats),
            "gac_header": np.array([(1, 2)], dtype=[("foo", "f4"), ("bar", "i4")]),
            "start_time": dt.datetime(2000, 1, 1),
            "end_time": dt.datetime(2000, 1, 1),
        }
        if with_orbital_parameters:
            attrs["orbital_parameters"] = {"tle": "my_tle"}
        return attrs

    @pytest.fixture
    def scene(self, scene_dataset_attrs, scene_lonlats):
        acq_time = np.array([0, 1], dtype="datetime64[s]")
        lons, lats = scene_lonlats
        scene = satpy.Scene()
        scene.attrs = {"sensor": "avhrr-1"}
        ch4_id = make_dataid(
            name="4",
            resolution=1234.0,
            wavelength="10.8um",
            modifiers=(),
            calibration="brightness_temperature",
        )
        lon_id = make_dataid(name="longitude", resolution=1234.0, modifiers=())
        lat_id = make_dataid(name="latitude", resolution=1234.0, modifiers=())
        qual_flags_id = make_dataid(name="qual_flags", resolution=1234.0, modifiers=())
        scene[ch4_id] = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=("y", "x"),
            coords={
                "acq_time": ("y", acq_time),
            },
            attrs=scene_dataset_attrs,
        )
        scene[lat_id] = xr.DataArray(
            lats,
            dims=("y", "x"),
            coords={
                "acq_time": ("y", acq_time),
            },
        )
        scene[lon_id] = xr.DataArray(
            lons,
            dims=("y", "x"),
            coords={
                "acq_time": ("y", acq_time),
            },
        )
        scene[qual_flags_id] = xr.DataArray(
            SEVEN_COLUMN_QUALITY_FLAGS,
            dims=("y", "num_flags"),
            coords={
                "acq_time": ("y", acq_time),
            },
        )
        return scene

    @pytest.fixture
    def attrs_exp(self, with_orbital_parameters):
        attrs = {
            "author": "Turtles",
            "end_time": "19700101T000001Z",
            "geospatial_lat_max": 4,
            "geospatial_lat_min": 1,
            "geospatial_lat_resolution": "1234.0 meters",
            "geospatial_lat_units": "degrees_north",
            "geospatial_lon_max": 8,
            "geospatial_lon_min": 5,
            "geospatial_lon_resolution": "1234.0 meters",
            "geospatial_lon_units": "degrees_east",
            "instrument": "Earth Remote Sensing Instruments > Passive Remote Sensing > "
            "Spectrometers/Radiometers > Imaging Spectrometers/Radiometers "
            "> AVHRR-1",
            "platform": "Earth Observation Satellites > NOAA POES > NOAA-15",
            "start_time": "19700101T000000Z",
            "sun_earth_distance_correction_factor": 0.9,
            "time_coverage_start": "19981026T005400Z",
            "version_calib_coeffs": "patmos-x 2012",
            "Conventions": "CF-1.8",
            "product_version": "1.2.3",
            "filename": "avhrr_gac_fdr_N15_19700101T000000Z_19700101T000001Z.nc",
            # Ten points per edge, walking the corners of the 2x2 swath clockwise from the first pixel.
            "geospatial_boundary": str([[5.0, 1.0]] * 10 + [[6.0, 2.0]] * 10 + [[8.0, 4.0]] * 10 + [[7.0, 3.0]] * 10),
        }
        if with_orbital_parameters:
            attrs["orbital_parameters_tle"] = "my_tle"
        return attrs

    @pytest.fixture
    def expected(self, attrs_exp):
        acq_time = xr.DataArray(
            np.array([0, 1], dtype="datetime64[s]"),
            dims="y",
            attrs={"standard_name": "time", "axis": "T"},
        )
        latitude = xr.DataArray(
            np.array([[1, 2], [3, 4]]),
            dims=("y", "x"),
            attrs={
                "name": "latitude",
                "standard_name": "latitude",
                "units": "degrees_north",
            },
        )
        longitude = xr.DataArray(
            np.array([[5, 6], [7, 8]]),
            dims=("y", "x"),
            attrs={
                "name": "longitude",
                "standard_name": "longitude",
                "units": "degrees_east",
            },
        )
        y = xr.DataArray(
            np.array([0, 1]),
            dims="y",
            attrs={"long_name": "Line number", "axis": "Y"},
        )
        x = xr.DataArray(
            np.array([0, 1]),
            dims="x",
            attrs={"long_name": "Pixel number", "axis": "X"},
        )
        bt_ch4 = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=("y", "x"),
            coords={
                "acq_time": acq_time,
                "latitude": latitude,
                "longitude": longitude,
                "y": y,
                "x": x,
            },
            attrs={
                "units": "my_units",
                "wavelength": "10.8um",
                "calibration": "brightness_temperature",
                "long_name": "my_long_name",
                "standard_name": "my_standard_name",
                "resolution": 1234.0,
                "modifiers": [],
            },
        )
        qual_flags = xr.DataArray(
            SEVEN_COLUMN_QUALITY_FLAGS,
            dims=("y", "num_flags"),
            coords={
                "acq_time": acq_time,
                "y": y,
            },
            attrs={
                "long_name": "qual_flags",
                "comment": "Seven binary quality flags are provided per "
                "scanline. See the num_flags coordinate for their "
                "meanings.",
            },
        )

        return xr.Dataset(
            {"brightness_temperature_channel_4": bt_ch4, "qual_flags": qual_flags},
            attrs=attrs_exp,
        )

    @pytest.fixture
    def writer(self):
        user_defined_attrs = {
            "Conventions": "CF-1.8",
            "author": "Turtles",
            "product_version": "1.2.3",
        }
        return NetcdfWriter(global_attrs=user_defined_attrs)

    @pytest.fixture
    def output_file(self, writer, scene):
        filename = writer.write(scene)
        yield filename
        os.unlink(filename)

    @staticmethod
    def _file_name_for_product_version(scene, out_dir, product_version):
        """Write the scene with the given product version and return the name the file gets."""
        writer = NetcdfWriter(output_dir=str(out_dir), fname_fmt="avhrr_fdr_{version_int:04d}.nc",
                              global_attrs={"Conventions": "CF-1.8", "product_version": product_version})
        return os.path.basename(writer.write(scene))

    def test_a_two_number_product_version_still_names_the_file(self, scene, tmp_path):
        """A product version such as 1.2 carries no patch number, and the file must still get its name."""
        assert self._file_name_for_product_version(scene, tmp_path, "1.2") == "avhrr_fdr_0120.nc"

    def test_a_three_number_product_version_names_the_file_with_all_three(self, scene, tmp_path):
        """The file name carries major, minor and patch number as one integer, for example 1.2.3 as 0123."""
        assert self._file_name_for_product_version(scene, tmp_path, "1.2.3") == "avhrr_fdr_0123.nc"

    def test_a_two_digit_major_version_takes_the_thousands(self, scene, tmp_path):
        """A major version of 10 or more simply adds digits in front, so 12.3.4 names the file with 1234."""
        assert self._file_name_for_product_version(scene, tmp_path, "12.3.4") == "avhrr_fdr_1234.nc"

    def test_a_minor_version_of_ten_or_more_is_refused(self, scene, tmp_path):
        """1.10.1 would collide with 2.0.1 in the file name, so the writer refuses it instead."""
        with pytest.raises(ValueError, match="Minor/patch versions > 9 are not supported"):
            self._file_name_for_product_version(scene, tmp_path, "1.10.1")

    def test_write(self, output_file, expected):
        with xr.open_dataset(output_file) as written:
            self._drop_dynamic_attrs(written.attrs)
            xr.testing.assert_identical(written, expected)

    def test_navigation_metadata_reaches_the_product(self, scene, writer):
        """Everything the georeferencing recorded has to survive into the file.

        The reader states on each pass what it fitted and how far the result can
        be trusted. Only part of that was forwarded, so a product could not say
        how many control points its navigation rested on, nor whether a clock
        measurement had reached the pass at all.
        """
        recorded = {
            "gcp_count": 431,
            "clock_table_covers_the_pass": True,
            "unexplained_displacement_in_pixels": 1.75,
            "navigation_metadata_schema_version": 3,
            "pre_alignment_applied": True,
        }
        # NetCDF has no boolean type, so the CF writer spells one as a string,
        # the same way the older ``georeferenced`` attribute has always appeared.
        on_file = {name: "true" if value is True else value for name, value in recorded.items()}

        scene["4"].attrs.update(recorded)
        filename = writer.write(scene)
        try:
            with xr.open_dataset(filename) as written:
                for name, value in on_file.items():
                    assert written.attrs[name] == value
        finally:
            os.unlink(filename)

    def test_the_navigation_record_reaches_the_product(self, scene, writer):
        """The one record the reader gathers travels as a whole, not name by name.

        Each hop between the reader and the file forwards attributes by name and
        keeps its own list of the names it knows, so a fact added at one end is
        dropped in silence unless every list along the way is edited. That is how
        the control-point count came to be missing from products while sitting on
        the dataset all along. Forwarding the record itself makes the lists a
        one-time cost: what it holds can grow without any hop being touched again.

        The CF writer flattens a mapping into ``parent_child`` names, which is how
        ``orbital_parameters`` has always appeared on these files.
        """
        scene["4"].attrs["navigation"] = {"gcp_count": 431, "schema_version": 3}

        filename = writer.write(scene)
        try:
            with xr.open_dataset(filename) as written:
                assert written.attrs["navigation_gcp_count"] == 431
        finally:
            os.unlink(filename)

    def _drop_dynamic_attrs(self, attrs):
        for drop_attrs in [
            "history",
            "date_created",
            "version_satpy",
            "version_pygac",
            "version_pygac_fdr",
        ]:
            attrs.pop(drop_attrs)
