#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for what the reader learns from a level 1b file name."""

from pygac_fdr.reader import attrs_from_filename


def test_a_noaa_file_name_gives_the_orbits_and_the_receiving_station():
    """The level 1b data records carry no orbit number; only the file name does.

    NOAA names give the start orbit in five digits, followed by the last two
    digits of the end orbit.
    """
    attrs = attrs_from_filename("NSS.GHRR.NL.D02187.S1904.E2058.B1831920.WI")
    assert attrs == {"orbit_number_start": 18319, "orbit_number_end": 18320, "ground_station": "WI"}


def test_an_esa_archive_name_gives_the_start_orbit_and_the_station_but_no_end_orbit():
    """ESA's archive renames each file and widens the start orbit to six digits.

    The two digits after the start orbit do not belong to the end orbit: the
    archive header of ESR.LHRR.M1.D16087.S2023.E2037.B01828628.BN keeps the
    original name, which shows the pass starting and ending in orbit 18286. An
    end orbit computed from them would be 18228, before the pass began, so
    none is given at all.
    """
    attrs = attrs_from_filename("ESR.LHRR.M1.D16087.S2023.E2037.B01828628.BN")
    assert attrs == {"orbit_number_start": 18286, "ground_station": "BN"}
