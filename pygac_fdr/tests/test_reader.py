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
