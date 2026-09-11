#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 pygac-fdr developers
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

"""Hand a pass to PPS alongside the FDR."""

import satpy

PPS_INPUTS = ("4",)


def scene_for_pps(scene):
    """Give PPS a scene of its own, holding only the datasets PPS reads and sharing their pixels."""
    copy = satpy.Scene()
    for dataset_id in scene.keys():
        if dataset_id["name"] in PPS_INPUTS:
            copy[dataset_id] = scene[dataset_id].copy(deep=False)
    return copy


def write_pps_file(scene, output_dir):
    """Write the scene as a PPS level1c file into the output directory."""
    from level1c4pps.lac2pps_lib import process_scene

    return process_scene(scene, out_path=str(output_dir))
