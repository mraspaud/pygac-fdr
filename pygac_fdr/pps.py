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


def scene_for_pps(scene):
    """Give PPS a scene of its own that shares the pixels with the given one."""
    copy = satpy.Scene()
    for dataset_id in scene.keys():
        copy[dataset_id] = scene[dataset_id].copy(deep=False)
    return copy
