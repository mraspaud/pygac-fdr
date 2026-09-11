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

import argparse
import logging
import os
import re
import tarfile
from contextlib import closing, suppress

from satpy.readers import FSFile

from pygac_fdr.config import read_config
from pygac_fdr.pps import scene_for_pps, write_pps_file
from pygac_fdr.reader import read_file
from pygac_fdr.utils import LOGGER_NAME, TarFileSystem, logging_on
from pygac_fdr.writer import NetcdfWriter

LOG = logging.getLogger(LOGGER_NAME)


def apply_chunk_size(controls, environment):
    """Check that the configured chunk size is one that can actually take effect.

    The only thing that reads a chunk size any more is pyresample, and it reads
    the environment variable once, as it is imported. By the time a configuration
    file has been parsed that has long happened, so a size named here cannot be
    made to take effect from here.

    A configuration that names no size takes dask's own default, as a run without
    the setting always has. One that names the size the environment already
    carries is describing what is already true. One that names anything else is
    refused, because the alternative is a run that used a chunk size other than
    the one it was given and never mentioned it.
    """
    wanted = controls.get("pytroll_chunk_size")
    if wanted is None:
        return
    if environment.get("PYTROLL_CHUNK_SIZE") == str(wanted):
        return
    raise ValueError(
        "pytroll_chunk_size is set to {0} in the configuration, but it can only be "
        "applied from the environment: run with PYTROLL_CHUNK_SIZE={0} instead.".format(
            wanted
        )
    )


class HideGzFSFile(FSFile):
    _gz_suffix = re.compile(r"\.gz$")

    def __fspath__(self):
        path = super().__fspath__()
        return self._gz_suffix.sub("", path)


def process_file(filename, config):
    LOG.info("Processing file {}".format(filename))
    try:
        scene = read_file(filename, reader_kwargs=config["controls"].get("reader_kwargs"))
        writer = NetcdfWriter(
            output_dir=config["output"].get("output_dir"),
            global_attrs=config.get("global_attrs"),
            gac_header_attrs=config.get("gac_header_attrs"),
            fname_fmt=config["output"].get("fname_fmt"),
            encoding=config["netcdf"].get("encoding"),
            engine=config["netcdf"].get("engine"),
            debug=config["controls"].get("debug"),
        )

        pps_config = config["output"].get("pps")
        pps_scene = scene_for_pps(scene) if pps_config else None
        writer.write(scene=scene)
        if pps_config:
            write_pps_file(pps_scene, pps_config["output_dir"], orbit_number=scene.attrs.get("orbit_number_start", 0))
        success = True
        if image_config := config["output"].get("image"):
            composite = image_config["composite"]
            scene.load([composite])
            area = image_config["area"]
            scn = scene.resample(area)
            overlay = None
            with suppress(KeyError):
                overlay = {'coast_dir': image_config["coastlines_dir"], 'color': 'red'}
            scn.save_dataset(composite, base_dir=config["output"].get("output_dir"), overlay=overlay)
    except Exception as err:
        if config["controls"]["debug"]:
            raise
        LOG.error("Error processing file {}: {}".format(filename, err))
        success = False
    return success


def process_tarball(tarball, config):
    LOG.info("Processing tarball {}".format(tarball))
    failures = []
    with closing(TarFileSystem(tarball)) as archive:
        filepaths = archive.find("", details=False, withdirs=False)
        for filepath in filepaths:
            filename = HideGzFSFile(filepath, fs=archive)
            success = process_file(filename, config)
            if not success:
                failures.append(filename)
    if failures:
        LOG.error("Could not process the following files: {}".format(failures))
    else:
        LOG.info("Successfully processed all files in {}".format(tarball))

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Read & calibrate AVHRR GAC data and write results to netCDF")
    parser.add_argument("--cfg", required=True, type=str, help="Path to pygac-fdr configuration file.")
    parser.add_argument(
        "--output-dir",
        help="Output directory. Overrides entry in the configuration file.",
    )
    parser.add_argument(
        "--pps-output-dir",
        help="Write a PPS level1c file for every pass into this directory. Overrides entry in the configuration file.",
    )
    parser.add_argument(
        "--tle-dir",
        help="Directory containing TLE files. Overrides entry in the configuration file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode, i.e. exit on first error. Otherwise continue with "
        "next file. Overrides entry in the configuration file.",
    )
    parser.add_argument("--verbose", action="store_true", help="Increase verbosity")
    parser.add_argument("--log-all", action="store_true", help="Enable logging for all modules")
    parser.add_argument("filenames", nargs="+", help="AVHRR GAC level 1b files to be processed")
    parser.add_argument("--georef", type=str, help="Path to reference GeoTiff file for georeferencing")
    parser.add_argument("--dem", type=str, help="Path to digital elevation model GeoTiff file for orthocorrection")
    parser.add_argument("--with-uncertainties", action="store_true", help="Compute uncertainties")
    args = parser.parse_args()
    logging_on(logging.DEBUG if args.verbose else logging.INFO, for_all=args.log_all)

    # Read config file. Convenience: Allow overriding of frequently used arguments from the command
    # line.
    config = read_config(args.cfg)
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    config["output"]["pps"] = {"output_dir": args.pps_output_dir}
    if args.tle_dir:
        config["controls"]["reader_kwargs"]["tle_dir"] = args.tle_dir
    if args.georef:
        config["controls"]["reader_kwargs"]["reference_image"] = args.georef
    if args.dem:
        config["controls"]["reader_kwargs"]["dem"] = args.dem
    if args.with_uncertainties:
        config["controls"]["reader_kwargs"]["compute_uncertainties"] = args.with_uncertainties
    if args.debug:
        config["controls"]["debug"] = args.debug

    # Process files
    apply_chunk_size(config["controls"], os.environ)
    for filename in args.filenames:
        if tarfile.is_tarfile(filename):
            process_tarball(filename, config)
        else:
            process_file(filename, config)
