#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the runner: its own settings, and the files it writes for a pass."""

import numpy as np
import pytest
from pyresample.geometry import SwathDefinition

from pygac_fdr.runners import run
from pygac_fdr.runners.run import apply_chunk_size
from pygac_fdr.tests.test_pps import _scene_pps_can_convert


def test_a_chunk_size_that_cannot_be_applied_is_refused():
    """A setting that silently does nothing is worse than one that is missing.

    satpy read the chunk size from a module attribute until 0.56, when the
    attribute went away; assigning it now succeeds, changes nothing, and warns
    nobody. pyresample still reads the environment variable, but only once, when it
    is first imported, which has already happened by the time a configuration file
    is parsed. So a chunk size named in configuration cannot be honoured from here,
    and saying so is the only honest thing left to do: a run that quietly used a
    different chunk size than it was told cost four passes in ninety-three.
    """
    with pytest.raises(ValueError, match="PYTROLL_CHUNK_SIZE"):
        apply_chunk_size({"pytroll_chunk_size": 512}, environment={})


def test_a_chunk_size_already_in_the_environment_is_accepted():
    """When the environment already says it, the setting is honoured and nothing is wrong.

    This is how a chunk size actually reaches pyresample: named before the process
    starts, so that it is there to be read at import. A configuration that agrees
    with the environment is describing what is already true, and a run given both
    should proceed.
    """
    apply_chunk_size({"pytroll_chunk_size": 512}, environment={"PYTROLL_CHUNK_SIZE": "512"})


def test_a_configuration_that_names_no_chunk_size_is_left_alone():
    """Most configurations say nothing about chunking, and that is not an error.

    Saying nothing means taking dask's own default, which is what a run without
    the setting has always done. Only a configuration that asks for something it
    cannot have is worth refusing.
    """
    apply_chunk_size({}, environment={})


def test_a_run_refuses_before_it_processes_anything(tmp_path, monkeypatch):
    """A chunk size that cannot take effect stops the run, not the thousandth pass.

    The check is worth nothing at the end of a campaign. Refusing while the first file
    is still unopened is what makes it a check rather than a postmortem, so it belongs
    before the loop over the files.
    """
    import sys

    from pygac_fdr.runners.run import main

    settings = tmp_path / "pygac-fdr.yaml"
    settings.write_text("controls:\n    pytroll_chunk_size: 1024\noutput:\n    output_dir: .\n")
    monkeypatch.delenv("PYTROLL_CHUNK_SIZE", raising=False)
    monkeypatch.setattr(sys, "argv", ["pygac-fdr-run", "--cfg", str(settings), "no_such_file.l1b"])

    with pytest.raises(ValueError, match="PYTROLL_CHUNK_SIZE"):
        main()


def _read_pass(filename, reader_kwargs=None):
    """Stand in for the level 1b reader: a pass shaped as read_file returns it, enough for the FDR and PPS writers."""
    scene = _scene_pps_can_convert()
    scene.attrs.update(sensor="avhrr-3", orbit_number_start=18286)
    area = SwathDefinition(scene["longitude"].values, scene["latitude"].values)
    for name in ("1", "2", "4"):
        scene[name].attrs.update(area=area, resolution=1050.0, sun_earth_distance_correction_factor=0.9,
                                 calib_coeffs_version="patmos-x 2012",
                                 gac_header=np.array([(1, 2)], dtype=[("foo", "f4"), ("bar", "i4")]))
    return scene


def _config(tmp_path, output):
    """A configuration writing the FDR into tmp_path/fdr, with whatever else the output section names."""
    (tmp_path / "fdr").mkdir()
    return {"controls": {"debug": True, "reader_kwargs": {}}, "netcdf": {},
            "global_attrs": {"Conventions": "CF-1.8", "product_version": "1.2.3"},
            "output": {"output_dir": str(tmp_path / "fdr"), **output}}


def test_a_pass_yields_a_pps_file_when_the_configuration_asks_for_one(tmp_path, monkeypatch):
    """One reading of the level 1b file serves both products, so PPS gets its file from the same run.

    The PPS file is named after the orbit the pass starts in, which the reader
    takes from the level 1b file name.
    """
    monkeypatch.setattr(run, "read_file", _read_pass)
    (tmp_path / "pps").mkdir()
    run.process_file("ESR.LHRR.M1.D16087.S2023.E2037.B01828628.BN",
                     _config(tmp_path, {"pps": {"output_dir": str(tmp_path / "pps")}}))
    assert [path.name for path in (tmp_path / "pps").iterdir()] == [
        "S_NWC_avhrr_noaa19_18286_20090701T1216000Z_20090701T1227000Z.nc"]


def test_a_pass_yields_no_pps_file_when_the_configuration_does_not_ask(tmp_path, monkeypatch):
    """PPS output is opted into, like the quicklook: a configuration without a pps block writes only the FDR."""
    monkeypatch.setattr(run, "read_file", _read_pass)
    config = _config(tmp_path, {})
    assert (run.process_file("ESR.LHRR.M1.D16087.S2023.E2037.B01828628.BN", config),
            [path.name for path in tmp_path.iterdir()]) == (True, ["fdr"])


def test_a_pass_whose_file_name_gives_no_orbit_still_yields_a_pps_file(tmp_path, monkeypatch):
    """The reader leaves out the start orbit when the level 1b file name follows no known pattern.

    Losing the PPS file over that would cost PPS a pass it can still use, so the
    file is written with orbit 00000, as level1c4pps does when no orbit is known.
    """
    def read_pass_without_orbit(filename, reader_kwargs=None):
        scene = _read_pass(filename, reader_kwargs)
        del scene.attrs["orbit_number_start"]
        return scene

    monkeypatch.setattr(run, "read_file", read_pass_without_orbit)
    (tmp_path / "pps").mkdir()
    run.process_file("not_a_level1b_name.l1b", _config(tmp_path, {"pps": {"output_dir": str(tmp_path / "pps")}}))
    assert [path.name for path in (tmp_path / "pps").iterdir()] == [
        "S_NWC_avhrr_noaa19_00000_20090701T1216000Z_20090701T1227000Z.nc"]
