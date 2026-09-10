#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the runner's own settings, as opposed to what it processes."""

import pytest

from pygac_fdr.runners.run import apply_chunk_size


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
