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
