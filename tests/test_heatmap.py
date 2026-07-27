"""Unit tests for the shared reference-relative / denoise transform.

`reference_relative_signal` is the recenter → inferCNV-denoise transform that
both the heatmap draws and `run --denoise-outputs` writes. These tests pin its
two guarantees: the reference (normal) noise floor collapses toward 0, while a
strong, coherent CNV event survives.
"""

import numpy as np
import pandas as pd
import pytest

from kopya.heatmap import reference_relative_signal


def _synthetic(seed=0):
    """25 normal + 15 tumor cells over 20 segments (2 chromosomes).

    Normals sit on a positive pedestal with small gene-to-gene noise; tumor cells
    carry a strong +0.4 gain on the first 4 segments (a minority, so per-cell
    median removal leaves it intact). tumor_score ranks tumor cells high.
    """
    rng = np.random.default_rng(seed)
    n_seg = 20
    n_norm, n_tum = 25, 15
    n_cells = n_norm + n_tum

    cn = 0.5 + rng.normal(0.0, 0.02, size=(n_cells, n_seg)).astype("float32")
    cn[n_norm:, 0:4] += 0.4  # planted gain on 4/20 segments for tumor cells

    cls = np.array(["normal"] * n_norm + ["tumor"] * n_tum)
    low_complexity = np.zeros(n_cells, dtype=bool)
    tumor_score = np.concatenate([
        rng.normal(2.0, 0.4, n_norm),
        rng.normal(20.0, 2.0, n_tum),
    ])
    segments = pd.DataFrame({
        "chr": ["chr1"] * 10 + ["chr2"] * 10,
        "n_genes": [10] * n_seg,
    })
    return cn, cls, low_complexity, tumor_score, segments, n_norm


def test_reference_relative_signal_collapses_floor_keeps_event():
    cn, cls, low_c, score, segments, n_norm = _synthetic()

    rc, is_ref, thr = reference_relative_signal(cn, cls, low_c, score, segments,
                                                sd_amplifier=1.0)

    assert rc.shape == cn.shape
    # Reference is a subset of the normal cells (below the score quantile).
    assert is_ref.sum() > 0
    assert is_ref[n_norm:].sum() == 0          # no tumor cell in the reference
    assert thr is not None

    normal_mag = np.abs(rc[:n_norm]).mean()
    tumor_event = rc[n_norm:, 0:4].mean()      # signed: should be a clear gain
    normal_event = rc[:n_norm, 0:4].mean()

    # Floor collapses: normal cells sit ~at 0 after denoise.
    assert normal_mag < 0.02
    # The planted gain survives, with the right sign and magnitude.
    assert tumor_event > 0.2
    # Normal cells show no event on those same segments.
    assert abs(normal_event) < 0.05
    # And tumor cells are clearly separated from normals in total signal.
    assert np.abs(rc[n_norm:]).sum(axis=1).mean() > 5 * np.abs(rc[:n_norm]).sum(axis=1).mean()


def test_reference_relative_signal_sd_amplifier_zero_recenters_only():
    """sd_amplifier=0 disables denoise: normals are centered but the floor is not collapsed."""
    cn, cls, low_c, score, segments, n_norm = _synthetic()

    rc0, _, _ = reference_relative_signal(cn, cls, low_c, score, segments, sd_amplifier=0.0)
    rc1, _, _ = reference_relative_signal(cn, cls, low_c, score, segments, sd_amplifier=1.0)

    # Recenter-only leaves the noise floor in place; denoise collapses it further.
    assert np.abs(rc0[:n_norm]).mean() >= np.abs(rc1[:n_norm]).mean()
    # The event survives either way.
    assert rc0[n_norm:, 0:4].mean() > 0.2


def test_reference_relative_signal_requires_reference():
    """No confident-diploid reference (all tumor) → a clear ValueError, not a crash."""
    cn, cls, low_c, score, segments, _ = _synthetic()
    cls = np.array(["tumor"] * len(cls))

    with pytest.raises(ValueError, match="no confident-diploid reference"):
        reference_relative_signal(cn, cls, low_c, score, segments)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_reference_relative_signal_rejects_nonfinite_sd(bad):
    """A non-finite sd_amplifier must raise, not silently poison the matrix."""
    cn, cls, low_c, score, segments, _ = _synthetic()
    with pytest.raises(ValueError, match="sd_amplifier must be finite"):
        reference_relative_signal(cn, cls, low_c, score, segments, sd_amplifier=bad)
