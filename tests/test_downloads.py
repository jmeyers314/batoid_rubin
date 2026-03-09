# tests/test_downloads.py
from copy import copy
from pathlib import Path
from unittest.mock import patch

import batoid
import numpy as np

import batoid_rubin


def test_ccd_height_map_dir_lazy():
    """ccd_height_map_dir should not resolve until accessed."""
    fiducial = batoid.Optic.fromYaml("LSST_r.yaml")

    with patch("batoid_rubin.builder.ensure_data_dir", wraps=batoid_rubin.utils.ensure_data_dir) as mock_ensure:
        builder = batoid_rubin.LSSTBuilder(fiducial)
        # __init__ resolved bend and fea, but not ccd
        called_args = [str(c.args[0]) for c in mock_ensure.call_args_list]
        assert not any("ccd" in a for a in called_args)

        mock_ensure.reset_mock()
        _ = builder.ccd_height_map_dir
        assert mock_ensure.call_count == 1
        assert "ccd" in str(mock_ensure.call_args.args[0])

        # Second access is cached
        mock_ensure.reset_mock()
        _ = builder.ccd_height_map_dir
        mock_ensure.assert_not_called()


def test_copy_preserves_resolved_ccd_dir():
    """copy(builder) should carry the already-resolved _ccd_height_map_dir."""
    fiducial = batoid.Optic.fromYaml("LSST_r.yaml")

    with patch("batoid_rubin.builder.ensure_data_dir", wraps=batoid_rubin.utils.ensure_data_dir) as mock_ensure:
        builder = batoid_rubin.LSSTBuilder(fiducial)
        # Resolve ccd_height_map_dir
        _ = builder.ccd_height_map_dir
        mock_ensure.reset_mock()

        builder2 = copy(builder)
        _ = builder2.ccd_height_map_dir
        mock_ensure.assert_not_called()


if __name__ == "__main__":
    test_ccd_height_map_dir_lazy()
    test_copy_preserves_resolved_ccd_dir()