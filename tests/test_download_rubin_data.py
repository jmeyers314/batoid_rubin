# tests/test_download_rubin_data.py
#
# Offline tests for the Zenodo downloader. Everything here mocks requests, so
# these never touch the network.
from collections import namedtuple
from hashlib import md5
from pathlib import Path
from unittest.mock import patch
import os

import pytest
import requests

import batoid_rubin.utils
from batoid_rubin.data.download_rubin_data import (
    ChecksumMismatch,
    _retryable,
    _with_retries,
    download_rubin_data,
)


Args = namedtuple("Args", ["dataset", "outdir", "retries"])


class FakeResponse:
    """Stand-in for a requests.Response."""

    def __init__(self, payload=None, content=b"", status_code=200):
        self._payload = payload
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code}", response=self
            )

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]


def make_record(files):
    """Build a Zenodo-shaped record payload from {name: content}."""
    return {
        "files": [
            {
                "key": name,
                "size": len(content),
                "checksum": "md5:" + md5(content).hexdigest(),
                "links": {"self": f"https://example.invalid/{name}"},
            }
            for name, content in files.items()
        ]
    }


def fake_get(files, corrupt=(), fail=()):
    """requests.get replacement serving a record and its files.

    corrupt : names to serve with content that won't match the checksum
    fail : names to serve as a 500
    """
    record = make_record(files)

    def _get(url, **kwargs):
        name = url.rsplit("/", 1)[-1]
        if name in files:
            if name in fail:
                return FakeResponse(status_code=500)
            content = b"garbage" if name in corrupt else files[name]
            return FakeResponse(content=content)
        return FakeResponse(payload=record)

    return _get


# --- retry helpers ---------------------------------------------------------

def test_retryable():
    """5xx and 429 are worth retrying; other 4xx are not."""
    def http_error(status):
        return requests.exceptions.HTTPError(
            "boom", response=FakeResponse(status_code=status)
        )

    assert not _retryable(http_error(404))
    assert not _retryable(http_error(403))
    assert _retryable(http_error(429))
    assert _retryable(http_error(500))
    assert _retryable(http_error(503))
    # Connection-level failures are always transient enough to retry.
    assert _retryable(requests.exceptions.ConnectionError("no route"))


def test_with_retries_succeeds_after_transient_failure():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise requests.exceptions.ConnectionError("boom")
        return "ok"

    assert _with_retries(flaky, retries=3, delay=0) == "ok"
    assert len(calls) == 3


def test_with_retries_reraises_when_exhausted():
    calls = []

    def always():
        calls.append(1)
        raise requests.exceptions.ConnectionError("boom")

    with pytest.raises(requests.exceptions.ConnectionError):
        _with_retries(always, retries=3, delay=0)
    # Initial attempt plus three retries.
    assert len(calls) == 4


def test_with_retries_does_not_retry_client_error():
    """A 404 won't fix itself, so don't waste retries on it."""
    calls = []

    def not_found():
        calls.append(1)
        raise requests.exceptions.HTTPError(
            "404", response=FakeResponse(status_code=404)
        )

    with pytest.raises(requests.exceptions.HTTPError):
        _with_retries(not_found, retries=3, delay=0)
    assert len(calls) == 1


# --- download_rubin_data ---------------------------------------------------

def test_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        download_rubin_data(Args("not_a_dataset", None, 3))


def test_download_writes_all_files(tmp_path):
    files = {"a.fits.gz": b"aaa", "b.yaml": b"bbb" * 5000}
    outdir = tmp_path / "ccd_height_map"

    with patch("requests.get", fake_get(files)):
        download_rubin_data(Args("ccd_height_map", outdir, 3))

    assert sorted(p.name for p in outdir.iterdir()) == ["a.fits.gz", "b.yaml"]
    for name, content in files.items():
        assert (outdir / name).read_bytes() == content


def test_download_leaves_no_staging_dir(tmp_path):
    """The staging directory must not survive a successful download."""
    outdir = tmp_path / "ccd_height_map"

    with patch("requests.get", fake_get({"a.fits.gz": b"aaa"})):
        download_rubin_data(Args("ccd_height_map", outdir, 3))

    assert [p.name for p in tmp_path.iterdir()] == ["ccd_height_map"]


def test_published_dir_is_readable(tmp_path):
    """mkdtemp is 0700; the published directory should follow the umask."""
    outdir = tmp_path / "ccd_height_map"

    with patch("requests.get", fake_get({"a.fits.gz": b"aaa"})):
        download_rubin_data(Args("ccd_height_map", outdir, 3))

    umask = os.umask(0o022)
    os.umask(umask)
    assert outdir.stat().st_mode & 0o777 == 0o777 & ~umask


def test_existing_nonempty_outdir_refused(tmp_path):
    """Never merge into a populated dataset directory."""
    outdir = tmp_path / "ccd_height_map"
    outdir.mkdir()
    (outdir / "mine.txt").write_text("keep me")

    with patch(
        "requests.get", side_effect=fake_get({"a.fits.gz": b"aaa"})
    ) as mock_get:
        with pytest.raises(FileExistsError):
            download_rubin_data(Args("ccd_height_map", outdir, 3))

    # Bailed out before touching the network, and left the file alone.
    mock_get.assert_not_called()
    assert (outdir / "mine.txt").read_text() == "keep me"


def test_existing_empty_outdir_allowed(tmp_path):
    """An empty directory carries no data, so it's safe to replace."""
    outdir = tmp_path / "ccd_height_map"
    outdir.mkdir()

    with patch("requests.get", fake_get({"a.fits.gz": b"aaa"})):
        download_rubin_data(Args("ccd_height_map", outdir, 3))

    assert (outdir / "a.fits.gz").read_bytes() == b"aaa"


def test_outdir_existing_as_file_refused(tmp_path):
    outdir = tmp_path / "ccd_height_map"
    outdir.write_text("not a directory")

    with pytest.raises(FileExistsError):
        download_rubin_data(Args("ccd_height_map", outdir, 3))


def test_checksum_mismatch(tmp_path):
    files = {"a.fits.gz": b"aaa"}
    outdir = tmp_path / "ccd_height_map"

    with patch("requests.get", fake_get(files, corrupt=["a.fits.gz"])):
        with pytest.raises(ChecksumMismatch):
            download_rubin_data(Args("ccd_height_map", outdir, 0))

    # Nothing published, and no staging directory left behind.
    assert not outdir.exists()
    assert list(tmp_path.iterdir()) == []


def test_partial_failure_publishes_nothing(tmp_path):
    """A failure on a later file must not leave a half-populated outdir."""
    files = {"a.fits.gz": b"aaa", "b.fits.gz": b"bbb", "c.fits.gz": b"ccc"}
    outdir = tmp_path / "ccd_height_map"

    with patch("requests.get", fake_get(files, fail=["b.fits.gz"])):
        with pytest.raises(requests.exceptions.HTTPError):
            download_rubin_data(Args("ccd_height_map", outdir, 0))

    assert not outdir.exists()
    assert list(tmp_path.iterdir()) == []


def test_retries_default_when_args_lacks_field(tmp_path):
    """Callers passing an args object without .retries still work."""
    LegacyArgs = namedtuple("LegacyArgs", ["dataset", "outdir"])
    outdir = tmp_path / "ccd_height_map"

    with patch("requests.get", fake_get({"a.fits.gz": b"aaa"})):
        download_rubin_data(LegacyArgs("ccd_height_map", outdir))

    assert (outdir / "a.fits.gz").read_bytes() == b"aaa"


# --- ensure_data_dir -------------------------------------------------------

def test_ensure_data_dir_treats_empty_dir_as_missing(tmp_path, monkeypatch):
    """An interrupted download leaves an empty dir; it should re-download."""
    monkeypatch.setenv("BATOID_RUBIN_DATA_DIR", str(tmp_path))
    (tmp_path / "ccd_height_map").mkdir()

    with patch("requests.get", fake_get({"a.fits.gz": b"aaa"})):
        resolved = batoid_rubin.utils.ensure_data_dir("ccd_height_map")

    assert (resolved / "a.fits.gz").read_bytes() == b"aaa"


def test_ensure_data_dir_keeps_populated_dir(tmp_path, monkeypatch):
    """A populated directory short-circuits without downloading."""
    monkeypatch.setenv("BATOID_RUBIN_DATA_DIR", str(tmp_path))
    existing = tmp_path / "ccd_height_map"
    existing.mkdir()
    (existing / "already_here.fits.gz").write_bytes(b"xxx")

    with patch("requests.get") as mock_get:
        resolved = batoid_rubin.utils.ensure_data_dir("ccd_height_map")

    mock_get.assert_not_called()
    assert resolved == existing


def test_ensure_data_dir_unknown_name(tmp_path, monkeypatch):
    monkeypatch.setenv("BATOID_RUBIN_DATA_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="Cannot infer directory"):
        batoid_rubin.utils.ensure_data_dir("not_a_dataset")


if __name__ == "__main__":
    pytest.main([__file__])
