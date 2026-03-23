"""Tests for the 5 new R2AuditLog methods via MockR2."""

import os
import tempfile


def test_upload_download_file(mock_r2):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(b"binary content here")
        src = f.name

    try:
        assert mock_r2.upload_file_from_disk(src, "test/file.bin")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            dst = f.name
        assert mock_r2.download_file_to_disk("test/file.bin", dst)
        with open(dst, "rb") as f:
            assert f.read() == b"binary content here"
        os.unlink(dst)
    finally:
        os.unlink(src)


def test_upload_download_text(mock_r2):
    assert mock_r2.upload_text("test/hello.txt", "hello world")
    result = mock_r2.download_text("test/hello.txt")
    assert result == "hello world"


def test_download_text_missing(mock_r2):
    assert mock_r2.download_text("nonexistent") is None


def test_key_exists(mock_r2):
    assert not mock_r2.key_exists("test/missing")
    mock_r2.upload_text("test/present", "data")
    assert mock_r2.key_exists("test/present")
