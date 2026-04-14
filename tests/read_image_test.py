"""Tests for image-reading support in the Read tool.

The Read tool must:
- Return a `ToolResult` with base64-encoded image data for supported image
  formats (PNG/JPEG/WebP/GIF) that fit within the payload limit.
- Compress oversized images to stay under `MAX_IMAGE_PAYLOAD_BYTES` and
  still return a `ToolResult`.
- Fall back to a text-only metadata string for unsupported image formats
  (BMP, TIFF, ...) and for non-image binary files.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from scider.tools.base import ToolContext, ToolResult
from scider.tools.fs.read_file import (
    MAX_IMAGE_PAYLOAD_BYTES,
    ReadFileTool,
    _compress_image_to_limit,
)


def _ctx() -> ToolContext:
    return ToolContext(agent_name="test")


def _write_png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    img = Image.new("RGB", size, color)
    img.save(path, format="PNG")


def _write_jpeg(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    img = Image.new("RGB", size, color)
    img.save(path, format="JPEG", quality=95)


class TestImageRead:
    def test_small_png_returns_tool_result_with_base64(self, tmp_path: Path):
        p = tmp_path / "tiny.png"
        _write_png(p, (16, 16), (255, 0, 0))

        tool = ReadFileTool()
        result = tool.call(_ctx(), file_path=str(p))

        assert isinstance(result, ToolResult)
        assert len(result.images) == 1
        img = result.images[0]
        assert img.media_type == "image/png"
        # base64 must be decodable and non-empty
        decoded = base64.b64decode(img.data)
        assert len(decoded) > 0
        # Text block must carry metadata the model can cite
        assert "[File:" in result.text
        assert "[Dimensions:" in result.text
        assert "[Image attached for model viewing]" in result.text

    def test_small_jpeg_keeps_jpeg_media_type(self, tmp_path: Path):
        p = tmp_path / "tiny.jpg"
        _write_jpeg(p, (32, 32), (0, 128, 0))

        tool = ReadFileTool()
        result = tool.call(_ctx(), file_path=str(p))

        assert isinstance(result, ToolResult)
        assert result.images[0].media_type == "image/jpeg"

    def test_unsupported_image_falls_back_to_text(self, tmp_path: Path):
        # BMP is in IMAGE_EXTENSIONS but NOT in SUPPORTED_IMAGE_MIME.
        p = tmp_path / "tiny.bmp"
        Image.new("RGB", (8, 8), (128, 128, 128)).save(p, format="BMP")

        tool = ReadFileTool()
        result = tool.call(_ctx(), file_path=str(p))

        assert isinstance(result, str)
        assert "[File:" in result
        assert "Format not supported for model viewing" in result

    def test_non_image_binary_still_returns_text(self, tmp_path: Path):
        p = tmp_path / "blob.bin"
        p.write_bytes(b"\x00\x01\x02\x03" * 256)

        tool = ReadFileTool()
        result = tool.call(_ctx(), file_path=str(p))

        assert isinstance(result, str)
        assert "[File:" in result
        assert "Binary file" in result

    def test_oversized_png_gets_compressed(self, tmp_path: Path):
        # Build a PNG large enough that it must be compressed. Random noise
        # gives poor PNG compression so the raw file exceeds 5 MB.
        import os

        import numpy as np

        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, (2500, 2500, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        p = tmp_path / "huge.png"
        img.save(p, format="PNG")
        assert os.path.getsize(p) > MAX_IMAGE_PAYLOAD_BYTES, "fixture not large enough"

        tool = ReadFileTool()
        result = tool.call(_ctx(), file_path=str(p))

        assert isinstance(result, ToolResult)
        assert len(result.images) == 1
        # After compression the decoded payload must be under the limit.
        decoded = base64.b64decode(result.images[0].data)
        assert len(decoded) <= MAX_IMAGE_PAYLOAD_BYTES
        # Compression collapses onto JPEG regardless of source format.
        assert result.images[0].media_type == "image/jpeg"
        assert "Compressed from" in result.text


class TestCompressHelper:
    def test_compress_returns_none_for_garbage(self):
        # Random bytes are not a decodable image — should fail gracefully.
        assert _compress_image_to_limit(b"\x00\x01not-an-image", 1024) is None

    def test_compress_small_image_is_noop_path(self):
        # A tiny image is already under any reasonable limit, but the
        # helper still re-encodes it to JPEG and returns a valid result.
        img = Image.new("RGB", (64, 64), (10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out = _compress_image_to_limit(buf.getvalue(), 10 * 1024 * 1024)
        assert out is not None
        data, media_type = out
        assert media_type == "image/jpeg"
        # Re-decode to verify we emitted a real JPEG.
        decoded = Image.open(io.BytesIO(data))
        assert decoded.size == (64, 64)
